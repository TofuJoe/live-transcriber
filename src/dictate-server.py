#!/usr/bin/env python3
"""Resident dictation daemon with phrase-level streaming.

Keeps a Whisper model warm in RAM and toggles recording on demand, so the
hotkey path never pays model-load cost. Listens on a unix socket for "toggle".

Audio streams continuously from parecord. A rolling Silero VAD pass finds
pauses; each phrase bounded by silence is transcribed and typed while you keep
talking. Committing only at pause boundaries means emitted text is never
revised -- important because we inject keystrokes into apps we don't own, where
"correcting" would mean backspacing over text we can't see.

  DICTATE_MODE=phrase  (default) type each phrase as you finish it
  DICTATE_MODE=single            type everything at once when you stop
"""

import functools
import os
import re
import socket
import subprocess
import sys
import threading

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

MODEL = os.environ.get("DICTATE_MODEL", "small.en")
MODE = os.environ.get("DICTATE_MODE", "phrase")
SOCKET_PATH = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "dictate.sock")
YDOTOOL_SOCKET = "/run/ydotoold/socket"

SR = 16000
SILENCE_MS = 500  # pause that ends a phrase
MAX_SEG_S = 25  # force a cut before Whisper's 30s window
READ_BYTES = 3200  # 100ms of s16le mono @16k

RECORD_CMD = [
    "parecord",
    "--format=s16le",
    "--rate=16000",
    "--channels=1",
    "--raw",
    "--latency-msec=100",  # without this parecord buffers and we get nothing
]

state_lock = threading.Lock()
session = None  # active Session while recording, else None


def notify(title, body="", urgency="normal", replace_id=None, transient=False,
           expire_ms=None):
    """Post a notification, returning its id so we can dismiss it later."""
    cmd = ["notify-send", "--app-name=Dictation", "-p", f"--urgency={urgency}"]
    if replace_id:
        cmd += ["-r", str(replace_id)]
    if transient:  # don't leave a copy behind in the notification tray
        cmd += ["-e"]
    if expire_ms is not None:
        cmd += ["-t", str(expire_ms)]
    cmd += [title, body]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    out = out.strip()
    return int(out) if out.isdigit() else None


def close_notification(nid):
    """Dismiss a notification outright. Critical-urgency banners never expire
    on their own, so the listening indicator must be closed explicitly."""
    if not nid:
        return
    subprocess.run(
        [
            "gdbus", "call", "--session",
            "--dest", "org.freedesktop.Notifications",
            "--object-path", "/org/freedesktop/Notifications",
            "--method", "org.freedesktop.Notifications.CloseNotification",
            str(nid),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


@functools.lru_cache(maxsize=1)
def hotkey_label():
    """Read the actual bound shortcut so notifications can't go stale."""
    path = (
        "/org/gnome/settings-daemon/plugins/media-keys/"
        "custom-keybindings/dictation/"
    )
    try:
        out = subprocess.run(
            [
                "gsettings",
                "get",
                f"org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:{path}",
                "binding",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip().strip("'")
    except Exception:
        return "the hotkey"
    if not out:
        return "the hotkey"
    # '<Super>a' -> 'Super+A'
    parts = re.findall(r"<([^>]+)>", out)
    key = re.sub(r"<[^>]+>", "", out)
    return "+".join(parts + [key.upper()]) if key else "+".join(parts)


def play_cue(sound):
    """Non-blocking audio cue, so recording state is audible without looking."""
    subprocess.Popen(
        ["canberra-gtk-play", "-f", f"/usr/share/sounds/freedesktop/stereo/{sound}.oga"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def type_text(text):
    """Inject text into whatever window has focus, and mirror to clipboard."""
    env = dict(os.environ, YDOTOOL_SOCKET=YDOTOOL_SOCKET)
    # Feed via stdin rather than argv: escaping is off for stdin, so apostrophes
    # and quotes in transcribed text are typed literally instead of interpreted.
    # key-hold defaults to 20ms, which with key-delay meant ~28ms/char -- a
    # 124-char phrase took 3.5s to type. 1/1 measured lossless at 2.2ms/char.
    subprocess.run(
        ["ydotool", "type", "--key-hold", "1", "--key-delay", "1", "--file", "-"],
        input=text.encode(),
        env=env,
        check=False,
    )
    subprocess.run(["wl-copy", "--", text], check=False)


def find_commit_point(audio):
    """Return a sample index to cut at, or None if the phrase is still open.

    A phrase is committable once VAD sees speech followed by SILENCE_MS of
    quiet. If the buffer grows past MAX_SEG_S we cut anyway rather than let it
    exceed the model's context window.
    """
    opts = VadOptions(
        threshold=0.5,
        min_silence_duration_ms=SILENCE_MS,
        speech_pad_ms=200,
    )
    stamps = get_speech_timestamps(audio, opts)
    if not stamps:
        # Pure silence: drop all but a short tail so the buffer can't grow.
        return len(audio) - SR if len(audio) > 3 * SR else None

    end = stamps[-1]["end"]
    if len(audio) - end >= SILENCE_MS / 1000 * SR:
        return min(end + int(0.2 * SR), len(audio))
    if len(audio) >= MAX_SEG_S * SR:
        return end
    return None


class Session:
    """One recording session: reader thread + phrase-committing worker."""

    def __init__(self, model, notif_id=None):
        self.model = model
        self.notif_id = notif_id  # the persistent "Listening…" banner
        self.buf = bytearray()
        self.buf_lock = threading.Lock()
        self.stopping = threading.Event()
        self.spoke = False  # emitted anything yet? controls leading space
        self.proc = subprocess.Popen(RECORD_CMD, stdout=subprocess.PIPE)
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.worker = threading.Thread(target=self._work_loop, daemon=True)
        self.reader.start()
        self.worker.start()

    def _read_loop(self):
        while not self.stopping.is_set():
            chunk = self.proc.stdout.read(READ_BYTES)
            if not chunk:
                break
            with self.buf_lock:
                self.buf.extend(chunk)

    def _take(self, n_samples=None):
        """Pull audio out of the buffer as float32, removing what we took."""
        with self.buf_lock:
            if n_samples is None:
                raw, self.buf = bytes(self.buf), bytearray()
            else:
                nbytes = n_samples * 2
                raw, self.buf = bytes(self.buf[:nbytes]), self.buf[nbytes:]
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _snapshot(self):
        with self.buf_lock:
            raw = bytes(self.buf)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _emit(self, audio):
        if audio.size < 0.3 * SR:  # too short to be speech
            return
        # condition_on_previous_text=False + repetition_penalty guard against
        # Whisper's repetition-loop failure mode on unclear or repetitive audio.
        segments, _ = self.model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            repetition_penalty=1.1,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return
        type_text((" " if self.spoke else "") + text)
        self.spoke = True

    def _work_loop(self):
        if MODE == "single":
            self.stopping.wait()
            return
        while not self.stopping.wait(0.4):
            audio = self._snapshot()
            if audio.size < SR:
                continue
            cut = find_commit_point(audio)
            if cut and cut > 0:
                self._emit(self._take(cut))

    def stop(self):
        """Terminate capture and flush whatever speech is left."""
        self.stopping.set()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.reader.join(timeout=2)
        self.worker.join(timeout=2)
        tail = self._take()
        if tail.size:
            notify("⏳ Transcribing…", replace_id=self.notif_id, urgency="critical")
            self._emit(tail)
        return self.spoke


def handle_toggle(model):
    global session
    with state_lock:
        if session is None:
            play_cue("message-new-instant")
            hk = hotkey_label()
            hint = (
                f"Speak — text appears as you pause. {hk} to stop."
                if MODE == "phrase"
                else f"Press {hk} again to transcribe."
            )
            # critical so GNOME keeps it on screen for the whole session --
            # otherwise there's no way to tell whether the mic is live.
            nid = notify("🎤 Listening…", hint, urgency="critical")
            session = Session(model, nid)
        else:
            play_cue("complete")
            s, session = session, None
            spoke = s.stop()
            close_notification(s.notif_id)
            title, body = (
                ("⏹️ Stopped", "Dictation ended.")
                if spoke
                else ("🔇 Nothing heard", "No speech detected.")
            )
            final = notify(
                title, body, urgency="low", transient=True, expire_ms=2000
            )
            # Belt and braces: GNOME doesn't always honour expire-time, so
            # close it ourselves too.
            threading.Timer(2.5, close_notification, args=(final,)).start()


def main():
    from faster_whisper import WhisperModel

    print(f"Loading model {MODEL} (mode={MODE})…", flush=True)
    # int8 on CPU: best speed/accuracy tradeoff without a discrete GPU.
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=8)
    print("Model ready.", flush=True)

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(4)
    print(f"Listening on {SOCKET_PATH}", flush=True)

    while True:
        conn, _ = server.accept()
        with conn:
            cmd = conn.recv(64).decode().strip()
            if cmd == "toggle":
                try:
                    handle_toggle(model)
                except Exception as exc:  # keep the daemon alive across failures
                    print(f"error: {exc}", file=sys.stderr, flush=True)
                    notify("⚠️ Dictation error", str(exc), urgency="critical")


if __name__ == "__main__":
    main()
