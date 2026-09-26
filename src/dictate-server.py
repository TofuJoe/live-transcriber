#!/usr/bin/env python3
"""Resident dictation daemon with phrase-level streaming.

Keeps a Whisper model warm in RAM and toggles recording on demand, so the
hotkey path never pays model-load cost. Listens on a unix socket for "toggle".

Audio streams continuously from parecord. A rolling Silero VAD pass finds
pauses; each phrase bounded by silence is transcribed and typed while you keep
talking. Committing only at pause boundaries means emitted text is never
revised -- important because we inject keystrokes into apps we don't own, where
"correcting" would mean backspacing over text we can't see.

Three threads, decoupled by queues so a slow or failing stage can never cost
audio: a reader appends capture to a byte buffer, a detector scans it for phrase
boundaries and hands each finished phrase off, and a single transcriber drains
that queue. Only the transcriber types, so phrases land in the order spoken.

With DICTATE_BACKEND=openvino the GPU model runs in a child process, so a GPU
eviction kills only that worker; the daemon drops to CPU and brings the GPU
back once memory allows. See Transcriber.

  DICTATE_MODE=phrase  (default) type each phrase as you finish it
  DICTATE_MODE=single            type everything at once when you stop
"""

import functools
import multiprocessing
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

MODEL = os.environ.get("DICTATE_MODEL", "small.en")
MODE = os.environ.get("DICTATE_MODE", "phrase")
BACKEND = os.environ.get("DICTATE_BACKEND", "faster-whisper")
OV_MODEL_DIR = os.environ.get(
    "DICTATE_OV_MODEL",
    os.path.expanduser("~/.local/share/voice-dictation/ov-model"),
)
OV_DEVICE = os.environ.get("DICTATE_OV_DEVICE", "GPU")
SOCKET_PATH = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "dictate.sock")
YDOTOOL_SOCKET = "/run/ydotoold/socket"

SR = 16000
SILENCE_MS = 500  # pause that ends a phrase
TAIL_MS = SILENCE_MS + 300  # window scanned to detect end-of-phrase
MAX_SEG_S = 25  # force a cut before Whisper's 30s window
SOFT_SEG_S = 20  # from here on, take any decent pause rather than wait for one
# Shortest pause worth cutting at once past SOFT_SEG_S. Measured intra-phrase
# gaps (the `pause-gaps` log) cluster at 32-128ms for stop consonants and
# word joins, then thin out above ~200ms, so this sits above the articulation
# noise and below anything a speaker would hear as a pause.
OPPORTUNISTIC_GAP_MS = 200
READ_BYTES = 3200  # 100ms of s16le mono @16k
# How often we check whether the phrase has ended. This is detection lag only:
# TAIL_MS decides *what* a boundary is, POLL_S decides how fast we notice one,
# so the two tune independently. At 0.4s the endpoint fired 800-1200ms after
# speech stopped (TAIL_MS + up to a full tick); 0.1s tightens that to 800-900ms
# and matches the rate audio actually arrives, so polling faster gains nothing.
# Cost is one VAD pass over TAIL_MS of audio per tick -- ~1.5ms, ~1.5% of a core.
POLL_S = 0.1

# GPU recovery. A GPU failure is almost always the xe driver banning our VM
# under memory pressure (see Transcriber), so retries are gated on headroom and
# back off: 30s, 60s, 2m, ... capped at 15m, reset after 10m of healthy GPU.
GPU_RETRY_MIN_S = 30
GPU_RETRY_MAX_S = 15 * 60
GPU_STABLE_S = 10 * 60
GPU_CHECK_S = 5
GPU_LOAD_TIMEOUT_S = 120  # first compile of the GPU kernels can be slow
GPU_MIN_AVAILABLE_MB = 2048
GPU_MAX_PSI_SOME = 5.0  # % of time some task stalled on memory, over 10s

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


def has_speech(audio, min_silence_ms=0):
    """Does this audio contain any speech at all?"""
    return bool(
        get_speech_timestamps(
            audio,
            VadOptions(
                threshold=0.5,
                min_speech_duration_ms=0,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=0,
            ),
        )
    )


def log_pause_gaps(audio):
    """Log intra-phrase silence gaps so TAIL_MS can be chosen from measured
    speech instead of guessed.

    Every gap logged here *survived* the current endpoint -- it is a pause we
    did not cut on. A shorter TAIL_MS of T would have split the phrase at any
    gap >= T, so this distribution is precisely the cost curve for lowering the
    threshold: pick T above the bulk of it and you keep clauses intact.

    Runs off the hot path (caller threads it) so it never adds to the latency
    it exists to reduce.
    """
    try:
        ts = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=0.5,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
            ),
        )
        if len(ts) < 2:
            return
        gaps = [
            round((b["start"] - a["end"]) / SR * 1000)
            for a, b in zip(ts, ts[1:])
        ]
        gaps = [g for g in gaps if g > 0]
        if gaps:
            print(f"pause-gaps ms tail={TAIL_MS} {gaps}", flush=True)
    except Exception as exc:  # diagnostics must never break dictation
        print(f"pause-gap logging failed: {exc}", flush=True)


def quietest_cut(window, search_s=5):
    """Index of the middle of the widest silence in the last `search_s` of
    `window`, or None if it is wall-to-wall speech."""
    start = max(0, window.size - int(search_s * SR))
    ts = get_speech_timestamps(
        window[start:],
        VadOptions(
            threshold=0.5,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
        ),
    )
    if len(ts) < 2:
        return None
    a, b = max(zip(ts, ts[1:]), key=lambda p: p[1]["start"] - p[0]["end"])
    gap = b["start"] - a["end"]
    return start + a["end"] + gap // 2 if gap > 0 else None


def opportunistic_cut(audio):
    """Earliest usable pause after SOFT_SEG_S, or None if speech is unbroken.

    Between SOFT_SEG_S and MAX_SEG_S a cut is already inevitable -- the 30s
    encoder window is closing in and no TAIL_MS pause has arrived to end the
    phrase normally. So relax what counts as a boundary: accept a pause far too
    short to end a phrase but long enough to fall between words. That trades a
    slightly early cut for not slicing through the middle of one.

    Earliest qualifying gap, not the widest. The buffer is still filling, so a
    wider gap may never arrive, and holding out for one spends the very margin
    this exists to use.

    Scans SOFT_SEG_S..now, so cost grows with how long the speaker has gone
    without pausing: 1.3ms at 21s, 6.4ms at the 25s cap, i.e. at most 6.4% of a
    core at POLL_S -- and only during a monologue that has run 20s unbroken. The
    scan resets as soon as a gap is found, because we cut there.
    """
    start = int(SOFT_SEG_S * SR)
    if audio.size <= start:
        return None
    ts = get_speech_timestamps(
        audio[start:],
        VadOptions(
            threshold=0.5,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
        ),
    )
    min_gap = int(OPPORTUNISTIC_GAP_MS / 1000 * SR)
    for a, b in zip(ts, ts[1:]):
        gap = b["start"] - a["end"]
        if gap >= min_gap:
            return start + a["end"] + gap // 2
    return None


def split_for_transcription(audio):
    """Break an oversized buffer into <= MAX_SEG_S pieces.

    Only the stop-path flush can hand us more than MAX_SEG_S. _work_loop caps
    live phrases, but MODE=single accumulates the entire session by design, and
    a detector thread that died leaves the buffer growing with nothing draining
    it -- which is how one GPU failure turned into a 72s chunk on 2026-08-02.

    Cutting on the quietest point near each boundary rather than at a hard index
    keeps words intact across the seam; a hard cut is the fallback when the
    speech really is continuous.
    """
    max_n = int(MAX_SEG_S * SR)
    out = []
    while audio.size > max_n:
        cut = quietest_cut(audio[:max_n]) or max_n
        out.append(audio[:cut])
        audio = audio[cut:]
    if audio.size:
        out.append(audio)
    return out


def find_commit_point(audio, speech_seen):
    """Decide whether to cut. Returns (cut_index_or_None, speech_seen).

    Scans only the last TAIL_MS of the buffer, not all of it. Scanning the whole
    buffer every poll is O(n) per poll and therefore O(n^2) over a phrase -- at a
    25s buffer that was 47ms of VAD every 400ms, ~12% of a core, growing the
    longer you talk. The tail is all we actually need: if it has gone quiet, the
    phrase is over.

    `speech_seen` carries the one bit of history the tail can't tell us -- did
    any speech occur since the last cut. Without it, the silence left in the
    buffer after a commit re-triggers immediately, emitting empty chunks.

    The cut lands where a full scan would. At the first poll with a silent tail,
    len(audio) ~= speech_end + TAIL_MS, so len(audio) - tail ~= speech_end.

    Past SOFT_SEG_S the tail-only rule stops being enough. A speaker who has run
    20s without a TAIL_MS pause is going to be cut at the cap regardless, so from
    there we widen the search and take progressively worse pauses rather than
    arrive at MAX_SEG_S with nowhere good to cut. See opportunistic_cut.
    """
    tail_n = int(TAIL_MS / 1000 * SR)
    if len(audio) < tail_n + int(0.3 * SR):
        return None, speech_seen  # not enough yet to judge

    if has_speech(audio[-tail_n:]):
        if len(audio) >= MAX_SEG_S * SR:
            # Out of room. Nothing cleared OPPORTUNISTIC_GAP_MS in the last 5s,
            # so settle for the widest gap there is -- a 60ms one still beats
            # cutting blind at the cap. Only truly gapless audio falls through.
            widest = quietest_cut(audio, search_s=MAX_SEG_S - SOFT_SEG_S)
            return (widest or len(audio)), False
        if len(audio) >= SOFT_SEG_S * SR:
            cut = opportunistic_cut(audio)
            if cut:
                return cut, False
        return None, True

    if not speech_seen:
        # Nothing but silence. Trim so the buffer can't grow unbounded; _emit
        # discards it without paying for transcription.
        return (len(audio) - tail_n, False) if len(audio) > 3 * SR else (None, False)

    # Tail is quiet and we heard speech: the phrase is done. Keep 200ms of
    # trailing silence in the chunk -- Whisper transcribes better with it.
    cut = len(audio) - tail_n + int(0.2 * SR)
    return (cut if cut > 0 else None), False


class FasterWhisperBackend:
    """CPU transcription via CTranslate2. Slower, but supports beam search,
    which is worth a few real word errors per hundred in noisy audio."""

    label = "faster-whisper CPU (beam=5)"

    def __init__(self, model_name):
        from faster_whisper import WhisperModel

        # int8 on CPU: best speed/accuracy tradeoff without a discrete GPU.
        self.model = WhisperModel(
            model_name, device="cpu", compute_type="int8", cpu_threads=8
        )

    def transcribe(self, audio):
        # condition_on_previous_text=False + repetition_penalty guard against
        # Whisper's repetition-loop failure mode on unclear or repetitive audio.
        segments, _ = self.model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            repetition_penalty=1.1,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class OpenVinoBackend:
    """iGPU transcription via OpenVINO. ~2.5x faster than the CPU backend.

    Greedy decoding only: openvino-genai 2026.2.1 cannot run beam search on
    GPU ("Not Implemented" on remote tensors at num_beams=2, logits/beam batch
    mismatch at 5). On clean speech that costs about one word error in ninety;
    in babble it is closer to three. See docs/findings.md.
    """

    label = f"openvino {OV_DEVICE} (greedy)"

    def __init__(self, model_dir, device):
        import openvino_genai as og

        if not os.path.isdir(model_dir):
            raise RuntimeError(
                f"OpenVINO model not found at {model_dir}. "
                "Run install.sh with DICTATE_BACKEND=openvino."
            )
        self.pipe = og.WhisperPipeline(model_dir, device=device)

    def transcribe(self, audio):
        return self.pipe.generate(audio).texts[0].strip()


def memory_headroom():
    """(MemAvailable in MiB, memory PSI "some" avg10 in percent).

    MemAvailable says how much could be had without swapping; PSI says whether
    the kernel is already stalling tasks to get it. Low RAM with no stall is
    fine (cache is cheap to drop), so the GPU gate checks both.
    """
    avail = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024
                break
    try:
        with open("/proc/pressure/memory") as f:
            psi = float(f.readline().split()[1].split("=")[1])
    except (OSError, IndexError, ValueError):
        psi = 0.0  # no PSI (kernel built without it): judge on RAM alone
    return avail, psi


def _gpu_worker(conn, model_dir, device):
    """Entry point of the GPU worker process: load, then serve phrases.

    Leaves only via os._exit, never by returning or raising. A dead pipeline's
    destructor calls clFinish on the banned VM and std::terminate()s whatever
    process it runs in; skipping finalizers is how that stays harmless here.
    """
    try:
        backend = OpenVinoBackend(model_dir, device)
    except Exception as exc:
        conn.send(("err", f"load failed: {exc}"))
        os._exit(1)
    conn.send(("ready", backend.label))
    while True:
        try:
            audio = conn.recv()
        except (EOFError, OSError):  # daemon went away
            os._exit(0)
        try:
            conn.send(("ok", backend.transcribe(audio)))
        except Exception as exc:
            # One failure means the VM is banned; every later call would fail
            # too. Report and die so the daemon can start a fresh process.
            conn.send(("err", str(exc)))
            os._exit(1)


class GpuWorker:
    """Daemon-side handle on one GPU worker process.

    Any failure -- an error reply, the process dying (the destructor abort
    included), or no reply in time (a GPU hang) -- kills the worker and raises.
    A worker is never reused after a failure; the Transcriber spawns a new one.
    """

    label = OpenVinoBackend.label

    def __init__(self):
        ctx = multiprocessing.get_context("spawn")  # no fork of a threaded daemon
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_gpu_worker,
            args=(child, OV_MODEL_DIR, OV_DEVICE),
            name="dictate-gpu",
            daemon=True,
        )
        self.proc.start()
        child.close()
        kind, msg = self._reply(GPU_LOAD_TIMEOUT_S)
        if kind != "ready":
            self.kill()
            raise RuntimeError(msg)

    def _reply(self, timeout):
        try:
            if not self.conn.poll(timeout):
                return "err", f"no reply in {timeout:.0f}s"
            return self.conn.recv()
        except (EOFError, OSError):
            self.proc.join(timeout=1)
            return "err", f"worker died (exit {self.proc.exitcode})"

    def transcribe(self, audio):
        try:
            self.conn.send(audio)
        except OSError as exc:
            self.kill()
            raise RuntimeError(f"worker unreachable: {exc}") from None
        # The iGPU runs ~19x realtime, so 2x realtime plus slack is a hang.
        kind, msg = self._reply(10 + 2 * audio.size / SR)
        if kind != "ok":
            self.kill()
            raise RuntimeError(msg)
        return msg

    def kill(self):
        # SIGKILL, not SIGTERM: nothing in the worker may get to run cleanup.
        self.proc.kill()
        self.proc.join(timeout=5)
        self.conn.close()


class Transcriber:
    """Owns the backends and moves between GPU and CPU as the GPU comes and goes.

    The iGPU has no VRAM: OpenVINO's buffers are system RAM that the kernel must
    pin into the GPU's address space. Under memory pressure that bind fails, the
    xe driver bans the GPU VM ("VM worker error: -12"), and every later inference
    raises CL_OUT_OF_RESOURCES. The ban lasts the life of the process, and the
    dead pipeline cannot even be freed: its destructor throws out of C++ and
    aborts the process (seen 2026-09-26, when that abort took the whole daemon
    down before the fallback phrase was typed).

    So the GPU lives in a separate process (GpuWorker) that can die or be killed
    without touching the daemon. On failure the phrase is re-run on the CPU
    backend, and a supervisor thread brings the GPU back with a fresh worker --
    a new process gets a new VM -- once the backoff has passed and memory has
    room. The CPU model is loaded only while needed and dropped on recovery, so
    the RAM it holds isn't what evicts the GPU again.
    """

    def __init__(self):
        self.lock = threading.Lock()  # guards the fields below
        self.gpu_io = threading.Lock()  # one request on the worker pipe at a time
        self.cpu_load = threading.Lock()
        self.gpu = None
        self.cpu = None
        self.failures = 0  # consecutive, drives the backoff
        self.retry_at = 0.0
        self.gpu_since = 0.0
        self.deferred = None  # last "not enough memory" reason, to log once
        if BACKEND == "openvino":
            if os.path.isdir(OV_MODEL_DIR):
                self._spawn_gpu()
                threading.Thread(target=self._supervise, daemon=True).start()
            else:
                print(
                    f"OpenVINO model not found at {OV_MODEL_DIR}; using CPU. "
                    "Run install.sh with DICTATE_BACKEND=openvino.",
                    flush=True,
                )
        if self.gpu is None:
            self._cpu()

    @property
    def label(self):
        gpu = self.gpu
        return gpu.label if gpu else FasterWhisperBackend.label

    def transcribe(self, audio):
        gpu = self.gpu
        if gpu is not None:
            try:
                with self.gpu_io:
                    return gpu.transcribe(audio)
            except Exception as exc:
                self._gpu_failed(gpu, exc)
        # Re-run on CPU rather than drop it: the phrase that killed the GPU
        # still gets typed.
        return self._cpu().transcribe(audio)

    def _cpu(self):
        with self.cpu_load:
            if self.cpu is None:
                self.cpu = FasterWhisperBackend(MODEL)
            return self.cpu

    def _gpu_failed(self, gpu, exc):
        with self.lock:
            if self.gpu is not gpu:
                return  # someone else already handled this worker
            self.gpu = None
            delay = self._schedule_retry()
        print(
            f"GPU worker failed ({exc}); on CPU, retrying GPU in {delay:.0f}s",
            flush=True,
        )
        notify(
            "⚠️ GPU transcription failed",
            "Using the CPU backend; the GPU comes back when memory allows.",
            urgency="normal",
        )

    def _schedule_retry(self):
        """Book the next GPU attempt. Call with self.lock held."""
        now = time.monotonic()
        if self.gpu_since and now - self.gpu_since >= GPU_STABLE_S:
            self.failures = 0  # it had recovered; this is a new episode
        self.gpu_since = 0.0
        self.failures += 1
        delay = min(GPU_RETRY_MIN_S * 2 ** (self.failures - 1), GPU_RETRY_MAX_S)
        self.retry_at = now + delay
        return delay

    def _spawn_gpu(self):
        try:
            gpu = GpuWorker()
        except Exception as exc:
            with self.lock:
                delay = self._schedule_retry()
            print(f"GPU worker failed to start ({exc}); retry in {delay:.0f}s",
                  flush=True)
            return False
        with self.lock:
            self.gpu = gpu
            self.gpu_since = time.monotonic()
        return True

    def _supervise(self):
        """Bring the GPU back after a failure, when backoff and memory allow."""
        while True:
            time.sleep(GPU_CHECK_S)
            with self.lock:
                if self.gpu is not None or time.monotonic() < self.retry_at:
                    continue
            avail, psi = memory_headroom()
            if avail < GPU_MIN_AVAILABLE_MB or psi > GPU_MAX_PSI_SOME:
                reason = f"{avail} MiB available, memory PSI {psi:.1f}%"
                if self.deferred is None:
                    print(f"GPU retry deferred: {reason}", flush=True)
                self.deferred = reason
                continue
            self.deferred = None
            print(f"retrying GPU ({avail} MiB available)", flush=True)
            if self._spawn_gpu():
                with self.cpu_load:
                    self.cpu = None  # give its RAM back to the GPU's headroom
                print(f"GPU restored: {self.gpu.label}", flush=True)
                notify("✅ GPU transcription restored", urgency="low",
                       transient=True, expire_ms=3000)


class Session:
    """One recording session: reader, phrase detector, and transcriber."""

    def __init__(self, backend, notif_id=None):
        self.backend = backend
        self.notif_id = notif_id  # the persistent "Listening…" banner
        self.buf = bytearray()
        self.buf_lock = threading.Lock()
        self.stopping = threading.Event()
        self.spoke = False  # emitted anything yet? controls leading space
        # Committed phrases waiting to be transcribed. Unbounded on purpose:
        # dropping queued audio is exactly the data loss this exists to prevent.
        # Depth stays near zero in practice -- the iGPU runs ~19x realtime and
        # the CPU fallback ~5x (docs/findings.md), so the queue absorbs bursts
        # rather than accumulating. _pending() reports the depth for stop().
        self.queue = queue.Queue()
        self.proc = subprocess.Popen(RECORD_CMD, stdout=subprocess.PIPE)
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.worker = threading.Thread(target=self._work_loop, daemon=True)
        self.typist = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.reader.start()
        self.worker.start()
        self.typist.start()

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

    def _commit(self, audio):
        """Hand a finished phrase to the transcriber. Must stay cheap: this runs
        on the detector thread, and anything slow here delays the *next* phrase
        boundary rather than the current phrase's text."""
        if audio.size < 0.3 * SR:  # too short to be speech
            return
        # One VAD pass per commit (not per poll) to avoid queueing ~1.5s of
        # transcription on a chunk that turns out to be pure silence.
        if not has_speech(audio):
            return
        self.queue.put(audio)

    def _pending(self):
        """Seconds of audio queued but not yet transcribed."""
        return sum(a.size for a in tuple(self.queue.queue) if a is not None) / SR

    def _emit(self, audio):
        text = self.backend.transcribe(audio)
        if not text:
            return
        type_text((" " if self.spoke else "") + text)
        self.spoke = True
        # After typing: diagnostics must not sit between speech and keystrokes.
        threading.Thread(
            target=log_pause_gaps, args=(audio,), daemon=True
        ).start()

    def _transcribe_loop(self):
        """Drain the phrase queue, one at a time. Single consumer, so phrases
        are typed in the order they were spoken."""
        while True:
            audio = self.queue.get()
            try:
                if audio is None:  # stop() has flushed everything
                    return
                self._emit(audio)
            except Exception as exc:
                # A phrase we cannot transcribe is lost, but the session is not:
                # keep draining so the rest of what was said still gets typed.
                print(
                    f"transcription failed, dropped {audio.size / SR:.1f}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                self.queue.task_done()

    def _work_loop(self):
        if MODE == "single":
            self.stopping.wait()
            return
        speech_seen = False
        while not self.stopping.wait(POLL_S):
            # Detection must outlive its own failures. When this thread died on
            # a backend exception, nothing drained self.buf and capture ran on
            # regardless -- the mic stayed live, the banner stayed up, and not a
            # word was typed until the daemon was restarted.
            try:
                audio = self._snapshot()
                if audio.size < SR:
                    continue
                cut, speech_seen = find_commit_point(audio, speech_seen)
                if cut and cut > 0:
                    self._commit(self._take(cut))
            except Exception as exc:
                print(f"phrase detection failed: {exc}", file=sys.stderr, flush=True)

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
        # The whole remaining buffer, not one phrase: in MODE=single that is the
        # entire session, and if the detector died it is everything since. Split
        # it so no piece exceeds what Whisper's 30s window can hold.
        for chunk in split_for_transcription(self._take()):
            self._commit(chunk)
        self.queue.put(None)  # sentinel: drain, then exit
        backlog = self._pending()
        if backlog:
            notify("⏳ Transcribing…", replace_id=self.notif_id, urgency="critical")
        # Wait for the backlog at a pessimistic 1x realtime, well under the ~5x
        # the slower (CPU) backend actually manages. A bounded join means a
        # wedged backend can't hold the toggle hostage forever; the typist is a
        # daemon thread and keeps draining if we give up early.
        self.typist.join(timeout=backlog + 30)
        if self.typist.is_alive():
            print(
                f"still transcribing {self._pending():.1f}s after stop",
                file=sys.stderr,
                flush=True,
            )
        return self.spoke


def handle_toggle(backend):
    global session
    with state_lock:
        if session is None:
            hk = hotkey_label()
            hint = (
                f"Speak — text appears as you pause. {hk} to stop."
                if MODE == "phrase"
                else f"Press {hk} again to transcribe."
            )
            # critical so GNOME keeps it on screen for the whole session --
            # otherwise there's no way to tell whether the mic is live.
            nid = notify("🎤 Listening…", hint, urgency="critical")
            session = Session(backend, nid)
        else:
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
    print(f"Loading backend={BACKEND} model={MODEL} mode={MODE}…", flush=True)
    backend = Transcriber()
    print(f"Ready: {backend.label}", flush=True)

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
                    handle_toggle(backend)
                except Exception as exc:  # keep the daemon alive across failures
                    print(f"error: {exc}", file=sys.stderr, flush=True)
                    notify("⚠️ Dictation error", str(exc), urgency="critical")


if __name__ == "__main__":
    main()
