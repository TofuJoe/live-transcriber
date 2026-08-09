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

The OpenVINO backend additionally runs in a child process, because its failure
mode is an uncatchable abort rather than an exception -- see _ov_worker.

  DICTATE_MODE=phrase  (default) type each phrase as you finish it
  DICTATE_MODE=single            type everything at once when you stop
  DICTATE_MODE=stream            LocalAgreement -- see _stream_loop
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

# How long the parent waits on the GPU child before declaring it dead. Load is
# generous because a cold child re-imports this module and builds the pipeline
# (~2s warm, far more if the model files have fallen out of page cache). The
# call bound only has to exceed a worst-case MAX_SEG_S phrase (~4s measured);
# anything near a minute means the GPU has wedged rather than slowed.
GPU_LOAD_TIMEOUT_S = 120
GPU_CALL_TIMEOUT_S = 60

# ---------------------------------------------------------- MODE=stream
# Floor between re-transcriptions. Usually not the binding constraint: a pass
# over the window costs more than this, and the loop paces itself on that.
STREAM_TICK_S = 0.5
STREAM_MIN_S = 1.0  # too little audio to be worth a pass
# How much audio must accumulate before a silence is allowed to end the window.
# This is the whole point of the mode: in phrase mode *any* TAIL_MS pause cuts,
# which is what starves the decoder of context (measured: 4 errors in 71 words
# at ~2s chunks, 0 at ~20s). Here a pause is ignored until the window is already
# long enough to decode well, so clauses accumulate instead of being isolated.
STREAM_FLUSH_S = 8
# Hard ceiling on a stream window, well under MAX_SEG_S. The iGPU has no VRAM,
# so every decode pins system RAM, and this mode decodes continuously rather
# than once per phrase -- windows that ran to 21.8s exhausted it and the driver
# returned CL_OUT_OF_RESOURCES, killing the worker. Capping the window caps the
# peak allocation. Cheap in accuracy: 8-15s is already far past the point where
# context stops paying (measured 2s -> 12s recovered nearly all of it).
STREAM_MAX_S = 15
# Words held back from the agreed prefix while the window is still open.
# Agreement means "two passes settled on this independently", but consecutive
# passes only differ by ~0.5s of audio, and the last word of a hypothesis is the
# one that has heard least of what follows it -- exactly the word that turns
# 'bye' into 'by' once the next one arrives. Holding one back costs a word of
# lag and buys the guarantee the mode exists for. The flush emits it.
STREAM_HOLDBACK = 1

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


def _agreement_key(word):
    """Compare words ignoring punctuation and case.

    Whisper re-punctuates as its window grows -- "words" becomes "words," once a
    following clause arrives, and a fragment that looked like a sentence loses
    its capital. Comparing raw strings would treat that as disagreement and stall
    the commit prefix on text the model never actually reconsidered.
    """
    return word.strip(".,!?;:-\"'").lower()


def resume_point(hyp, typed):
    """Where to pick up in `hyp` given `typed` has already been sent.

    Indices are not safe to carry across decodes. Each pass re-segments what it
    hears -- the same speech comes back as a different number of words when
    punctuation shifts or a compound splits -- so hyp[len(typed):] silently
    repeats or drops text once the count moves. Measured: a 21-word passage
    typed as 23-24 words, with "to the harbor" emitted twice.

    Anchoring on the tail we actually typed survives re-segmentation, because it
    matches on content rather than position. Searching backwards takes the
    latest occurrence, so a phrase the speaker genuinely repeated resumes after
    the second one rather than replaying it.
    """
    if not typed:
        return 0
    anchor = [_agreement_key(w) for w in typed[-4:]]
    keys = [_agreement_key(w) for w in hyp]
    for start in range(len(keys) - len(anchor), -1, -1):
        if keys[start:start + len(anchor)] == anchor:
            return start + len(anchor)
    return len(typed)  # anchor gone entirely; index is the least-bad guess


def agreed_prefix(hyp, prev):
    """How many leading words two consecutive hypotheses agree on.

    This is LocalAgreement-2: a word is committed once it has survived one
    re-decode with more audio behind it. Whisper revises the tail of its output
    as context arrives and leaves the head alone, so agreement across two passes
    is a good proxy for "this will not change again" -- which is the only
    guarantee that matters here, because typed keystrokes cannot be recalled.
    """
    n = 0
    for a, b in zip(hyp, prev):
        if _agreement_key(a) != _agreement_key(b):
            break
        n += 1
    return n


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


class WorkerDied(RuntimeError):
    """The GPU child process is gone, or has stopped answering."""


def _ov_worker(model_dir, device, req_q, res_q):
    """Child process: owns the OpenVINO pipeline and nothing else.

    Isolation is the whole point of this function existing. When the xe driver
    bans the GPU VM, OpenVINO tears its command stream down on a background
    thread whose destructor throws; with no handler on that thread,
    std::terminate aborts the process. Python never gets a chance to intervene
    -- an `except` around generate() catches the *first* error and logs a tidy
    fallback, and the process dies anyway a few seconds later. Reproduced and
    measured in docs/findings.md.

    So the pipeline lives behind a process boundary. When it aborts, the parent
    finds a corpse instead of sharing one.

    Nothing here may raise: a traceback escaping this function would kill the
    worker for reasons the parent would then misread as a GPU fault.
    """
    try:
        import openvino_genai as og

        pipe = og.WhisperPipeline(model_dir, device=device)
    except BaseException as exc:  # noqa: BLE001 -- report, never propagate
        res_q.put(("fatal", f"{type(exc).__name__}: {exc}"))
        return

    res_q.put(("ready", None))
    while True:
        audio = req_q.get()
        if audio is None:  # parent is shutting us down
            return
        try:
            res_q.put(("ok", pipe.generate(audio).texts[0].strip()))
        except BaseException as exc:  # noqa: BLE001
            res_q.put(("err", f"{type(exc).__name__}: {exc}"))


class OpenVinoBackend:
    """iGPU transcription via OpenVINO, running in a child process.

    ~2.5x faster than the CPU backend. Greedy decoding only: openvino-genai
    2026.2.1 cannot run beam search on GPU ("Not Implemented" on remote tensors
    at num_beams=2, logits/beam batch mismatch at 5). On clean speech that costs
    about one word error in ninety; in babble it is closer to three.

    The parent must never import openvino itself. Touching the GPU from this
    process would put the abort back in the one place we are protecting.

    Requests are strictly one at a time -- a single transcriber thread drives
    this -- so a plain request/response pair over two queues is enough, with no
    correlation ids. Any desync means the worker is being discarded anyway.
    """

    label = f"openvino {OV_DEVICE} (greedy, isolated)"

    def __init__(self, model_dir, device):
        if not os.path.isdir(model_dir):
            raise RuntimeError(
                f"OpenVINO model not found at {model_dir}. "
                "Run install.sh with DICTATE_BACKEND=openvino."
            )
        self.model_dir, self.device = model_dir, device
        self._start()

    def _start(self):
        # spawn, not fork: a forked child would inherit this process's threads
        # and allocator state, and we want the GPU stack built from clean.
        ctx = multiprocessing.get_context("spawn")
        self.req_q, self.res_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(
            target=_ov_worker,
            args=(self.model_dir, self.device, self.req_q, self.res_q),
            daemon=True,  # never outlive the daemon
        )
        self.proc.start()
        kind, payload = self._await(GPU_LOAD_TIMEOUT_S)
        if kind != "ready":
            self.close()
            raise RuntimeError(payload or "GPU worker failed to start")

    def _await(self, timeout):
        """Wait for the worker's reply, watching for its death as well.

        Polling rather than a bare blocking get: an aborted worker sends
        nothing, so a plain get(timeout=...) would stall for the full timeout on
        the common failure. Checking is_alive() between short waits turns a
        SIGABRT into an immediate WorkerDied.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.res_q.get(timeout=0.2)
            except queue.Empty:
                pass
            if not self.proc.is_alive():
                # -6 is SIGABRT, the signature of the GPU VM ban.
                raise WorkerDied(f"worker exited with code {self.proc.exitcode}")
            if time.monotonic() >= deadline:
                raise WorkerDied(f"worker silent for {timeout:.0f}s")

    def transcribe(self, audio):
        try:
            self.req_q.put(audio)
        except Exception as exc:
            raise WorkerDied(f"could not reach worker: {exc}") from exc
        kind, payload = self._await(GPU_CALL_TIMEOUT_S)
        if kind == "ok":
            return payload
        # The worker answered but the GPU refused. In the observed failure the
        # first symptom is exactly this -- a catchable CL error -- and the abort
        # follows moments later, so treat it as fatal to the worker, not as a
        # bad phrase.
        raise WorkerDied(payload)

    def restart(self):
        self.close()
        self._start()

    def close(self):
        """Tear the worker down. Must never raise: every caller is already on a
        failure path."""
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.is_alive():
                self.req_q.put(None)
                proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


def make_backend():
    """Build the configured backend, falling back to CPU if OpenVINO is
    unavailable -- a missing GPU stack should degrade, not break dictation."""
    if BACKEND == "openvino":
        try:
            return OpenVinoBackend(OV_MODEL_DIR, OV_DEVICE)
        except Exception as exc:
            print(f"openvino backend unavailable ({exc}); using CPU", flush=True)
            notify(
                "⚠️ OpenVINO unavailable",
                "Fell back to the CPU backend.",
                urgency="normal",
            )
    return FasterWhisperBackend(MODEL)


class Transcriber:
    """Owns the active backend and survives its death mid-session.

    The iGPU has no VRAM: OpenVINO's buffers are system RAM that the kernel must
    pin into the GPU's address space. Under memory pressure that bind fails and
    the xe driver bans the GPU VM ("VM worker error: -12").

    The ban is scoped to the process holding the VM -- which, since the pipeline
    moved into a child, is no longer this one. That buys a recovery the old
    in-process design could not have: a *fresh* worker gets a fresh VM and
    usually comes back on the GPU. Worth one attempt, because the CPU backend
    costs ~2.2x per phrase. If the replacement dies too, the machine genuinely
    has no room for the iGPU and we stop asking for the rest of the session.

    The failed audio is re-run on the new backend rather than dropped, so the
    phrase that triggered the fallback still gets typed.
    """

    def __init__(self):
        self.backend = make_backend()
        self.lock = threading.Lock()
        self.gpu_retried = False

    @property
    def label(self):
        return self.backend.label

    def transcribe(self, audio):
        # Loops rather than retrying once: the replacement may itself be a GPU
        # worker that fails immediately. _replace() gives up the GPU on the
        # second failure, so this terminates on the CPU backend at worst.
        while True:
            try:
                return self.backend.transcribe(audio)
            except Exception as exc:
                with self.lock:
                    if not isinstance(self.backend, OpenVinoBackend):
                        raise  # already on CPU; nothing left to fall back to
                    self.backend = self._replace(self.backend, exc)

    def _replace(self, dead, exc):
        """Swap in a working backend after a GPU failure. Caller holds the lock."""
        print(f"GPU worker failed ({exc})", file=sys.stderr, flush=True)
        if not self.gpu_retried:
            self.gpu_retried = True
            try:
                dead.restart()
                print("restarted the GPU worker", flush=True)
                return dead
            except Exception as exc2:
                print(f"GPU worker restart failed ({exc2})", file=sys.stderr,
                      flush=True)
        dead.close()
        notify(
            "⚠️ GPU transcription failed",
            "Switched to the CPU backend for the rest of this session.",
            urgency="normal",
        )
        return FasterWhisperBackend(MODEL)


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
        # MODE=stream state. Words of the current window already typed, and
        # whether this window has heard speech yet -- the same bit find_commit_point
        # carries as `speech_seen`, for the same reason.
        self.streamed = []  # words typed from the current window, in order
        self.stream_heard = False
        # Raised by stop() once the reader is joined and the buffer is final.
        # The stream thread flushes on its own; see _stream_loop.
        self.flush_ready = threading.Event()
        self.proc = subprocess.Popen(RECORD_CMD, stdout=subprocess.PIPE)
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        # stream transcribes and types on its own thread, so it needs no queue
        # and no typist: it is already a single consumer, and pacing the loop on
        # the decode is what keeps the window from outrunning the GPU.
        self.worker = threading.Thread(
            target=self._stream_loop if MODE == "stream" else self._work_loop,
            daemon=True,
        )
        self.typist = (
            None if MODE == "stream"
            else threading.Thread(target=self._transcribe_loop, daemon=True)
        )
        self.reader.start()
        self.worker.start()
        if self.typist:
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

    def _pending_stream_audio(self):
        """Is there enough left in the buffer that the flush will take a moment?"""
        with self.buf_lock:
            return len(self.buf) / 2 > STREAM_MIN_S * SR

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

    def _say(self, words):
        """Type newly committed words. Append-only: this never revises."""
        if not words:
            return
        type_text((" " if self.spoke else "") + " ".join(words))
        self.spoke = True
        self.streamed.extend(words)  # the anchor resume_point aligns against

    def _stream_loop(self):
        """Re-decode a growing window and commit only what has stopped changing.

        Phrase mode cuts at every TAIL_MS pause and decodes each ~2s fragment
        alone, which is where the word errors come from -- 'by Thursday' becomes
        'Bye Thursday' because nothing in the fragment says otherwise. Here the
        audio simply accumulates, so by the time a word is committed the decoder
        has seen the clauses either side of it.

        Committing the agreed prefix rather than the whole hypothesis is what
        makes that safe to type live. The tail of a Whisper hypothesis churns as
        context arrives; the head does not. Emitting only the part that survived
        a second pass keeps the stream append-only, which is the constraint this
        daemon cannot break -- we inject keystrokes into windows we don't own and
        cannot backspace over what we can't see.

        Cost: the GPU runs continuously while you speak instead of once per
        phrase. That is the real price of this mode, and the reason it is opt-in.
        """
        prev = []
        tail_n = int(TAIL_MS / 1000 * SR)
        while not self.stopping.wait(STREAM_TICK_S):
            # As in _work_loop: this thread must outlive its own failures, or
            # capture runs on with nothing draining it and nothing is ever typed.
            try:
                audio = self._snapshot()
                if audio.size < STREAM_MIN_S * SR:
                    continue

                if not self.stream_heard:
                    # Full scan only until speech is confirmed. After that the
                    # tail check below is enough, so the O(n) pass doesn't run
                    # every tick on a growing buffer.
                    if not has_speech(audio):
                        if audio.size > 3 * SR:  # keep silence from accumulating
                            self._take(audio.size - tail_n)
                        continue
                    self.stream_heard = True

                # A pause only ends the window once there is enough audio to
                # have decoded well; shorter pauses are deliberately ignored.
                tail_quiet = not has_speech(audio[-tail_n:])
                ended = tail_quiet and audio.size >= STREAM_FLUSH_S * SR
                full = audio.size >= STREAM_MAX_S * SR

                if tail_quiet and not (ended or full):
                    # Paused mid-thought, window too short to close. Decoding
                    # again would spend the GPU on audio that hasn't changed,
                    # and worse, the identical hypothesis would come back and
                    # look like agreement -- the same guess counted twice rather
                    # than two passes that independently settled. That is how a
                    # pause used to commit a trailing word early, undoing the
                    # context this mode exists to preserve. Wait for speech.
                    continue

                t0 = time.monotonic()
                hyp = self.backend.transcribe(audio).split()
                decode_s = time.monotonic() - t0
                # The number that decides whether this mode is usable: a pass
                # must stay well under the audio it covers, or the committed
                # text falls further behind the speaker the longer they talk.
                print(
                    f"stream window={audio.size / SR:.1f}s decode={decode_s * 1000:.0f}ms "
                    f"committed={len(self.streamed)}/{len(hyp)}",
                    flush=True,
                )

                start = resume_point(hyp, self.streamed)
                if ended or full:
                    # Window is closed, so the hypothesis is final rather than
                    # provisional: commit all of it, not just the agreed prefix.
                    # Take exactly what we decoded -- the reader has appended
                    # more since the snapshot, and that belongs to the next one.
                    self._say(hyp[start:])
                    self._take(audio.size)
                    prev, self.streamed, self.stream_heard = [], [], False
                    continue

                n = agreed_prefix(hyp, prev) - STREAM_HOLDBACK
                if n > start:
                    self._say(hyp[start:n])
                prev = hyp
            except Exception as exc:
                print(f"stream loop failed: {exc}", file=sys.stderr, flush=True)

        # Session over. Flush here rather than on the caller's thread: this must
        # stay the only thread that ever calls the backend. OpenVinoBackend
        # pairs each request with the next reply and carries no correlation ids,
        # so a second concurrent caller crosses them -- one thread takes the
        # other's answer, the loser waits out GPU_CALL_TIMEOUT_S, and the retry
        # in Transcriber.transcribe then holds its lock through a 120s worker
        # restart. That wedged the accept loop for eleven minutes on 2026-08-09,
        # which looks from the outside like the hotkey being dead.
        if self.flush_ready.wait(timeout=10):
            try:
                self._stream_flush()
            except Exception as exc:
                print(f"stream flush failed: {exc}", file=sys.stderr, flush=True)

    def _stream_flush(self):
        """Type whatever the last window had not committed yet.

        Runs on the stop path once the reader and stream threads are joined, so
        the buffer is final and nothing races us for it.
        """
        audio = self._take()
        first = True
        for chunk in split_for_transcription(audio):
            if chunk.size < 0.3 * SR or not has_speech(chunk):
                continue
            words = self.backend.transcribe(chunk).split()
            # Only the first chunk overlaps text already typed; later chunks are
            # audio this session has never decoded, so they start at zero.
            self._say(words[resume_point(words, self.streamed) if first else 0:])
            first = False
        self.streamed = []

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
        if MODE == "stream":
            # The buffer is final now the reader is done, so release the stream
            # thread to flush it. We wait for that thread rather than decoding
            # here, so exactly one thread ever drives the backend.
            if self._pending_stream_audio():
                notify("⏳ Transcribing…", replace_id=self.notif_id,
                       urgency="critical")
            self.flush_ready.set()
            # Generous but bounded: one in-flight decode plus the flush. A
            # wedged backend must not hold the toggle -- and therefore the whole
            # accept loop -- hostage.
            self.worker.join(timeout=GPU_CALL_TIMEOUT_S + 30)
            if self.worker.is_alive():
                print("stream thread still flushing after stop",
                      file=sys.stderr, flush=True)
            return self.spoke
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
                else f"Speak — text appears as you talk. {hk} to stop."
                if MODE == "stream"
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
