# Findings

Measurements behind the tuning decisions. All on Fedora 44, GNOME Wayland,
i5-1240P (16 threads, no discrete GPU), 16GB RAM, 57.6Wh battery,
`faster-whisper` `small.en` int8, 8 CPU threads.

Recorded so these questions don't get re-litigated from intuition.

---

## Latency budget

Per-phrase latency, before and after tuning:

```
before:  700ms endpoint + 1.5s transcribe + 3.5s typing  ~= 5.7s
after:   500ms endpoint + 1.5s transcribe + 0.27s typing ~= 2.3s
```

The dominant term was **typing**, not the model. This was counter-intuitive
and cost the most time to find.

### ydotool typing speed — the big one

`ydotool type --key-hold` defaults to **20ms**, separate from `--key-delay`.
Setting only `--key-delay` leaves ~28ms/char. Measured on a 124-char phrase:

| Setting | Total | Per char |
|---|---|---|
| `hold=20 delay=8` (naive) | 3.51s | 28.3ms |
| `hold=5 delay=2` | 0.90s | 7.3ms |
| **`hold=1 delay=1`** (chosen) | **0.27s** | **2.2ms** |
| `hold=0 delay=0` | 0.02s | 0.2ms |

Both `1/1` and `0/0` verified **character-exact** by typing into a `zenity
--entry` and comparing its echoed output — including capitals, punctuation and
digits.

`1/1` chosen over `0/0` because the fidelity test only covers GTK on native
Wayland. A nonzero hold matters for:

- **Modifier sequencing** — `Shift`+`h` with identical timestamps can be
  evaluated before the modifier applies, yielding `hello` for `Hello`.
- **Electron / Chromium** — throttles synthetic input in its own event loop.
- **XWayland** — a second event queue with its own coalescing.
- **Parsec** — forwards input over the network; compressed bursts reorder.

There is a ceiling too: a very large `key-hold` triggers kernel auto-repeat
(`hhhhh`). It's a band, not a monotonic tradeoff.

### Transcription cost

Natural speech, warm model:

| Audio | Time | Realtime factor |
|---|---|---|
| 5s | 1.45s | 3.4x |
| 8s | 1.52s | 5.3x |
| 12s | 1.77s | 6.8x |
| 20s | 3.69s | 5.4x |

Roughly **1.5s fixed cost per call**, scaling linearly beyond that. The fixed
component is the encoder: Whisper pads every input to a 30s window regardless
of clip length.

### Endpointing (`SILENCE_MS`)

Set to **500ms**. Do not drop to 300ms.

Because of the ~1.5s fixed cost, shorter phrases do **not** make text appear
sooner — the latency floor is transcription, not the pause. Lowering the
threshold adds more full-price calls, fragments sentences (each transcribed
without context, hurting punctuation and capitalization), and risks crossing
the point where the daemon transcribes slower than you speak. Past that point
the buffer grows without bound and it falls progressively behind.

---

## Rejected options

### distil-small.en — slower, not faster

| Model | 8s clip | 20s clip | WER* |
|---|---|---|---|
| `small.en` | **1.92s** | **2.04s** | 2.3% |
| `distil-small.en` | 2.23s | 2.62s | 3.4% |

Distil-Whisper freezes the encoder and distills only the **decoder** (12 layers
to 2). Its advertised ~2x applies to long-form audio where decoding dominates.
Short dictation phrases are **encoder-bound**, and `distil-small.en` shares
`small.en`'s encoder exactly — so there is nothing to save.

This also explains why `base.en` *is* 2.6x faster (0.58s vs 1.50s on 8s): it has
a genuinely smaller encoder.

> **For short-phrase dictation, encoder size is the only lever that matters.**

\* WER measured against espeak-ng synthetic speech — indicative only, not
representative of real-voice accuracy. `small.en` was also penalized for writing
"harbor"/"gray" against a British-spelled reference, so its real accuracy is
better than the figure suggests.

### beam_size

`beam_size=1` vs `5` is ~0.1s on an 8s clip. Irrelevant; kept at 5.

### NPU / new hardware

Not worth it. Battery impact is already negligible (below). Also
`faster-whisper` is CTranslate2, which targets **CPU and CUDA only** — there is
no Intel NPU path. Using one would mean migrating to OpenVINO, and Whisper's
autoregressive decoder (dynamic shapes, growing KV cache) maps poorly to NPUs
anyway; typically only the encoder offloads.

Unexplored and free: OpenVINO on the existing Iris Xe iGPU. Needs
`intel-compute-runtime` + Level Zero. Expectation is *comparable* to CPU at
lower power, not dramatically faster — Xe shares memory bandwidth with the CPU.

---

## Power

Measured via Intel RAPL package energy (requires root).

| State | Cost |
|---|---|
| Idle, daemon resident | **0.0% CPU** — 700MB RAM, no compute |
| Listening (mic open, VAD polling) | **1.8% CPU**, ~0.2W |
| Transcribing (1.5s burst) | **~19-28W package** vs ~4W idle |

Against a 57.6Wh battery:

| Usage | Drain |
|---|---|
| 100 phrases/day | 0.75 Wh — 1.3% of a charge |
| 300 phrases/day | 2.26 Wh — 3.9% |
| 600 phrases/day | 4.53 Wh — 7.9% |

Negligible. It is a spiky load with a tiny duty cycle, and **zero** when not
listening — there is no wake-word detection.

Caveat: RAPL covers CPU and iGPU only, not display or wifi. Valid for comparing
states, understates absolute system draw. Transcribing draw varied 28.4W then
18.8W across rounds — turbo boost, then the package settling to its sustained
power limit.

---

## Whisper behaviour

### Punctuation is predicted, not dictated

Whisper was trained on punctuated, cased transcripts, so punctuation is emitted
as ordinary tokens inferred from prosody and language modelling. There is **no
spoken-command vocabulary**. Saying *"i went to the store period"* produces:

> I went to the store, period,

Phrase mode has a side effect here: each phrase is transcribed independently, so
Whisper has no idea whether it is mid-sentence. A phrase continuing a thought
may come out uncapitalized, or a fragment may get a spurious capital and period.

### Repetition loops

Whisper's decoder can fall into a self-reinforcing attractor where repeating is
always the highest-probability continuation. It never emits an end-of-transcript
token, so it runs to the 448-token cap — then Whisper detects the anomaly via
`compression_ratio_threshold` (default 2.4) and **retries at higher
temperature**, costing several full decode passes.

Measured on deliberately repetitive audio (25s slice, 377 chars of ground truth):

| Config | Output | vs ground truth |
|---|---|---|
| `condition_on_previous_text=True`, no penalty | 988 chars | **2.62x — invented speech** |
| `condition_on_previous_text=False`, `repetition_penalty=1.1` | 368 chars | **0.98x — near-exact** |

Hence both guards in `_emit()`. `condition_on_previous_text` is the dangerous
one: Whisper normally feeds each 30s window's output forward as context, so a
loop in one window **propagates** — the well-documented hallucination cascade.

Cost of the guard: no cross-phrase context, so capitalization at phrase joins is
occasionally wrong. Accepted — a stray capital is cosmetic, a repetition loop is
a wall of garbage.

Trigger in practice is low-information audio (silence, noise, music, coughing).
`vad_filter=True` removes most of it before the model sees it, and phrase mode
contains any loop to a single phrase.

---

## Environment gotchas

- **`parecord --raw` into a pipe buffers and yields nothing on SIGTERM.**
  `--latency-msec=100` is required for incremental flushes. First streaming
  attempt read exactly 0 bytes.
- **`audioop` was removed in Python 3.13+.** Use `array` for WAV amplitude checks.
- **ydotool defaults to `/run/user/1000/.ydotool_socket`**, not our socket path.
  The daemon sets `YDOTOOL_SOCKET` explicitly.
- **`/dev/uinput` needs root.** We run `ydotoold` as a system service owning the
  socket to uid 1000, rather than adding the user to the `input` group — group
  changes require a full re-login.
- **Critical-urgency notifications never self-expire.** They must be closed via
  `org.freedesktop.Notifications.CloseNotification` over gdbus.
- **GNOME doesn't always honour notification expire-time.** Belt-and-braces with
  an explicit `threading.Timer` close.
- **`Super+D` is GNOME's show-desktop** and unavailable.

---

## Benchmarking discipline

Two wrong conclusions were published to the user before being caught. Both from
sloppy method:

1. **Benchmarked repetitive synthetic audio** (one sentence x14) and reported
   superlinear blowup — 25s audio taking 30s. That was the repetition-loop
   pathology, not real scaling. Natural speech is linear at 5-6x realtime.
   *Always benchmark ASR with varied natural text.*
2. **Ran two benchmarks concurrently**, inflating per-call cost from ~1.5s to
   ~2.4s; and measured power before earlier load had settled, producing the
   impossible result that transcribing drew *less* power than idle.
   *Run benchmarks sequentially, with a settling period, interleaved and
   repeated.*
