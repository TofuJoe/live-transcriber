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

### VAD rescan — was O(n^2), now O(1)

The worker polls every 0.4s. The original `find_commit_point` ran Silero VAD
over the **entire** buffer each poll, so cost grew with phrase length:

| Buffer | Full scan | Tail scan | % of a core (full scan, per 0.4s poll) |
|---|---|---|---|
| 5s | 6.2ms | 1.1ms | 2.0% |
| 10s | 12.6ms | 1.1ms | 4.8% |
| 20s | 25.3ms | 1.1ms | 10.2% |
| 25s | 34.2ms | 1.2ms | 11.9% |

Total work over a phrase was O(n^2) — for a 25s phrase, ~1.5 core-seconds of
VAD against ~12 core-seconds of transcription, so ~11% of compute spent
re-scanning already-scanned audio.

Only the **tail** is needed: if the last `TAIL_MS` has gone quiet, the phrase is
over. Cost is now flat regardless of phrase length.

The tail alone is not quite sufficient — it can't tell whether any speech
happened since the last cut, so the silence left in the buffer after a commit
re-triggers immediately and emits empty chunks. First attempt produced 4 commits
where the old code produced 2. Fixed by threading one bit of state
(`speech_seen`) through the poll loop.

Verified byte-identical transcription output against the old implementation on a
two-phrase stream, at 2.6x lower total VAD cost.

`_emit` also gates on `has_speech` now: one VAD pass per *commit* (not per poll)
avoids paying ~1.5s of transcription on a chunk that turns out to be silence.

### Sample rate and bit depth — no headroom here

Asked whether lowering either would save power. It would not:

- **16kHz mono is fixed by the model.** Whisper's feature extractor builds an
  80-bin log-Mel spectrogram from 16kHz audio. Recording at 48kHz would mean
  downsampling before inference — strictly more work. Capturing at 16k lets
  PipeWire do the conversion once in its own pipeline.
- **`s16` is already the sensible floor.** ~96dB dynamic range against speech
  occupying ~40dB, and Whisper converts to float32 regardless.

| Format | Data rate |
|---|---|
| current (16k s16 mono) | **31 kB/s** |
| s32 instead | 62 kB/s |
| hardware native (48k s32 4ch) | 750 kB/s |

At 31 kB/s a full minute is under 2MB. Audio I/O is nowhere near the energy
budget — transcription is ~90% of it.

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

### RNNoise noise suppression — actively harmful

Tested via ffmpeg's built-in `arnndn` filter across three independently trained
models. WER against the same reference utterance:

| Model | Clean | Babble 5dB SNR |
|---|---|---|
| none | 2.3% | **40.9%** |
| `sh` (somnolent-hogwash) | 2.3% | 63.6% |
| `bd` (beguiling-drafter) | 0.0% | 65.9% |
| `mp` (marathon-prescription) | 0.0% | 58.0% |

Neutral on clean speech (the 0.0% vs 2.3% is one or two words on a single
utterance — noise), and **consistently much worse on babble**.

RNNoise suppresses *non-speech* noise; babble is speech, so it cannot remove it
and instead distorts the target speaker while trying. Whisper was trained on
680k hours of noisy real-world audio and is already noise-robust — what it is
*not* robust to is unfamiliar front-end processing artifacts. Front-end
enhancement degrading robust ASR is a well-established result; RNNoise optimises
perceptual quality for human listeners, a different objective.

Not installed. Fedora has no package for it anyway (`arnndn` is built into
ffmpeg; models come from GregorR/rnnoise-models).

**Gotcha:** RNNoise runs at 48kHz. ffmpeg auto-resamples input up but leaves the
output at 48kHz — feed that to a 16kHz pipeline unchanged and it plays at a
third speed. Chain `aresample=48000,arnndn=...,aresample=16000`.

### Iris Xe iGPU via OpenVINO — works, ~2.4x faster, but costs accuracy

Tested properly rather than assumed. Installed `intel-compute-runtime` +
`intel-level-zero` (Fedora repos) and `openvino` / `openvino-genai` (PyPI), using
the pre-converted `OpenVINO/whisper-small.en-int8-ov` model. `/dev/dri/renderD128`
is world-readable, so no group changes were needed.

**Encoder only**, interleaved, thread parity, single process:

| Backend | median | vs current |
|---|---|---|
| CTranslate2 CPU (current) | 3021ms | 1.00x |
| OpenVINO CPU | 2287ms | 1.32x |
| OpenVINO iGPU | 1109ms | **2.72x** |

**End-to-end**, mean WER across clean / white-10dB / babble-10dB / babble-5dB:

| Backend | 5s clip | mean WER |
|---|---|---|
| **ct2 CPU beam=5 (current)** | 2285ms | **13.6%** |
| ov CPU beam=5 | 2131ms | 15.9% |
| ov GPU greedy | **953ms** | 16.2% |

**Verdict: shipped as a switchable backend** (`DICTATE_BACKEND=openvino`), not
as the default. Measured in the installed daemon: 656ms vs 1456ms on a 5s
phrase, ~2.2x. See "What the WER gap actually was" below — the accuracy cost is
much smaller than the raw WER numbers suggested.

**Why the GPU is fast: it can't do beam search.** `openvino-genai` 2026.2.1 fails
beam search on GPU — `Not Implemented` on remote tensors at `num_beams=2`, and
`Logits batch size doesn't match the number of beams` at 5. It works on OpenVINO
CPU, so this is a GPU-path limitation, not a Whisper one.

**Beam search is worth +4.5% WER, consistently:**

| Condition | beam=1 | beam=5 | gain |
|---|---|---|---|
| clean | 4.5% | 0.0% | +4.5 |
| white 10dB | 10.2% | 6.8% | +3.4 |
| babble 10dB | 15.9% | 10.2% | +5.7 |
| babble 5dB | 42.0% | 37.5% | +4.5 |
| **mean** | 18.2% | 13.6% | **+4.5** |

Consistent across all four conditions, and it costs only ~6% more time on CPU
(2220ms vs 2363ms). That is why `beam_size=5` stays.

OpenVINO CPU *with* beam=5 is only 7% faster than CTranslate2 and still less
accurate (15.9% vs 13.6%), so it isn't a free win either.

**Method note:** the first end-to-end comparison was unfair — `openvino-genai`
defaults to `num_beams=1` while faster-whisper was on `beam_size=5`. It looked
like a 2.8x speedup at equal accuracy; matching decoders showed the speed was
partly bought with greedy decoding. At matched greedy both produce *identical*
4.5% WER, which is how the beam-search explanation was isolated.

Side finding: `repetition_penalty=1.1` costs a little accuracy on clean speech
(2.3% vs 0.0% WER). Kept anyway — it guards the repetition-loop failure mode,
which is far more destructive than a single word error.

### What the WER gap actually was — mostly spelling

The "+4.5% WER" figure above is **substantially inflated**, and it nearly drove
the wrong decision. Inspecting the individual error sites on clean speech:

```
beam=5:  0 error sites in 88 words
beam=1:  3 error sites in 88 words
    'harbour' -> 'harbor'      <- spelling variant, not an error
    'come in' -> 'coming'      <- genuine error
    'grey'    -> 'gray'        <- spelling variant, not an error
```

The reference text is British-spelled; `small.en` is trained largely on American
English. **Greedy's real error rate on clean speech is ~1 word in 88, not 4.5%.**

Worse, the model's spelling choice wanders with unrelated parameters — `beam=5`
scored 2.3% in one run and 0.0% in another purely because `repetition_penalty`
flipped `harbour`/`harbor`. A meaningful fraction of *every* WER number in this
document is spelling noise.

Genuine error counts, beam=5 vs beam=1:

| Condition | beam=5 | beam=1 | real difference |
|---|---|---|---|
| clean | 0 | 1 | 1 word in 88 |
| white 10dB | 4 | 5 | 1 |
| babble 10dB | 6 | 9 | 3 |

In noise the errors are real (`'grey sky'->'race guy'`, `'bought'->'walked'`) and
beam search genuinely helps. In a quiet room it barely matters — which is what
made the GPU backend worth shipping.

**Lesson: never act on an aggregate WER without reading the actual errors.**
Normalise spelling variants, or inspect the diff. This was the only measurement
mistake in this project that changed a recommendation rather than just a number.

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

## iGPU reliability — the VM ban, and why the in-process fallback never worked

Investigated 2026-08-04 after dictation went silent mid-session. Worth reading
before touching `Transcriber`: the original design here was wrong in a way that
looked right in the logs.

### Symptom

```
22:47:40  kernel: xe 0000:00:02.0: [drm] VM worker error: -12
22:47:40  kernel: xe 0000:00:02.0: [drm] Trying to schedule after vm is closed or banned
22:47:41  python: GPU backend died (...); falling back to CPU
22:47:55  systemd: Main process exited, code=dumped, status=6/ABRT
22:47:58  systemd: Started dictation.service
```

The fallback fires and logs the right thing. **Then the process dies anyway,
14 seconds later.** systemd restarts it, but the live `Session` goes with it —
the recording, the "Listening…" banner, and every phrase queued in
`self.queue`. From the user's seat dictation simply stops, which reads as
"transcription is taking way too long" rather than as a crash.

Not a one-off: same `-12` on 2026-08-02 21:44, followed by restarts at 21:50,
22:01, 22:06 and 22:17.

### Root cause

`-12` is `ENOMEM`. The iGPU has no VRAM, so OpenVINO's buffers are system RAM
the kernel must pin into the GPU's address space; under memory pressure that
bind fails and the `xe` driver bans the VM.

The abort is the interesting part. Reproduced by running `generate()` in a loop:

```
RuntimeError: ... clWaitForEvents, error code: -14 CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
terminate called after throwing an instance of 'ov::Exception'
  what(): ... [GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES
→ core dumped
```

Two distinct events. Python sees the **first** — a catchable `RuntimeError` —
and the `except` in `Transcriber.transcribe` handles it correctly. OpenVINO then
tears its command stream down on a **background thread**, that destructor throws
with no handler on the thread, and `std::terminate` aborts the process.

> **No Python `except` can catch a C++ exception on a thread you don't own.**
> The old docstring's claim — "the sole recovery is to stop using the GPU …
> falling back to CPU keeps dictation working" — was untestable optimism. The
> fallback ran to completion and the process died regardless.

A catchable error from the GPU is therefore not a bad phrase, it is the first
symptom of a dying context. `OpenVinoBackend.transcribe` treats it as fatal to
the worker.

### Fix: a process boundary

The pipeline moved into a spawned child (`_ov_worker`). The parent never imports
`openvino` — doing so would put the abort back in the process being protected.

Verified by sending the child `SIGABRT`, the exact signal from the incident:

| | before | after |
|---|---|---|
| daemon survives | no — `status=6/ABRT` | **yes** |
| live session survives | no | yes |
| queued phrases | lost | typed |
| detection | n/a | `WorkerDied: exit code -6` in **2.4s** |
| the failed phrase | lost | re-run on the replacement |

Detection polls `is_alive()` between short queue waits rather than blocking on
`get(timeout=…)`. An aborted worker sends *nothing*, so a bare blocking get
would stall for the full 60s timeout on the most common failure.

**The ban is per-process**, which is a recovery the old design could not have
had: a fresh worker gets a fresh VM and usually comes back on the GPU. So the
policy is restart-once, then CPU permanently — one retry is worth it at ~2.2x
per phrase, and a second failure means the machine genuinely has no room.

### The other half: stop losing the pages

Isolation makes the ban survivable; it doesn't make it rarer. The trigger was
ordinary memory pressure, and the daemon had **no protection from reclaim** —
`MemoryMin=0`, with 248.9MB swapped out in a 22-minute run.

`memory.min` is capped by every ancestor cgroup, so a drop-in on
`dictation.service` alone is **inert**. All of these had to move:

| cgroup | before | after |
|---|---|---|
| `user@1000.service` | 250M | 1300M |
| ` ├─ session.slice` | 250M *(claimed all of it)* | 250M |
| ` └─ app.slice` | **0** | 900M |
| `    └─ dictation.service` | **0** | **900M** |

The top three are owned by `uresourced` (`/etc/uresourced.conf`), not systemd
units. `[SessionSlice]` **defaults to `[ActiveUser]`**, so raising the ceiling
alone does nothing — session.slice inherits the increase and swallows it. It
must be pinned explicitly.

Sized from observed usage: 514MB RSS fresh, 723MB cgroup current, 913MB peak.

`MemoryLow` is set on the leaf but is currently **inert** — uresourced pins
`memory.low` to 0 up the chain, so nothing propagates. `MemoryMin` does the real
work; the `MemoryLow` line is intent, not effect.

### Non-finding: there is no memory leak

The pre-crash process showed a 1.6GB peak against 514MB fresh, which looked like
a leak and is not one. 40 sequential `generate()` calls:

```
after warm-up:  anon = 310.0 MB
  iter   5:     anon = 319.4 MB (+9.4)
  iter  20:     anon = 319.5 MB (+9.5)
```

One-time +9.5MB, then flat. The 1.6GB is `MemoryPeak` counting **page cache**
(360MB of model file cache at a fresh start alone) plus the ~500MB FasterWhisper
model loaded during the failed fallback. Rules out a fast leak; 20 iterations
can't rule out a very slow one.

**Harness caveat, per the discipline section below:** the loop that reproduced
the abort held a *second* copy of the model on the iGPU alongside the running
daemon's, roughly doubling GPU memory demand. It is good evidence for the abort
*mechanism*, not evidence that the daemon alone fails at ~20 phrases.

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

## Noise robustness

WER vs SNR, same reference utterance. Synthetic speech with synthetic noise, so
treat absolute numbers as indicative — the *shape* is the finding.

| Interference | 20dB | 10dB | 5dB | 0dB |
|---|---|---|---|---|
| Steady (HVAC, hum) | 6.8% | 13.6% | 11.4% | 38.6% |
| Babble (competing speech) | 5.7% | 11.4% | **43.2%** | **100%** |

*(clean baseline 2.3%)*

**Steady noise degrades gracefully; competing speech falls off a cliff between
10dB and 5dB.** Whisper has no speaker targeting — it does not know which voice
is yours. At 0dB babble it emitted nothing at all, which is the safe failure.

Practical consequence: a close-talk/headset mic is worth more than any software
change, because it moves you from the ~5dB regime to the ~20dB regime where WER
is around 6%.

## Microphone array

`Mic1` exposes 4 channels (`s32le 4ch 48000Hz`). Measured per-channel:

| Channel | RMS | Verdict |
|---|---|---|
| 0 | 0.0167 | live |
| 1 | 0.0000 | **dead — not a mic** |
| 2 | 0.0174 | live |
| 3 | 0.0179 | live |

On Windows this array is processed in the Intel SST DSP by proprietary OEM
firmware (beamforming, NR, AEC) and applications see one clean mono stream. On
Linux, SOF is open firmware without those blobs, so the raw array is exposed and
any processing must happen in software.

**The dead channel is harmless.** Averaging `(ch0+0+ch2+ch3)/4` is exactly ¾ of
`(ch0+ch2+ch3)/3` — a uniform scalar that attenuates signal and noise equally.
SNR is identical; it costs 2.5dB of level at −41dBFS, still ~55dB above the
16-bit floor, and Whisper normalises input anyway.

**The naive downmix is already a crude beamformer.** Averaging is delay-and-sum
with zero delay — a broadside beamformer, steered perpendicular to the array,
which for a laptop lid is roughly where the user's face is. Channels 0 and 2
correlated only **0.15** on ambient noise (spatially incoherent diffuse noise
cancels on averaging) while coherent point-source speech sums constructively.
That is a real ~√3 SNR gain obtained for free.

Steered beamforming (MVDR, delay-and-sum with real delays) would need physical
mic geometry in millimetres, which OEMs don't publish and ACPI rarely exposes.
Not worth reverse-engineering for the margin over broadside averaging.

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
3. **Fed 48kHz audio to a 16kHz pipeline** while testing RNNoise, reporting 100%
   WER across every condition. The audio was playing at a third speed; the
   denoiser was fine. *Assert the sample rate when loading audio* — the reader
   now does.

Common thread: every wrong conclusion came from the harness, not the system
under test. A result that looks dramatic (superlinear blowup, negative power
draw, total failure) is far more likely to be a measurement bug than a real
discovery. Check the harness first.

### base.en — fast enough, not accurate enough (tried in real use 2026-07-20)

Ran `OpenVINO/whisper-base.en-int8-ov` as the live daily-driver model, then
reverted the same day. Latency was not the problem; accuracy was.

| | Params | Enc/dec layers | Width | On-disk (int8 OV) |
|---|---|---|---|---|
| `small.en` | 244M | 12 | 768 | 245 MB |
| `base.en` | 74M | 6 | 512 | 81 MB |

**Failure mode: confabulation, not garbling.** `base.en` emitted fluent,
grammatical words that were never said — plausible enough to read past, which is
worse than obvious noise because you don't catch it while dictating. A weaker
decoder leans harder on its language prior when the acoustic evidence is
ambiguous, so errors arrive pre-laundered into reasonable English.

Two things compounded it:

- **Greedy decoding.** The OpenVINO backend runs greedy (no beam search). On
  `small.en` that costs ~1 word; on `base.en` it stacks a weaker LM on top of no
  search, so a bad first token has nothing to recover it. `base.en` +
  `beam_size=5` on the CPU backend would fare better — but hands back the speed
  that motivated the switch.
- **Technical vocabulary.** btrfs, ydotool, OpenVINO, systemd. Proper nouns and
  jargon are precisely where parameter count pays, and they are most of what
  gets dictated here.

> **`small.en` is the floor for this workload.** The accuracy budget is already
> spent on greedy decoding (worth it: 2.2x). Spending it a second time on a
> smaller encoder compounds two hits for a shrinking latency return — see the
> budget note below.

**Why the latency case had weakened anyway.** The lever ranking in this doc was
written when transcription cost ~1456ms on CPU. OpenVINO cut that to ~656ms, so
the endpoint silence — not the model — became the dominant term. Moving to
`base.en` would have recovered ~350ms of a ~2.3s budget while the ~800-1200ms
endpoint sat untouched. Wrong knob.

The model dir is kept at `~/.local/share/voice-dictation/ov-model-base.en`;
re-test by pointing `DICTATE_OV_MODEL` at it via a systemd drop-in.

### Moonshine v2 — the non-padding architecture doesn't help here (2026-07-20)

Whisper pads every clip to a 30s encoder window, so a 3s phrase costs what a 25s
one does. [Moonshine v2](https://arxiv.org/html/2602.12241v1) (MIT, English-only,
onnxruntime) encodes variable length with no zero-padding — on paper the exact fix
for short-phrase dictation. Tested against the deployed `small.en`. **Rejected.**

Latency vs clip duration, same 20s recording of natural speech truncated to each
length, best of 3 after a warm-up, sequential:

| clip | `small.en`/iGPU | moonshine-M/CPU (245M) | moonshine-S/CPU (123M) |
|---:|---:|---:|---:|
| **2s** | **421ms** | 441ms | 418ms |
| **3s** | **464ms** | 896ms | 799ms |
| **5s** | **517ms** | 1720ms | 1033ms |
| 8s | 736ms | 2880ms | 2112ms |
| 12s | 869ms | 4402ms | 3322ms |
| 20s | 1041ms | 7058ms | 4917ms |

**The mechanism is real; the magnitude is not.** The curves behave exactly as the
architectures predict — Whisper is near-flat (2.5x cost over a 10x duration range;
fixed encoder dominates, only the decoder scales), Moonshine scales ~linearly (16x
over the same range). But the crossover lands at **~2s**, and that is the only
point where Moonshine is competitive. Across the 3-5s phrases this tool actually
handles, `small.en` is 2-3x faster.

Whisper's fixed encoder cost is only **421ms on the iGPU** — low enough that the
padding waste never becomes the bottleneck it looks like on paper.

**Accuracy was a tie.** On real voice both produced the same text, differing on one
phrase ("which ... warn against" vs "with ... worn against"); `small.en` was
correct. On a synthetic-TTS smoke test Moonshine-M got `openvino` right where
`small.en` wrote "opinveno" — a single data point on unrepresentative audio, noted
only because jargon is the axis that killed `base.en`.

**Caveat, stated honestly:** Moonshine ran on CPU, `small.en` on the iGPU.
A fair fight needs `onnxruntime-openvino`. But Moonshine would need ~3x from iGPU
acceleration just to draw level at 5s, and it would still be scaling linearly
against a flat curve — so it loses again on anything longer. Not pursued.

> **The useful corollary:** transcription cost barely moves between a 2s and a 5s
> phrase (421ms -> 517ms). Cutting phrases sooner buys **nothing** in transcription
> time — it only changes when the clock starts. The endpoint silence is the entire
> remaining latency budget. See the pause-gap logging added the same day.

Harness kept in the session scratchpad (`sweep.py`, `compare.sh`,
`bench_moonshine.py`, `bench_whisper.py`); models cached under
`~/.cache/moonshine_voice` (665MB) — delete if not revisiting.
