# live-transcriber

iPhone-style push-to-dictate voice input for Fedora / GNOME Wayland, running
Whisper entirely locally on CPU.

Press a hotkey, speak, and text appears in whatever window has focus — typed as
you pause, not batched up until you stop. No cloud, no account, no network.

Built and tuned on Fedora 44, GNOME Wayland, i5-1240P (no discrete GPU).

## How it works

```
Super+A ──> dictate-toggle ──unix socket──> dictation daemon (resident)
                                                  │
                                    parecord ─────┤ streams 16kHz mono PCM
                                                  │
                                    Silero VAD ───┤ finds pauses (500ms)
                                                  │
                                    faster-whisper┤ transcribes each phrase
                                                  │
                                    ydotool ──────┴─> types into focused window
                                    wl-copy ────────> mirrors to clipboard
```

The daemon is **resident** — it keeps the Whisper model warm in RAM so the
hotkey never pays model-load cost (~8s cold, ~1s warm). The hotkey client is a
thin script that pokes a unix socket.

Text is committed **only at pause boundaries**. That is a deliberate design
choice: true streaming ASR constantly revises earlier words, and since we inject
keystrokes into apps we don't control, "correcting" would mean backspacing over
text we can't see. If the cursor moved, that destroys the wrong content. A
phrase bounded by silence on both sides is stable, so emitted text is never
revised.

## Install

```sh
./install.sh
```

Idempotent — re-run it after editing `src/dictate-server.py`. Optional overrides:

```sh
DICTATE_MODEL=base.en DICTATE_HOTKEY='<Super><Alt>d' ./install.sh
```

Needs `sudo` for one thing only: the ydotool daemon, which requires
`/dev/uinput` to create a virtual input device.

## Usage

| Action | Key |
|---|---|
| Start / stop dictation | `Super+A` |

A chime and a persistent notification confirm the mic is live. In `phrase` mode
text appears each time you pause; pressing the hotkey again ends the session and
flushes whatever is left.

Punctuation and capitalization are automatic — Whisper predicts them from
prosody and language modelling. You **cannot** dictate them explicitly: saying
"period" types the word *period*. There is no spoken-command vocabulary.

## Configuration

Runtime settings live in the systemd unit:

```sh
systemctl --user edit dictation      # override DICTATE_MODEL / DICTATE_MODE
systemctl --user restart dictation
```

| Variable | Default | Notes |
|---|---|---|
| `DICTATE_BACKEND` | `openvino` | Intel iGPU; `faster-whisper` is CPU-only |
| `DICTATE_OV_DEVICE` | `GPU` | `CPU` or `HETERO:GPU,CPU` also work |
| `DICTATE_MODEL` | `small.en` | `base.en` is ~2.6x faster, less accurate |
| `DICTATE_MODE` | `phrase` | `stream` is more accurate; `single` waits until stop |

### Choosing a mode

| Mode | Feedback | Context per decode | GPU load |
|---|---|---|---|
| `phrase` (default) | per pause | ~2s | one pass per phrase |
| `stream` | per word | up to 25s | continuous while speaking |
| `single` | none until stop | up to 25s | one pass per 25s at stop |

**Chunking, not the decoder, is the main source of word errors.** `phrase` cuts
at every pause and decodes each ~2s fragment with nothing around it, so `by
Thursday` comes out `Bye Thursday` and `on short phrases` becomes `A short
phrase is` — the fragment contains nothing that would rule them out. Measured on
one 36s passage, same backend and audio throughout:

| Mode | Word errors |
|---|---|
| `phrase` | 5/71 — 7.0% |
| `stream` | **1/71 — 1.4%** |
| `single` | 0/71 — 0.0% |

`stream` re-decodes a growing window and types each word once it has survived a
second pass (LocalAgreement-2), so it gets `single`'s context while still typing
as you talk. Because committed text is never revised, it stays safe to inject
into windows we don't own.

The cost is power: the GPU decodes continuously while you speak rather than once
per phrase. `phrase` remains the default for that reason.

```sh
dictate-mode status    # which mode is live
dictate-mode stream    # accurate, live feedback, continuous GPU
dictate-mode phrase    # back to the low-power default
```

Post-processing the *text* cannot substitute for this: every one of those
substitutions is a correctly-spelled English word (`hunspell` flags none of
them), so there is nothing for a dictionary pass to catch. The context has to
reach the decoder, not a corrector downstream.

### Choosing a backend

| Backend | 5s phrase | Decoding | Use when |
|---|---|---|---|
| `openvino` (default) | ~660ms | greedy | Speed matters — quiet room, everyday dictation |
| `faster-whisper` | ~1450ms | beam=5 | Accuracy matters — noisy rooms, technical terms |

**openvino is ~2.2x faster but greedy-only** — `openvino-genai` 2026.2.1 cannot
run beam search on GPU. In a quiet room that costs roughly one word error in
ninety; in babble it is closer to three in ninety. See
[docs/findings.md](docs/findings.md).

Switch at any time with `dictate-backend`. Both models stay on disk, so it is
just a daemon restart — about 1s to CPU, 2s to GPU:

```sh
dictate-backend status           # which one is live
dictate-backend openvino         # switch to the iGPU
dictate-backend faster-whisper   # switch back to CPU
```

It writes a systemd drop-in at
`~/.config/systemd/user/dictation.service.d/backend.conf` rather than editing
the unit, so your choice survives re-running `install.sh`. It waits for the
daemon to actually report ready and prints the backend that loaded — so a
failed switch is visible rather than silent.

The daemon logs `Ready: <backend>` on startup, and **falls back to
`faster-whisper` automatically** if OpenVINO or its model is unavailable, so a
broken GPU stack degrades rather than breaking dictation. `install.sh` does the
same: if it cannot see a GPU it warns and installs the CPU backend instead of
failing, so a machine without an Intel iGPU still gets working dictation.

The GPU backend runs in a **child process**. When the kernel bans the GPU VM
under memory pressure, OpenVINO aborts in a way no Python handler can catch, so
the pipeline is isolated behind a process boundary — the worker dies, the daemon
and your in-flight session do not. It restarts the worker once, then stays on
CPU for the rest of the session.

`faster-whisper` is always installed regardless of backend — it supplies the
Silero VAD used for endpointing.

Tunables in `src/dictate-server.py`:

| Constant | Default | Notes |
|---|---|---|
| `SILENCE_MS` | `500` | Endpointing: pause that commits a phrase |
| `MAX_SEG_S` | `25` | Force a cut before Whisper's 30s window |

`--key-hold`/`--key-delay` on the ydotool call control typing speed. **Do not
raise these without reading [docs/findings.md](docs/findings.md)** — the
defaults were the single largest source of perceived latency.

## Troubleshooting

**Nothing types.** Check the daemon and the ydotool socket:

```sh
systemctl --user status dictation
ls -l /run/ydotoold/socket          # should be owned by you
journalctl --user -u dictation -e
```

**Dropped characters or wrong capitalization in one specific app.** Raise
`--key-hold` from `1` to `5` in `type_text()`. Electron and XWayland apps
process synthetic input more slowly than native GTK.

**A "Listening" notification is stuck on screen.** It's `urgency=critical` so
it never self-expires; if the daemon was killed mid-session nothing closed it.
Dismiss it by hand.

**A phrase comes out as the same clause repeated several times.** Whisper's
repetition-loop failure mode, usually triggered by noise passing the VAD. It is
contained to that one phrase. See findings.

## Layout

```
src/dictate-server.py          resident daemon
bin/dictate-toggle             hotkey client
bin/dictate-backend            switch CPU <-> iGPU backend
bin/dictate-mode               switch phrase / stream / single
systemd/dictation.service      user service (templated)
systemd/ydotool-override.conf  system drop-in (templated)
install.sh                     deploys all of the above
docs/findings.md               measurements behind every tuning decision
```

## License

[MIT](LICENSE).

Note on dependencies: `ydotool` is AGPL-3.0, but it is invoked as a separate
process rather than linked, so its terms don't extend to this project. The
Python stack this actually imports — `faster-whisper`, `ctranslate2`,
`onnxruntime` — is MIT, and the Whisper models themselves are MIT.

## Requirements

Fedora with GNOME on Wayland; `ydotool`, `pipewire-utils`, `libnotify`,
`libcanberra-gtk3`, `wl-clipboard`, `glib2`, `python3`. `install.sh` installs
any that are missing.
