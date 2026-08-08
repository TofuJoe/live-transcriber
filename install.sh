#!/usr/bin/env bash
# Install the live-transcriber dictation stack on Fedora / GNOME Wayland.
#
# Idempotent: safe to re-run after editing src/dictate-server.py.
# Needs sudo only for the ydotool daemon (virtual input device).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARE="$HOME/.local/share/voice-dictation"
BIN="$HOME/.local/bin"
UNIT="$HOME/.config/systemd/user"

MODEL="${DICTATE_MODEL:-small.en}"
MODE="${DICTATE_MODE:-phrase}"
HOTKEY="${DICTATE_HOTKEY:-<Super>a}"
BACKEND="${DICTATE_BACKEND:-openvino}"         # openvino | faster-whisper
OV_DEVICE="${DICTATE_OV_DEVICE:-GPU}"          # GPU | CPU | HETERO:GPU,CPU
OV_REPO="OpenVINO/whisper-small.en-int8-ov"

# Memory reservation for the daemon. Sized from observed usage: 514MB RSS
# fresh, 723MB cgroup current, 913MB peak. SESSION_MIN is what uresourced
# already gave session.slice; ACTIVE_USER_MIN must cover both children.
MEM_MIN="${DICTATE_MEMORY_MIN:-900M}"
MEM_LOW="${DICTATE_MEMORY_LOW:-1200M}"
SESSION_MIN="${DICTATE_SESSION_MIN:-250M}"
ACTIVE_USER_MIN="${DICTATE_ACTIVE_USER_MIN:-1300M}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] || say "warning: session is not Wayland (\$XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-unset})"

# ---------------------------------------------------------------- packages
say "Checking dependencies"
missing=()
for pkg in ydotool pipewire-utils libnotify libcanberra-gtk3 wl-clipboard glib2 python3; do
  rpm -q "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if ((${#missing[@]})); then
  say "Installing: ${missing[*]}"
  sudo dnf install -y "${missing[@]}"
fi

# ------------------------------------------------------- ydotool daemon
# Wayland won't let us inject keystrokes into other apps, so we need a
# virtual input device via /dev/uinput. Rather than adding the user to the
# `input` group (which needs a re-login), run ydotoold as root and hand the
# socket to this user.
say "Configuring ydotoold (socket owned by uid $(id -u))"
echo uinput | sudo tee /etc/modules-load.d/uinput.conf >/dev/null
sudo modprobe uinput
sudo mkdir -p /etc/systemd/system/ydotool.service.d
sed -e "s/@UID@/$(id -u)/" -e "s/@GID@/$(id -g)/" \
  "$REPO/systemd/ydotool-override.conf" \
  | sudo tee /etc/systemd/system/ydotool.service.d/override.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now ydotool.service
sudo systemctl restart ydotool.service

# ------------------------------------------------------------------ venv
say "Setting up Python environment"
mkdir -p "$SHARE" "$BIN" "$UNIT"
[[ -d "$SHARE/venv" ]] || python3 -m venv "$SHARE/venv"
"$SHARE/venv/bin/pip" install --quiet --upgrade pip
# faster-whisper is always needed: it supplies the Silero VAD used for
# endpointing, regardless of which backend does the transcribing.
"$SHARE/venv/bin/pip" install --quiet faster-whisper

if [[ "$BACKEND" == "openvino" ]]; then
  say "Setting up OpenVINO backend (device=$OV_DEVICE)"
  # Intel GPU compute stack: OpenCL + Level Zero. /dev/dri/renderD128 is
  # world-readable on Fedora, so no group membership is needed.
  ovpkgs=()
  for pkg in intel-compute-runtime intel-level-zero oneapi-level-zero; do
    rpm -q "$pkg" >/dev/null 2>&1 || ovpkgs+=("$pkg")
  done
  ((${#ovpkgs[@]})) && sudo dnf install -y "${ovpkgs[@]}"
  "$SHARE/venv/bin/pip" install --quiet openvino openvino-genai
  # Probe before the 245MB download -- no point fetching a model we can't run.
  if "$SHARE/venv/bin/python" -c "
import openvino as ov
devs=ov.Core().available_devices
print('  OpenVINO devices:', devs)
raise SystemExit(0 if any(d.startswith('GPU') for d in devs) or '$OV_DEVICE'=='CPU' else 1)
"; then
    if [[ ! -d "$SHARE/ov-model" ]]; then
      say "Downloading $OV_REPO (~245MB)"
      "$SHARE/venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$OV_REPO", local_dir="$SHARE/ov-model")
PY
    fi
  else
    # Degrade rather than refuse. This used to be a hard failure, which was
    # unreachable while faster-whisper was the default; now that openvino is,
    # any machine without an Intel iGPU would hit it on a plain ./install.sh.
    # The daemon already falls back to CPU when the GPU stack is missing, so
    # the installer matches that rather than contradicting it.
    say "warning: OpenVINO cannot see a GPU -- installing the faster-whisper backend"
    say "         check intel-compute-runtime, then: dictate-backend openvino"
    BACKEND=faster-whisper
  fi
fi

# ----------------------------------------------------------------- files
say "Installing daemon and client"
install -m 0644 "$REPO/src/dictate-server.py" "$SHARE/dictate-server.py"
install -m 0755 "$REPO/bin/dictate-toggle" "$BIN/dictate-toggle"
install -m 0755 "$REPO/bin/dictate-backend" "$BIN/dictate-backend"
install -m 0755 "$REPO/bin/dictate-mode" "$BIN/dictate-mode"
sed -e "s/@MODEL@/$MODEL/" -e "s/@MODE@/$MODE/" \
    -e "s/@BACKEND@/$BACKEND/" -e "s|@OV_DEVICE@|$OV_DEVICE|" \
  "$REPO/systemd/dictation.service" > "$UNIT/dictation.service"

# ---------------------------------------------------- memory protection
# The daemon holds a ~500MB model resident. Unprotected it gets reclaimed under
# memory pressure (248.9MB swapped out in a 22-minute run), which for the
# OpenVINO backend is fatal rather than merely slow: the iGPU has no VRAM, so
# its buffers are system RAM the kernel must pin, and a failed pin makes the xe
# driver ban the GPU VM. See docs/findings.md.
#
# cgroup v2 caps memory.min at every ancestor, so all three levels must move --
# service, app.slice, and the active-user reservation. Setting only the service
# does nothing at all.
say "Reserving memory for the daemon (MemoryMin=$MEM_MIN)"
mkdir -p "$UNIT/dictation.service.d" "$UNIT/app.slice.d"
sed -e "s/@MEM_MIN@/$MEM_MIN/" -e "s/@MEM_LOW@/$MEM_LOW/" \
  "$REPO/systemd/dictation-memory.conf" > "$UNIT/dictation.service.d/50-memory.conf"
sed -e "s/@MEM_MIN@/$MEM_MIN/" \
  "$REPO/systemd/app-slice-memory.conf" > "$UNIT/app.slice.d/50-dictation-memory.conf"

if [[ -f /etc/uresourced.conf ]]; then
  # uresourced owns user@.service's memory.min and rewrites it on session
  # changes, so `systemctl set-property` would not survive. Its config is the
  # only durable lever, and it has no drop-in directory.
  say "Raising the uresourced active-user reservation to $ACTIVE_USER_MIN"
  sudo python3 - "$ACTIVE_USER_MIN" "$SESSION_MIN" <<'PY'
import pathlib, re, shutil, sys

active, session = sys.argv[1], sys.argv[2]
path = pathlib.Path("/etc/uresourced.conf")
backup = pathlib.Path("/etc/uresourced.conf.pre-dictation")
if not backup.exists():
    shutil.copy2(path, backup)


def set_key(text, section, key, value):
    """Set key=value within [section], leaving comments and other keys alone."""
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == section)
    except StopIteration:
        return text.rstrip("\n") + f"\n\n{section}\n{key}={value}\n"
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].lstrip().startswith("[")), len(lines))
    for j in range(start + 1, end):
        # Matches the commented-out defaults too, which is how SessionSlice
        # ships -- it must be set explicitly or it inherits ActiveUser.
        if re.match(rf"\s*#?\s*{key}\s*=", lines[j]):
            lines[j] = f"{key}={value}"
            break
    else:
        lines.insert(start + 1, f"{key}={value}")
    return "\n".join(lines) + "\n"


text = path.read_text()
text = set_key(text, "[ActiveUser]", "MemoryMin", active)
# [SessionSlice] defaults to [ActiveUser]. Without an explicit value it
# inherits the raise and swallows the whole reservation, leaving app.slice at
# zero -- the exact state this is meant to fix.
text = set_key(text, "[SessionSlice]", "MemoryMin", session)
path.write_text(text)
PY
  sudo systemctl restart uresourced
else
  # No uresourced: nothing sets user@.service's memory.min, so it defaults to
  # 0 and would cap everything below it.
  say "uresourced not present; reserving via user@$(id -u).service directly"
  sudo mkdir -p "/etc/systemd/system/user@$(id -u).service.d"
  printf '[Service]\nMemoryMin=%s\n' "$ACTIVE_USER_MIN" \
    | sudo tee "/etc/systemd/system/user@$(id -u).service.d/50-dictation-memory.conf" >/dev/null
  sudo systemctl daemon-reload
fi

# --------------------------------------------------------------- hotkey
say "Binding hotkey: $HOTKEY"
KEYPATH=/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/dictation/
SCHEMA="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$KEYPATH"
existing=$(gsettings get org.gnome.settings-daemon.plugins.media-keys custom-keybindings)
if [[ "$existing" != *"$KEYPATH"* ]]; then
  if [[ "$existing" == "@as []" || "$existing" == "[]" ]]; then
    gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "['$KEYPATH']"
  else
    gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \
      "${existing%]}, '$KEYPATH']"
  fi
fi
gsettings set "$SCHEMA" name 'Voice Dictation Toggle'
gsettings set "$SCHEMA" command "$BIN/dictate-toggle"
gsettings set "$SCHEMA" binding "$HOTKEY"

# --------------------------------------------------------------- service
say "Starting dictation daemon (first run downloads the model)"
systemctl --user daemon-reload
systemctl --user enable --now dictation.service
systemctl --user restart dictation.service

# Confirm the reservation survived the cgroup hierarchy. Worth checking rather
# than assuming: the failure mode is silent -- every level accepts the setting
# and the effective value is still 0 if any ancestor caps it.
UID_N="$(id -u)"
CG="/sys/fs/cgroup/user.slice/user-$UID_N.slice/user@$UID_N.service/app.slice/dictation.service/memory.min"
if [[ -r "$CG" ]]; then
  eff=$(<"$CG")
  if [[ "$eff" == "0" ]]; then
    say "warning: memory reservation is not in effect (memory.min=0)."
    say "         an ancestor cgroup is capping it; see docs/findings.md"
  else
    say "Memory reserved: $((eff / 1024 / 1024))MB protected from reclaim"
  fi
fi

for _ in $(seq 1 60); do
  if journalctl --user -u dictation.service --since "-2min" 2>/dev/null | grep -q "Listening on"; then
    say "Ready. Press ${HOTKEY} to dictate."
    exit 0
  fi
  systemctl --user is-active --quiet dictation.service || die "daemon died; see: journalctl --user -u dictation -e"
  sleep 2
done
die "daemon did not become ready; see: journalctl --user -u dictation -e"
