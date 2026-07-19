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
"$SHARE/venv/bin/pip" install --quiet faster-whisper

# ----------------------------------------------------------------- files
say "Installing daemon and client"
install -m 0644 "$REPO/src/dictate-server.py" "$SHARE/dictate-server.py"
install -m 0755 "$REPO/bin/dictate-toggle" "$BIN/dictate-toggle"
sed -e "s/@MODEL@/$MODEL/" -e "s/@MODE@/$MODE/" \
  "$REPO/systemd/dictation.service" > "$UNIT/dictation.service"

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

for _ in $(seq 1 60); do
  if journalctl --user -u dictation.service --since "-2min" 2>/dev/null | grep -q "Listening on"; then
    say "Ready. Press ${HOTKEY} to dictate."
    exit 0
  fi
  systemctl --user is-active --quiet dictation.service || die "daemon died; see: journalctl --user -u dictation -e"
  sleep 2
done
die "daemon did not become ready; see: journalctl --user -u dictation -e"
