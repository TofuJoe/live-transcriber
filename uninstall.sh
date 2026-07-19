#!/usr/bin/env bash
# Remove the live-transcriber dictation stack.
# Leaves ydotool itself installed; only reverts our drop-in.

set -euo pipefail

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

say "Stopping dictation daemon"
systemctl --user disable --now dictation.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/dictation.service"
systemctl --user daemon-reload

say "Removing files"
rm -rf "$HOME/.local/share/voice-dictation"
rm -f "$HOME/.local/bin/dictate-toggle"

say "Unbinding hotkey"
KEYPATH=/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/dictation/
existing=$(gsettings get org.gnome.settings-daemon.plugins.media-keys custom-keybindings)
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \
  "$(printf '%s' "$existing" | sed "s|'$KEYPATH', *||; s|, *'$KEYPATH'||; s|'$KEYPATH'||")" 2>/dev/null || true

say "Reverting ydotoold override"
sudo rm -f /etc/systemd/system/ydotool.service.d/override.conf
sudo rmdir /etc/systemd/system/ydotool.service.d 2>/dev/null || true
sudo systemctl daemon-reload
sudo systemctl restart ydotool.service 2>/dev/null || true

say "Done. 'sudo dnf remove ydotool' if you no longer need it."
