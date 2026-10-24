#!/usr/bin/env bash
# Installs the background Google Drive model upload as a user systemd timer.
# Safe to re-run after editing the unit files or the sync script. Does NOT
# configure the rclone remote itself -- that is a one-time interactive step
# tied to your Google account; see docs/DRIVE_SYNC.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"

mkdir -p "${UNIT_DIR}"
# The repo is not under $HOME, so the service gets its absolute path.
sed "s|@REPO_ROOT@|${REPO_ROOT}|" "${SCRIPT_DIR}/systemd/drone-ai-model-sync.service" \
  > "${UNIT_DIR}/drone-ai-model-sync.service"
cp "${SCRIPT_DIR}/systemd/drone-ai-model-sync.timer" "${UNIT_DIR}/"
chmod +x "${SCRIPT_DIR}/sync_models_to_drive.sh"

systemctl --user daemon-reload
systemctl --user enable --now drone-ai-model-sync.timer

if ! loginctl show-user "$(whoami)" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
  echo "Warning: lingering is not enabled for $(whoami)." >&2
  echo "Without it the timer only runs while you are logged in. Enable with:" >&2
  echo "  sudo loginctl enable-linger $(whoami)" >&2
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "Warning: rclone is not installed yet (sudo apt install rclone)." >&2
elif ! rclone listremotes 2>/dev/null | grep -qx "gdrive-models:"; then
  echo "Warning: the rclone remote 'gdrive-models' is not configured yet." >&2
  echo "See docs/DRIVE_SYNC.md for the one-time Google Drive authorization." >&2
fi

echo "Installed. Check status with: systemctl --user list-timers drone-ai-model-sync.timer"
echo "Check logs with: tail -f ${REPO_ROOT}/logs/drive_sync.log"
