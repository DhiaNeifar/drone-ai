#!/usr/bin/env bash
# Pushes finished training runs to Google Drive via rclone. A run is uploaded
# once it has done.json (written by the trainer when training succeeds), and
# only the files listed in its upload.txt (best and last weights, done.json)
# are kept in the run's Drive folder; anything else there is moved to the
# Drive trash. Safe to run repeatedly and with no network: rclone only
# transfers files that differ from the remote, so a failed upload is retried
# on the next invocation. Driven by
# drone-ai-model-sync.timer; run by hand with --progress to watch an upload.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_FILE="${DRIVE_SYNC_CONFIG:-$HOME/.config/drone-ai/drive_sync.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

RUNS_ROOT="${DRIVE_SYNC_RUNS_ROOT:-${REPO_ROOT}/runs}"
REMOTE_NAME="${DRIVE_SYNC_REMOTE_NAME:-gdrive-models}"
# Empty: runs go straight into the remote's root_folder_id (the shared models folder).
REMOTE_PATH="${DRIVE_SYNC_REMOTE_PATH-}"
LOG_FILE="${DRIVE_SYNC_LOG_FILE:-${REPO_ROOT}/logs/drive_sync.log}"
LOCK_FILE="${DRIVE_SYNC_LOCK_FILE:-${REPO_ROOT}/logs/.drive_sync.lock}"
RCLONE_FLAGS=(--retries 5 --low-level-retries 10 --checkers 4 --transfers 2
  --contimeout 15s --timeout 60s)
[[ "${1:-}" == "--progress" ]] && RCLONE_FLAGS+=(--progress)

mkdir -p "$(dirname "${LOG_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "$(date -Is) skip: a sync is already running" >> "${LOG_FILE}"
  exit 0
fi

log() { echo "$(date -Is) $*" | tee -a "${LOG_FILE}"; }

if ! command -v rclone >/dev/null 2>&1; then
  log "rclone is not installed; see docs/DRIVE_SYNC.md"
  exit 1
fi

if ! rclone listremotes 2>/dev/null | grep -qx "${REMOTE_NAME}:"; then
  log "remote '${REMOTE_NAME}:' is not configured; see docs/DRIVE_SYNC.md"
  exit 1
fi

synced=0
failed=0
for run_dir in "${RUNS_ROOT}"/*/; do
  run_dir="${run_dir%/}"
  [[ -f "${run_dir}/done.json" && -f "${run_dir}/upload.txt" ]] || continue
  run="$(basename "${run_dir}")"
  if rclone sync "${run_dir}" "${REMOTE_NAME}:${REMOTE_PATH:+${REMOTE_PATH}/}${run}" \
      --files-from "${run_dir}/upload.txt" --delete-excluded "${RCLONE_FLAGS[@]}" 2>> "${LOG_FILE}"; then
    synced=$((synced + 1))
  else
    failed=$((failed + 1))
    log "FAILED ${run} (will retry next run)"
  fi
done

log "done: ${synced} run(s) up to date on ${REMOTE_NAME}:${REMOTE_PATH}, ${failed} failed"
[[ "${failed}" -eq 0 ]]
