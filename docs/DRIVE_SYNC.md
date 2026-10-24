# Uploading finished models to Google Drive

When a training run finishes successfully, the trainer writes `done.json`
(model, best epoch, validation mAP) and `upload.txt` (the files to share) into
`runs/<timestamp>_<experiment>/`. A user systemd timer runs
`scripts/sync_models_to_drive.sh` every five minutes and pushes each completed
run to `<run>/` inside the shared models folder with `rclone sync`, which only
transfers files that differ from the remote and moves anything else in that
run's Drive folder to the trash. If the machine is offline, the
run uploads on the first tick after the connection returns.

Only the best and last weights (plus `done.json`) are uploaded, never
plots or per-epoch checkpoints:

| Model | Files |
|---|---|
| YOLO | `weights/best.pt`, `weights/last.pt` |
| D-FINE | `best_stg1.pth`, `best_stg2.pth`, `last.pth` |
| RT-DETRv2 | `best.pth`, `last.pth` |
| RF-DETR | `checkpoint_best_total.pth`, `last*.pth` |

Resuming a run (`--resume`) removes its `done.json`, so nothing uploads
while it trains further; the Drive copy is refreshed when the resumed run
finishes.

## One-time setup

### 1. Create a remote with write access

The `gdrive` remote used by `fetch-drive` is read-only, so model uploads use
a separate remote, `gdrive-models`. Use your own Google Cloud OAuth client
(the one created for DracoViLoc works; see steps 2 and 4 of DracoViLoc's
`docs/DRIVE_SYNC.md`), because rclone's shared default client hits Google's
rate limit.

```bash
rclone config create gdrive-models drive scope=drive \
  root_folder_id=1ptxQRwCKWE8-DYbIK-FYwNptAi5xLl7g \
  client_id=YOUR_CLIENT_ID client_secret=YOUR_CLIENT_SECRET
```

`root_folder_id` is the ID at the end of the shared folder's link
(`drive.google.com/drive/folders/<ID>`).

A browser opens once to log in. On a machine without a browser, add
`config_is_local=false` and follow the printed `rclone authorize` step.
Check it:

```bash
rclone lsd gdrive-models:
```

To upload into a subfolder of it instead, create
`~/.config/drone-ai/drive_sync.env`:

```bash
DRIVE_SYNC_REMOTE_NAME=gdrive-models
DRIVE_SYNC_REMOTE_PATH=some/subfolder
```

### 2. Install the timer

```bash
scripts/install_drive_sync.sh
sudo loginctl enable-linger "$USER"   # keep the timer running when logged out
```

### 3. Share the folder

Anyone with the shared folder's link sees each run as it uploads.

## Day to day

```bash
# Mark a run that finished before this existed, or a run whose trainer was started earlier
python -m drone_ai.train mark-done runs/<run>

# Upload now with a progress bar instead of waiting for the timer
scripts/sync_models_to_drive.sh --progress

systemctl --user list-timers drone-ai-model-sync.timer
tail -f logs/drive_sync.log
```
