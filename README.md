# Drone YOLO Training

This repository is set up to train YOLO object detectors on the Anti-UAV-RGBT
dataset. The raw dataset contains videos plus frame-wise JSON annotations, so it
must be converted before Ultralytics YOLO can train on it.

## Environment

```bash
conda create -n drone-yolo python=3.11 -y
conda activate drone-yolo

pip install --upgrade pip
pip install ultralytics opencv-python torch torchvision tqdm pandas pyyaml matplotlib
conda install -c conda-forge ffmpeg -y
```

Verify the environment:

```bash
python -c "import torch, cv2, ultralytics; print('torch', torch.__version__); print('mps', torch.backends.mps.is_available()); print('cv2', cv2.__version__)"
yolo checks
```

## Check Dataset

```bash
find Anti-UAV-RGBT -maxdepth 1 -type d
find Anti-UAV-RGBT/train -mindepth 1 -maxdepth 1 -type d | wc -l
find Anti-UAV-RGBT/val -mindepth 1 -maxdepth 1 -type d | wc -l
find Anti-UAV-RGBT/test -mindepth 1 -maxdepth 1 -type d | wc -l

find Anti-UAV-RGBT -maxdepth 3 -type f -name "*.mp4" | wc -l
find Anti-UAV-RGBT -maxdepth 3 -type f -name "*.json" | wc -l
```

Inspect one video and annotation pair:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
  -of default=nw=1 \
  Anti-UAV-RGBT/train/20190925_101846_1_2/visible.mp4

python -c "import json; p='Anti-UAV-RGBT/train/20190925_101846_1_2/visible.json'; d=json.load(open(p)); print(d.keys()); print(len(d['exist']), len(d['gt_rect'])); print(d['gt_rect'][0])"
```

Preview boxes on a video:

```bash
python scripts/view_antiuav_boxes.py \
  --sequence Anti-UAV-RGBT/train/20190925_101846_1_2 \
  --modality visible \
  --start-frame 0 \
  --delay 30
```

Browse the extracted frames in `gem_dataset`:

```bash
python scripts/view_gem_frames.py
```

Use Left/Right to move between frames, Up/Down to move between sequence
folders, and Q or Esc to close the OpenCV window. The viewer also accepts
A/D and W/S as fallback controls. To browse a different copy of the dataset,
pass its path with `--root`.

## Convert To YOLO

The raw `Anti-UAV-RGBT` directory is only read by the converter. The generated
YOLO dataset is written to `--output`, so keep `--output` outside
`Anti-UAV-RGBT`.

Recommended first dataset creation command:

```bash
conda activate drone-yolo

python scripts/convert_antiuav_to_yolo.py \
  --root Anti-UAV-RGBT \
  --output datasets/antiuav_yolo_visible_s10 \
  --modalities visible \
  --stride 10 \
  --workers 4 \
  --overwrite
```

This creates a YOLO dataset from visible-camera frames, keeping every 10th frame.
It is a good first training dataset because it is much smaller than extracting
every frame.

Useful parameters:

```text
--root
  Location of the original Anti-UAV-RGBT dataset. The script reads from this
  directory and does not modify it.

--output
  Location where the YOLO-format dataset will be created.

--modalities
  Which video streams to convert. Use "visible", "infrared", or both:
  --modalities visible infrared

--stride
  Frame sampling interval. --stride 1 keeps every frame. --stride 10 keeps
  frame 0, 10, 20, 30, ...

--workers
  Number of parallel worker threads. Use 4 on this Mac as a reasonable default.
  Increase only if disk and CPU are not saturated.

--max-sequences-per-split
  Debug/testing limiter. For example, 1 converts only one train sequence, one
  val sequence, and one test sequence. Do not use this for the real dataset.

--overwrite
  Deletes and recreates only the output directory. It does not touch
  Anti-UAV-RGBT.
```

Quick smoke conversion for testing the converter:

```bash
python scripts/convert_antiuav_to_yolo.py \
  --root Anti-UAV-RGBT \
  --output outputs/antiuav_yolo_smoke \
  --modalities visible \
  --stride 100 \
  --max-sequences-per-split 1 \
  --workers 4 \
  --overwrite
```

Full visible-frame conversion:

```bash
python scripts/convert_antiuav_to_yolo.py \
  --root Anti-UAV-RGBT \
  --output datasets/antiuav_yolo_visible_full \
  --modalities visible \
  --stride 1 \
  --workers 4 \
  --overwrite
```

The converter writes:

```text
datasets/.../images/train
datasets/.../images/val
datasets/.../images/test
datasets/.../labels/train
datasets/.../labels/val
datasets/.../labels/test
datasets/.../dataset.yaml
datasets/.../conversion_stats.json
```

## Train

All training runs are saved as `runs/DD_MM_YYYY_HH_MM_SS_<experiment>`:

```bash
python scripts/train_drone_model.py \
  --experiment yolo11n_drone_merged \
  --model yolo11n.pt \
  --data datasets/drone_merged/dataset.yaml \
  --epochs 60 \
  --imgsz 640 \
  --batch 32 \
  --device mps
```

All annotated videos and inference metadata are saved as
`artifacts/DD_MM_YYYY_HH_MM_SS_<experiment>`:

```bash
python scripts/predict_videos_with_progress.py \
  --experiment yolo11n_drone_merged_real_scenario \
  --model runs/<training-run>/weights/best.pt \
  --source real-scenario \
  --device mps
```

For a legacy job that was started before this layout was introduced, wait for
its process to exit and then move it without hand-building the directory name:

```bash
python scripts/finalize_experiment.py training runs/old_name --experiment old_name
python scripts/finalize_experiment.py artifact path/to/predictions --experiment old_predictions
```

Smoke train from config:

```bash
python scripts/train_drone_model.py \
  --experiment antiuav_smoke \
  --model yolo11n.yaml \
  --data outputs/antiuav_yolo_smoke/dataset.yaml \
  --epochs 1 \
  --imgsz 320 \
  --batch 2 \
  --workers 0
```

First real run:

```bash
python scripts/train_drone_model.py \
  --experiment antiuav_visible_s10 \
  --model yolo11n.pt \
  --data datasets/antiuav_yolo_visible_s10/dataset.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --workers 0 \
  --device mps
```

If Ultralytics falls back to CPU, remove `device=mps` and reduce `batch`.
# drone-ai
