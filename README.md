# Drone Detection with YOLO

This repository contains the training, review, preprocessing, and inference
tools for a single-class YOLO drone detector. The processed dataset is stored
separately in the private Hugging Face dataset repository
[`DhiaNeifar/Akkurad_Drone_Detection`](https://huggingface.co/datasets/DhiaNeifar/Akkurad_Drone_Detection).

The Git repository contains code only. Follow the quick start below to download
the dataset and begin training on a new machine.

## Training Server Quick Start

### 1. Prerequisites

The server needs:

- Git and Conda
- Access to the private Hugging Face dataset
- Approximately 40 GB of free disk space for the package and extracted data
- An NVIDIA GPU for the training command shown below

Clone the code:

```bash
git clone https://github.com/DhiaNeifar/drone-ai.git
cd drone-ai
```

Create the Python environment:

```bash
conda create -n drone-yolo python=3.11 -y
conda activate drone-yolo

python -m pip install --upgrade pip
python -m pip install \
  ultralytics \
  huggingface_hub \
  opencv-python-headless \
  pyyaml \
  tqdm \
  pandas \
  matplotlib
```

Confirm that PyTorch can use the server GPU:

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

Do not start a long training run unless `CUDA: True` is printed.

### 2. Authenticate to Hugging Face

The Hugging Face account or token must have read access to the private dataset:

```bash
hf auth login
hf auth whoami
```

Never commit a Hugging Face token to this repository. For a non-interactive
server, provide the token through the server's secret manager as `HF_TOKEN`.

### 3. Download the Dataset

The dataset is stored as 17 TAR shards to avoid transferring approximately
198,000 files individually. Download the shards and metadata:

```bash
mkdir -p datasets/drone_merged_package

hf download DhiaNeifar/Akkurad_Drone_Detection \
  --repo-type dataset \
  --include "shards/*.tar" \
  --include "shard_manifest.json" \
  --include "dataset.yaml" \
  --include "manifest.csv" \
  --include "audit.json" \
  --include "near_duplicate_candidates.csv" \
  --include "annotation_preview.jpg" \
  --include "dataset_card.md" \
  --local-dir datasets/drone_merged_package
```

The command is resumable. Run the same command again if the SSH connection or
download is interrupted.

Extract the package into the YOLO directory expected by the training script:

```bash
python scripts/extract_yolo_dataset_package.py \
  --package datasets/drone_merged_package \
  --output datasets/drone_merged \
  --overwrite
```

Extraction verifies every shard SHA-256 checksum and checks that image and
label counts match the package manifest. A successful extraction ends with:

```text
YOLO dataset: .../datasets/drone_merged/dataset.yaml
```

Confirm the configuration exists:

```bash
test -f datasets/drone_merged/dataset.yaml && echo "Dataset ready"
```

After successful extraction, `datasets/drone_merged_package` is no longer
needed for training and may be removed to recover approximately 18 GB.

### 4. Train

Start a single-GPU YOLO11 nano training run:

```bash
conda activate drone-yolo

python scripts/train_drone_model.py \
  --experiment yolo11n_drone_merged \
  --model yolo11n.pt \
  --data datasets/drone_merged/dataset.yaml \
  --epochs 60 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0
```

The first run downloads the base YOLO checkpoint if it is not already cached.
Training output is written under:

```text
runs/DD_MM_YYYY_HH_MM_SS_yolo11n_drone_merged/
```

The best checkpoint is `weights/best.pt` inside that run directory. If CUDA
runs out of memory, reduce `--batch` to `8` or `4`. For multiple GPUs
supported by Ultralytics, use a device value such as `--device 0,1`.

For an SSH server, run training inside `tmux`:

```bash
tmux new -s drone-train
# Run the training command. Detach with Ctrl-b followed by d.
tmux attach -t drone-train
```

## Dataset

The current processed build contains images with at least one valid class `0`
(`drone`) bounding box.

| Split | Images | Objects |
| --- | ---: | ---: |
| Train | 78,117 | 82,171 |
| Validation | 12,578 | 12,733 |
| Test | 8,415 | 8,693 |
| **Total** | **99,110** | **103,597** |

The extracted YOLO layout is:

```text
datasets/drone_merged/
|-- images/{train,val,test}/<hash-prefix>/...
|-- labels/{train,val,test}/<hash-prefix>/...
|-- dataset.yaml
|-- manifest.csv
|-- audit.json
|-- near_duplicate_candidates.csv
`-- README.md
```

`manifest.csv` records source provenance and hashes. `audit.json` records
exclusions, annotation repairs, duplicates, and final counts. Related sequence
frames are assigned to one split to prevent train/validation leakage.

The dataset is excluded by `.gitignore`; `git pull` does not download,
modify, or remove local dataset files.

## Review Labels

Use the desktop viewer to inspect images with their YOLO bounding boxes. Install
`opencv-python` instead of `opencv-python-headless` on the review machine:

```bash
conda activate drone-yolo
python -m pip install opencv-python

python scripts/view_yolo_dataset.py \
  --dataset datasets/drone_merged \
  --split train
```

Use Left/Right or A/D to navigate and Q or Esc to close the viewer. Add
`--source gem_new` to restrict review to the newly converted GEM video frames.

## Update Code or Dataset

Pull code changes without touching the local dataset:

```bash
cd drone-ai
git pull --ff-only origin main
conda activate drone-yolo
```

When a new dataset version is published, rerun the download command with the
same `--local-dir`, then rerun the extraction command.

## Rebuild and Upload the Dataset

Most training servers do not need this section. Rebuilding requires all source
datasets under `datasets/`, plus `pyarrow` and desktop OpenCV.

Convert the Label Studio GEM video export:

```bash
conda activate drone-yolo
python -m pip install pyarrow opencv-python

python scripts/prepare_gem_new_yolo.py \
  --root datasets/gem_new_videos \
  --overwrite
```

Rebuild all supported sources, remove exact duplicates, apply annotation
filters, and regenerate the manifest and audit:

```bash
python scripts/build_merged_drone_dataset.py --overwrite
```

Review the rebuilt dataset before publishing it. Then package it into
approximately 1 GiB TAR shards:

```bash
python scripts/package_yolo_dataset.py \
  --dataset datasets/drone_merged \
  --output datasets/drone_merged_hf \
  --shard-size-gb 1 \
  --overwrite
```

Authenticate and upload the package to the private dataset repository:

```bash
hf auth login

python scripts/upload_private_hf_dataset.py \
  DhiaNeifar/Akkurad_Drone_Detection \
  --dataset datasets/drone_merged_hf \
  --workers 2
```

The uploader refuses raw, unpackaged datasets and public repositories. The
upload resumes from its local Hugging Face cache after an interruption.

## Video Inference

Run inference with a trained checkpoint:

```bash
python scripts/predict_videos_with_progress.py \
  --experiment drone_merged_real_scenario \
  --model runs/<training-run>/weights/best.pt \
  --source real-scenario \
  --device 0
```

Inference artifacts are written under `artifacts/`.

## Troubleshooting

- `401` or `403` from Hugging Face: run `hf auth whoami` and confirm that
  the account has access to `DhiaNeifar/Akkurad_Drone_Detection`.
- Interrupted dataset download: rerun the identical `hf download` command.
- Checksum mismatch during extraction: rerun the download, then extract again.
- `CUDA: False`: install a PyTorch build compatible with the server's NVIDIA
  driver and CUDA environment before training.
- CUDA out-of-memory during training: reduce `--batch`.
- OpenCV display errors on a server: keep `opencv-python-headless`; use the
  interactive viewer only on a desktop session.
