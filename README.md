# Drone Detection with YOLO

This repository contains the training, review, preprocessing, and inference
tools for a single-class YOLO drone detector. The processed dataset is stored
separately in the private Hugging Face dataset repository
[`DhiaNeifar/Akkurad_Drone_Detection`](https://huggingface.co/datasets/DhiaNeifar/Akkurad_Drone_Detection).

The Git repository contains code only. Follow the quick start below to download
the dataset and begin training on a new machine.

### Package for Transfer

To avoid downloading tens of thousands of individual files, package a complete
sharded YOLO dataset into a small number of TAR files before uploading it:

```bash
python scripts/drone_data.py package-yolo \
  --dataset datasets/drone_dataset \
  --output datasets/drone_dataset_hf \
  --shard-size-gb 1 \
  --overwrite
```

The packager can generate `manifest.csv` from an already-sharded
`images/<split>` and `labels/<split>` tree. It verifies every image has a label
and writes checksums to `shard_manifest.json`. Upload only the resulting
package directory, including `shards/*.tar`, then download those TAR files on
the server and extract them with `extract-package`.

## Annotation Toolkit

Detection annotations must remain separate from image pixels. A box rendered
into a training image creates a visual shortcut: the model can learn the box
color instead of the object. The tools below always keep extracted frames clean
and store coordinates in JSON, CSV, or YOLO TXT files.

### Recursively Extract Videos

The extractor scans every subdirectory for `.avi`, `.m4v`, `.mkv`, `.mov`,
`.mp4`, and `.webm` files. For an input such as `/data/Recordings`, it creates
`/data/Recordings_annotations` beside it and mirrors the source hierarchy. Each
video gets its own sequence directory so videos with the same filename cannot
collide:

```text
/data/Recordings/site_a/camera.mp4
    -> /data/Recordings_annotations/site_a/camera/frames/*.jpg
    -> /data/Recordings_annotations/site_a/camera/camera.json
    -> /data/Recordings_annotations/site_a/camera/camera.csv
```

Run it with:

```bash
conda activate drone-yolo
python scripts/drone_data.py extract-frames /data/Recordings --workers 4
```

The default JPEG quality is 95. The extractor refuses to replace an existing
output unless `--overwrite` is supplied. It also writes
`extraction_manifest.json` with source paths, video SHA-256 hashes, dimensions,
frame rates, and decoded frame counts. A matching `<video>.json` or
`<video>.csv` sidecar is carried forward; otherwise empty JSON/CSV annotations
are initialized.

### Build the C++ Annotation Editor

The native editor keeps navigation responsive on large frame collections. It
requires a desktop OpenCV development package, not only the Python wheel.

macOS:

```bash
brew install cmake opencv
cmake -S annotation_editor -B build/annotation_editor -DCMAKE_BUILD_TYPE=Release
cmake --build build/annotation_editor --parallel
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev
cmake -S annotation_editor -B build/annotation_editor -DCMAKE_BUILD_TYPE=Release
cmake --build build/annotation_editor --parallel
```

Open an annotation workspace:

```bash
build/annotation_editor/drone_annotation_editor \
  datasets/recordings_clean_annotated
```

For a non-GUI structure and annotation parse check, append `--check`.

The editor recursively discovers directories named `frames` and automatically
loads the first applicable format:

| Format | Supported layout |
| --- | --- |
| Sequence JSON | `<sequence>/<sequence>.json` with `exist` and pixel `gt_rect` arrays |
| Bounding-box CSV | One row per frame with `frame`, optional `drone_exist`/`class_id`, and pixel `x,y,width,height` columns |
| YOLO TXT | `<sequence>/labels/<frame-stem>.txt` with normalized `class x_center y_center width height` rows |

Use `--format json`, `--format csv`, or `--format yolo` to override automatic
detection. New workspaces default to sequence JSON. JSON and CSV store one box
per frame; YOLO TXT supports multiple boxes and preserves class IDs.

Editor controls:

- Left-drag replaces or creates the current box.
- Shift+left-drag adds a box when editing YOLO TXT.
- Right-click inside a box deletes it.
- Mouse wheel, Left/Right, or A/D changes frames; Up/Down or W/S changes sequences.
- Comma/period skips backward/forward by 10 frames.
- `N` jumps to the next frame without a box.
- `C` copies boxes from the previous frame; `X` clears the current frame.
- `Z` and `Y` undo and redo edits; `B` toggles box visibility.
- Ctrl+mouse-wheel or `+`/`-` zooms up to 8x; middle-click recenters and
  `0` resets the view. Mouse edits remain coordinate-correct while zoomed.
- Every edit, undo, and redo is written immediately through a temporary file;
  `Q` or Esc exits with no separate save step.

You can also open a YOLO dataset directly:

```bash
build/annotation_editor/drone_annotation_editor \
  datasets/drone_merged \
  --format yolo
```

The old Python-only viewers have been removed. Use this editor for JSON, CSV,
and YOLO review/editing.

## Repository Structure

The repository intentionally keeps the public command surface small:

```text
annotation_editor/          Native image/box editor
scripts/drone_data.py       Dataset extraction, conversion, validation, package, upload
scripts/drone_experiment.py Training and inference
scripts/drone_tools/        Implementation modules behind drone_data.py
runs/                       Timestamped training runs and checkpoints
artifacts/                  Timestamped test/inference outputs
datasets/                   Local raw, working, clean, and packaged datasets
```

Do not create ad hoc output roots such as `outputs/` or `real-scenario/`.
Training belongs in `runs/`; testing and inference belong in `artifacts/`.

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
mkdir -p datasets/drone_merged_hf

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
  --local-dir datasets/drone_merged_hf
```

The command is resumable. Run the same command again if the SSH connection or
download is interrupted.

Extract the package into the YOLO directory expected by the training script:

```bash
python scripts/drone_data.py extract-package \
  --package datasets/drone_merged_hf \
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

After successful extraction, `datasets/drone_merged_hf` is no longer
needed for training and may be removed to recover approximately 18 GB.

### 4. Train

Start a single-GPU YOLO11 nano training run:

```bash
conda activate drone-yolo

python scripts/drone_experiment.py train \
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

### Validate YOLO Format

A valid Ultralytics detection dataset has a `dataset.yaml`, matching image and
label paths, integer class IDs declared in `names`, and one label row per object
in normalized `class x_center y_center width height` format. Coordinates must be
finite, centers must be in `[0,1]`, sizes must be in `(0,1]`, and no box may
cross an image boundary.

Run the repository validator before packaging, uploading, or training:

```bash
python scripts/drone_data.py validate-yolo datasets/drone_merged
```

The command exits nonzero for malformed rows, missing image/label pairs,
unknown classes, invalid geometry, duplicated image references, or orphan label
files. Empty images should normally have an empty `.txt` label; use
`--allow-missing-labels` only when missing files intentionally mean background.
Before a release, add `--check-images` to decode every image and catch corrupt
media; this is intentionally optional because it is slower on large datasets.

## Review Labels

Use the native annotation editor for label review. For YOLO data, launch:

```bash
build/annotation_editor/drone_annotation_editor datasets/drone_merged --format yolo
```

## Update Code or Dataset

Pull code changes without touching the local dataset:

```bash
cd drone-ai
git pull --ff-only origin main
conda activate drone-yolo
```

When a new dataset version is published, rerun the download command with the
same `--local-dir`, then rerun the extraction command.

## Dataset Lifecycle

`datasets/` contains raw source folders and local working products. The folder
that matters for training is:

```text
datasets/drone_merged/
```

That is the cleaned YOLO dataset. The folder that matters for Hugging Face is:

```text
datasets/drone_merged_hf/
```

That is the packaged shard set uploaded to the private dataset repository.

When you add new sample folders such as `datasets/dataset_samples0` or
`datasets/dataset_samples1`, they are not automatically part of training. The
normal flow is:

1. Extract videos into an annotation workspace.
2. Annotate or review with the C++ editor.
3. Put the reviewed workspace under `datasets/` as either
   `recordings_clean_annotated`, any folder ending in `_annotations`, or a
   `dataset_samples*` folder.
4. Rebuild `datasets/drone_merged` from all approved sources.
5. Validate it.
6. Repackage into TAR shards.
7. Upload the package to Hugging Face.

For example:

```bash
python scripts/drone_data.py extract-frames \
  datasets/Recordings \
  --output datasets/recordings_clean_annotated \
  --workers 4

build/annotation_editor/drone_annotation_editor datasets/recordings_clean_annotated

python scripts/drone_data.py build-merged --overwrite
python scripts/drone_data.py validate-yolo datasets/drone_merged
python scripts/drone_data.py package-yolo --overwrite
python scripts/drone_data.py upload-hf DhiaNeifar/Akkurad_Drone_Detection
```

The current builder creates a clean, deterministic full dataset. It deduplicates
by image hash and writes a manifest, so rerunning it after adding samples does
not duplicate old images. Hugging Face uploads are resumable and only changed
files are transferred from the local upload cache, but because TAR shards are
archives, changing the dataset can still rewrite affected shards. A true
append-only shard workflow would require a package manifest that freezes old
shards and creates only new shard files for new samples; that is the next
logical improvement if rebuild time becomes painful.

## Rebuild and Upload the Dataset

Most training servers do not need this section. Rebuilding requires all source
datasets under `datasets/`, plus `pyarrow` and desktop OpenCV.

Convert the Label Studio GEM video export:

```bash
conda activate drone-yolo
python -m pip install pyarrow opencv-python

python scripts/drone_data.py prepare-gem \
  --root datasets/gem_new_videos \
  --overwrite
```

Rebuild all supported sources, remove exact duplicates, apply annotation
filters, and regenerate the manifest and audit:

```bash
python scripts/drone_data.py build-merged --overwrite
```

Review the rebuilt dataset before publishing it. Then package it into
approximately 1 GiB TAR shards:

```bash
python scripts/drone_data.py package-yolo \
  --dataset datasets/drone_merged \
  --output datasets/drone_merged_hf \
  --shard-size-gb 1 \
  --overwrite
```

Authenticate and upload the package to the private dataset repository:

```bash
hf auth login

python scripts/drone_data.py upload-hf \
  DhiaNeifar/Akkurad_Drone_Detection \
  --dataset datasets/drone_merged_hf \
  --workers 2
```

The uploader refuses raw, unpackaged datasets and public repositories. The
upload resumes from its local Hugging Face cache after an interruption.

## Video Inference

Run inference with a trained checkpoint:

```bash
python scripts/drone_experiment.py predict \
  --experiment drone_merged_real_scenario \
  --model runs/<training-run>/weights/best.pt \
  --source /path/to/test/videos \
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
