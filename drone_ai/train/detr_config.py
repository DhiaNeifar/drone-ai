"""Shared runner for D-FINE and RT-DETRv2, which use the same YAML config system.

Each run writes ``runs/<timestamp>_<name>/config.yml``. That file includes the
stock model config from ``external/`` and overrides only the dataset paths,
class count, output folder, batch size, and epoch schedule, then launches the
framework's own training script with that config.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from drone_ai.paths import timestamped_experiment_name
from drone_ai.train.publish import mark_complete, unmark


COCO_ROOT = Path("datasets/drone_dataset_coco")
WEIGHTS_ROOT = Path("weights")
# MetricLogger lines: "Epoch: [3/72]  [ 100/3804]  eta: ..." (D-FINE), "Epoch: [3]  [ 100/3804]" (RT-DETR),
# and "Test:  [ 10/511]" during validation.
PROGRESS_LINE = re.compile(r"(?:Epoch: \[(\d+)(?:/\d+)?\]|(Test):)\s+\[\s*(\d+)/(\d+)\]")


def add_common_arguments(parser: argparse.ArgumentParser, default_epochs: int, default_batch: int) -> None:
    parser.add_argument("--experiment", help="Short experiment name (required unless --resume).")
    parser.add_argument("--data", type=Path, default=COCO_ROOT, help="COCO export from `export-coco`.")
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Total epochs, including any already done.")
    parser.add_argument(
        "--resume", type=Path, metavar="RUN_DIR",
        help="Continue an existing run from its last.pth up to --epochs, writing to the same folder.",
    )
    parser.add_argument("--batch", type=int, default=default_batch, help="Total batch size.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device index, such as 0 or 0,1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not start from COCO weights.")


def download_weights(url: str) -> Path:
    import torch

    target = WEIGHTS_ROOT / Path(url).name
    if not target.is_file():
        WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}", flush=True)
        torch.hub.download_url_to_file(url, str(target))
    return target


def free_port() -> int:
    """A free local port, so several runs can train at the same time."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def check_coco(data: Path) -> None:
    for folder in ("train", "valid"):
        if not (data / folder / "_annotations.coco.json").is_file():
            raise SystemExit(f"COCO export not found under {data}; run: python -m drone_ai.data export-coco")


def run(
    args: argparse.Namespace,
    repo: Path,
    train_script: str,
    base_config: Path,
    overrides: dict,
    epochs_key: str,
    weights_url: str | None,
) -> int:
    """Write the run config and launch ``train_script`` from inside ``repo``."""
    check_coco(args.data)
    data = args.data.resolve()
    if args.resume:
        output = args.resume.resolve()
        if not (output / "last.pth").is_file():
            raise SystemExit(f"No last.pth in {output}; nothing to resume")
        unmark(output)
    elif args.experiment:
        output = Path("runs").resolve() / timestamped_experiment_name(args.experiment, datetime.now())
        output.mkdir(parents=True, exist_ok=False)
    else:
        raise SystemExit("--experiment is required for a new run")
    done = len((output / "log.txt").read_text().splitlines()) if (output / "log.txt").is_file() else 0
    stop_epoch = max(1, round(args.epochs * 0.9))
    config = {
        "__include__": [str(base_config.resolve())],
        "output_dir": str(output),
        "print_freq": 10,
        "checkpoint_freq": 1,
        "num_classes": 1,
        "remap_mscoco_category": False,
        epochs_key: args.epochs,
        "train_dataloader": {
            "total_batch_size": args.batch,
            "num_workers": args.workers,
            "dataset": {
                "img_folder": str(data / "train"),
                "ann_file": str(data / "train" / "_annotations.coco.json"),
                # Strong augmentation stops for the final ~10% of epochs, as in the stock schedules.
                "transforms": {"policy": {"epoch": stop_epoch}},
            },
            "collate_fn": {"stop_epoch": stop_epoch},
        },
        "val_dataloader": {
            "num_workers": args.workers,
            "dataset": {
                "img_folder": str(data / "valid"),
                "ann_file": str(data / "valid" / "_annotations.coco.json"),
            },
        },
    }
    for key, value in overrides.items():
        config[key] = {**config.get(key, {}), **value} if isinstance(value, dict) else value
    config_path = output / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Training output: {output}", flush=True)

    command = [
        sys.executable, "-m", "torch.distributed.run", f"--master_port={free_port()}",
        f"--nproc_per_node={len(args.device.split(','))}",
        train_script, "-c", str(config_path), "--use-amp", f"--seed={args.seed}",
    ]
    if args.resume:
        command += ["-r", str(output / "last.pth")]
        print(f"Resuming after epoch {done} of {args.epochs}", flush=True)
    elif weights_url and not args.no_pretrained:
        command += ["-t", str(download_weights(weights_url).resolve())]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device, "PYTHONUNBUFFERED": "1"}
    print("Running: " + " ".join(command), flush=True)
    returncode = run_with_progress(command, repo, environment, args.epochs, output / "train.log", done)
    if returncode == 0:
        mark_complete(output)
    return returncode


def run_with_progress(
    command: list[str], cwd: Path, environment: dict, epochs: int, log_path: Path, done: int = 0
) -> int:
    """Run the trainer, turning its periodic log lines into epoch and batch progress bars.

    Every line goes to ``log_path``; lines that are not progress updates (losses
    summaries, COCO AP tables, errors) are also printed above the bars.
    """
    from tqdm import tqdm

    epoch_bar = tqdm(total=epochs, initial=done, desc="Epochs", unit="epoch", position=0, dynamic_ncols=True)
    batch_bar = tqdm(total=0, desc="Train", unit="batch", position=1, dynamic_ncols=True)
    current = (None, None)
    with log_path.open("a", encoding="utf-8") as log, subprocess.Popen(
        command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as process:
        for line in process.stdout:
            log.write(line)
            match = PROGRESS_LINE.search(line)
            if not match:
                if line.strip():
                    tqdm.write(line.rstrip())
                continue
            epoch, test, done, total = match.groups()
            phase = ("test", current[1]) if test else ("train", int(epoch))
            if phase != current:
                if phase[0] == "train" and current[0] is not None and phase[1] != current[1]:
                    epoch_bar.update(phase[1] - epoch_bar.n)
                batch_bar.reset(total=int(total))
                batch_bar.set_description("Validate" if test else f"Train epoch {int(epoch) + 1}")
                current = phase
            batch_bar.update(int(done) - batch_bar.n)
            if "eta:" in line and (loss := re.search(r"loss: ([\d.]+)", line)):
                batch_bar.set_postfix(loss=loss.group(1), refresh=False)
        returncode = process.wait()
    if returncode == 0:
        epoch_bar.update(epochs - epoch_bar.n)
    batch_bar.close()
    epoch_bar.close()
    print(f"Full trainer log: {log_path}")
    return returncode
