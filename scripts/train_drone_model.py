#!/usr/bin/env python3
"""Train YOLO with a timestamped directory under runs/."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from experiment_paths import timestamped_experiment_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a drone detector and save it under runs/<timestamp>_<experiment>."
    )
    parser.add_argument("--experiment", required=True, help="Short experiment name.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model or checkpoint.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/drone_merged/dataset.yaml"),
        help="YOLO dataset configuration.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-period", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset configuration not found: {args.data}")

    run_name = timestamped_experiment_name(args.experiment, datetime.now())
    runs_root = Path("runs").resolve()
    output = runs_root / run_name
    print(f"Training output: {output}", flush=True)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        save_period=args.save_period,
        project=str(runs_root),
        name=run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
