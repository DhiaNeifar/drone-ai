#!/usr/bin/env python3
"""Train a YOLO model (external/ultralytics) into runs/<timestamp>_<name>."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from drone_ai.paths import timestamped_experiment_name
from drone_ai.train.publish import mark_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Short experiment name.")
    parser.add_argument("--model", default="weights/yolo11n.pt", help="Base model or checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("datasets/drone_dataset/dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-period", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

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
    mark_complete(Path(model.trainer.save_dir), "yolo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
