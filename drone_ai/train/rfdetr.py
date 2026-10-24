#!/usr/bin/env python3
"""Train RF-DETR (external/rf-detr) on the COCO export into runs/<timestamp>_<name>."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from drone_ai.paths import timestamped_experiment_name
from drone_ai.train.detr_config import COCO_ROOT, check_coco
from drone_ai.train.publish import mark_complete


SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Short experiment name.")
    parser.add_argument("--size", choices=sorted(SIZES), default="small")
    parser.add_argument("--data", type=Path, default=COCO_ROOT, help="COCO export from `export-coco`.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2, help="Effective batch = batch * grad-accum.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    import rfdetr

    args = parse_args()
    check_coco(args.data)
    output = Path("runs").resolve() / timestamped_experiment_name(args.experiment, datetime.now())
    output.mkdir(parents=True, exist_ok=False)
    print(f"Training output: {output}", flush=True)
    model = getattr(rfdetr, SIZES[args.size])()
    model.train(
        dataset_dir=str(args.data.resolve()),
        output_dir=str(output),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.workers,
        early_stopping=True,
        early_stopping_patience=args.patience,
        seed=args.seed,
        progress_bar="tqdm",
    )
    mark_complete(output, "rfdetr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
