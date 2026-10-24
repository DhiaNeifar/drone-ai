#!/usr/bin/env python3
"""Train D-FINE (external/D-FINE) on the COCO export into runs/<timestamp>_<name>."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_ai.train import detr_config


REPO = Path("external/D-FINE")
WEIGHTS = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{size}_coco.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], default="s")
    detr_config.add_common_arguments(parser, default_epochs=72, default_batch=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return detr_config.run(
        args,
        repo=REPO,
        train_script="train.py",
        base_config=REPO / f"configs/dfine/custom/dfine_hgnetv2_{args.size}_custom.yml",
        # The COCO checkpoint already contains a trained backbone; skip the ImageNet download.
        overrides={} if args.no_pretrained else {"HGNetv2": {"pretrained": False}},
        epochs_key="epochs",
        weights_url=WEIGHTS.format(size=args.size),
    )


if __name__ == "__main__":
    raise SystemExit(main())
