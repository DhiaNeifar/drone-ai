#!/usr/bin/env python3
"""Train RT-DETRv2 (external/RT-DETR/rtdetrv2_pytorch) on the COCO export into runs/<timestamp>_<name>."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_ai.train import detr_config


REPO = Path("external/RT-DETR/rtdetrv2_pytorch")
RELEASES = "https://github.com/lyuwenyu/storage/releases/download"
SIZES = {
    "s": ("rtdetrv2_r18vd_120e_coco.yml", f"{RELEASES}/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
    "m": ("rtdetrv2_r34vd_120e_coco.yml", f"{RELEASES}/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth"),
    "l": ("rtdetrv2_r50vd_6x_coco.yml", f"{RELEASES}/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth"),
    "x": ("rtdetrv2_r101vd_6x_coco.yml", f"{RELEASES}/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(SIZES), default="s")
    detr_config.add_common_arguments(parser, default_epochs=72, default_batch=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_name, weights_url = SIZES[args.size]
    return detr_config.run(
        args,
        repo=REPO,
        train_script="tools/train.py",
        base_config=REPO / "configs/rtdetrv2" / config_name,
        # The COCO checkpoint already contains a trained backbone; skip the ImageNet download.
        overrides={} if args.no_pretrained else {"PResNet": {"pretrained": False}},
        epochs_key="epoches",
        weights_url=weights_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())
