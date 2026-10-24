#!/usr/bin/env python3
"""Export the YOLO dataset as COCO JSON for RF-DETR, D-FINE, and RT-DETR.

Images are not copied. Each split folder holds relative symlinks to the
dataset's hash-prefix image folders plus ``_annotations.coco.json``::

    datasets/drone_dataset_coco/
    |-- train/_annotations.coco.json, 00 -> ../../drone_dataset/images/train/00, ...
    |-- valid/...
    `-- test/...

This is the Roboflow COCO layout RF-DETR detects automatically, and each split
folder plus its JSON is the ``img_folder``/``ann_file`` pair D-FINE and
RT-DETR expect. The single class is category id 0 (``drone``).

``--train-fraction`` and ``--val-fraction`` export a fixed subset (chosen from
each image's SHA-256, so reruns pick the same images). Consecutive video frames
are near-duplicates, so a subset trains much faster at little cost; the test
split is always exported in full for the final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm


SPLIT_FOLDERS = {"train": "train", "val": "valid", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/drone_dataset"))
    parser.add_argument("--output", type=Path, default=Path("datasets/drone_dataset_coco"))
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of train images to keep.")
    parser.add_argument("--val-fraction", type=float, default=1.0, help="Fraction of val images to keep.")
    return parser.parse_args()


def keep(row: dict[str, str], fraction: float) -> bool:
    return fraction >= 1.0 or int(row["sha256"][:12], 16) / 16**12 < fraction


def class_names(dataset: Path) -> list[str]:
    names = yaml.safe_load((dataset / "dataset.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=int)]
    return list(names)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    names = class_names(dataset)
    categories = [{"id": index, "name": name, "supercategory": "none"} for index, name in enumerate(names)]

    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_FOLDERS}
    fractions = {"train": args.train_fraction, "val": args.val_fraction, "test": 1.0}
    for row in rows:
        if keep(row, fractions[row["split"]]):
            by_split[row["split"]].append(row)

    for split, folder in SPLIT_FOLDERS.items():
        split_dir = output / folder
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        image_root = dataset / "images" / split
        for prefix in sorted(path.name for path in image_root.iterdir() if path.is_dir()):
            os.symlink(os.path.relpath(image_root / prefix, split_dir), split_dir / prefix)

        images, annotations, boxes = [], [], Counter()
        ordered = sorted(by_split[split], key=lambda item: item["output_image"])
        for image_id, row in enumerate(tqdm(ordered, desc=f"Export {folder}", unit="image", dynamic_ncols=True), start=1):
            relative = Path(row["output_image"]).relative_to(Path("images") / split)
            width, height = int(row["width"]), int(row["height"])
            images.append({"id": image_id, "file_name": str(relative), "width": width, "height": height})
            label = (dataset / "labels" / split / relative).with_suffix(".txt")
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, x, y, w, h = line.split()
                box_width, box_height = float(w) * width, float(h) * height
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [
                            round(float(x) * width - box_width / 2, 3),
                            round(float(y) * height - box_height / 2, 3),
                            round(box_width, 3),
                            round(box_height, 3),
                        ],
                        "area": round(box_width * box_height, 3),
                        "iscrowd": 0,
                    }
                )
                boxes[split] += 1
        document = {
            "info": {"description": f"COCO export of {dataset.name}", "version": "1.0"},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(document), encoding="utf-8")
        print(f"{folder}: {len(images)} image(s), {boxes[split]} box(es) -> {split_dir}")
    print(f"COCO dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
