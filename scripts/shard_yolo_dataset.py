#!/usr/bin/env python3
"""Shard a flat YOLO dataset into hash-prefix directories for Hub storage."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

from tqdm import tqdm


SPLITS = ("train", "val", "test")
HASH_PATTERN = re.compile(r"_([0-9a-f]{20})(?:\.[^.]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/drone_merged"),
        help="YOLO dataset root.",
    )
    return parser.parse_args()


def image_bucket(path: Path) -> str:
    match = HASH_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Image name does not contain the expected hash: {path}")
    return match.group(1)[:2]


def move_split(root: Path, split: str) -> int:
    image_root = root / "images" / split
    label_root = root / "labels" / split
    images = sorted(path for path in image_root.rglob("*") if path.is_file())
    moved = 0

    for image in tqdm(images, desc=f"Shard {split}", unit="pair"):
        bucket = image_bucket(image)
        image_destination = image_root / bucket / image.name
        label_destination = label_root / bucket / f"{image.stem}.txt"

        current_label = label_root / image.relative_to(image_root).with_suffix(".txt")
        if not current_label.is_file():
            flat_label = label_root / f"{image.stem}.txt"
            current_label = flat_label if flat_label.is_file() else current_label
        if not current_label.is_file():
            raise FileNotFoundError(f"Missing label for {image}")

        if image != image_destination:
            if image_destination.exists():
                raise FileExistsError(image_destination)
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image), str(image_destination))
            moved += 1

        if current_label != label_destination:
            if label_destination.exists():
                raise FileExistsError(label_destination)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current_label), str(label_destination))

    return moved


def update_manifest(root: Path) -> None:
    manifest = root / "manifest.csv"
    temporary = manifest.with_suffix(".csv.tmp")
    with manifest.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header")
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                name = Path(row["output_image"]).name
                bucket = row["sha256"][:2]
                row["output_image"] = f"images/{row['split']}/{bucket}/{name}"
                writer.writerow(row)
    os.replace(temporary, manifest)


def validate(root: Path) -> None:
    for split in SPLITS:
        image_root = root / "images" / split
        label_root = root / "labels" / split
        images = [path for path in image_root.rglob("*") if path.is_file()]
        labels = [path for path in label_root.rglob("*.txt") if path.is_file()]
        if len(images) != len(labels):
            raise RuntimeError(f"{split}: {len(images)} images != {len(labels)} labels")
        for image in images:
            relative = image.relative_to(image_root).with_suffix(".txt")
            if not (label_root / relative).is_file():
                raise RuntimeError(f"Missing paired label: {relative}")
        oversized = [
            directory
            for path in (image_root, label_root)
            for directory in path.rglob("*")
            if directory.is_dir()
            and sum(1 for child in directory.iterdir() if child.is_file()) >= 10_000
        ]
        if oversized:
            raise RuntimeError(f"Folders still contain 10,000 or more files: {oversized}")
        print(f"{split}: {len(images):,} image/label pairs")


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    if not (root / "dataset.yaml").is_file() or not (root / "manifest.csv").is_file():
        raise SystemExit(f"Not a prepared YOLO dataset: {root}")

    moved = sum(move_split(root, split) for split in SPLITS)
    update_manifest(root)
    validate(root)
    print(f"Moved {moved:,} image/label pairs into hash-prefix directories.")


if __name__ == "__main__":
    main()
