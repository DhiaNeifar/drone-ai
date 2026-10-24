#!/usr/bin/env python3
"""Validate an Ultralytics YOLO detection dataset and its label geometry."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset root or its dataset.yaml file.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat images without label files as valid empty images.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Decode every image to detect corrupt files (slower).",
    )
    parser.add_argument("--max-errors", type=int, default=100)
    return parser.parse_args()


def resolve_image_root(config_root: Path, value: object) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(str(item))
        paths.append(path if path.is_absolute() else config_root / path)
    return paths


def label_for_image(dataset_root: Path, image: Path) -> Path:
    try:
        relative = image.relative_to(dataset_root / "images")
        return (dataset_root / "labels" / relative).with_suffix(".txt")
    except ValueError:
        parts = list(image.parts)
        if "images" not in parts:
            return image.with_suffix(".txt")
        position = len(parts) - 1 - parts[::-1].index("images")
        parts[position] = "labels"
        return Path(*parts).with_suffix(".txt")


def main() -> int:
    args = parse_args()
    config_path = args.dataset.expanduser().resolve()
    if config_path.is_dir():
        config_path /= "dataset.yaml"
    if not config_path.is_file():
        raise SystemExit(f"Dataset configuration not found: {config_path}")
    if args.max_errors < 1:
        raise SystemExit("--max-errors must be positive")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"Invalid YAML mapping: {config_path}")
    yaml_root = config_path.parent
    configured_root = Path(str(config.get("path", yaml_root)))
    dataset_root = configured_root if configured_root.is_absolute() else yaml_root / configured_root
    dataset_root = dataset_root.resolve()
    names = config.get("names")
    if isinstance(names, dict):
        class_ids = {int(value) for value in names.keys()}
    elif isinstance(names, list):
        class_ids = set(range(len(names)))
    else:
        raise SystemExit("dataset.yaml must define names as a list or integer-keyed mapping")
    if not class_ids:
        raise SystemExit("dataset.yaml defines no classes")

    errors = []
    counts: Counter[str] = Counter()
    seen_images: set[Path] = set()
    expected_labels: set[Path] = set()
    cv2_module = None
    if args.check_images:
        import cv2

        cv2_module = cv2
    for split in ("train", "val", "test"):
        if split not in config:
            if split == "train":
                errors.append("dataset.yaml does not define train")
            continue
        for image_root in resolve_image_root(dataset_root, config[split]):
            if not image_root.exists():
                errors.append(f"{split}: image path does not exist: {image_root}")
                continue
            images = (
                [image_root]
                if image_root.is_file() and image_root.suffix.casefold() in IMAGE_EXTENSIONS
                else [
                    path
                    for path in image_root.rglob("*")
                    if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
                ]
            )
            if not images:
                errors.append(f"{split}: no supported images under {image_root}")
            for image in images:
                image = image.resolve()
                if image in seen_images:
                    errors.append(f"{split}: image appears more than once: {image}")
                    continue
                seen_images.add(image)
                counts[f"{split}_images"] += 1
                if args.check_images:
                    decoded = cv2_module.imread(str(image), cv2_module.IMREAD_UNCHANGED)
                    if decoded is None or decoded.size == 0:
                        errors.append(f"{split}: unreadable image: {image}")
                label = label_for_image(dataset_root, image)
                expected_labels.add(label)
                if not label.is_file():
                    counts[f"{split}_missing_labels"] += 1
                    if not args.allow_missing_labels:
                        errors.append(f"{split}: missing label for {image}")
                    continue
                counts[f"{split}_labels"] += 1
                for line_number, line in enumerate(
                    label.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        errors.append(
                            f"{label}:{line_number}: expected 5 fields, found {len(fields)}"
                        )
                        continue
                    try:
                        class_value = float(fields[0])
                        coordinates = [float(value) for value in fields[1:]]
                    except ValueError:
                        errors.append(f"{label}:{line_number}: non-numeric field")
                        continue
                    class_id = int(class_value)
                    if class_value != class_id or class_id not in class_ids:
                        errors.append(f"{label}:{line_number}: invalid class {fields[0]}")
                    if not all(math.isfinite(value) for value in coordinates):
                        errors.append(f"{label}:{line_number}: non-finite coordinate")
                        continue
                    center_x, center_y, width, height = coordinates
                    if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
                        errors.append(f"{label}:{line_number}: center outside [0,1]")
                    if not (0 < width <= 1 and 0 < height <= 1):
                        errors.append(f"{label}:{line_number}: invalid width or height")
                    tolerance = 1e-6
                    if (
                        center_x - width / 2 < -tolerance
                        or center_x + width / 2 > 1 + tolerance
                        or center_y - height / 2 < -tolerance
                        or center_y + height / 2 > 1 + tolerance
                    ):
                        errors.append(f"{label}:{line_number}: box crosses image boundary")
                    counts[f"{split}_boxes"] += 1

    labels_root = dataset_root / "labels"
    if labels_root.is_dir():
        resolved_expected_labels = {path.resolve() for path in expected_labels}
        for label in labels_root.rglob("*.txt"):
            if label.resolve() not in resolved_expected_labels:
                counts["orphan_labels"] += 1
                errors.append(f"orphan label without an image: {label}")

    summary = {
        "dataset_yaml": str(config_path),
        "dataset_root": str(dataset_root),
        "classes": len(class_ids),
        **dict(sorted(counts.items())),
        "errors": len(errors),
    }
    print(yaml.safe_dump(summary, sort_keys=False).rstrip())
    if errors:
        print("\nValidation errors:")
        for error in errors[: args.max_errors]:
            print(f"- {error}")
        if len(errors) > args.max_errors:
            print(f"- ... {len(errors) - args.max_errors} more")
        return 1
    print("\nYOLO detection dataset is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
