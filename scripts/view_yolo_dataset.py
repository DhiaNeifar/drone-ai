#!/usr/bin/env python3
"""Browse a YOLO dataset with its bounding boxes overlaid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
LEFT_KEYS = {2, 81, 63234, 65361, 2424832}
RIGHT_KEYS = {3, 83, 63235, 65363, 2555904}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/drone_merged"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--source", help="Only show rows from this manifest source.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based starting image index.")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    parser.add_argument("--save-frame", type=Path, help="Write one annotated image and exit.")
    return parser.parse_args()


def discover_images(root: Path, split: str, source: str | None) -> list[Path]:
    if source:
        manifest = root / "manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            return sorted(
                root / row["output_image"]
                for row in rows
                if row["split"] == split and row["source"] == source
            )
    return sorted(
        path
        for path in (root / "images" / split).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path(root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(root / "images")
    return (root / "labels" / relative).with_suffix(".txt")


def render(root: Path, image_path: Path, index: int, total: int, max_width: int, max_height: int):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read {image_path}")
    height, width = image.shape[:2]
    annotation_path = label_path(root, image_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing paired label: {annotation_path}")
    box_count = 0
    for line_number, line in enumerate(annotation_path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {annotation_path}:{line_number}")
        class_id, x, y, box_width, box_height = map(float, fields)
        x1 = round((x - box_width / 2) * width)
        y1 = round((y - box_height / 2) * height)
        x2 = round((x + box_width / 2) * width)
        y2 = round((y + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, f"drone ({int(class_id)})", (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
        )
        box_count += 1

    status_height = 76
    canvas = cv2.copyMakeBorder(image, status_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(
        canvas, f"{index + 1}/{total} | boxes={box_count} | {image_path.relative_to(root)}",
        (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "Left/Right or A/D: navigate | Q/Esc: quit", (14, 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
    )
    scale = min(max_width / canvas.shape[1], max_height / canvas.shape[0], 1.0)
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas


def main() -> int:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    if args.max_width <= 0 or args.max_height <= 0:
        raise ValueError("--max-width and --max-height must be positive")
    images = discover_images(root, args.split, args.source)
    if not images:
        raise RuntimeError(f"No matching {args.split} images under {root}")
    index = min(max(args.start, 0), len(images) - 1)
    window = "YOLO annotation review"
    while True:
        rendered = render(root, images[index], index, len(images), args.max_width, args.max_height)
        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_frame), rendered):
                raise RuntimeError(f"Could not write {args.save_frame}")
            print(f"Saved {args.save_frame}")
            return 0
        cv2.imshow(window, rendered)
        key = cv2.waitKeyEx(0)
        if key in (27, ord("q"), ord("Q")):
            break
        if key in LEFT_KEYS or key in (ord("a"), ord("A")):
            index = max(0, index - 1)
        elif key in RIGHT_KEYS or key in (ord("d"), ord("D")):
            index = min(len(images) - 1, index + 1)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
