#!/usr/bin/env python3
"""Convert Anti-UAV-RGBT videos and annotations into a YOLO detection dataset."""

from __future__ import annotations

import argparse
import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


@dataclass
class SplitStats:
    sequences: int = 0
    frames: int = 0
    positive_frames: int = 0
    empty_frames: int = 0
    skipped_frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Anti-UAV-RGBT frames and convert gt_rect annotations to "
            "YOLO class x_center y_center width height labels."
        )
    )
    parser.add_argument(
        "--root",
        default=Path("Anti-UAV-RGBT"),
        type=Path,
        help="Anti-UAV-RGBT dataset root.",
    )
    parser.add_argument(
        "--output",
        default=Path("datasets/antiuav_yolo"),
        type=Path,
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["visible"],
        choices=("visible", "infrared"),
        help="Modalities to convert.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=("train", "val", "test"),
        help="Dataset splits to convert.",
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="Keep every Nth frame. Use larger values for quick smoke tests.",
    )
    parser.add_argument(
        "--max-sequences-per-split",
        type=int,
        help="Limit sequences per split for quick conversion tests.",
    )
    parser.add_argument(
        "--workers",
        default=max(1, min(4, (os.cpu_count() or 2) - 1)),
        type=int,
        help="Number of parallel worker threads for sequence conversion.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip frames where the drone does not exist instead of writing empty labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        choices=(".jpg", ".png"),
        help="Image format for extracted frames.",
    )
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_annotation(path: Path) -> tuple[list[int], list[list[float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["exist"], data["gt_rect"]


def box_to_yolo(rect: list[float], image_width: int, image_height: int) -> str | None:
    if len(rect) != 4:
        return None

    x, y, w, h = [float(value) for value in rect]
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + w))
    y2 = max(0.0, min(float(image_height), y + h))

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 1.0 or clipped_h <= 1.0:
        return None

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    norm_w = clipped_w / image_width
    norm_h = clipped_h / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"


def sequence_dirs(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def convert_sequence(
    sequence_dir: Path,
    split: str,
    modality: str,
    output: Path,
    stride: int,
    skip_empty: bool,
    image_ext: str,
) -> SplitStats:
    stats = SplitStats(sequences=1)
    video_path = sequence_dir / f"{modality}.mp4"
    annotation_path = sequence_dir / f"{modality}.json"

    if not video_path.exists() or not annotation_path.exists():
        print(f"skip missing pair: {video_path} / {annotation_path}")
        return stats

    exists, boxes = load_annotation(annotation_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"skip unreadable video: {video_path}")
        return stats

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(total_frames, len(exists), len(boxes))

    images_dir = output / "images" / split
    labels_dir = output / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    while frame_index < frame_limit:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            stats.skipped_frames += 1
            frame_index += stride
            continue

        yolo_label = None
        if int(exists[frame_index]) == 1:
            yolo_label = box_to_yolo(boxes[frame_index], width, height)

        if yolo_label is None and skip_empty:
            stats.empty_frames += 1
            frame_index += stride
            continue

        stem = f"{sequence_dir.name}_{modality}_{frame_index:06d}"
        image_path = images_dir / f"{stem}{image_ext}"
        label_path = labels_dir / f"{stem}.txt"

        cv2.imwrite(str(image_path), frame)
        label_path.write_text(yolo_label or "", encoding="utf-8")

        stats.frames += 1
        if yolo_label:
            stats.positive_frames += 1
        else:
            stats.empty_frames += 1

        frame_index += stride

    cap.release()
    return stats


def write_dataset_yaml(output: Path) -> None:
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "drone"},
    }
    with (output / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def add_stats(total: SplitStats, part: SplitStats) -> None:
    total.sequences += part.sequences
    total.frames += part.frames
    total.positive_frames += part.positive_frames
    total.empty_frames += part.empty_frames
    total.skipped_frames += part.skipped_frames


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.root}")

    root = args.root.resolve()
    output = args.output.resolve()
    if output == root or is_relative_to(output, root):
        raise ValueError(
            f"Refusing to write output inside the raw dataset root: {root}. "
            "Choose an output path like datasets/antiuav_yolo_visible_s10."
        )

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists and is not empty: {args.output}. Use --overwrite.")

    all_stats: dict[str, dict[str, int]] = {}
    for split in args.splits:
        split_stats = SplitStats()
        sequences = sequence_dirs(args.root, split)
        if args.max_sequences_per_split:
            sequences = sequences[: args.max_sequences_per_split]

        jobs = [
            {
                "sequence_dir": sequence_dir,
                "split": split,
                "modality": modality,
                "output": args.output,
                "stride": args.stride,
                "skip_empty": args.skip_empty,
                "image_ext": args.image_ext,
            }
            for sequence_dir in sequences
            for modality in args.modalities
        ]
        print(f"{split}: converting {len(jobs)} video/annotation pairs with {args.workers} workers")

        if args.workers == 1:
            for job in jobs:
                add_stats(split_stats, convert_sequence(**job))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(convert_sequence, **job) for job in jobs]
                completed = 0
                for future in as_completed(futures):
                    add_stats(split_stats, future.result())
                    completed += 1
                    if completed == len(futures) or completed % 10 == 0:
                        print(f"{split}: completed {completed}/{len(futures)}")

        all_stats[split] = split_stats.__dict__
        print(
            f"{split}: sequences={split_stats.sequences} frames={split_stats.frames} "
            f"positive={split_stats.positive_frames} empty={split_stats.empty_frames} "
            f"skipped={split_stats.skipped_frames}"
        )

    write_dataset_yaml(args.output)
    with (args.output / "conversion_stats.json").open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
        f.write("\n")
    print(f"wrote {args.output / 'dataset.yaml'}")
    print(f"wrote {args.output / 'conversion_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
