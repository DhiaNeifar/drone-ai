#!/usr/bin/env python3
"""Browse image frames in the GEM dataset with OpenCV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "gem_dataset"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
WINDOW_NAME = "GEM dataset frame viewer"

# OpenCV reports arrow keys differently across operating systems and GUI backends.
LEFT_KEYS = {2, 81, 63234, 65361, 2424832}
UP_KEYS = {0, 82, 63232, 65362, 2490368}
RIGHT_KEYS = {3, 83, 63235, 65363, 2555904}
DOWN_KEYS = {1, 84, 63233, 65364, 2621440}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse GEM dataset frames with arrow keys."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum displayed image width (default: 1600).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Maximum displayed image height (default: 900).",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[object]:
    """Return a key that sorts frame_2 before frame_10."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def discover_sequences(root: Path) -> list[tuple[Path, list[Path]]]:
    sequences = []
    for sequence_dir in sorted((path for path in root.iterdir() if path.is_dir())):
        frames_dir = sequence_dir / "frames"
        if not frames_dir.is_dir():
            continue
        frames = sorted(
            (
                path
                for path in frames_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=natural_key,
        )
        if frames:
            sequences.append((sequence_dir, frames))
    return sequences


def fit_to_window(image, max_width: int, max_height: int):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def add_status_bar(
    image,
    sequence_dir: Path,
    sequence_index: int,
    sequence_count: int,
    frame_path: Path,
    frame_index: int,
    frame_count: int,
):
    lines = (
        f"Folder {sequence_index + 1}/{sequence_count} | Frame {frame_index + 1}/{frame_count}",
        f"{sequence_dir.name} / {frame_path.name}",
        "Left/Right: frame | Up/Down: folder | Q or Esc: quit",
    )
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    for line_index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (14, 27 + line_index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if args.max_width <= 0 or args.max_height <= 0:
        raise ValueError("--max-width and --max-height must be positive")

    sequences = discover_sequences(root)
    if not sequences:
        raise RuntimeError(f"No images found in sequence frames directories under {root}")

    print(f"Found {len(sequences)} folders and {sum(len(frames) for _, frames in sequences)} frames")
    print("Controls: Left/Right = frame, Up/Down = folder, Q/Esc = quit")

    sequence_index = 0
    frame_indices = [0] * len(sequences)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            sequence_dir, frames = sequences[sequence_index]
            frame_index = frame_indices[sequence_index]
            frame_path = frames[frame_index]
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"OpenCV could not read image: {frame_path}")

            image = fit_to_window(image, args.max_width, args.max_height)
            image = add_status_bar(
                image,
                sequence_dir,
                sequence_index,
                len(sequences),
                frame_path,
                frame_index,
                len(frames),
            )
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKeyEx(0)

            if key in (27, ord("q"), ord("Q")):
                break
            if key in LEFT_KEYS or key in (ord("a"), ord("A")):
                frame_indices[sequence_index] = max(0, frame_index - 1)
            elif key in RIGHT_KEYS or key in (ord("d"), ord("D")):
                frame_indices[sequence_index] = min(len(frames) - 1, frame_index + 1)
            elif key in UP_KEYS or key in (ord("w"), ord("W")):
                sequence_index = max(0, sequence_index - 1)
            elif key in DOWN_KEYS or key in (ord("s"), ord("S")):
                sequence_index = min(len(sequences) - 1, sequence_index + 1)
            elif key == -1 and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
