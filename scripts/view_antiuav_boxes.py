#!/usr/bin/env python3
"""View Anti-UAV-RGBT videos with frame-wise bounding boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Anti-UAV-RGBT gt_rect annotations on visible/infrared video."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        type=Path,
        help="Path to a sequence directory, e.g. Anti-UAV-RGBT/train/20190925_101846_1_2",
    )
    parser.add_argument(
        "--modality",
        default="visible",
        choices=("visible", "infrared"),
        help="Video/annotation pair to view.",
    )
    parser.add_argument(
        "--start-frame",
        default=0,
        type=int,
        help="Frame index to start from.",
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="Advance this many frames after each displayed frame.",
    )
    parser.add_argument(
        "--delay",
        default=30,
        type=int,
        help="Playback delay in milliseconds. Use 0 for frame-by-frame.",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        help="Save one annotated frame to this path and exit.",
    )
    return parser.parse_args()


def draw_annotation(frame, exists: int, rect: list[float], frame_index: int):
    height, width = frame.shape[:2]
    color = (0, 255, 0) if exists else (128, 128, 128)

    if exists and len(rect) == 4:
        x, y, w, h = [int(round(v)) for v in rect]
        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(0, min(width - 1, x + w))
        y2 = max(0, min(height - 1, y + h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            "drone",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    label = f"frame {frame_index} | exist={exists} | q quit, space pause, n next"
    cv2.putText(
        frame,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> int:
    args = parse_args()
    sequence = args.sequence
    video_path = sequence / f"{args.modality}.mp4"
    annotation_path = sequence / f"{args.modality}.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    exists = annotation["exist"]
    boxes = annotation["gt_rect"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"{video_path} | frames={total_frames} annotations={len(boxes)} "
        f"size={width}x{height} fps={fps:.2f}"
    )

    frame_index = max(0, args.start_frame)
    paused = args.delay == 0

    while frame_index < total_frames and frame_index < len(boxes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        frame = draw_annotation(frame, int(exists[frame_index]), boxes[frame_index], frame_index)

        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_frame), frame)
            print(f"saved {args.save_frame}")
            break

        cv2.imshow("Anti-UAV-RGBT annotations", frame)
        key = cv2.waitKey(0 if paused else args.delay) & 0xFF

        if key == ord("q") or key == 27:
            break
        if key == ord(" "):
            paused = not paused
        if paused and key != ord("n"):
            continue

        frame_index += max(1, args.stride)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
