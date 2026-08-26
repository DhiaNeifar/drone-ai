#!/usr/bin/env python3
"""Convert annotated GEM video frames from Label Studio to a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from view_gem_video_annotations import drone_tracks, find_video, interpolate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/gem_new_videos"))
    parser.add_argument("--output", type=Path, help="Default: <root>/yolo")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clip_box(box: tuple[float, ...]) -> tuple[float, float, float, float] | None:
    x, y, width, height = box
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(100.0, x + width), min(100.0, y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 200, (y1 + y2) / 200, (x2 - x1) / 100, (y2 - y1) / 100)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or root / "yolo").resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {output}; pass --overwrite to rebuild")
        shutil.rmtree(output)
    images_root, labels_root = output / "images", output / "labels"
    images_root.mkdir(parents=True)
    labels_root.mkdir(parents=True)

    tasks = json.loads((root / "export_label_studio.json").read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        tasks = [tasks]
    audit: Counter[str] = Counter()
    videos = []
    for task in tasks:
        video_url = task.get("data", {}).get("video_url", "")
        video_path = find_video(root / "videos", video_url)
        if video_path is None:
            audit["missing_videos"] += 1
            continue
        tracks = drone_tracks(task)
        if not tracks:
            audit["videos_without_tracks"] += 1
            continue
        videos.append(video_path.name)
        image_dir = images_root / video_path.stem
        label_dir = labels_root / video_path.stem
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            audit["unreadable_videos"] += 1
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=total, desc=video_path.stem, unit="frame")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = (frame_index + 1) / fps
                boxes = [
                    normalized
                    for track in tracks
                    if (box := interpolate(timestamp, track)) is not None
                    if (normalized := clip_box(box)) is not None
                ]
                if boxes:
                    stem = f"frame_{frame_index:06d}"
                    image_path = image_dir / f"{stem}.jpg"
                    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError(f"Could not write {image_path}")
                    label_text = "".join(
                        f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n"
                        for x, y, width, height in boxes
                    )
                    (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")
                    audit["images"] += 1
                    audit["boxes"] += len(boxes)
                else:
                    audit["excluded_unboxed_frames"] += 1
                frame_index += 1
                progress.update(1)
        finally:
            progress.close()
            capture.release()

    config = {"path": str(output), "train": "images", "names": {0: "drone"}}
    (output / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = {"videos": videos, **dict(audit)}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"YOLO dataset: {output / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
