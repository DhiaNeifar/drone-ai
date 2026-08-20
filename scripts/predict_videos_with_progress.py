#!/usr/bin/env python3
"""Run YOLO inference over a folder of videos with frame-level progress bars."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from experiment_paths import timestamped_experiment_name


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict every video in a directory and show frame-level progress."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="YOLO checkpoint path.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("real-scenario"),
        help="Input video or directory containing videos.",
    )
    parser.add_argument("--experiment", required=True, help="Short experiment name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Process every Nth frame. Larger values run faster.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, such as cpu, 0 (first CUDA GPU), or mps.",
    )
    return parser.parse_args()


def frame_count(video_path: Path, stride: int) -> int | None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return math.ceil(frames / stride) if frames > 0 else None


def main() -> None:
    args = parse_args()
    started_at = datetime.now()
    if args.vid_stride < 1:
        raise SystemExit("--vid-stride must be at least 1")
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    print(f"[1/4] Scanning {args.source} ...", flush=True)
    if args.source.is_file():
        videos = [args.source] if args.source.suffix.lower() in VIDEO_EXTENSIONS else []
    else:
        videos = sorted(
            path
            for path in args.source.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    if not videos:
        raise SystemExit(f"No supported videos found in {args.source}")
    counts = {video: frame_count(video, args.vid_stride) for video in videos}
    known_total = sum(count for count in counts.values() if count is not None)
    print(f"      Found {len(videos)} video(s), {known_total:,} readable frame(s).")

    print(f"[2/4] Loading model {args.model} on {args.device} ...", flush=True)
    load_started = time.perf_counter()
    model = YOLO(str(args.model))
    print(f"      Model loaded in {time.perf_counter() - load_started:.1f}s.")

    artifact_name = timestamped_experiment_name(args.experiment, started_at)
    output = Path("artifacts").resolve() / artifact_name
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "inference.json"
    metadata = {
        "status": "running",
        "started_at": started_at.isoformat(timespec="seconds"),
        "experiment": args.experiment,
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()),
        "video_count": len(videos),
        "videos": [video.name for video in videos],
        "confidence": args.conf,
        "image_size": args.imgsz,
        "video_stride": args.vid_stride,
        "device": args.device,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[3/4] Running inference; output: {output}", flush=True)
    overall = tqdm(
        total=known_total or None,
        desc="All videos",
        unit="frame",
        dynamic_ncols=True,
        position=0,
    )
    inference_started = time.perf_counter()
    try:
        for index, video in enumerate(videos, start=1):
            results = model.predict(
                source=str(video),
                stream=True,
                save=True,
                project=str(output.parent),
                name=output.name,
                exist_ok=True,
                conf=args.conf,
                imgsz=args.imgsz,
                vid_stride=args.vid_stride,
                device=args.device,
                verbose=False,
            )
            with tqdm(
                results,
                total=counts[video],
                desc=f"Video {index}/{len(videos)}: {video.name}",
                unit="frame",
                dynamic_ncols=True,
                position=1,
                leave=False,
            ) as current:
                for result in current:
                    detections = len(result.boxes) if result.boxes is not None else 0
                    inference_ms = result.speed.get("inference", 0.0)
                    current.set_postfix(
                        detections=detections,
                        inference=f"{inference_ms:.1f}ms",
                        refresh=False,
                    )
                    overall.update(1)
    except BaseException:
        metadata["status"] = "failed"
        metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        overall.close()

    metadata["status"] = "complete"
    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["elapsed_seconds"] = round(time.perf_counter() - inference_started, 3)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[4/4] Complete. Annotated videos are in {output}")


if __name__ == "__main__":
    main()
