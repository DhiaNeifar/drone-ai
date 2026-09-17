#!/usr/bin/env python3
"""Train and run YOLO experiments with repository-standard output paths."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from experiment_paths import timestamped_experiment_name


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train YOLO into runs/<timestamp>_<name>.")
    train_parser.add_argument("--experiment", required=True, help="Short experiment name.")
    train_parser.add_argument("--model", default="yolo11n.pt", help="Base model or checkpoint.")
    train_parser.add_argument("--data", type=Path, default=Path("datasets/drone_merged/dataset.yaml"))
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=32)
    train_parser.add_argument("--device", default="mps")
    train_parser.add_argument("--workers", type=int, default=8)
    train_parser.add_argument("--patience", type=int, default=15)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--save-period", type=int, default=-1)

    predict_parser = subparsers.add_parser("predict", help="Run video inference into artifacts/<timestamp>_<name>.")
    predict_parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint path.")
    predict_parser.add_argument("--source", type=Path, required=True, help="Input video or video directory.")
    predict_parser.add_argument("--experiment", required=True, help="Short experiment name.")
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--imgsz", type=int, default=640)
    predict_parser.add_argument("--vid-stride", type=int, default=1)
    predict_parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    if not args.data.is_file():
        raise SystemExit(f"Dataset configuration not found: {args.data}")
    run_name = timestamped_experiment_name(args.experiment, datetime.now())
    runs_root = Path("runs").resolve()
    output = runs_root / run_name
    print(f"Training output: {output}", flush=True)
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        save_period=args.save_period,
        project=str(runs_root),
        name=run_name,
        exist_ok=False,
    )


def frame_count(video_path: Path, stride: int) -> int | None:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return math.ceil(frames / stride) if frames > 0 else None


def predict(args: argparse.Namespace) -> None:
    from tqdm import tqdm
    from ultralytics import YOLO

    started_at = datetime.now()
    if args.vid_stride < 1:
        raise SystemExit("--vid-stride must be at least 1")
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")
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
    print(f"Found {len(videos)} video(s), {known_total:,} readable frame(s).")

    model = YOLO(str(args.model))
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
    overall = tqdm(total=known_total or None, desc="All videos", unit="frame", dynamic_ncols=True)
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
                desc=f"Video {index}/{len(videos)}",
                unit="frame",
                leave=False,
            ) as current:
                for result in current:
                    detections = len(result.boxes) if result.boxes is not None else 0
                    current.set_postfix(detections=detections, refresh=False)
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
    print(f"Complete. Annotated videos are in {output}")


def main() -> int:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
