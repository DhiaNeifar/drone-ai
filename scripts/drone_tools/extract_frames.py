#!/usr/bin/env python3
"""Recursively extract videos into an annotation-ready mirrored directory tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass
class ExtractionResult:
    source_video: str
    sequence: str
    frames: int
    fps: float
    width: int
    height: int
    annotation_format: str
    video_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Folder to scan recursively for videos.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: a sibling folder named <root>_annotations.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_videos(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    )


def validate_json_sidecar(path: Path, frame_count: int) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    exists = data.get("exist")
    boxes = data.get("gt_rect")
    if not isinstance(exists, list) or not isinstance(boxes, list):
        raise ValueError(f"Unsupported JSON annotation schema: {path}")
    if len(exists) != frame_count or len(boxes) != frame_count:
        raise ValueError(
            f"Annotation/video length mismatch for {path}: "
            f"exist={len(exists)}, gt_rect={len(boxes)}, frames={frame_count}"
        )
    return data


def copy_or_initialize_annotations(
    video: Path,
    sequence_dir: Path,
    frame_count: int,
    fps: float,
) -> str:
    json_source = video.with_suffix(".json")
    csv_source = video.with_suffix(".csv")
    json_destination = sequence_dir / f"{video.stem}.json"
    csv_destination = sequence_dir / f"{video.stem}.csv"

    csv_fieldnames = None
    csv_rows = None
    if csv_source.is_file():
        with csv_source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            csv_fieldnames = reader.fieldnames
            csv_rows = list(reader)
        if not csv_fieldnames or len(csv_rows) != frame_count:
            raise ValueError(
                f"Annotation/video length mismatch for {csv_source}: "
                f"rows={len(csv_rows)}, frames={frame_count}"
            )

    if json_source.is_file():
        data = validate_json_sidecar(json_source, frame_count)
        json_destination.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        annotation_format = "json"
    elif csv_source.is_file():
        shutil.copy2(csv_source, csv_destination)
        return "csv"
    else:
        data = {"exist": [0] * frame_count, "gt_rect": [[] for _ in range(frame_count)]}
        json_destination.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        annotation_format = "json"

    if csv_source.is_file():
        if "drone_exist" not in csv_fieldnames:
            csv_fieldnames.append("drone_exist")
        for index, row in enumerate(csv_rows):
            row["drone_exist"] = str(int(bool(data["exist"][index])))
        with csv_destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    else:
        exists = data["exist"]
        with csv_destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frame", "timestamp", "drone_exist", "audio_exist"))
            for index, present in enumerate(exists):
                writer.writerow((index, f"{index / fps:.3f}", int(bool(present)), 0))
    return annotation_format


def extract_video(
    video: Path,
    source_root: Path,
    output_root: Path,
    jpeg_quality: int,
) -> ExtractionResult:
    import cv2
    from tqdm import tqdm

    relative = video.relative_to(source_root)
    sequence_dir = output_root / relative.parent / video.stem
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"OpenCV could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {video}")

    frame_index = 0
    progress = tqdm(
        total=expected or None,
        desc=str(relative),
        unit="frame",
        position=0,
        leave=False,
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            destination = frames_dir / f"frame_{frame_index:06d}_t{timestamp:.2f}s.jpg"
            if not cv2.imwrite(
                str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            ):
                raise OSError(f"Could not write frame: {destination}")
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()
    if frame_index == 0:
        raise ValueError(f"No frames could be decoded: {video}")
    if expected > 0 and frame_index != expected:
        raise ValueError(
            f"Decoded frame count mismatch for {video}: {frame_index} != {expected}"
        )

    annotation_format = copy_or_initialize_annotations(
        video, sequence_dir, frame_index, fps
    )
    return ExtractionResult(
        source_video=str(video),
        sequence=str(sequence_dir.relative_to(output_root)),
        frames=frame_index,
        fps=fps,
        width=width,
        height=height,
        annotation_format=annotation_format,
        video_sha256=sha256(video),
    )


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input folder does not exist: {root}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else root.with_name(f"{root.name}_annotations")
    )
    if output == root or output.is_relative_to(root):
        raise SystemExit("Output must not be the input folder or one of its descendants")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {output}; pass --overwrite to rebuild")
        shutil.rmtree(output)

    videos = find_videos(root)
    if not videos:
        raise SystemExit(f"No supported videos found under {root}")
    output.mkdir(parents=True)
    print(f"Found {len(videos)} videos under {root}")
    print(f"Writing annotation workspace to {output}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_video, video, root, output, args.jpeg_quality): video
            for video in videos
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise

    results.sort(key=lambda result: result.sequence)
    manifest = {
        "format": "recursive-video-frames-v1",
        "source_root": str(root),
        "output_root": str(output),
        "video_count": len(results),
        "frame_count": sum(result.frames for result in results),
        "jpeg_quality": args.jpeg_quality,
        "videos": [asdict(result) for result in results],
    }
    (output / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("video_count", "frame_count")}, indent=2))
    print(f"Annotation workspace: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
