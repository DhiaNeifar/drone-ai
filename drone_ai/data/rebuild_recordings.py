#!/usr/bin/env python3
"""Re-extract clean recording frames and retain their existing annotations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm


@dataclass(frozen=True)
class RecordingMatch:
    recording: str
    annotation_set: str
    sequence: str


MATCHES = (
    RecordingMatch("10_09_2026_08_45_34", "dataset_samples1", "camera"),
    RecordingMatch("10_09_2026_09_14_13", "dataset_samples0", "vd_7"),
    RecordingMatch("10_09_2026_09_16_20", "dataset_samples1", "camera_1"),
    RecordingMatch("10_09_2026_09_19_56", "dataset_samples0", "vd_8"),
    RecordingMatch("10_09_2026_09_27_38", "dataset_samples1", "camera_2"),
    RecordingMatch("10_09_2026_09_30_51", "dataset_samples0", "vd_9"),
    RecordingMatch("10_09_2026_09_35_03", "dataset_samples0", "vd_10"),
    RecordingMatch("10_09_2026_09_39_10", "dataset_samples1", "camera_3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/recordings_clean_annotated"),
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_annotations(path: Path) -> tuple[dict[str, object], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    exists = data.get("exist")
    boxes = data.get("gt_rect")
    if not isinstance(exists, list) or not isinstance(boxes, list):
        raise ValueError(f"Invalid annotation structure: {path}")
    if len(exists) != len(boxes):
        raise ValueError(
            f"Annotation length mismatch in {path}: "
            f"exist={len(exists)}, gt_rect={len(boxes)}"
        )
    for index, (present, box) in enumerate(zip(exists, boxes)):
        if bool(present) and (not isinstance(box, list) or len(box) != 4):
            raise ValueError(f"Positive frame {index} has no valid box in {path}")
    return data, len(exists)


def load_and_validate_csv(path: Path, expected_frames: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "frame" not in fieldnames or "drone_exist" not in fieldnames:
        raise ValueError(f"CSV is missing required columns: {path}")
    if len(rows) != expected_frames:
        raise ValueError(
            f"CSV length mismatch in {path}: {len(rows)} != {expected_frames}"
        )
    for index, row in enumerate(rows):
        if int(row["frame"]) != index:
            raise ValueError(f"Non-contiguous frame index at row {index} in {path}")
    return fieldnames, rows


def write_reconciled_csv(
    destination: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    exists: list[object],
) -> int:
    corrections = 0
    for row, present in zip(rows, exists):
        value = str(int(bool(present)))
        corrections += row["drone_exist"] != value
        row["drone_exist"] = value
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return corrections


def clip_boxes(data: dict[str, object], width: int, height: int) -> int:
    boxes = data["gt_rect"]
    exists = data["exist"]
    corrections = 0
    for index, (present, box) in enumerate(zip(exists, boxes)):
        if not present:
            continue
        x, y, box_width, box_height = (float(value) for value in box)
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + box_width))
        y2 = max(0.0, min(float(height), y + box_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Frame {index} has an empty box after clipping: {box}")
        clipped = [x1, y1, x2 - x1, y2 - y1]
        clipped = [int(value) if value.is_integer() else value for value in clipped]
        if clipped != box:
            boxes[index] = clipped
            corrections += 1
    return corrections


def extract_sequence(
    video_path: Path,
    frames_dir: Path,
    expected_frames: int,
    jpeg_quality: int,
) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {video_path}")

    frames_dir.mkdir(parents=True)
    frame_index = 0
    progress = tqdm(total=expected_frames, desc=frames_dir.parent.name, unit="frame")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= expected_frames:
                raise ValueError(
                    f"Video has more frames than its annotations: {video_path}"
                )
            timestamp = frame_index / fps
            destination = frames_dir / f"frame_{frame_index:06d}_t{timestamp:.2f}s.jpg"
            written = cv2.imwrite(
                str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not written:
                raise OSError(f"Could not write frame: {destination}")
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        capture.release()

    if frame_index != expected_frames:
        raise ValueError(
            f"Decoded frame count mismatch for {video_path}: "
            f"{frame_index} != {expected_frames}"
        )
    return frame_index, fps, width, height


def main() -> int:
    args = parse_args()
    datasets_root = args.datasets_root.resolve()
    output = args.output.resolve()
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {output}; pass --overwrite to rebuild")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    report: dict[str, object] = {
        "output": str(output),
        "jpeg_quality": args.jpeg_quality,
        "sequences": [],
    }
    total_frames = 0
    total_positive_frames = 0

    for match in MATCHES:
        video_path = datasets_root / "Recordings" / match.recording / "camera.mp4"
        annotation_dir = datasets_root / match.annotation_set / match.sequence
        annotation_path = annotation_dir / f"{match.sequence}.json"
        csv_path = annotation_dir / f"{match.sequence}.csv"
        for required in (video_path, annotation_path, csv_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        annotations, expected_frames = load_annotations(annotation_path)
        csv_fieldnames, csv_rows = load_and_validate_csv(csv_path, expected_frames)
        sequence_dir = output / match.recording
        frame_count, fps, width, height = extract_sequence(
            video_path,
            sequence_dir / "frames",
            expected_frames,
            args.jpeg_quality,
        )
        clipped_box_corrections = clip_boxes(annotations, width, height)
        (sequence_dir / f"{match.recording}.json").write_text(
            json.dumps(annotations, indent=1) + "\n", encoding="utf-8"
        )
        csv_corrections = write_reconciled_csv(
            sequence_dir / f"{match.recording}.csv",
            csv_fieldnames,
            csv_rows,
            annotations["exist"],
        )

        positive_frames = sum(bool(value) for value in annotations["exist"])
        total_frames += frame_count
        total_positive_frames += positive_frames
        report["sequences"].append(
            {
                "sequence": match.recording,
                "annotation_sequence": match.sequence,
                "recording": match.recording,
                "video": str(video_path),
                "annotation_source": str(annotation_dir),
                "frames": frame_count,
                "positive_frames": positive_frames,
                "csv_presence_corrections": csv_corrections,
                "clipped_box_corrections": clipped_box_corrections,
                "fps": fps,
                "width": width,
                "height": height,
            }
        )

    report["sequence_count"] = len(MATCHES)
    report["frames"] = total_frames
    report["positive_frames"] = total_positive_frames
    report["negative_frames"] = total_frames - total_positive_frames
    report["csv_presence_corrections"] = sum(
        sequence["csv_presence_corrections"] for sequence in report["sequences"]
    )
    report["clipped_box_corrections"] = sum(
        sequence["clipped_box_corrections"] for sequence in report["sequences"]
    )
    (output / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
