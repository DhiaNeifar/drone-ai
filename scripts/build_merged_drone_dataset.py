#!/usr/bin/env python3
"""Build one clean, single-class YOLO drone dataset from local sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from tqdm import tqdm


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".jpg",
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}
SPLITS = ("train", "val", "test")
SOURCE_PRIORITY = {
    "gem_new": 115,
    "gem": 110,
    "midgard": 105,
    "antiuav": 100,
    "pathik": 90,
    "kaggle1_xml": 85,
    "kaggle2": 80,
    "kaggle3": 70,
    "rigoleto": 60,
    "kaggle1_yolo": 50,
}


@dataclass
class Candidate:
    source: str
    original: str
    group: str
    source_split: str | None
    boxes: list[tuple[float, float, float, float]]
    image_hash: str
    visual_hash: str
    image_format: str
    width: int
    height: int
    image_path: Path | None = None
    parquet_path: Path | None = None
    parquet_row: int | None = None
    negative_kind: str = ""
    removed_classes: Counter[str] = field(default_factory=Counter)
    split: str | None = None


@dataclass
class Audit:
    candidates: Counter[str] = field(default_factory=Counter)
    accepted: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    removed_objects: Counter[str] = field(default_factory=Counter)
    repaired_boxes: Counter[str] = field(default_factory=Counter)
    duplicates: Counter[str] = field(default_factory=Counter)
    near_duplicates: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("datasets/drone_merged"))
    parser.add_argument(
        "--antiuav-stride",
        type=int,
        default=3,
        help="Keep every Nth already-extracted Anti-UAV frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=10,
        help="Maximum 512-bit difference-hash distance to report across sources.",
    )
    parser.add_argument(
        "--min-box-pixels",
        type=float,
        default=8.0,
        help="Drop an image if any drone box is narrower or shorter than this many pixels.",
    )
    return parser.parse_args()


def stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def inspect_image(data: bytes) -> tuple[str, int, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        image_format = (image.format or "").upper()
        width, height = image.size
    if image_format not in FORMAT_EXTENSIONS or width < 2 or height < 2:
        raise ValueError(f"unsupported or invalid image: format={image_format} size={width}x{height}")
    return image_format, width, height


def difference_hash(data: bytes) -> str:
    """Return a 512-bit perceptual hash resilient to encoding and size changes."""
    with Image.open(io.BytesIO(data)) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L")
        horizontal = grayscale.resize((17, 16), Image.Resampling.LANCZOS)
        vertical = grayscale.resize((16, 17), Image.Resampling.LANCZOS)
    horizontal_pixels = list(horizontal.getdata())
    vertical_pixels = list(vertical.getdata())
    bits = 0
    for y in range(16):
        row = y * 17
        for x in range(16):
            bits = (bits << 1) | (horizontal_pixels[row + x] > horizontal_pixels[row + x + 1])
    for y in range(16):
        row = y * 16
        next_row = (y + 1) * 16
        for x in range(16):
            bits = (bits << 1) | (vertical_pixels[row + x] > vertical_pixels[next_row + x])
    return f"{bits:0128x}"


def image_info(path: Path) -> tuple[str, str, int, int, str]:
    data = path.read_bytes()
    image_format, width, height = inspect_image(data)
    return hashlib.sha256(data).hexdigest(), image_format, width, height, difference_hash(data)


def clip_yolo_box(
    values: Iterable[float], width: int, height: int
) -> tuple[tuple[float, float, float, float] | None, bool]:
    x, y, w, h = values
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    clipped = (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))
    repaired = any(abs(a - b) > 1e-9 for a, b in zip((x1, y1, x2, y2), clipped))
    x1, y1, x2, y2 = clipped
    if (x2 - x1) * width < 1.0 or (y2 - y1) * height < 1.0:
        return None, repaired
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), repaired


def parse_yolo_label(
    path: Path,
    width: int,
    height: int,
    keep_class: int = 0,
    class_names: dict[int, str] | None = None,
) -> tuple[list[tuple[float, float, float, float]], Counter[str], int, int]:
    boxes: list[tuple[float, float, float, float]] = []
    removed: Counter[str] = Counter()
    repaired = 0
    malformed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            malformed += 1
            continue
        try:
            class_id = int(float(fields[0]))
            values = tuple(float(value) for value in fields[1:])
        except ValueError:
            malformed += 1
            continue
        if class_id != keep_class:
            removed[(class_names or {}).get(class_id, f"class_{class_id}")] += 1
            continue
        box, was_repaired = clip_yolo_box(values, width, height)
        repaired += int(was_repaired)
        if box is None:
            malformed += 1
        else:
            boxes.append(box)
    return boxes, removed, repaired, malformed


def add_local_candidate(
    candidates: list[Candidate],
    audit: Audit,
    *,
    source: str,
    image_path: Path,
    label_path: Path,
    group: str,
    source_split: str | None,
    keep_class: int = 0,
    class_names: dict[int, str] | None = None,
) -> None:
    if not label_path.is_file():
        audit.skipped[f"{source}:missing_label"] += 1
        return
    try:
        image_hash, image_format, width, height, visual_hash = image_info(image_path)
        boxes, removed, repaired, malformed = parse_yolo_label(
            label_path, width, height, keep_class, class_names
        )
    except (OSError, UnidentifiedImageError, ValueError):
        audit.skipped[f"{source}:unreadable_image"] += 1
        return
    audit.repaired_boxes[source] += repaired
    audit.skipped[f"{source}:malformed_boxes"] += malformed
    audit.removed_objects.update({f"{source}:{key}": value for key, value in removed.items()})
    negative_kind = "non_target" if removed and not boxes else ("native_empty" if not boxes else "")
    candidates.append(
        Candidate(
            source=source,
            original=str(image_path),
            group=group,
            source_split=source_split,
            boxes=boxes,
            image_hash=image_hash,
            visual_hash=visual_hash,
            image_format=image_format,
            width=width,
            height=height,
            image_path=image_path,
            negative_kind=negative_kind,
            removed_classes=removed,
        )
    )
    audit.candidates[source] += 1


def ingest_antiuav(root: Path, stride: int, candidates: list[Candidate], audit: Audit) -> None:
    base = root / "antiuav_yolo"
    for split in SPLITS:
        image_dir, label_dir = base / "images" / split, base / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for path in tqdm(images, desc=f"Anti-UAV {split}", unit="image"):
            match = re.search(r"_(\d+)$", path.stem)
            if match and int(match.group(1)) % stride:
                audit.skipped["antiuav:stride"] += 1
                continue
            sequence = re.sub(r"_visible_\d+$", "", path.stem)
            add_local_candidate(
                candidates,
                audit,
                source="antiuav",
                image_path=path,
                label_path=label_dir / f"{path.stem}.txt",
                group=f"antiuav:{sequence}",
                source_split=split,
            )


def ingest_kaggle1_xml(root: Path, candidates: list[Candidate], audit: Audit) -> None:
    base = root / "Drone Dataset Kaggle 1" / "dataset_xml_format" / "dataset_xml_format"
    for xml_path in tqdm(sorted(base.glob("*.xml")), desc="Kaggle 1 XML", unit="image"):
        try:
            xml_root = ET.parse(xml_path).getroot()
            filename = xml_root.findtext("filename")
            image_path = base / filename if filename else None
            if image_path is None or not image_path.is_file():
                matches = [p for p in base.glob(f"{xml_path.stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
                image_path = matches[0] if matches else None
            if image_path is None:
                audit.skipped["kaggle1_xml:missing_image"] += 1
                continue
            image_hash, image_format, width, height, visual_hash = image_info(image_path)
            boxes = []
            for obj in xml_root.findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                if name != "drone":
                    audit.removed_objects[f"kaggle1_xml:{name or 'unknown'}"] += 1
                    continue
                node = obj.find("bndbox")
                if node is None:
                    audit.skipped["kaggle1_xml:malformed_boxes"] += 1
                    continue
                xmin, ymin, xmax, ymax = (
                    float(node.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")
                )
                raw = ((xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height,
                       (xmax - xmin) / width, (ymax - ymin) / height)
                box, repaired = clip_yolo_box(raw, width, height)
                audit.repaired_boxes["kaggle1_xml"] += int(repaired)
                if box is not None:
                    boxes.append(box)
            candidates.append(
                Candidate(
                    source="kaggle1_xml",
                    original=str(image_path),
                    group=f"kaggle1:{image_hash}",
                    source_split=None,
                    boxes=boxes,
                    image_hash=image_hash,
                    visual_hash=visual_hash,
                    image_format=image_format,
                    width=width,
                    height=height,
                    image_path=image_path,
                    negative_kind="native_empty" if not boxes else "",
                )
            )
            audit.candidates["kaggle1_xml"] += 1
        except (ET.ParseError, OSError, UnidentifiedImageError, ValueError):
            audit.skipped["kaggle1_xml:invalid_record"] += 1


def ingest_flat_yolo(
    base: Path,
    source: str,
    candidates: list[Candidate],
    audit: Audit,
) -> None:
    images = sorted(path for path in base.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for path in tqdm(images, desc=source, unit="image"):
        if source == "kaggle3":
            match = re.match(r"(video\d+)_", path.stem, re.IGNORECASE)
            group = f"kaggle3:{match.group(1).lower()}" if match else f"kaggle3:{path.stem}"
        else:
            group = f"{source}:{path.stem}"
        add_local_candidate(
            candidates,
            audit,
            source=source,
            image_path=path,
            label_path=base / f"{path.stem}.txt",
            group=group,
            source_split=None,
        )


def ingest_kaggle2(root: Path, candidates: list[Candidate], audit: Audit) -> None:
    base = root / "Drone Dataset Kaggle 2" / "drone-detection-new.v5-new-train.yolov8"
    names = {0: "airplane", 1: "drone", 2: "helicopter"}
    for split_name, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        image_dir, label_dir = base / split_name / "images", base / split_name / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for path in tqdm(images, desc=f"Kaggle 2 {split}", unit="image"):
            original_stem = re.sub(r"\.rf\.[^.]+$", "", path.stem)
            add_local_candidate(
                candidates,
                audit,
                source="kaggle2",
                image_path=path,
                label_path=label_dir / f"{path.stem}.txt",
                group=f"kaggle2:{original_stem}",
                source_split=split,
                keep_class=1,
                class_names=names,
            )


def ingest_rigoleto(root: Path, candidates: list[Candidate], audit: Audit) -> None:
    base = root / "Rigoleto-drone-detection" / "data"
    for split_name, split in (("train", "train"), ("val", "val"), ("test", "test")):
        image_dir, label_dir = base / split_name / "images", base / split_name / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for path in tqdm(images, desc=f"Rigoleto {split}", unit="image"):
            add_local_candidate(
                candidates,
                audit,
                source="rigoleto",
                image_path=path,
                label_path=label_dir / f"{path.stem}.txt",
                group=f"rigoleto:{path.stem}",
                source_split=split,
            )


def ingest_midgard(root: Path, candidates: list[Candidate], audit: Audit) -> None:
    base = root / "MIDGARD"
    image_dirs = sorted(path for path in base.glob("*/*/images") if path.is_dir())
    for image_dir in image_dirs:
        scene = image_dir.parent
        annotation_dir = scene / "annotation"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for image_path in tqdm(images, desc=f"MIDGARD {scene.name}", unit="image"):
            annotation_path = annotation_dir / f"annot_{image_path.stem.removeprefix('image_')}.csv"
            if not annotation_path.is_file():
                audit.skipped["midgard:missing_label"] += 1
                continue
            try:
                image_hash, image_format, width, height, visual_hash = image_info(image_path)
                boxes = []
                with annotation_path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.reader(handle):
                        if len(row) < 5:
                            audit.skipped["midgard:malformed_boxes"] += 1
                            continue
                        x, y, box_width, box_height = map(float, row[1:5])
                        if not any((x, y, box_width, box_height)):
                            continue
                        raw = (
                            (x + box_width / 2) / width,
                            (y + box_height / 2) / height,
                            box_width / width,
                            box_height / height,
                        )
                        box, repaired = clip_yolo_box(raw, width, height)
                        audit.repaired_boxes["midgard"] += int(repaired)
                        if box is None:
                            audit.skipped["midgard:malformed_boxes"] += 1
                        else:
                            boxes.append(box)
                candidates.append(
                    Candidate(
                        source="midgard",
                        original=str(image_path),
                        group=f"midgard:{scene.relative_to(base)}",
                        source_split=None,
                        boxes=boxes,
                        image_hash=image_hash,
                        visual_hash=visual_hash,
                        image_format=image_format,
                        width=width,
                        height=height,
                        image_path=image_path,
                        negative_kind="native_empty" if not boxes else "",
                    )
                )
                audit.candidates["midgard"] += 1
            except (OSError, UnidentifiedImageError, ValueError):
                audit.skipped["midgard:invalid_record"] += 1


def interpolate_gem_track(sequence: list[dict[str, object]]) -> dict[int, tuple[float, float, float, float]]:
    """Interpolate Label Studio video boxes, retaining enabled frame intervals."""
    points = sorted(sequence, key=lambda point: int(point["frame"]))
    interpolated: dict[int, tuple[float, float, float, float]] = {}
    for left, right in zip(points, points[1:]):
        start, end = int(left["frame"]), int(right["frame"])
        if not bool(left.get("enabled")) or end <= start:
            continue
        left_box = tuple(float(left[key]) for key in ("x", "y", "width", "height"))
        right_box = tuple(float(right[key]) for key in ("x", "y", "width", "height"))
        final_frame = end if bool(right.get("enabled")) else end - 1
        for frame in range(start, final_frame + 1):
            ratio = (frame - start) / (end - start)
            interpolated[frame] = tuple(
                first + ratio * (second - first) for first, second in zip(left_box, right_box)
            )
    if points and bool(points[-1].get("enabled")):
        point = points[-1]
        interpolated[int(point["frame"])] = tuple(
            float(point[key]) for key in ("x", "y", "width", "height")
        )
    return interpolated


def ingest_gem(root: Path, candidates: list[Candidate], audit: Audit) -> None:
    base = root / "gem_dataset"
    for sequence_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        csv_files = list(sequence_dir.glob("*.csv"))
        json_files = list(sequence_dir.glob("*.json"))
        frames_dir = sequence_dir / "frames"
        if not csv_files or not json_files or not frames_dir.is_dir():
            audit.skipped["gem:unannotated_sequence"] += 1
            continue
        with json_files[0].open(encoding="utf-8") as handle:
            tasks = json.load(handle)
        results = tasks[0].get("annotations", [{}])[0].get("result", []) if tasks else []
        tracks = [
            interpolate_gem_track(result["value"].get("sequence", []))
            for result in results
            if result.get("type") == "videorectangle"
            and "Drone" in result.get("value", {}).get("labels", [])
        ]
        presence: dict[int, bool] = {}
        with csv_files[0].open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                presence[int(row["frame"])] = bool(int(row["drone_exist"]))
        frame_paths = sorted(
            path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in tqdm(frame_paths, desc=f"GEM {sequence_dir.name}", unit="image"):
            match = re.match(r"frame_(\d+)_", image_path.name)
            if not match:
                audit.skipped["gem:invalid_frame_name"] += 1
                continue
            frame = int(match.group(1))
            try:
                image_hash, image_format, width, height, visual_hash = image_info(image_path)
                boxes = []
                if presence.get(frame, False):
                    # Label Studio video frames are one-based; extracted filenames are zero-based.
                    for track in tracks:
                        percent_box = track.get(frame + 1)
                        if percent_box is None:
                            continue
                        x, y, box_width, box_height = percent_box
                        raw = ((x + box_width / 2) / 100, (y + box_height / 2) / 100,
                               box_width / 100, box_height / 100)
                        box, repaired = clip_yolo_box(raw, width, height)
                        audit.repaired_boxes["gem"] += int(repaired)
                        if box is not None:
                            boxes.append(box)
                if presence.get(frame, False) and not boxes:
                    audit.skipped["gem:positive_without_box"] += 1
                    continue
                candidates.append(
                    Candidate(
                        source="gem",
                        original=str(image_path),
                        group=f"gem:{sequence_dir.name}",
                        source_split=None,
                        boxes=boxes,
                        image_hash=image_hash,
                        visual_hash=visual_hash,
                        image_format=image_format,
                        width=width,
                        height=height,
                        image_path=image_path,
                        negative_kind="native_empty" if not boxes else "",
                    )
                )
                audit.candidates["gem"] += 1
            except (OSError, UnidentifiedImageError, ValueError):
                audit.skipped["gem:invalid_record"] += 1


def ingest_gem_new(root: Path, candidates: list[Candidate], audit: Audit) -> None:
    """Ingest the YOLO frames prepared from the new GEM videos."""
    base = root / "gem_new_videos" / "yolo"
    images_dir, labels_dir = base / "images", base / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        audit.skipped["gem_new:missing_prepared_dataset"] += 1
        return
    images = sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in tqdm(images, desc="GEM new", unit="image"):
        relative = image_path.relative_to(images_dir)
        add_local_candidate(
            candidates,
            audit,
            source="gem_new",
            image_path=image_path,
            label_path=(labels_dir / relative).with_suffix(".txt"),
            group=f"gem_new:{relative.parent}",
            source_split=None,
        )


def ingest_pathik(root: Path, candidates: list[Candidate], audit: Audit) -> None:
    base = root / "pathikg-drone-detection-dataset" / "data"
    files = sorted(base.glob("*.parquet"))
    for parquet_path in files:
        original_split = "test" if parquet_path.name.startswith("test-") else "train"
        parquet_file = pq.ParquetFile(parquet_path)
        row_index = 0
        progress = tqdm(total=parquet_file.metadata.num_rows, desc=f"Pathik {parquet_path.name}", unit="image")
        for batch in parquet_file.iter_batches(
            columns=["width", "height", "objects", "image", "image_id"],
            batch_size=128,
            use_threads=False,
        ):
            for record in batch.to_pylist():
                try:
                    data = record["image"]["bytes"]
                    image_format, actual_width, actual_height = inspect_image(data)
                    width, height = int(record["width"]), int(record["height"])
                    if (width, height) != (actual_width, actual_height):
                        width, height = actual_width, actual_height
                        audit.skipped["pathik:metadata_size_repaired"] += 1
                    boxes = []
                    for x, y, box_width, box_height in record["objects"]["bbox"] or []:
                        raw = (
                            (x + box_width / 2) / width,
                            (y + box_height / 2) / height,
                            box_width / width,
                            box_height / height,
                        )
                        box, repaired = clip_yolo_box(raw, width, height)
                        audit.repaired_boxes["pathik"] += int(repaired)
                        if box is not None:
                            boxes.append(box)
                        else:
                            audit.skipped["pathik:malformed_boxes"] += 1
                    image_hash = hashlib.sha256(data).hexdigest()
                    visual_hash = difference_hash(data)
                    original_name = record["image"].get("path") or f"image_{record['image_id']}"
                    candidates.append(
                        Candidate(
                            source="pathik",
                            original=f"{parquet_path}:{row_index}:{original_name}",
                            group=f"pathik:{image_hash}",
                            source_split=original_split,
                            boxes=boxes,
                            image_hash=image_hash,
                            visual_hash=visual_hash,
                            image_format=image_format,
                            width=width,
                            height=height,
                            parquet_path=parquet_path,
                            parquet_row=row_index,
                            negative_kind="native_empty" if not boxes else "",
                        )
                    )
                    audit.candidates["pathik"] += 1
                except (OSError, UnidentifiedImageError, ValueError, TypeError):
                    audit.skipped["pathik:invalid_record"] += 1
                row_index += 1
                progress.update(1)
        progress.close()


def filter_detection_candidates(
    candidates: list[Candidate], min_box_pixels: float, audit: Audit
) -> list[Candidate]:
    retained = []
    for candidate in candidates:
        if not candidate.boxes:
            audit.skipped[f"{candidate.source}:no_drone_boxes"] += 1
            continue
        undersized = [
            box
            for box in candidate.boxes
            if box[2] * candidate.width < min_box_pixels
            or box[3] * candidate.height < min_box_pixels
        ]
        if undersized:
            audit.skipped[f"{candidate.source}:undersized_box_image"] += 1
            audit.removed_objects[f"{candidate.source}:undersized_box"] += len(undersized)
            continue
        retained.append(candidate)
    return retained


def deduplicate(candidates: list[Candidate], audit: Audit) -> list[Candidate]:
    by_hash: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_hash[candidate.image_hash].append(candidate)
    selected = []
    for records in by_hash.values():
        records.sort(
            key=lambda item: (
                SOURCE_PRIORITY[item.source],
                bool(item.boxes),
                len(item.boxes),
            ),
            reverse=True,
        )
        winner = records[0]
        selected.append(winner)
        for duplicate in records[1:]:
            audit.duplicates[f"{duplicate.source}->kept:{winner.source}"] += 1
    return selected


class HashTree:
    def __init__(self) -> None:
        self.root: tuple[int, dict[int, object]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            current, children = node
            distance = (value ^ current).bit_count()
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child  # type: ignore[assignment]

    def query(self, value: int, threshold: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches = []
        pending = [self.root]
        while pending:
            current, children = pending.pop()
            distance = (value ^ current).bit_count()
            if distance <= threshold:
                matches.append((distance, current))
            lower, upper = distance - threshold, distance + threshold
            pending.extend(child for edge, child in children.items() if lower <= edge <= upper)
        return matches


def find_near_duplicates(
    candidates: list[Candidate], threshold: int, audit: Audit
) -> list[dict[str, object]]:
    tree = HashTree()
    by_visual_hash: dict[int, list[Candidate]] = defaultdict(list)
    rows = []
    for candidate in sorted(candidates, key=lambda item: (item.source, item.image_hash)):
        value = int(candidate.visual_hash, 16)
        matches = []
        for distance, matched_hash in tree.query(value, threshold):
            for other in by_visual_hash[matched_hash]:
                if other.source == candidate.source:
                    continue
                aspect_a = candidate.width / candidate.height
                aspect_b = other.width / other.height
                if abs(aspect_a - aspect_b) / max(aspect_a, aspect_b) > 0.02:
                    continue
                matches.append((distance, other))
        for distance, other in sorted(matches, key=lambda item: item[0])[:3]:
            key = f"{candidate.source}<->{other.source}"
            audit.near_duplicates[key] += 1
            rows.append(
                {
                    "hamming_distance": distance,
                    "source_a": other.source,
                    "image_a": other.original,
                    "sha256_a": other.image_hash,
                    "source_b": candidate.source,
                    "image_b": candidate.original,
                    "sha256_b": candidate.image_hash,
                }
            )
        if value not in by_visual_hash:
            tree.add(value)
        by_visual_hash[value].append(candidate)
    return rows


def write_near_duplicate_report(output: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "hamming_distance", "source_a", "image_a", "sha256_a",
        "source_b", "image_b", "sha256_b",
    ]
    with (output / "near_duplicate_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assign_splits(candidates: list[Candidate]) -> None:
    group_splits: dict[str, str] = {}
    for candidate in candidates:
        if candidate.group in group_splits:
            candidate.split = group_splits[candidate.group]
            continue
        if candidate.source_split == "test":
            split = "test"
        elif candidate.source == "pathik" and candidate.source_split == "train":
            split = "val" if stable_fraction(candidate.group) < 0.10 else "train"
        elif candidate.source_split in {"train", "val"}:
            split = candidate.source_split
        else:
            value = stable_fraction(candidate.group)
            split = "test" if value < 0.10 else ("val" if value < 0.20 else "train")
        group_splits[candidate.group] = split
        candidate.split = split


def output_path(output: Path, candidate: Candidate) -> Path:
    extension = FORMAT_EXTENSIONS[candidate.image_format]
    return (
        output
        / "images"
        / str(candidate.split)
        / candidate.image_hash[:2]
        / f"{candidate.source}_{candidate.image_hash[:20]}{extension}"
    )


def label_path(output: Path, candidate: Candidate) -> Path:
    return (
        output
        / "labels"
        / str(candidate.split)
        / candidate.image_hash[:2]
        / f"{candidate.source}_{candidate.image_hash[:20]}.txt"
    )


def materialize_local(candidate: Candidate, destination: Path) -> None:
    assert candidate.image_path is not None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if candidate.image_format == "GIF":
        with Image.open(candidate.image_path) as image:
            image.seek(0)
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        return
    if candidate.image_format == "JPEG":
        data = candidate.image_path.read_bytes()
        if not data.endswith(b"\xff\xd9"):
            destination.write_bytes(data + b"\xff\xd9")
            return
    # Training libraries may repair image files in place, so outputs must not
    # share inodes with immutable source data.
    shutil.copy2(candidate.image_path, destination)


def materialize_pathik(candidates: list[Candidate], output: Path) -> None:
    by_file: dict[Path, dict[int, Candidate]] = defaultdict(dict)
    for candidate in candidates:
        if candidate.parquet_path is not None and candidate.parquet_row is not None:
            by_file[candidate.parquet_path][candidate.parquet_row] = candidate
    for parquet_path, wanted in sorted(by_file.items()):
        parquet_file = pq.ParquetFile(parquet_path)
        row_index = 0
        progress = tqdm(total=len(wanted), desc=f"Extract {parquet_path.name}", unit="image")
        for batch in parquet_file.iter_batches(columns=["image"], batch_size=128, use_threads=False):
            for record in batch.column(0).to_pylist():
                candidate = wanted.get(row_index)
                if candidate is not None:
                    destination = output_path(output, candidate)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(record["bytes"])
                    progress.update(1)
                row_index += 1
        progress.close()


def write_label(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for x, y, w, h in boxes)
    path.write_text(text, encoding="utf-8")


def write_preview(output: Path, per_source: int = 4) -> None:
    with (output / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    source_counts: Counter[str] = Counter()
    for row in rows:
        if row["split"] != "train" or source_counts[row["source"]] >= per_source:
            continue
        selected.append(row)
        source_counts[row["source"]] += 1
    if not selected:
        return

    tile_width, tile_height, header_height = 320, 260, 20
    columns = 4
    rows_count = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image_path = output / row["output_image"]
        image_relative = image_path.relative_to(output / "images")
        annotation_path = (output / "labels" / image_relative).with_suffix(".txt")
        with Image.open(image_path) as source_image:
            source_image = ImageOps.exif_transpose(source_image).convert("RGB")
            source_width, source_height = source_image.size
            scale = min(tile_width / source_width, (tile_height - header_height) / source_height)
            resized = source_image.resize(
                (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
            )
        x_offset = (index % columns) * tile_width + (tile_width - resized.width) // 2
        y_offset = (index // columns) * tile_height + header_height
        sheet.paste(resized, (x_offset, y_offset))
        draw.text(
            ((index % columns) * tile_width + 4, (index // columns) * tile_height + 4),
            f"{row['source']} | {row['split']}",
            fill="black",
        )
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            _, x, y, width, height = line.split()
            x, y, width, height = map(float, (x, y, width, height))
            x1 = x_offset + (x - width / 2) * resized.width
            y1 = y_offset + (y - height / 2) * resized.height
            x2 = x_offset + (x + width / 2) * resized.width
            y2 = y_offset + (y + height / 2) * resized.height
            draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=2)
    sheet.save(output / "annotation_preview.jpg", quality=92)


def write_outputs(candidates: list[Candidate], output: Path, audit: Audit) -> None:
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    locals_only = [candidate for candidate in candidates if candidate.image_path is not None]
    for candidate in tqdm(locals_only, desc="Link local images", unit="image"):
        destination = output_path(output, candidate)
        materialize_local(candidate, destination)
    materialize_pathik(candidates, output)

    manifest_fields = [
        "output_image", "split", "source", "original", "sha256", "visual_hash", "group", "source_split",
        "width", "height", "boxes", "negative_kind", "removed_classes",
    ]
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for candidate in sorted(candidates, key=lambda item: (str(item.split), item.source, item.image_hash)):
            image_destination = output_path(output, candidate)
            label_destination = label_path(output, candidate)
            write_label(label_destination, candidate.boxes)
            audit.accepted[f"{candidate.source}:{candidate.split}"] += 1
            writer.writerow(
                {
                    "output_image": str(image_destination.relative_to(output)),
                    "split": candidate.split,
                    "source": candidate.source,
                    "original": candidate.original,
                    "sha256": candidate.image_hash,
                    "visual_hash": candidate.visual_hash,
                    "group": candidate.group,
                    "source_split": candidate.source_split or "",
                    "width": candidate.width,
                    "height": candidate.height,
                    "boxes": len(candidate.boxes),
                    "negative_kind": candidate.negative_kind,
                    "removed_classes": json.dumps(candidate.removed_classes, sort_keys=True),
                }
            )


def final_validation(candidates: list[Candidate], output: Path) -> dict[str, object]:
    expected = Counter(candidate.split for candidate in candidates)
    objects = Counter()
    negatives = Counter()
    missing = []
    for candidate in tqdm(candidates, desc="Final validation", unit="image"):
        image_path = output_path(output, candidate)
        annotation_path = label_path(output, candidate)
        if not image_path.is_file() or not annotation_path.is_file():
            missing.append(str(image_path))
            continue
        objects[str(candidate.split)] += len(candidate.boxes)
        negatives[str(candidate.split)] += int(not candidate.boxes)
    actual = {
        split: sum(1 for path in (output / "images" / split).rglob("*") if path.is_file())
        for split in SPLITS
    }
    if missing or any(actual[split] != expected[split] for split in SPLITS):
        raise RuntimeError(f"final validation failed: missing={len(missing)} expected={expected} actual={actual}")
    return {
        "images": actual,
        "objects": dict(objects),
        "negative_images": dict(negatives),
        "total_images": sum(actual.values()),
        "total_objects": sum(objects.values()),
    }


def write_dataset_card(output: Path, summary: dict[str, object], audit: Audit) -> None:
    images = summary["images"]
    objects = summary["objects"]
    negatives = summary["negative_images"]
    assert isinstance(images, dict) and isinstance(objects, dict) and isinstance(negatives, dict)
    source_counts = Counter()
    for key, count in audit.accepted.items():
        source_counts[key.split(":", 1)[0]] += count
    source_lines = "\n".join(f"| `{source}` | {count:,} |" for source, count in sorted(source_counts.items()))
    text = f"""---
pretty_name: Merged Single-Class Drone Detection Dataset
task_categories:
  - object-detection
tags:
  - image
  - object-detection
  - yolo
  - drone-detection
license: other
viewer: false
---

# Merged Single-Class Drone Detection Dataset

Cleaned single-class YOLO data. Every retained image has at least one class `0`
(`drone`) box. Images without a drone box are excluded.

> This dataset combines sources with different or unresolved redistribution
> terms. Keep the Hub repository private unless every retained source has been
> cleared for public redistribution.

## Summary

| Split | Images | Objects | Negative images |
| --- | ---: | ---: | ---: |
| Train | {images.get('train', 0):,} | {objects.get('train', 0):,} | {negatives.get('train', 0):,} |
| Validation | {images.get('val', 0):,} | {objects.get('val', 0):,} | {negatives.get('val', 0):,} |
| Test | {images.get('test', 0):,} | {objects.get('test', 0):,} | {negatives.get('test', 0):,} |
| **Total** | **{summary['total_images']:,}** | **{summary['total_objects']:,}** | **{sum(negatives.values()):,}** |

| Source | Retained images |
| --- | ---: |
{source_lines}

## Layout

`dataset.yaml` is directly usable by Ultralytics. Images and matching labels
are under `images/<split>/<hash-prefix>/` and `labels/<split>/<hash-prefix>/`.
`manifest.csv` records provenance and hashes, while `audit.json` records all
repairs, exclusions, exact duplicates, and counts.

Exact duplicates are removed using SHA-256. `near_duplicate_candidates.csv`
lists high-confidence cross-source perceptual-hash matches for manual review;
these are not automatically removed because adjacent video frames can be
legitimate distinct training samples. Images without a drone box are excluded,
as are images containing a box smaller than the configured native-pixel limit.

`annotation_preview.jpg` is a quick contact sheet. For interactive review run:

```bash
python scripts/view_yolo_dataset.py --dataset datasets/drone_merged
```

MIDGARD pixel boxes and GEM Label Studio video boxes are converted to normalized
YOLO coordinates. GEM interpolation is gated by its manually exported
`drone_exist` value. Sources without usable bounding boxes are excluded.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.antiuav_stride < 1:
        raise SystemExit("--antiuav-stride must be at least 1")
    if not 0 <= args.near_duplicate_distance <= 512:
        raise SystemExit("--near-duplicate-distance must be in [0, 512]")
    if args.min_box_pixels < 1:
        raise SystemExit("--min-box-pixels must be at least 1")
    if not args.datasets_root.is_dir():
        raise SystemExit(f"datasets root not found: {args.datasets_root}")
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output}; pass --overwrite to rebuild")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    audit = Audit()
    candidates: list[Candidate] = []
    ingest_antiuav(args.datasets_root, args.antiuav_stride, candidates, audit)
    ingest_kaggle1_xml(args.datasets_root, candidates, audit)
    ingest_flat_yolo(
        args.datasets_root / "Drone Dataset Kaggle 1" / "drone_dataset_yolo" / "dataset_txt",
        "kaggle1_yolo",
        candidates,
        audit,
    )
    ingest_kaggle2(args.datasets_root, candidates, audit)
    ingest_flat_yolo(
        args.datasets_root / "Drone Dataset Kaggle 3" / "Database1" / "Database1",
        "kaggle3",
        candidates,
        audit,
    )
    ingest_rigoleto(args.datasets_root, candidates, audit)
    ingest_midgard(args.datasets_root, candidates, audit)
    ingest_gem(args.datasets_root, candidates, audit)
    ingest_gem_new(args.datasets_root, candidates, audit)
    ingest_pathik(args.datasets_root, candidates, audit)

    audit.skipped["drone_bird:incomplete_download"] = 1
    audit.skipped["classification_dataset:no_detection_boxes"] = 1
    audit.skipped["foggy_img:no_detection_boxes"] = 1
    audit.skipped["drone_tracking:point_annotations_and_archived_frames"] = 1
    candidates = filter_detection_candidates(candidates, args.min_box_pixels, audit)
    candidates = deduplicate(candidates, audit)
    near_duplicates = find_near_duplicates(candidates, args.near_duplicate_distance, audit)
    assign_splits(candidates)
    write_outputs(candidates, args.output, audit)
    write_near_duplicate_report(args.output, near_duplicates)
    write_preview(args.output)
    summary = final_validation(candidates, args.output)

    dataset_config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "drone"},
    }
    with (args.output / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_config, handle, sort_keys=False)
    report = {
        "parameters": {
            "antiuav_stride": args.antiuav_stride,
            "min_box_pixels": args.min_box_pixels,
        },
        "summary": summary,
        "candidates": dict(sorted(audit.candidates.items())),
        "accepted": dict(sorted(audit.accepted.items())),
        "skipped": dict(sorted(audit.skipped.items())),
        "removed_objects": dict(sorted(audit.removed_objects.items())),
        "repaired_boxes": dict(sorted(audit.repaired_boxes.items())),
        "duplicates": dict(sorted(audit.duplicates.items())),
        "near_duplicate_candidates": dict(sorted(audit.near_duplicates.items())),
    }
    with (args.output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_dataset_card(args.output, summary, audit)
    print(json.dumps(summary, indent=2))
    print(f"Dataset: {args.output / 'dataset.yaml'}")
    print(f"Audit:   {args.output / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
