#!/usr/bin/env python3
"""Convert extracted annotation archives to YOLO and append them to the dataset.

Each archive in state ``extracted`` is scanned for sequence folders, meaning a
folder with a ``frames/`` subfolder plus one annotation JSON. Two formats are
supported:

- annotation editor JSON: ``{"exist": [...], "gt_rect": [[x, y, w, h], ...]}``
  in pixels, one entry per frame;
- Label Studio video export: a task list with ``videorectangle`` tracks of
  ``Drone`` keyframes in percent, interpolated between keyframes.

``verif/`` folders (annotator check renders with boxes drawn in) are never
read. Frames without a drone box and frames with an undersized box are not
added, matching the dataset build rules. Exact duplicates of images already in
the dataset are skipped, so re-uploaded sequences add nothing.

Splits are assigned per recording day, parsed from the sequence name, so
near-identical footage from one session never lands in both train and test.
A day already in the dataset keeps its split; each new day (largest first)
goes to the split furthest below its 80/10/10 share of all Drive data.
Existing images never move.

New rows are appended to ``manifest.csv`` after each archive's files are
written, and the archive moves to state ``ingested`` in the ledger.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from drone_ai.data.build_merged import (
    IMAGE_EXTENSIONS,
    SPLITS,
    Audit,
    Candidate,
    clip_yolo_box,
    filter_detection_candidates,
    image_info,
    label_path,
    materialize_local,
    output_path,
    write_label,
)
from drone_ai.data.ledger import INCOMING_ROOT, Ledger, now


SOURCE = "drive"
SPLIT_TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}
log = logging.getLogger(__name__)


class SequenceError(Exception):
    pass


@dataclass
class ArchiveResult:
    record: dict[str, object]
    sequences: list[dict[str, object]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    totals: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incoming", type=Path, default=INCOMING_ROOT)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/drone_dataset"))
    parser.add_argument("--min-box-pixels", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8, help="Processes used to hash images.")
    parser.add_argument("--retry-failed", action="store_true", help="Also retry archives in state ingest_failed.")
    parser.add_argument("--dry-run", action="store_true", help="Convert and report, but write nothing.")
    return parser.parse_args()


def find_sequences(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.rglob("frames")
        if path.is_dir() and "verif" not in path.relative_to(root).parts
    )


def recording_day(name: str) -> str | None:
    """Recording date from names like 2026_10_09_11_44_26 or 18_09_2026_10_19_04."""
    if match := re.match(r"(20\d{2})_(\d{1,2})_(\d{1,2})(?:_|$)", name):
        year, month, day = match.groups()
    elif match := re.match(r"(\d{1,2})_(\d{1,2})_(20\d{2})(?:_|$)", name):
        day, month, year = match.groups()
    else:
        return None
    return f"{year}-{int(month):02d}-{int(day):02d}"


def annotation_file(sequence: Path) -> Path:
    preferred = sequence / f"{sequence.name}.json"
    if preferred.is_file():
        return preferred
    found = sorted(sequence.glob("*.json"))
    if len(found) != 1:
        raise SequenceError(f"expected one annotation JSON, found {len(found)}")
    return found[0]


def editor_boxes(data: dict, frame_count: int) -> dict[int, list[tuple[float, float, float, float, str]]]:
    """Boxes per frame index from editor JSON, in pixels (x, y, w, h)."""
    exists, rects = data["exist"], data["gt_rect"]
    if len(exists) != frame_count or len(rects) != frame_count:
        raise SequenceError(f"annotation length {len(exists)}/{len(rects)} does not match {frame_count} frames")
    boxes = {}
    for index, (exist, rect) in enumerate(zip(exists, rects)):
        if exist and isinstance(rect, list) and len(rect) == 4:
            boxes[index] = [(*map(float, rect), "px")]
    return boxes


def label_studio_boxes(
    tasks: list, frame_count: int, stats: Counter[str]
) -> dict[int, list[tuple[float, float, float, float, str]]]:
    """Boxes per frame index from a Label Studio video export, in percent (x, y, w, h).

    An enabled keyframe's box is interpolated linearly up to the next keyframe;
    the last keyframe covers only its own frame. A disabled keyframe ends the
    track and gets no box itself. This reproduces the annotators' verif/ renders
    frame for frame.
    """
    if len(tasks) != 1:
        raise SequenceError(f"expected one Label Studio task, found {len(tasks)}")
    annotations = [item for item in tasks[0].get("annotations", []) if not item.get("was_cancelled")]
    if not annotations:
        raise SequenceError("no completed annotation")
    if len(annotations) > 1:
        stats["multiple_annotations_used_latest"] += 1
        log.warning("Task has %d annotations; using the most recent one", len(annotations))
    annotation = max(annotations, key=lambda item: item.get("id", 0))

    boxes: dict[int, list[tuple[float, float, float, float, str]]] = {}
    for result in annotation.get("result", []):
        value = result.get("value", {})
        if result.get("type") != "videorectangle" or "Drone" not in (value.get("labels") or []):
            stats["ignored_results"] += 1
            continue
        declared = value.get("framesCount")
        if declared is not None and int(declared) != frame_count:
            raise SequenceError(f"framesCount {declared} does not match {frame_count} frames")
        keyframes = sorted(value.get("sequence", []), key=lambda point: int(point["frame"]))
        if not keyframes:
            stats["empty_tracks"] += 1
            continue
        stats["tracks"] += 1
        for current, following in zip(keyframes, [*keyframes[1:], None]):
            if float(current.get("rotation", 0) or 0) != 0:
                stats["rotated_keyframes_skipped"] += 1
                continue
            start = int(current["frame"])
            if not current.get("enabled", True):
                continue
            if following is None:
                end = start
            else:
                end = int(following["frame"]) - 1
            for frame in range(start, min(end, frame_count) + 1):
                if following is None or frame == start:
                    values = tuple(float(current[key]) for key in ("x", "y", "width", "height"))
                else:
                    ratio = (frame - start) / (int(following["frame"]) - start)
                    values = tuple(
                        float(current[key]) + (float(following[key]) - float(current[key])) * ratio
                        for key in ("x", "y", "width", "height")
                    )
                boxes.setdefault(frame - 1, []).append((*values, "pct"))
    return boxes


def to_yolo(
    box: tuple[float, float, float, float, str], width: int, height: int
) -> tuple[tuple[float, float, float, float] | None, bool]:
    x, y, box_width, box_height, unit = box
    scale_x, scale_y = (width, height) if unit == "px" else (100.0, 100.0)
    return clip_yolo_box(
        ((x + box_width / 2) / scale_x, (y + box_height / 2) / scale_y, box_width / scale_x, box_height / scale_y),
        width,
        height,
    )


def sequence_candidates(
    sequence: Path, group: str, pool: ProcessPoolExecutor, audit: Audit, stats: Counter[str]
) -> tuple[str, list[Candidate]]:
    frames = sorted(path for path in (sequence / "frames").iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not frames:
        raise SequenceError("no frames")
    data = json.loads(annotation_file(sequence).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "exist" in data and "gt_rect" in data:
        kind, boxes_by_frame = "editor", editor_boxes(data, len(frames))
    elif isinstance(data, list):
        kind, boxes_by_frame = "label_studio", label_studio_boxes(data, len(frames), stats)
    else:
        raise SequenceError("unrecognized annotation format")

    stats["frames"] += len(frames)
    stats["frames_without_box"] += len(frames) - len(boxes_by_frame)
    annotated = [(index, frames[index]) for index in sorted(boxes_by_frame)]
    candidates = []
    for (index, frame), info in zip(annotated, pool.map(image_info, [frame for _, frame in annotated], chunksize=64)):
        image_hash, image_format, width, height, visual_hash = info
        yolo_boxes = []
        for box in boxes_by_frame[index]:
            clipped, repaired = to_yolo(box, width, height)
            audit.repaired_boxes[SOURCE] += int(repaired)
            if clipped is None:
                audit.removed_objects[f"{SOURCE}:degenerate_box"] += 1
            else:
                yolo_boxes.append(clipped)
        candidates.append(
            Candidate(
                source=SOURCE,
                original=str(frame),
                group=group,
                source_split=None,
                boxes=yolo_boxes,
                image_hash=image_hash,
                visual_hash=visual_hash,
                image_format=image_format,
                width=width,
                height=height,
                image_path=frame,
            )
        )
    return kind, candidates


def load_manifest(dataset: Path) -> tuple[list[str], set[str], dict[str, str], Counter[str]]:
    """Return manifest fields, known hashes, split per Drive group, and Drive images per split."""
    group_splits: dict[str, str] = {}
    drive_counts: Counter[str] = Counter()
    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        hashes = set()
        for row in reader:
            hashes.add(row["sha256"])
            if row["source"] == SOURCE:
                group_splits[row["group"]] = row["split"]
                drive_counts[row["split"]] += 1
        return list(reader.fieldnames or []), hashes, group_splits, drive_counts


def convert_archive(
    record: dict[str, object], args: argparse.Namespace, known_hashes: set[str], pool: ProcessPoolExecutor
) -> ArchiveResult:
    root = Path(str(record["extracted_dir"]))
    archive_name = str(root.relative_to(args.incoming / "extracted"))
    sequences = find_sequences(root)
    if not sequences:
        raise SequenceError(f"no sequence folders with frames/ under {root}")

    result = ArchiveResult(record)
    for sequence in sequences:
        name = str(sequence.relative_to(root))
        day = recording_day(sequence.name)
        group = f"{SOURCE}:{day}" if day else f"{SOURCE}:{archive_name}/{name}"
        audit, stats = Audit(), Counter()
        try:
            kind, candidates = sequence_candidates(sequence, group, pool, audit, stats)
        except (SequenceError, KeyError, ValueError, json.JSONDecodeError) as error:
            log.error("  %s: skipped, %s", name, error)
            result.sequences.append({"name": name, "error": str(error)})
            result.totals["sequences_failed"] += 1
            continue
        retained = filter_detection_candidates(candidates, args.min_box_pixels, audit)
        fresh = []
        for candidate in retained:
            if candidate.image_hash in known_hashes:
                stats["duplicates"] += 1
                continue
            known_hashes.add(candidate.image_hash)
            fresh.append(candidate)
        if day is None and fresh:
            log.warning("  %s: no recording date in the name; it forms its own split group", name)
        entry = {
            "name": name,
            "format": kind,
            "group": group,
            "frames": stats["frames"],
            "frames_without_box": stats["frames_without_box"],
            "undersized_box_frames": audit.skipped[f"{SOURCE}:undersized_box_image"],
            "duplicates_skipped": stats["duplicates"],
            "added": len(fresh),
            "boxes": sum(len(candidate.boxes) for candidate in fresh),
            **{key: stats[key] for key in ("tracks", "empty_tracks", "ignored_results", "rotated_keyframes_skipped") if stats[key]},
        }
        result.sequences.append(entry)
        result.candidates.extend(fresh)
        for key in ("frames", "frames_without_box", "duplicates_skipped", "undersized_box_frames", "added", "boxes"):
            result.totals[key] += entry[key]
    return result


def assign_group_splits(
    candidates: list[Candidate], group_splits: dict[str, str], drive_counts: Counter[str]
) -> dict[str, str]:
    """Give each new group the split furthest below its target share; known groups keep theirs."""
    counts = Counter(drive_counts)
    sizes: Counter[str] = Counter(candidate.group for candidate in candidates)
    for group, size in sizes.items():
        if group in group_splits:
            counts[group_splits[group]] += size
    total = sum(counts.values()) + sum(size for group, size in sizes.items() if group not in group_splits)
    assigned = {}
    for group, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        if group in group_splits:
            continue
        split = max(SPLITS, key=lambda name: SPLIT_TARGETS[name] * total - counts[name])
        assigned[group] = split
        counts[split] += size
    for candidate in candidates:
        candidate.split = group_splits.get(candidate.group) or assigned[candidate.group]
    return assigned


def manifest_row(dataset: Path, candidate: Candidate) -> dict[str, object]:
    return {
        "output_image": str(output_path(dataset, candidate).relative_to(dataset)),
        "split": candidate.split,
        "source": candidate.source,
        "original": candidate.original,
        "sha256": candidate.image_hash,
        "visual_hash": candidate.visual_hash,
        "group": candidate.group,
        "source_split": "",
        "width": candidate.width,
        "height": candidate.height,
        "boxes": len(candidate.boxes),
        "negative_kind": "",
        "removed_classes": "{}",
    }


def write_archive(result: ArchiveResult, dataset: Path, fields: list[str]) -> None:
    for candidate in result.candidates:
        materialize_local(candidate, output_path(dataset, candidate))
        write_label(label_path(dataset, candidate), candidate.boxes)
    with (dataset / "manifest.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for candidate in sorted(result.candidates, key=lambda item: (str(item.split), item.image_hash)):
            writer.writerow(manifest_row(dataset, candidate))


def record_in_audit(dataset: Path, entries: list[dict[str, object]]) -> None:
    path = dataset / "audit.json"
    audit = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    audit.setdefault("incoming_ingests", []).extend(entries)
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not (args.dataset / "manifest.csv").is_file():
        raise SystemExit(f"Dataset manifest not found: {args.dataset / 'manifest.csv'}")
    fields, known_hashes, group_splits, drive_counts = load_manifest(args.dataset)
    log.info("Dataset %s has %d images (%d from Drive)", args.dataset, len(known_hashes), sum(drive_counts.values()))

    with Ledger(args.incoming) as ledger:
        statuses = ("extracted", "ingesting", "ingest_failed") if args.retry_failed else ("extracted", "ingesting")
        pending = sorted(
            (record for record in ledger.by_status(*statuses) if record.get("extracted_dir")),
            key=lambda record: str(record["drive_path"]),
        )
        log.info("Archives to ingest: %d%s", len(pending), " (dry run)" if args.dry_run else "")

        results, failures = [], 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, record in enumerate(pending, start=1):
                log.info("[%d/%d] Converting %s", index, len(pending), record["drive_path"])
                try:
                    results.append(convert_archive(record, args, known_hashes, pool))
                except (SequenceError, OSError) as error:
                    failures += 1
                    log.error("Ingest failed for %s: %s", record["drive_path"], error)
                    if not args.dry_run:
                        ledger.set_status(record, "ingest_failed", str(error), error=str(error))

        assigned = assign_group_splits(
            [candidate for result in results for candidate in result.candidates], group_splits, drive_counts
        )
        for group, split in sorted(assigned.items()):
            log.info("New split group %s -> %s", group, split)

        audit_entries, added = [], Counter()
        for result in results:
            record = result.record
            splits = Counter(str(candidate.split) for candidate in result.candidates)
            for entry in result.sequences:
                if "group" in entry and entry["added"]:
                    entry["split"] = group_splits.get(str(entry["group"])) or assigned[str(entry["group"])]
                    log.info(
                        "  %s [%s]: %d frames, %d added to %s, %d without box, %d undersized, %d duplicates",
                        entry["name"], entry["format"], entry["frames"], entry["added"], entry["split"],
                        entry["frames_without_box"], entry["undersized_box_frames"], entry["duplicates_skipped"],
                    )
                elif "group" in entry:
                    log.info("  %s [%s]: %d frames, nothing new", entry["name"], entry["format"], entry["frames"])
            log.info("%s: %d image(s) added %s", record["drive_path"], result.totals["added"], dict(splits))
            if result.totals["sequences_failed"]:
                failures += 1
            added.update(splits)
            if args.dry_run:
                continue
            ledger.set_status(record, "ingesting")
            write_archive(result, args.dataset, fields)
            report = {"sequences": result.sequences, "splits": dict(splits), "totals": dict(result.totals)}
            ledger.set_status(
                record, "ingested", f"{result.totals['added']} image(s) added", ingest=report, ingested_at=now(), error=None
            )
            audit_entries.append({"archive": record["drive_path"], "drive_id": record["drive_id"], "md5": record["md5"], "at": now(), **report})

        if audit_entries:
            record_in_audit(args.dataset, audit_entries)
        log.info("Ingest complete: %d image(s) added %s, %d archive(s) with problems", sum(added.values()), dict(added), failures)
        if failures:
            raise SystemExit(f"{failures} archive(s) had problems; see the log and incoming-status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
