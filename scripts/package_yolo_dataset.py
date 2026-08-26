#!/usr/bin/env python3
"""Package a YOLO directory into upload-friendly TAR shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm


METADATA_FILES = (
    "dataset.yaml",
    "manifest.csv",
    "audit.json",
    "near_duplicate_candidates.csv",
    "annotation_preview.jpg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/drone_merged"))
    parser.add_argument("--output", type=Path, default=Path("datasets/drone_merged_hf"))
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=1.0,
        help="Approximate maximum payload per TAR shard (default: 1 GiB).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def label_path(root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(root / "images")
    return (root / "labels" / relative).with_suffix(".txt")


def write_hub_readme(output: Path, audit: dict, shard_count: int) -> None:
    summary = audit["summary"]
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

Private single-class YOLO dataset packaged into {shard_count} TAR shards for
reliable transfer. It contains {summary['total_images']:,} images and
{summary['total_objects']:,} class `0` (`drone`) bounding boxes.

The TAR members preserve the original `images/<split>/...` and
`labels/<split>/...` paths. Download and extract them with the repository tool:

```bash
python scripts/extract_yolo_dataset_package.py \\
  --package datasets/drone_merged_package \\
  --output datasets/drone_merged \\
  --overwrite
```

See `shard_manifest.json` for per-shard sizes, checksums, image counts, and box
counts. `manifest.csv` records sample provenance and `audit.json` records build
filtering and validation.

This dataset combines sources with different or unresolved redistribution
terms. Keep this repository private unless every retained source has been
cleared for public redistribution.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.dataset.resolve()
    output = args.output.resolve()
    if args.shard_size_gb <= 0:
        raise SystemExit("--shard-size-gb must be positive")
    for name in ("dataset.yaml", "manifest.csv", "audit.json", "README.md"):
        if not (root / name).is_file():
            raise SystemExit(f"missing required dataset file: {root / name}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {output}; pass --overwrite to rebuild")
        shutil.rmtree(output)
    shards_dir = output / "shards"
    shards_dir.mkdir(parents=True)

    for name in METADATA_FILES:
        source = root / name
        if source.is_file():
            shutil.copy2(source, output / name)
    shutil.copy2(root / "README.md", output / "dataset_card.md")

    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_split: dict[str, list[dict[str, str]]] = {split: [] for split in ("train", "val", "test")}
    for row in rows:
        by_split[row["split"]].append(row)

    target_size = round(args.shard_size_gb * 1024**3)
    shard_records = []
    for split, split_rows in by_split.items():
        shard_index = 0
        archive = None
        archive_path = None
        payload_size = 0
        image_count = 0
        box_count = 0

        def close_shard() -> None:
            nonlocal archive, archive_path, payload_size, image_count, box_count
            if archive is None or archive_path is None:
                return
            archive.close()
            shard_records.append(
                {
                    "path": str(archive_path.relative_to(output)),
                    "size": archive_path.stat().st_size,
                    "sha256": sha256(archive_path),
                    "split": split,
                    "images": image_count,
                    "boxes": box_count,
                }
            )
            archive = None
            archive_path = None
            payload_size = 0
            image_count = 0
            box_count = 0

        for row in tqdm(split_rows, desc=f"Package {split}", unit="image"):
            image = root / row["output_image"]
            label = label_path(root, image)
            if not image.is_file() or not label.is_file():
                raise RuntimeError(f"missing image/label pair: {image} | {label}")
            pair_size = image.stat().st_size + label.stat().st_size
            if archive is not None and payload_size + pair_size > target_size:
                close_shard()
                shard_index += 1
            if archive is None:
                archive_path = shards_dir / f"{split}-{shard_index:05d}.tar"
                archive = tarfile.open(archive_path, mode="w")
            archive.add(image, arcname=str(image.relative_to(root)), recursive=False)
            archive.add(label, arcname=str(label.relative_to(root)), recursive=False)
            payload_size += pair_size
            image_count += 1
            box_count += int(row["boxes"])
        close_shard()

    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "yolo-tar-shards-v1",
        "dataset": audit["summary"],
        "shards": shard_records,
        "total_shards": len(shard_records),
        "total_bytes": sum(record["size"] for record in shard_records),
    }
    (output / "shard_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_hub_readme(output, audit, len(shard_records))
    split_counts = Counter(record["split"] for record in shard_records)
    print(
        json.dumps(
            {
                "shards": dict(split_counts),
                "total_shards": len(shard_records),
                "total_bytes": manifest["total_bytes"],
            },
            indent=2,
        )
    )
    print(f"Package: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
