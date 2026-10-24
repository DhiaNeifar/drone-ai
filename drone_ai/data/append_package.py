#!/usr/bin/env python3
"""Append newly ingested images to an existing TAR-shard package without touching old shards.

The package's own ``manifest.csv`` lists every image already packaged. Rows of
the dataset manifest that are not in it are written to new shards, numbered
after the last shard of each split, and appended to ``shard_manifest.json``.
Existing shards stay byte-identical, so an upload only transfers the new ones.
Image and box counts in the package and in the dataset's ``audit.json`` are
recomputed from the manifest, which ``extract-package`` verifies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from drone_ai.data.package_yolo import METADATA_FILES, label_path, sha256, write_hub_readme


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/drone_dataset"))
    parser.add_argument("--package", type=Path, default=Path("datasets/drone_dataset_hf"))
    parser.add_argument("--shard-size-gb", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary(rows: list[dict[str, str]]) -> dict[str, object]:
    images = Counter(row["split"] for row in rows)
    objects: Counter[str] = Counter()
    negatives: Counter[str] = Counter()
    for row in rows:
        objects[row["split"]] += int(row["boxes"])
        negatives[row["split"]] += int(row["boxes"]) == 0
    return {
        "images": {split: images[split] for split in SPLITS},
        "objects": {split: objects[split] for split in SPLITS},
        "negative_images": {split: negatives[split] for split in SPLITS},
        "total_images": sum(images.values()),
        "total_objects": sum(objects.values()),
    }


def next_shard_index(records: list[dict[str, object]], split: str) -> int:
    indices = [
        int(match.group(1))
        for record in records
        if (match := re.fullmatch(rf"shards/{split}-(\d+)\.tar", str(record["path"])))
    ]
    return max(indices, default=-1) + 1


def main() -> int:
    args = parse_args()
    dataset, package = args.dataset.resolve(), args.package.resolve()
    shard_manifest_path = package / "shard_manifest.json"
    if not shard_manifest_path.is_file():
        raise SystemExit(f"No package at {package}; build one first with package-yolo")
    shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))

    rows = read_manifest(dataset / "manifest.csv")
    packaged = {row["output_image"] for row in read_manifest(package / "manifest.csv")}
    current = {row["output_image"] for row in rows}
    if missing := packaged - current:
        raise SystemExit(f"{len(missing)} packaged image(s) are no longer in the dataset; rebuild with package-yolo")
    new_rows = [row for row in rows if row["output_image"] not in packaged]
    by_split = {split: [row for row in new_rows if row["split"] == split] for split in SPLITS}
    print(f"Packaged: {len(packaged)} image(s); new: {len(new_rows)} " + str({k: len(v) for k, v in by_split.items()}))
    if not new_rows:
        print("Nothing to append.")
        return 0
    if args.dry_run:
        return 0

    target_size = round(args.shard_size_gb * 1024**3)
    new_records = []
    for split, split_rows in by_split.items():
        index = next_shard_index(shard_manifest["shards"], split)
        archive = archive_path = None
        payload = images = boxes = 0
        for row in tqdm(split_rows, desc=f"Append {split}", unit="image"):
            image = dataset / row["output_image"]
            label = label_path(dataset, image)
            if not image.is_file() or not label.is_file():
                raise RuntimeError(f"missing image/label pair: {image} | {label}")
            size = image.stat().st_size + label.stat().st_size
            if archive is not None and payload + size > target_size:
                archive.close()
                new_records.append({"path": str(archive_path.relative_to(package)), "split": split, "images": images, "boxes": boxes})
                archive, payload, images, boxes, index = None, 0, 0, 0, index + 1
            if archive is None:
                archive_path = package / "shards" / f"{split}-{index:05d}.tar"
                if archive_path.exists():
                    raise RuntimeError(f"refusing to overwrite existing shard: {archive_path}")
                archive = tarfile.open(archive_path, mode="w")
            archive.add(image, arcname=row["output_image"], recursive=False)
            archive.add(label, arcname=str(label.relative_to(dataset)), recursive=False)
            payload, images, boxes = payload + size, images + 1, boxes + int(row["boxes"])
        if archive is not None:
            archive.close()
            new_records.append({"path": str(archive_path.relative_to(package)), "split": split, "images": images, "boxes": boxes})

    for record in new_records:
        path = package / str(record["path"])
        record.update(size=path.stat().st_size, sha256=sha256(path))
        print(f"New shard {record['path']}: {record['images']} image(s), {record['size'] / 1e9:.2f} GB")

    totals = summary(rows)
    audit_path = dataset / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["summary"] = totals
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    shard_manifest["shards"].extend(new_records)
    shard_manifest["dataset"] = totals
    shard_manifest["total_shards"] = len(shard_manifest["shards"])
    shard_manifest["total_bytes"] = sum(int(record["size"]) for record in shard_manifest["shards"])
    for name in METADATA_FILES:
        if (dataset / name).is_file():
            shutil.copy2(dataset / name, package / name)
    temporary = shard_manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(shard_manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(shard_manifest_path)
    write_hub_readme(package, audit, shard_manifest["total_shards"])
    print(json.dumps({"dataset": totals, "total_shards": shard_manifest["total_shards"]}, indent=2))
    print(f"Package updated: {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
