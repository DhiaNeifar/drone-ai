#!/usr/bin/env python3
"""Verify and extract a sharded YOLO dataset package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
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
    parser.add_argument("--package", type=Path, default=Path("datasets/drone_merged_package"))
    parser.add_argument("--output", type=Path, default=Path("datasets/drone_merged"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, output: Path) -> None:
    output_prefix = str(output.resolve()) + "/"
    for member in archive.getmembers():
        destination = (output / member.name).resolve()
        if not str(destination).startswith(output_prefix) or not member.isfile():
            raise RuntimeError(f"unsafe TAR member: {member.name}")
    archive.extractall(output)


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    output = args.output.resolve()
    manifest_path = package / "shard_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing package manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "yolo-tar-shards-v1":
        raise SystemExit(f"unsupported package format: {manifest.get('format')}")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for record in tqdm(manifest["shards"], desc="Extract shards", unit="shard"):
        shard = package / record["path"]
        if not shard.is_file():
            raise RuntimeError(f"missing shard: {shard}")
        if not args.skip_checksums and sha256(shard) != record["sha256"]:
            raise RuntimeError(f"checksum mismatch: {shard}")
        with tarfile.open(shard, mode="r") as archive:
            safe_extract(archive, output)

    for name in METADATA_FILES:
        source = package / name
        if source.is_file():
            shutil.copy2(source, output / name)
    dataset_card = package / "dataset_card.md"
    if dataset_card.is_file():
        shutil.copy2(dataset_card, output / "README.md")

    expected = manifest["dataset"]["images"]
    actual_images = {
        split: sum(1 for path in (output / "images" / split).rglob("*") if path.is_file())
        for split in ("train", "val", "test")
    }
    actual_labels = {
        split: sum(1 for path in (output / "labels" / split).rglob("*.txt"))
        for split in ("train", "val", "test")
    }
    if actual_images != expected or actual_labels != expected:
        raise RuntimeError(
            f"extraction validation failed: expected={expected}, "
            f"images={actual_images}, labels={actual_labels}"
        )
    print(json.dumps({"images": actual_images, "labels": actual_labels}, indent=2))
    print(f"YOLO dataset: {output / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
