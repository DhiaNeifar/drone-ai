#!/usr/bin/env python3
"""Download the packaged dataset shards from the private Hugging Face repository."""

from __future__ import annotations

import argparse
from pathlib import Path


PACKAGE_PATTERNS = [
    "shards/*.tar",
    "shard_manifest.json",
    "dataset.yaml",
    "manifest.csv",
    "audit.json",
    "near_duplicate_candidates.csv",
    "annotation_preview.jpg",
    "dataset_card.md",
    "README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="DhiaNeifar/Akkurad_Drone_Detection")
    parser.add_argument("--revision", help="Branch, tag, or commit (default: main).")
    parser.add_argument("--output", type=Path, default=Path("datasets/drone_dataset_hf"))
    return parser.parse_args()


def main() -> int:
    from huggingface_hub import snapshot_download

    args = parse_args()
    print(f"Downloading {args.repo_id} into {args.output}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.output),
        allow_patterns=PACKAGE_PATTERNS,
    )
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
