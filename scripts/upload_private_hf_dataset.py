#!/usr/bin/env python3
"""Create or update a private Hugging Face dataset repository."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Hugging Face repository ID, such as user/drone-dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/drone_merged"),
        help="Prepared dataset root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent upload workers (Hugging Face default when omitted).",
    )
    return parser.parse_args()


def validate_layout(root: Path) -> None:
    required = ("README.md", "dataset.yaml", "manifest.csv", "audit.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"Dataset is missing required files: {', '.join(missing)}")
    oversized = []
    for directory in (path for path in root.rglob("*") if path.is_dir()):
        if sum(1 for child in directory.iterdir() if child.is_file()) >= 10_000:
            oversized.append(directory)
    if oversized:
        raise SystemExit(
            "Dataset must be sharded before upload; oversized folders: "
            + ", ".join(str(path) for path in oversized)
        )


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    validate_layout(root)

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing dependency: install huggingface_hub or run this script in the "
            "drone-yolo conda environment."
        ) from error

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    repository = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
    if not repository.private:
        raise SystemExit(f"Refusing upload because {args.repo_id} is not private")

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(root),
        private=True,
        ignore_patterns=[
            "*.cache",
            "**/*.cache",
            ".cache/**",
            ".DS_Store",
            "**/.DS_Store",
        ],
        num_workers=args.workers,
        print_report=True,
    )
    print(f"Uploaded to private dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
