#!/usr/bin/env python3
"""Move a completed legacy experiment into the canonical timestamped layout."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from experiment_paths import TIMESTAMP_FORMAT, timestamped_experiment_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize an old output directory after its process has exited."
    )
    parser.add_argument("kind", choices=("training", "artifact"))
    parser.add_argument("source", type=Path, help="Completed directory to move.")
    parser.add_argument("--experiment", required=True, help="Short experiment name.")
    parser.add_argument(
        "--started-at",
        help="Optional start time in DD_MM_YYYY_HH_MM_SS format; defaults to creation time.",
    )
    parser.add_argument(
        "--attach",
        type=Path,
        action="append",
        default=[],
        help="File to move into an artifact's model/ directory (repeatable).",
    )
    return parser.parse_args()


def source_creation_time(path: Path) -> datetime:
    stat = path.stat()
    created = getattr(stat, "st_birthtime", stat.st_ctime)
    return datetime.fromtimestamp(created)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")
    if args.kind == "training" and args.attach:
        raise SystemExit("--attach is only valid for artifacts")

    started_at = (
        datetime.strptime(args.started_at, TIMESTAMP_FORMAT)
        if args.started_at
        else source_creation_time(source)
    )
    root = Path("runs" if args.kind == "training" else "artifacts").resolve()
    destination = root / timestamped_experiment_name(args.experiment, started_at)
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}")

    missing = [path for path in args.attach if not path.is_file()]
    if missing:
        raise SystemExit(f"Attachment not found: {missing[0]}")

    root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))

    attached: list[str] = []
    if args.attach:
        model_dir = destination / "model"
        model_dir.mkdir()
        for path in args.attach:
            target = model_dir / path.name
            shutil.move(str(path), str(target))
            attached.append(str(target))

    record = {
        "finalized_at": datetime.now().isoformat(timespec="seconds"),
        "original_path": str(source),
        "destination": str(destination),
        "attached_files": attached,
    }
    (destination / "finalization.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
