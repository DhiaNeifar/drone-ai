#!/usr/bin/env python3
"""One entry point for dataset extraction, cleaning, validation, and Hub sync."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


COMMANDS = {
    "extract-frames": "extract_frames.py",
    "rebuild-recordings": "rebuild_recordings.py",
    "convert-antiuav": "convert_antiuav.py",
    "prepare-gem": "prepare_gem_new.py",
    "build-merged": "build_merged.py",
    "validate-yolo": "validate_yolo.py",
    "package-yolo": "package_yolo.py",
    "extract-package": "extract_package.py",
    "upload-hf": "upload_hf.py",
    "shard-yolo": "shard_yolo.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module_path = Path(__file__).resolve().parent / "drone_tools" / COMMANDS[args.command]
    sys.argv = [f"{Path(__file__).name} {args.command}", *args.args]
    runpy.run_path(str(module_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
