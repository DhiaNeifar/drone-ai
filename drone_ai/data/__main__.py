"""Dataset pipeline: fetch, extract, clean, validate, package, and Hub sync."""

from __future__ import annotations

from drone_ai.cli import dispatch


COMMANDS = {
    "fetch-drive": "fetch_drive",
    "extract-archives": "extract_archives",
    "ingest-incoming": "ingest_incoming",
    "incoming-status": "ledger",
    "extract-frames": "extract_frames",
    "rebuild-recordings": "rebuild_recordings",
    "convert-antiuav": "convert_antiuav",
    "prepare-gem": "prepare_gem_new",
    "build-merged": "build_merged",
    "validate-yolo": "validate_yolo",
    "export-coco": "export_coco",
    "package-yolo": "package_yolo",
    "package-append": "append_package",
    "extract-package": "extract_package",
    "download-hf": "download_hf",
    "upload-hf": "upload_hf",
    "shard-yolo": "shard_yolo",
}


if __name__ == "__main__":
    raise SystemExit(dispatch("data", COMMANDS, __doc__))
