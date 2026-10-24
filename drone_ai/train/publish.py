#!/usr/bin/env python3
"""Mark finished training runs for upload to Google Drive.

Every trainer calls ``mark_complete`` once training exits successfully. It
writes two files into the run folder:

- ``upload.txt``: the files to share (best and last weights, plus
  ``done.json``), one path per line, relative to the run folder.
- ``done.json``: model type, finish time, and best validation metrics. It is
  written last, so its presence means the run is complete.

``scripts/sync_models_to_drive.sh`` (driven by a systemd timer) uploads only
runs that have ``done.json``, and only the files listed in ``upload.txt``.

As a command, it marks runs that finished before this existed::

    python -m drone_ai.train mark-done runs/<run> [runs/<run> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


UPLOAD_PATTERNS = {
    "yolo": ["weights/best.pt", "weights/last.pt"],
    # D-FINE keeps best_stg1.pth and, once strong augmentation stops, best_stg2.pth; RT-DETRv2 keeps best.pth.
    "dfine": ["best*.pth", "last.pth"],
    "rtdetr": ["best*.pth", "last.pth"],
    "rfdetr": ["checkpoint_best_total.pth", "last*.pth"],
}
WEIGHT_SUFFIXES = {".pt", ".pth"}


def detect_model(run: Path) -> str:
    if (run / "args.yaml").is_file() or (run / "weights" / "best.pt").is_file():
        return "yolo"
    if (run / "checkpoint_best_total.pth").is_file():
        return "rfdetr"
    config = run / "config.yml"
    if config.is_file():
        text = config.read_text(encoding="utf-8")
        if "external/D-FINE" in text:
            return "dfine"
        if "external/RT-DETR" in text:
            return "rtdetr"
    raise SystemExit(f"Cannot tell which model trained {run}")


def best_metrics(run: Path, model: str) -> dict:
    """Best validation mAP and the epoch (1-based) it was reached."""
    if model == "yolo" and (run / "results.csv").is_file():
        with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
            rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
        if rows:
            best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
            return {
                "epoch": int(best["epoch"]),
                "epochs_trained": len(rows),
                "val_map50_95": round(float(best["metrics/mAP50-95(B)"]), 4),
                "val_map50": round(float(best["metrics/mAP50(B)"]), 4),
            }
    if model in {"dfine", "rtdetr"} and (run / "log.txt").is_file():
        epochs = [json.loads(line) for line in (run / "log.txt").read_text(encoding="utf-8").splitlines() if line]
        scored = [entry for entry in epochs if entry.get("test_coco_eval_bbox")]
        if scored:
            best = max(scored, key=lambda entry: entry["test_coco_eval_bbox"][0])
            return {
                "epoch": best["epoch"] + 1,
                "epochs_trained": len(epochs),
                "val_map50_95": round(best["test_coco_eval_bbox"][0], 4),
                "val_map50": round(best["test_coco_eval_bbox"][1], 4),
            }
    return {}


def mark_complete(run: Path, model: str | None = None) -> Path:
    run = run.resolve()
    model = model or detect_model(run)
    files = sorted({str(path.relative_to(run)) for pattern in UPLOAD_PATTERNS[model] for path in run.glob(pattern)})
    if not any(Path(name).suffix in WEIGHT_SUFFIXES for name in files):
        raise SystemExit(f"No best weights found in {run}; not marking it for upload")
    (run / "upload.txt").write_text("\n".join([*files, "done.json"]) + "\n", encoding="utf-8")
    done = {
        "state": "complete",
        "run": run.name,
        "model": model,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": best_metrics(run, model),
        "files": files,
    }
    temporary = run / "done.json.tmp"
    temporary.write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run / "done.json")
    print(f"Marked for Drive upload: {run.name} ({model}, {len(files)} file(s))", flush=True)
    return run / "done.json"


def unmark(run: Path) -> None:
    """Hold a run back from upload while it trains further (``--resume``)."""
    (run / "done.json").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Finished run folders.")
    parser.add_argument("--model", choices=sorted(UPLOAD_PATTERNS), help="Override the detected model type.")
    args = parser.parse_args()
    for run in args.runs:
        mark_complete(run, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
