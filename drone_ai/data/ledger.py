#!/usr/bin/env python3
"""Show the state of every annotation archive fetched from Google Drive.

The ledger (``datasets/incoming/ledger.json``) is the single source of truth
for the incoming pipeline. Each Drive file version gets one record keyed by
``<drive id>:<md5>`` that moves through these states:

    new -> downloading -> downloaded -> extracting -> extracted
                -> download_failed          -> extract_failed
    duplicate (same content as another record; not downloaded again)
    ignored   (not a .zip/.rar archive)

Every transition is appended to the record's ``events`` list, so the history
of a single archive can be read without digging through log files.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path


INCOMING_ROOT = Path("datasets/incoming")
LEDGER_VERSION = 1


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Ledger:
    """JSON ledger with an exclusive lock so two pipeline runs cannot collide."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "ledger.json"
        self.data: dict[str, object] = {"version": LEDGER_VERSION, "drive_folder": None, "records": {}}
        self._lock_handle = None

    def __enter__(self) -> "Ledger":
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.root / ".ledger.lock").open("w")
        try:
            fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another pipeline run is using {self.path}; wait for it to finish")
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            for record in self.records.values():
                record.setdefault("status", "new")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.save()
        fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
        self._lock_handle.close()

    @property
    def records(self) -> dict[str, dict[str, object]]:
        return self.data["records"]  # type: ignore[return-value]

    def save(self) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def set_status(self, record: dict[str, object], status: str, detail: str | None = None, **fields: object) -> None:
        """Move a record to a new state, log the event, and persist immediately."""
        record["status"] = status
        record["updated_at"] = now()
        record.update(fields)
        event = {"at": record["updated_at"], "status": status}
        if detail:
            event["detail"] = detail
        record.setdefault("events", []).append(event)  # type: ignore[union-attr]
        self.save()

    def by_status(self, *statuses: str) -> list[dict[str, object]]:
        return [record for record in self.records.values() if record["status"] in statuses]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incoming", type=Path, default=INCOMING_ROOT)
    parser.add_argument("--all", action="store_true", help="List every record, not only problems.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.incoming / "ledger.json").is_file():
        raise SystemExit(f"No ledger yet at {args.incoming / 'ledger.json'}; run fetch-drive first")
    with Ledger(args.incoming) as ledger:
        records = sorted(ledger.records.values(), key=lambda record: str(record["drive_path"]))
        print(f"Drive folder: {ledger.data.get('drive_folder')}")
        print(f"Records: {len(records)}")
        for status, count in sorted(Counter(str(record["status"]) for record in records).items()):
            print(f"  {status:16} {count}")
        failed = [record for record in records if str(record["status"]).endswith("_failed")]
        shown = records if args.all else failed
        if shown:
            print()
        for record in shown:
            print(f"[{record['status']}] {record['drive_path']}")
            if record.get("error"):
                print(f"    error: {record['error']}")
            print(f"    last update: {record.get('updated_at')}, attempts: {record.get('attempts', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
