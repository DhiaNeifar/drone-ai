#!/usr/bin/env python3
"""Download new annotation archives (.zip/.rar) from a Google Drive folder with rclone.

Archives are kept permanently under ``datasets/incoming/archives/`` mirroring
the Drive folder layout. ``datasets/incoming/ledger.json`` records every Drive
file by ID and MD5, so a rerun only downloads files that are new or changed.

One-time setup (read-only access to your Google account):

    rclone config create gdrive drive scope=drive.readonly

On a server without a browser, rclone prints an ``rclone authorize`` command to
run on a machine with a browser; paste the token it returns back into the
server prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from drone_ai.data.ledger import INCOMING_ROOT, Ledger, now


ARCHIVE_PATTERN = re.compile(r"\.(zip|rar|r\d{2,3})$", re.IGNORECASE)
# Annotators sometimes upload archives without an extension; Drive still reports the type.
ARCHIVE_MIME_SUFFIX = {
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/x-rar": ".rar",
    "application/x-rar-compressed": ".rar",
    "application/vnd.rar": ".rar",
}
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--folder",
        help="Drive folder URL or ID. Remembered in the ledger, so it is only needed on the first run.",
    )
    parser.add_argument("--remote", default="gdrive", help="rclone remote name (default: gdrive).")
    parser.add_argument("--incoming", type=Path, default=INCOMING_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded and stop.")
    return parser.parse_args()


def parse_folder(value: str) -> tuple[str, str | None]:
    """Return (folder_id, resource_key) from a Drive folder URL or a bare ID."""
    if "://" not in value:
        return value, None
    url = urlparse(value)
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", url.path)
    if not match:
        raise SystemExit(f"Not a Google Drive folder link: {value}")
    resource_key = parse_qs(url.query).get("resourcekey", [None])[0]
    return match.group(1), resource_key


def rclone_binary() -> str:
    binary = shutil.which("rclone") or str(Path.home() / ".local/bin/rclone")
    if not Path(binary).is_file():
        raise SystemExit("rclone is not installed; see https://rclone.org/install/")
    return binary


def run_rclone(rclone: str, *args: str) -> str:
    command = [rclone, *args]
    log.debug("Running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stderr.strip():
        log.debug("rclone stderr:\n%s", result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"rclone exit {result.returncode}")
    return result.stdout


def check_remote(rclone: str, remote: str) -> None:
    remotes = run_rclone(rclone, "listremotes").split()
    if f"{remote}:" not in remotes:
        raise SystemExit(
            f"rclone remote '{remote}' is not configured. Run once:\n\n"
            f"    rclone config create {remote} drive scope=drive.readonly\n"
        )


def list_folder(rclone: str, remote: str, folder_id: str, resource_key: str | None) -> list[dict[str, object]]:
    options = f"root_folder_id={folder_id}"
    if resource_key:
        options += f",resource_key={resource_key}"
    output = run_rclone(rclone, "lsjson", f"{remote},{options}:", "--recursive", "--files-only", "--hash")
    return json.loads(output)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_suffix(drive_path: str, mime_type: str | None) -> str | None:
    """Return the suffix to add to the local file name ('' if it already has one), or None if not an archive."""
    if ARCHIVE_PATTERN.search(drive_path):
        return ""
    return ARCHIVE_MIME_SUFFIX.get(str(mime_type))


def local_target(archives: Path, drive_path: str, suffix: str, md5: str, taken: set[str]) -> Path:
    """Mirror the Drive path; add an MD5 suffix if a different version already uses the name."""
    target = archives / (drive_path + suffix)
    if not target.resolve().is_relative_to(archives.resolve()):
        raise SystemExit(f"Refusing Drive path outside the archive folder: {drive_path}")
    if str(target) in taken or target.exists():
        target = target.with_name(f"{target.stem}__{md5[:8]}{target.suffix}")
    return target


def download(rclone: str, remote: str, drive_id: str, target: Path) -> None:
    partial = target.with_name(target.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    try:
        run_rclone(rclone, "backend", "copyid", f"{remote}:", drive_id, str(partial))
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(target)


def main() -> int:
    args = parse_args()
    rclone = rclone_binary()
    check_remote(rclone, args.remote)
    archives = args.incoming / "archives"

    with Ledger(args.incoming) as ledger:
        folder_value = args.folder or ledger.data.get("drive_folder")
        if not folder_value:
            raise SystemExit("No Drive folder known yet; pass --folder <link> on the first run")
        folder_id, resource_key = parse_folder(str(folder_value))
        previous = ledger.data.get("drive_folder")
        if previous and parse_folder(str(previous))[0] != folder_id:
            log.warning("Drive folder changed from %s to %s", previous, folder_value)
        ledger.data["drive_folder"] = folder_value
        ledger.data["last_listing_at"] = now()

        log.info("Listing Drive folder %s", folder_id)
        try:
            entries = list_folder(rclone, args.remote, folder_id, resource_key)
        except RuntimeError as error:
            raise SystemExit(f"Could not list the Drive folder: {error}")
        log.info("Drive folder contains %d file(s)", len(entries))

        records = ledger.records
        known_md5 = {
            str(record["md5"]): key
            for key, record in records.items()
            if record["status"] not in {"download_failed", "ignored", "duplicate"}
        }
        taken = {str(record["archive"]) for record in records.values() if record.get("archive")}
        pending: list[dict[str, object]] = []
        skipped = ignored = duplicates = 0

        for entry in sorted(entries, key=lambda item: str(item["Path"])):
            drive_path = str(entry["Path"])
            md5 = str((entry.get("Hashes") or {}).get("md5") or f"size{entry['Size']}-{entry['ModTime']}")
            key = f"{entry['ID']}:{md5}"
            record = records.get(key)
            if record and record["drive_path"] != drive_path:
                log.info("Renamed on Drive: %s -> %s (same file, not downloaded again)", record["drive_path"], drive_path)
                record.setdefault("previous_drive_paths", []).append(record["drive_path"])
                ledger.set_status(record, str(record["status"]), f"renamed on Drive from {record['drive_path']}", drive_path=drive_path)
            if record and record["status"] not in {"new", "download_failed", "downloading", "ignored"}:
                skipped += 1
                log.debug("Already handled (%s): %s", record["status"], drive_path)
                continue
            if record is None:
                record = {
                    "key": key,
                    "status": "new",
                    "drive_id": entry["ID"],
                    "drive_path": drive_path,
                    "size": entry["Size"],
                    "md5": md5,
                    "drive_modified": entry["ModTime"],
                    "mime_type": entry.get("MimeType"),
                    "first_seen_at": now(),
                    "attempts": 0,
                    "events": [],
                }
                records[key] = record
                older = [r for r in records.values() if r["drive_id"] == entry["ID"] and r is not record]
                if older:
                    log.warning("File changed on Drive since last fetch, downloading new version: %s", drive_path)
            suffix = archive_suffix(drive_path, entry.get("MimeType"))
            if suffix is None:
                ignored += 1
                log.warning("Ignoring non-archive file: %s", drive_path)
                ledger.set_status(record, "ignored", f"not a .zip/.rar file (type {entry.get('MimeType')})")
                continue
            if md5 in known_md5 and known_md5[md5] != key:
                duplicates += 1
                original = records[known_md5[md5]]
                log.warning("Duplicate upload of %s, skipping: %s", original["drive_path"], drive_path)
                ledger.set_status(record, "duplicate", f"same content as {original['drive_path']}", duplicate_of=known_md5[md5])
                continue
            if suffix:
                log.warning("Archive without extension, saving as %s: %s", suffix, drive_path)
            record["local_suffix"] = suffix
            pending.append(record)
            known_md5[md5] = key

        total_bytes = sum(int(record["size"]) for record in pending)
        log.info(
            "New archives: %d (%.2f GB); already fetched: %d; duplicates: %d; ignored: %d",
            len(pending), total_bytes / 1e9, skipped, duplicates, ignored,
        )
        if args.dry_run:
            for record in pending:
                log.info("Would download: %s (%.1f MB)", record["drive_path"], int(record["size"]) / 1e6)
            return 0

        free = shutil.disk_usage(args.incoming).free
        if total_bytes > free * 0.9:
            raise SystemExit(f"Not enough disk space: need {total_bytes / 1e9:.1f} GB, {free / 1e9:.1f} GB free")

        failures = 0
        for index, record in enumerate(pending, start=1):
            drive_path = str(record["drive_path"])
            target = Path(record["archive"]) if record.get("archive") else local_target(archives, drive_path, str(record.get("local_suffix", "")), str(record["md5"]), taken)
            taken.add(str(target))
            record["attempts"] = int(record.get("attempts", 0)) + 1
            ledger.set_status(record, "downloading", archive=str(target))
            log.info("[%d/%d] Downloading %s (%.1f MB)", index, len(pending), drive_path, int(record["size"]) / 1e6)
            try:
                download(rclone, args.remote, str(record["drive_id"]), target)
                local_md5 = md5sum(target)
                if not str(record["md5"]).startswith("size") and local_md5 != record["md5"]:
                    target.unlink()
                    raise RuntimeError(f"MD5 mismatch: Drive {record['md5']}, local {local_md5}")
            except (RuntimeError, OSError) as error:
                failures += 1
                log.error("Download failed for %s: %s", drive_path, error)
                ledger.set_status(record, "download_failed", str(error), error=str(error))
                continue
            ledger.set_status(record, "downloaded", downloaded_at=now(), error=None)
            log.info("Saved %s", target)

        log.info("Fetch complete: %d downloaded, %d failed", len(pending) - failures, failures)
        if failures:
            raise SystemExit(f"{failures} download(s) failed; rerun fetch-drive to retry them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
