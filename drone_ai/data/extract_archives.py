#!/usr/bin/env python3
"""Extract downloaded annotation archives into one clean folder per archive.

Each archive in state ``downloaded`` is unpacked to
``datasets/incoming/extracted/<archive path without suffix>/``. Extraction goes
to a ``.partial`` folder that is renamed only on success, so an interrupted
run never leaves a half-extracted folder that looks complete.

Safety: entries with absolute paths or ``..`` components and symlinks are
refused, OS junk (``__MACOSX``, ``.DS_Store``, ``Thumbs.db``) is dropped, and
the uncompressed size is checked against free disk space first.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import stat
import subprocess
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from drone_ai.data.ledger import INCOMING_ROOT, Ledger, now


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
ANNOTATION_EXTENSIONS = {".json", ".csv", ".txt", ".xml", ".yaml", ".yml", ".parquet"}
JUNK_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
RAR_VOLUME = re.compile(r"(\.part(\d+)\.rar|\.r\d{2,3})$", re.IGNORECASE)
log = logging.getLogger(__name__)


class ExtractionError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incoming", type=Path, default=INCOMING_ROOT)
    parser.add_argument("--retry-failed", action="store_true", help="Also retry archives in state extract_failed.")
    return parser.parse_args()


def is_junk(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return any(part == "__MACOSX" or part in JUNK_NAMES or part.startswith("._") for part in parts)


def check_member_name(name: str) -> None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
        raise ExtractionError(f"unsafe path inside archive: {name!r}")


def extract_zip(archive: Path, destination: Path) -> int:
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in members:
            check_member_name(member.filename)
            if member.flag_bits & 0x1:
                raise ExtractionError(f"password-protected entry: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ExtractionError(f"symlink inside archive: {member.filename}")
        files = [member for member in members if not member.is_dir() and not is_junk(member.filename)]
        check_disk_space(destination, sum(member.file_size for member in files))
        for member in files:
            target = destination / member.filename.replace("\\", "/")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reading to the end verifies the entry CRC and raises BadZipFile on corruption.
            with bundle.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 8 * 1024 * 1024)
    return len(files)


def run_unrar(*args: str) -> str:
    unrar = shutil.which("unrar")
    if unrar is None:
        raise ExtractionError("unrar is not installed")
    command = [unrar, *args]
    log.debug("Running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
        raise ExtractionError(f"unrar {args[0]} failed (exit {result.returncode}): {detail[-1] if detail else ''}")
    return result.stdout


def rar_unpacked_size(listing: str) -> int:
    """Sum the sizes from ``unrar vt`` technical listing."""
    return sum(int(match) for match in re.findall(r"^\s*Size:\s*(\d+)", listing, re.MULTILINE))


def extract_rar(archive: Path, destination: Path) -> int:
    names = [name for name in run_unrar("lb", "-p-", str(archive)).splitlines() if name.strip()]
    for name in names:
        check_member_name(name)
    check_disk_space(destination, rar_unpacked_size(run_unrar("vt", "-p-", str(archive))))
    run_unrar("t", "-p-", "-idq", str(archive))
    destination.mkdir(parents=True, exist_ok=True)
    run_unrar("x", "-p-", "-o-", "-ol-", "-idq", "-y", str(archive), f"{destination}/")
    for path in sorted(destination.rglob("*"), reverse=True):
        relative = str(path.relative_to(destination))
        if path.is_symlink():
            raise ExtractionError(f"symlink inside archive: {relative}")
        if is_junk(relative):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return sum(1 for path in destination.rglob("*") if path.is_file())


def check_disk_space(destination: Path, required: int) -> None:
    free = shutil.disk_usage(destination.parent).free
    if required > free * 0.9:
        raise ExtractionError(f"not enough disk space: need {required / 1e9:.1f} GB, {free / 1e9:.1f} GB free")


def content_report(root: Path) -> dict[str, object]:
    """Summarize what an extracted archive contains, to choose the right converter."""
    extensions: Counter[str] = Counter()
    total_bytes = 0
    frame_dirs = label_dirs = 0
    annotation_files: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir():
            frame_dirs += path.name.lower() in {"frames", "images"}
            label_dirs += path.name.lower() == "labels"
            continue
        suffix = path.suffix.lower() or "<none>"
        extensions[suffix] += 1
        total_bytes += path.stat().st_size
        if suffix in ANNOTATION_EXTENSIONS and len(annotation_files) < 20:
            annotation_files.append(str(path.relative_to(root)))
    return {
        "files": sum(extensions.values()),
        "bytes": total_bytes,
        "images": sum(count for ext, count in extensions.items() if ext in IMAGE_EXTENSIONS),
        "videos": sum(count for ext, count in extensions.items() if ext in VIDEO_EXTENSIONS),
        "annotation_files": sum(count for ext, count in extensions.items() if ext in ANNOTATION_EXTENSIONS),
        "extensions": dict(extensions.most_common()),
        "frames_or_images_dirs": frame_dirs,
        "labels_dirs": label_dirs,
        "top_level": sorted(path.name for path in root.iterdir())[:30],
        "annotation_examples": sorted(annotation_files),
    }


def archive_format(archive: Path) -> str:
    """Identify the archive by its magic bytes rather than trusting the file name."""
    with archive.open("rb") as handle:
        header = handle.read(8)
    if header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06"):
        return "zip"
    if header.startswith(b"Rar!\x1a\x07"):
        return "rar"
    raise ExtractionError(f"not a zip or rar archive (header {header[:8]!r})")


def extract_one(record: dict[str, object], extracted_root: Path, archives_root: Path) -> tuple[Path, int]:
    archive = Path(str(record["archive"]))
    if not archive.is_file():
        raise ExtractionError(f"archive file is missing: {archive}")
    relative = archive.relative_to(archives_root)
    destination = extracted_root / re.sub(r"\.part\d+$", "", str(relative.with_suffix("")), flags=re.IGNORECASE)
    if destination.exists():
        raise ExtractionError(f"destination already exists: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        if archive_format(archive) == "zip":
            count = extract_zip(archive, partial)
        else:
            count = extract_rar(archive, partial)
    except zipfile.BadZipFile as error:
        shutil.rmtree(partial, ignore_errors=True)
        raise ExtractionError(f"corrupt zip: {error}") from error
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(destination)
    return destination, count


def main() -> int:
    args = parse_args()
    archives_root = args.incoming / "archives"
    extracted_root = args.incoming / "extracted"
    with Ledger(args.incoming) as ledger:
        statuses = ("downloaded", "extracting", "extract_failed") if args.retry_failed else ("downloaded", "extracting")
        pending = sorted(ledger.by_status(*statuses), key=lambda record: str(record["drive_path"]))
        log.info("Archives to extract: %d", len(pending))
        failures = 0
        for index, record in enumerate(pending, start=1):
            name = str(record["drive_path"])
            volume = RAR_VOLUME.search(name)
            if volume and (volume.group(2) is None or int(volume.group(2)) > 1):
                log.info("[%d/%d] %s is a later RAR volume; it is extracted with part 1", index, len(pending), name)
                ledger.set_status(record, "extracted", "later RAR volume, extracted with part 1", volume_of_set=True)
                continue
            log.info("[%d/%d] Extracting %s", index, len(pending), name)
            ledger.set_status(record, "extracting")
            try:
                destination, count = extract_one(record, extracted_root, archives_root)
            except (ExtractionError, OSError) as error:
                failures += 1
                log.error("Extraction failed for %s: %s", name, error)
                ledger.set_status(record, "extract_failed", str(error), error=str(error))
                continue
            report = content_report(destination)
            ledger.set_status(
                record,
                "extracted",
                f"{count} file(s)",
                extracted_dir=str(destination),
                extracted_at=now(),
                content=report,
                error=None,
            )
            log.info(
                "Extracted %d file(s) to %s: %d image(s), %d video(s), %d annotation file(s)",
                count, destination, report["images"], report["videos"], report["annotation_files"],
            )
            log.debug("Content report for %s: %s", name, report)
        log.info("Extraction complete: %d extracted, %d failed", len(pending) - failures, failures)
        if failures:
            raise SystemExit(f"{failures} archive(s) failed to extract; see incoming-status for details")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
