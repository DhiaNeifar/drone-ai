"""Shared logging: one log file per command run plus an append-only run history.

Every command started through ``python -m drone_ai.<group>`` writes:

- ``logs/<timestamp>_<command>.log``: full DEBUG log, including anything the
  command printed to stdout and the traceback of an uncaught error.
- ``logs/history.jsonl``: one JSON line per run with its status
  (``ok``/``failed``/``interrupted``), duration, arguments, and log file.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from drone_ai.paths import timestamped_experiment_name


LOG_ROOT = Path("logs")
HISTORY_FILE = LOG_ROOT / "history.jsonl"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_FORMAT = "%(levelname)s %(message)s"


class _StdoutToLog:
    """Mirror print() output of a command into the run log file."""

    def __init__(self, stream, logger: logging.Logger) -> None:
        self._stream = stream
        self._logger = logger
        self._buffer = ""

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.info(line)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _exit_status(code: object) -> tuple[str, int, str | None]:
    if code is None or code == 0:
        return "ok", 0, None
    if isinstance(code, int):
        return "failed", code, f"exit code {code}"
    return "failed", 1, str(code)


@contextmanager
def run_logging(command: str, argv: list[str]) -> Iterator[Path]:
    """Log one command run to its own file and record the outcome in history."""
    started_at = datetime.now()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{timestamped_experiment_name(command, started_at)}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    stdout_logger = logging.getLogger("stdout")
    stdout_logger.propagate = False
    stdout_logger.addHandler(file_handler)
    original_stdout = sys.stdout
    sys.stdout = _StdoutToLog(original_stdout, stdout_logger)

    log = logging.getLogger("drone_ai.run")
    log.info("Run %s started; log file: %s", command, log_path)
    log.debug("Arguments: %s", argv)
    status, exit_code, error = "ok", 0, None
    started = time.perf_counter()
    try:
        yield log_path
    except SystemExit as exc:
        status, exit_code, error = _exit_status(exc.code)
        if status != "ok":
            log.error("Run failed: %s", error)
        raise
    except KeyboardInterrupt:
        status, exit_code, error = "interrupted", 130, "KeyboardInterrupt"
        log.error("Run interrupted by user")
        raise
    except BaseException as exc:
        status, exit_code, error = "failed", 1, f"{type(exc).__name__}: {exc}"
        log.error("Run failed with an uncaught error:\n%s", traceback.format_exc())
        raise
    finally:
        elapsed = round(time.perf_counter() - started, 3)
        log.info("Run %s finished: %s in %.1fs", command, status, elapsed)
        sys.stdout = original_stdout
        entry = {
            "command": command,
            "argv": argv,
            "status": status,
            "exit_code": exit_code,
            "error": error,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "log_file": str(log_path),
        }
        with HISTORY_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        for handler in (file_handler, console_handler):
            root.removeHandler(handler)
        stdout_logger.removeHandler(file_handler)
        file_handler.close()
