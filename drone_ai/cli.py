"""Shared dispatcher for the ``python -m drone_ai.<group> <command>`` entry points."""

from __future__ import annotations

import argparse
import importlib
import sys

from drone_ai.logs import run_logging


def dispatch(group: str, commands: dict[str, str], description: str | None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m drone_ai.{group}", description=description)
    parser.add_argument("command", choices=sorted(commands))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    module = importlib.import_module(f"drone_ai.{group}.{commands[args.command]}")
    sys.argv = [f"python -m drone_ai.{group} {args.command}", *args.args]
    if any(arg in {"-h", "--help"} for arg in args.args):
        return module.main() or 0
    with run_logging(f"{group}-{args.command}", args.args):
        code = module.main()
        if code:
            raise SystemExit(code)
    return 0
