"""Train detection models."""

from __future__ import annotations

from drone_ai.cli import dispatch


COMMANDS = {
    "yolo": "yolo",
    "rfdetr": "rfdetr",
    "dfine": "dfine",
    "rtdetr": "rtdetr",
    "mark-done": "publish",
}


if __name__ == "__main__":
    raise SystemExit(dispatch("train", COMMANDS, __doc__))
