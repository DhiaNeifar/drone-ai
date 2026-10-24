"""Evaluate trained models and run inference."""

from __future__ import annotations

from drone_ai.cli import dispatch


COMMANDS = {
    "yolo-eval": "yolo_evaluate",
    "yolo-predict": "yolo_predict",
}


if __name__ == "__main__":
    raise SystemExit(dispatch("test", COMMANDS, __doc__))
