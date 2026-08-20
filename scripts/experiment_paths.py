"""Shared naming rules for training runs and inference artifacts."""

from __future__ import annotations

import re
from datetime import datetime


TIMESTAMP_FORMAT = "%d_%m_%Y_%H_%M_%S"


def clean_experiment_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    if not name:
        raise ValueError("Experiment name must contain a letter or number")
    return name


def timestamped_experiment_name(experiment: str, started_at: datetime) -> str:
    timestamp = started_at.strftime(TIMESTAMP_FORMAT)
    return f"{timestamp}_{clean_experiment_name(experiment)}"
