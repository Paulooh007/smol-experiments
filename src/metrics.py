"""Small CSV metric logging helpers for experiment scripts.

The training scripts in this repo should work in plain terminals, notebooks,
and Modal jobs. CSV logs are boring in the best way: easy to append, easy to
inspect, and easy to turn into dashboards later without committing to a hosted
tracking service.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvMetricLogger:
    """Append metric rows to a CSV file, creating the header when needed."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        should_write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if should_write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **row: Any) -> None:
        clean = {name: row.get(name, "") for name in self.fieldnames}
        self._writer.writerow(clean)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CsvMetricLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None
