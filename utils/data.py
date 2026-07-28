from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRIC_COLUMNS = {
    "rho",
    "iterations",
    "n_levels",
    "elapsed_sec",
    "assembly_sec",
    "amg_setup_sec",
    "solve_sec",
    "setup_plus_solve_sec",
    "l2_error",
    "h1_seminorm_error",
    "energy_error",
    "residual_initial",
    "residual_final",
}


def _coerce(value: str) -> Any:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


@dataclass(frozen=True)
class SampleRecord:
    sample_meta: dict[str, Any]
    metrics: dict[str, Any]


class SampleRecordRepository:
    def __init__(self, records: list[SampleRecord]) -> None:
        self._records = records

    @classmethod
    def from_glob(cls, pattern: str) -> "SampleRecordRepository":
        records: list[SampleRecord] = []
        for filename in sorted(glob.glob(pattern, recursive=True)):
            path = Path(filename)
            if path.suffix.lower() != ".csv":
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    row = {key: _coerce(value) for key, value in raw.items()}
                    metrics = {
                        key: value
                        for key, value in row.items()
                        if key in METRIC_COLUMNS and value is not None
                    }
                    sample_meta = {
                        key: value
                        for key, value in row.items()
                        if key not in METRIC_COLUMNS and value is not None
                    }
                    records.append(
                        SampleRecord(sample_meta=sample_meta, metrics=metrics)
                    )
        return cls(records)

    def all(self) -> list[SampleRecord]:
        return list(self._records)
