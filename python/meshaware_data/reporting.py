from __future__ import annotations

import csv
import glob
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_FIELDS = (
    "scale",
    "sample_id",
    "matrix_id",
    "mesh_family",
    "pattern",
    "epsilon",
    "refinement",
    "h",
    "h_max",
    "theta",
    "amg_backend",
    "boomeramg_profile",
    "amg_smoother",
    "amg_relaxation_weight",
    "repeat",
    "cells",
    "background_cells",
    "dofs",
    "nnz",
    "rho",
    "iterations",
    "n_levels",
    "grid_complexity",
    "operator_complexity",
    "residual_initial",
    "residual_final",
    "assembly_sec",
    "amg_setup_sec",
    "solve_sec",
    "elapsed_sec",
    "setup_plus_solve_sec",
    "l2_error",
    "h1_seminorm_error",
    "energy_error",
    "matrix_path",
)

METRIC_COLUMNS = frozenset(
    {
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
)


def _coerce_csv_value(value: str) -> Any:
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
    """Read generated trial-report CSVs for plotting and analysis."""

    def __init__(self, records: list[SampleRecord]) -> None:
        self._records = records

    @classmethod
    def from_glob(cls, pattern: str) -> SampleRecordRepository:
        records: list[SampleRecord] = []
        for filename in sorted(glob.glob(pattern, recursive=True)):
            path = Path(filename)
            if path.suffix.lower() != ".csv":
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    row = {
                        key: _coerce_csv_value(value)
                        for key, value in raw.items()
                    }
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
                        SampleRecord(
                            sample_meta=sample_meta,
                            metrics=metrics,
                        )
                    )
        return cls(records)

    def all(self) -> list[SampleRecord]:
        return list(self._records)


def load_json_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("schema_version") != 1:
            raise ValueError(f"Unsupported record schema in {path}")
        records.append(record)
    return records


def report_row(record: dict[str, Any], scale: str) -> dict[str, Any]:
    setup = float(record["amg_setup_time_seconds"])
    solve = float(record["solve_time_seconds"])
    return {
        "scale": scale,
        "sample_id": record["sample_id"],
        "matrix_id": record["matrix_id"],
        "mesh_family": record["mesh_family"],
        "pattern": record["pattern"],
        "epsilon": record["epsilon"],
        "refinement": record["level"],
        "h": record["h_nominal"],
        "h_max": record["h_max"],
        "theta": record["theta"],
        "amg_backend": record.get("amg_backend", "boomeramg"),
        "boomeramg_profile": record.get("boomeramg_profile", "default"),
        "amg_smoother": record.get(
            "amg_smoother", "symmetric-gauss-seidel"
        ),
        "amg_relaxation_weight": record.get("amg_relaxation_weight", 1.0),
        "repeat": record["repeat"],
        "cells": record["cells"],
        "background_cells": record.get("background_cells", record["cells"]),
        "dofs": record["dofs"],
        "nnz": record["nnz"],
        "rho": record["convergence_factor"],
        "iterations": record["cg_iterations"],
        "n_levels": record["amg_levels"],
        "grid_complexity": record.get("grid_complexity", 0.0),
        "operator_complexity": record.get("operator_complexity", 0.0),
        "residual_initial": record["residual_initial"],
        "residual_final": record["residual_final"],
        "assembly_sec": record["assembly_time_seconds"],
        "amg_setup_sec": setup,
        "solve_sec": solve,
        # Existing plots define elapsed_sec as solve-only time. The combined
        # value remains available under its unambiguous name.
        "elapsed_sec": solve,
        "setup_plus_solve_sec": setup + solve,
        "l2_error": record["l2_error"],
        "h1_seminorm_error": record["h1_seminorm_error"],
        "energy_error": record["energy_error"],
        "matrix_path": record["matrix_path"],
    }


def write_trial_report(
    records: list[dict[str, Any]], destination: Path, scale: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_row(record, scale) for record in records)


def write_optimal_theta_summary(
    records: list[dict[str, Any]], destination: Path
) -> None:
    by_matrix_theta: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for record in records:
        by_matrix_theta[(str(record["matrix_id"]), float(record["theta"]))].append(
            record
        )

    aggregated: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (matrix_id, theta), group in by_matrix_theta.items():
        aggregated[matrix_id].append(
            {
                "theta": theta,
                "mean_rho": statistics.mean(
                    float(record["convergence_factor"]) for record in group
                ),
                "mean_setup_sec": statistics.mean(
                    float(record["amg_setup_time_seconds"]) for record in group
                ),
                "mean_solve_sec": statistics.mean(
                    float(record["solve_time_seconds"]) for record in group
                ),
                "repeats": float(len(group)),
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "matrix_id",
        "theta_min_rho",
        "mean_rho",
        "theta_min_total_time",
        "mean_setup_plus_solve_sec",
        "theta_count",
        "minimum_repeats",
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for matrix_id, candidates in sorted(aggregated.items()):
            # Explicit deterministic tie-break rules are part of the schema.
            best_rho = min(
                candidates,
                key=lambda row: (
                    row["mean_rho"],
                    row["mean_solve_sec"],
                    row["theta"],
                ),
            )
            best_time = min(
                candidates,
                key=lambda row: (
                    row["mean_setup_sec"] + row["mean_solve_sec"],
                    row["mean_rho"],
                    row["theta"],
                ),
            )
            writer.writerow(
                {
                    "matrix_id": matrix_id,
                    "theta_min_rho": best_rho["theta"],
                    "mean_rho": best_rho["mean_rho"],
                    "theta_min_total_time": best_time["theta"],
                    "mean_setup_plus_solve_sec": best_time["mean_setup_sec"]
                    + best_time["mean_solve_sec"],
                    "theta_count": len(candidates),
                    "minimum_repeats": int(
                        min(candidate["repeats"] for candidate in candidates)
                    ),
                }
            )


def build_family_reports(family_root: Path, scale: str) -> int:
    records = load_json_records((family_root / "records").glob("*.json"))
    if not records:
        return 0
    write_trial_report(
        records, family_root / "diffusion_reports" / "trials.csv", scale
    )
    write_optimal_theta_summary(
        records, family_root / "summaries" / "optimal_theta.csv"
    )
    return len(records)
