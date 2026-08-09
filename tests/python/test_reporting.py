from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.reporting import (
    SampleRecordRepository,
    report_row,
    write_optimal_theta_summary,
)


def record(theta: float, rho: float, setup: float, solve: float, repeat: int):
    return {
        "matrix_id": "matrix-a",
        "sample_id": f"sample-{theta}-{repeat}",
        "schema_version": 1,
        "mesh_family": "quadrilateral",
        "pattern": "vertical_split",
        "epsilon": 0.0,
        "level": 3,
        "h_nominal": 0.125,
        "h_max": 0.1767,
        "theta": theta,
        "repeat": repeat,
        "cells": 256,
        "background_cells": 256,
        "dofs": 289,
        "nnz": 1889,
        "convergence_factor": rho,
        "grid_complexity": 1.25,
        "operator_complexity": 1.5,
        "cg_iterations": 8,
        "amg_levels": 4,
        "residual_initial": 1.0,
        "residual_final": 1e-8,
        "assembly_time_seconds": 0.01,
        "amg_setup_time_seconds": setup,
        "solve_time_seconds": solve,
        "l2_error": 0.1,
        "h1_seminorm_error": 0.2,
        "energy_error": 0.2,
        "matrix_path": "matrix.petsc",
    }


class ReportingTests(unittest.TestCase):
    def test_elapsed_is_solve_only_and_total_is_explicit(self) -> None:
        row = report_row(record(0.2, 0.2, 3.0, 2.0, 0), "smoke")
        self.assertEqual(row["elapsed_sec"], 2.0)
        self.assertEqual(row["setup_plus_solve_sec"], 5.0)
        self.assertEqual(row["n_levels"], 4)
        self.assertEqual(row["amg_backend"], "boomeramg")
        self.assertEqual(row["boomeramg_profile"], "default")
        self.assertEqual(row["amg_smoother"], "symmetric-gauss-seidel")
        self.assertEqual(row["amg_relaxation_weight"], 1.0)
        self.assertEqual(row["grid_complexity"], 1.25)
        self.assertEqual(row["operator_complexity"], 1.5)

    def test_optimal_theta_has_two_objectives(self) -> None:
        records = [
            record(0.2, 0.10, 5.0, 2.0, 0),
            record(0.2, 0.12, 5.0, 2.0, 1),
            record(0.5, 0.20, 1.0, 1.0, 0),
            record(0.5, 0.22, 1.0, 1.0, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimal.csv"
            write_optimal_theta_summary(records, path)
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(float(row["theta_min_rho"]), 0.2)
        self.assertEqual(float(row["theta_min_total_time"]), 0.5)
        self.assertEqual(int(row["minimum_repeats"]), 2)

    def test_existing_plot_repository_reads_derived_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            path.write_text(
                "pattern,epsilon,h,theta,rho,iterations,elapsed_sec\n"
                "vertical_split,0,0.125,0.24,0.1,8,0.002\n",
                encoding="utf-8",
            )
            loaded = SampleRecordRepository.from_glob(str(path)).all()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].sample_meta["pattern"], "vertical_split")
        self.assertEqual(loaded[0].metrics["iterations"], 8)


if __name__ == "__main__":
    unittest.main()
