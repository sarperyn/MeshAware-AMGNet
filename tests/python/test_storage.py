from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.schema import load_experiment_config
from meshaware_data.storage import estimate_experiment_storage


class StorageEstimateTests(unittest.TestCase):
    def load(self, name: str):
        return load_experiment_config(REPO_ROOT / "configs" / f"{name}.json")

    def test_batch_smoke_counts(self) -> None:
        report = estimate_experiment_storage(self.load("batch_smoke"))
        self.assertEqual(report["totals"]["matrices"], 3)
        self.assertEqual(report["totals"]["trial_records"], 12)
        self.assertGreater(report["totals"]["petsc_matrix_bytes"], 0)
        self.assertGreater(report["totals"]["csr_matrix_bytes"], 0)
        self.assertLess(
            report["totals"]["retained_matrix_bytes"],
            report["totals"]["petsc_matrix_bytes"],
        )

    def test_large_file_count_warning(self) -> None:
        report = estimate_experiment_storage(self.load("large"))
        self.assertEqual(report["totals"]["matrices"], 1152)
        self.assertEqual(report["totals"]["trial_records"], 1_425_600)
        self.assertEqual(
            report["largest_solver_case"]["mesh_family"], "polygonal"
        )
        self.assertEqual(report["largest_solver_case"]["level"], 10)
        self.assertTrue(
            any("100,000 files" in warning for warning in report["warnings"])
        )

    def test_simplex_dg_size_model(self) -> None:
        report = estimate_experiment_storage(self.load("medium_simplex_dg"))
        self.assertEqual(report["totals"]["matrices"], 192)
        self.assertEqual(report["totals"]["trial_records"], 1920)
        level_10 = next(
            row for row in report["breakdown"] if row["level"] == 10
        )
        self.assertEqual(level_10["cells"], 8_388_608)
        self.assertEqual(level_10["dofs_per_matrix"], 25_165_824)
        self.assertEqual(
            level_10["model"],
            "structured_p1_sipg_conservative_12_entries_per_dof",
        )


if __name__ == "__main__":
    unittest.main()
