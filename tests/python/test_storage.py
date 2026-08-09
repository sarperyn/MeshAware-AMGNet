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


if __name__ == "__main__":
    unittest.main()
