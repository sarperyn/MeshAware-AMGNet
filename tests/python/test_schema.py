from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.schema import load_experiment_config


class ExperimentConfigTests(unittest.TestCase):
    def load(self, name: str):
        return load_experiment_config(REPO_ROOT / "configs" / f"{name}.json")

    def test_expected_h_grids(self) -> None:
        self.assertEqual(self.load("small").h_values, (0.125, 0.0625, 0.015625))
        self.assertEqual(
            self.load("medium").h_values,
            (0.125, 0.03125, 0.00390625, 0.0009765625),
        )
        self.assertEqual(
            self.load("large").h_values,
            (
                0.125,
                0.0625,
                0.03125,
                0.015625,
                0.0078125,
                0.00390625,
                0.001953125,
                0.0009765625,
            ),
        )

    def test_theta_grids_are_inclusive(self) -> None:
        for name, count in (("small", 5), ("medium", 10), ("large", 25)):
            values = self.load(name).theta_values
            self.assertEqual(len(values), count)
            self.assertAlmostEqual(values[0], 0.02)
            self.assertAlmostEqual(values[-1], 0.90)

    def test_paper_reference_grid(self) -> None:
        config = self.load("paper_reference")
        self.assertEqual(config.theta_values, (0.24, 0.48, 0.72))
        self.assertEqual(
            config.patterns, ("vertical_stripes_4", "checkerboard_4x4")
        )

    def test_large_repeat_schedule_matches_paper(self) -> None:
        config = self.load("large")
        self.assertEqual(
            tuple(config.repeats_by_level[level] for level in config.levels),
            (200, 100, 50, 20, 10, 7, 5, 4),
        )

    def test_smoke_expands_to_one_trial(self) -> None:
        config = self.load("smoke")
        self.assertEqual(config.trial_count, 1)
        self.assertEqual(config.warmup_runs, 1)
        self.assertEqual(len(tuple(config.iter_trials())), 1)

    def test_polygonal_validation_configs(self) -> None:
        smoke = self.load("polygonal_smoke")
        convergence = self.load("polygonal_convergence")
        self.assertEqual(smoke.mesh_families, ("polygonal",))
        self.assertEqual(smoke.trial_count, 1)
        self.assertEqual(convergence.levels, (4, 5, 6))

    def test_batch_smoke_grid(self) -> None:
        config = self.load("batch_smoke")
        self.assertEqual(config.trial_count, 12)
        self.assertEqual(config.theta_values, (0.2, 0.4))
        self.assertEqual(config.repeats, 2)

    def test_duplicate_theta_values_are_rejected(self) -> None:
        config = replace(self.load("batch_smoke"), theta_values=(0.2, 0.2))
        with self.assertRaisesRegex(ValueError, "must be unique"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
