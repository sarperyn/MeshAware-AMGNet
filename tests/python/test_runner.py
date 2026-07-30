from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_experiments", REPO_ROOT / "scripts" / "run_experiments.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
run_experiments = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(run_experiments)

from meshaware_data.schema import load_experiment_config


class BatchRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_experiment_config(
            REPO_ROOT / "configs" / "batch_smoke.json"
        )
        self.trials = list(self.config.iter_trials())

    def test_groups_theta_and_repeats_by_matrix(self) -> None:
        groups = run_experiments.group_trials_by_matrix(
            self.trials, self.config.high_region
        )
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(len(group) == 4 for _, group in groups))

    def test_batch_command_carries_timing_protocol(self) -> None:
        _, group = run_experiments.group_trials_by_matrix(
            self.trials, self.config.high_region
        )[0]
        command = run_experiments.command_for_batch(
            Path("driver"),
            self.config,
            group[0],
            Path("records"),
            Path("matrix.petsc"),
            (0.2, 0.4),
            2,
            write_matrix=False,
            skip_existing_records=True,
        )
        self.assertEqual(command[command.index("--theta-values") + 1], "0.2,0.4")
        self.assertEqual(command[command.index("--repeats") + 1], "2")
        self.assertEqual(
            command[command.index("--warmup-runs") + 1],
            str(self.config.warmup_runs),
        )
        self.assertEqual(
            command[command.index("--amg-smoother") + 1],
            "symmetric-gauss-seidel",
        )
        self.assertAlmostEqual(
            float(command[command.index("--jacobi-damping") + 1]),
            2.0 / 3.0,
        )
        self.assertIn("--skip-matrix-write", command)
        self.assertIn("--skip-existing-records", command)


if __name__ == "__main__":
    unittest.main()
