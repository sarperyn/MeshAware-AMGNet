from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.evaluation import (
    EvaluationConfig,
    evaluate,
    grouped_bootstrap_intervals,
    regression_metrics,
    theta_decisions,
)
from meshaware_ml.training import train

from tests.python.test_ml_training import (
    tiny_run_config,
    write_synthetic_index,
)


def prediction(
    *,
    sample_id: str,
    matrix_id: str,
    source_hash: str,
    theta: float,
    target: float,
    predicted: float,
    family: str = "simplex",
) -> dict:
    error = predicted - target
    return {
        "sample_id": sample_id,
        "matrix_id": matrix_id,
        "matrix_sha256": source_hash,
        "mesh_family": family,
        "level": 3,
        "pattern": "vertical_split",
        "epsilon": 0.0,
        "theta": theta,
        "target_rho": target,
        "predicted_rho": predicted,
        "error": error,
        "absolute_error": abs(error),
        "squared_error": error * error,
    }


class MetricTests(unittest.TestCase):
    def test_regression_metrics_and_range_counts(self) -> None:
        rows = [
            prediction(
                sample_id="a",
                matrix_id="m1",
                source_hash="a" * 64,
                theta=0.2,
                target=0.0,
                predicted=-0.1,
            ),
            prediction(
                sample_id="b",
                matrix_id="m2",
                source_hash="b" * 64,
                theta=0.4,
                target=1.0,
                predicted=1.1,
            ),
        ]
        metrics = regression_metrics(rows)
        self.assertTrue(
            math.isclose(metrics["rmse"], 0.1, abs_tol=1.0e-15)
        )
        self.assertTrue(
            math.isclose(metrics["mae"], 0.1, abs_tol=1.0e-15)
        )
        self.assertEqual(metrics["predictions_below_zero"], 1)
        self.assertEqual(metrics["predictions_above_one"], 1)

    def test_grouped_bootstrap_is_deterministic(self) -> None:
        rows = [
            prediction(
                sample_id=f"s{index}",
                matrix_id=f"m{index // 2}",
                source_hash=chr(97 + index // 2) * 64,
                theta=0.2 + 0.1 * (index % 2),
                target=0.2 + 0.1 * index,
                predicted=0.21 + 0.09 * index,
            )
            for index in range(6)
        ]
        first = grouped_bootstrap_intervals(
            rows,
            group_key="matrix_sha256",
            replicates=50,
            seed=2026,
            confidence_level=0.95,
        )
        second = grouped_bootstrap_intervals(
            rows,
            group_key="matrix_sha256",
            replicates=50,
            seed=2026,
            confidence_level=0.95,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["groups"], 3)

    def test_theta_decision_reports_true_regret(self) -> None:
        rows = [
            prediction(
                sample_id="a",
                matrix_id="m",
                source_hash="a" * 64,
                theta=0.2,
                target=0.2,
                predicted=0.4,
            ),
            prediction(
                sample_id="b",
                matrix_id="m",
                source_hash="a" * 64,
                theta=0.4,
                target=0.3,
                predicted=0.1,
            ),
        ]
        decisions, metrics = theta_decisions(rows)
        self.assertFalse(decisions[0]["exact_optimum"])
        self.assertTrue(
            math.isclose(decisions[0]["regret"], 0.1, abs_tol=1.0e-15)
        )
        self.assertEqual(metrics["exact_optimum_rate"], 0.0)


class LockedEvaluationTests(unittest.TestCase):
    def test_evaluation_is_locked_reused_and_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, samples_path, splits_path = write_synthetic_index(root)
            run = tiny_run_config(
                root,
                dataset_root,
                samples_path,
                splits_path,
                max_epochs=1,
            )
            train(run)
            evaluation_config_path = root / "evaluation.json"
            evaluation_config_path.write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            config = EvaluationConfig(
                config_path=evaluation_config_path,
                training_run=run,
                checkpoint_path=run.output_dir / "best.pt",
                phase3_summary_path=run.output_dir / "summary.json",
                output_dir=root / "results" / "test_v1",
                report_path=root / "reports" / "phase4.md",
                bootstrap_replicates=20,
                bootstrap_seed=2026,
                confidence_level=0.95,
                create_plots=False,
            )
            metrics, lock = evaluate(config)
            self.assertEqual(metrics["overall"]["samples"], 1)
            self.assertEqual(lock["test_contract"]["model_updates"], 0)
            self.assertTrue(
                (config.output_dir / "evaluation_lock.json").is_file()
            )
            self.assertTrue(config.report_path.is_file())

            reused_metrics, reused_lock = evaluate(config)
            self.assertEqual(metrics, reused_metrics)
            self.assertEqual(lock, reused_lock)

            predictions = config.output_dir / "predictions.csv"
            predictions.write_text(
                predictions.read_text(encoding="utf-8") + "corruption\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                evaluate(config)


if __name__ == "__main__":
    unittest.main()
