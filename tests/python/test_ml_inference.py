from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.amg_integration import (
    AMGProblem,
    assemble_command,
    solve_command,
    validate_existing_workflow,
)
from meshaware_ml.inference import RhoPredictor, parse_theta_values
from meshaware_ml.training import train

from tests.python.test_ml_pipeline import write_matrix_npz
from tests.python.test_ml_training import (
    tiny_run_config,
    write_synthetic_index,
)


class PredictorTests(unittest.TestCase):
    def test_theta_validation(self) -> None:
        self.assertEqual(
            parse_theta_values((0.4, 0.2)), (0.2, 0.4)
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_theta_values((0.2, 0.2))
        with self.assertRaisesRegex(ValueError, r"in \(0, 1\)"):
            parse_theta_values((0.0, 0.2))

    def test_verified_predictor_is_deterministic_and_checks_level(self) -> None:
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
            matrix_path = root / "matrix.npz"
            write_matrix_npz(
                matrix_path,
                matrix_id="inference_matrix",
                source_hash="e" * 64,
                size=100,
            )
            predictor = RhoPredictor(
                run,
                run.output_dir / "best.pt",
                run.output_dir / "summary.json",
                device="cpu",
            )
            first = predictor.recommend_matrix(
                matrix_path, (0.2, 0.4, 0.99)
            )
            second = predictor.recommend_matrix(
                matrix_path, (0.2, 0.4, 0.99)
            )
            self.assertEqual(first["predictions"], second["predictions"])
            self.assertEqual(first["recommendation"], second["recommendation"])
            self.assertEqual(first["model_updates"], 0)
            self.assertEqual(first["matrix"]["source_sha256"], "e" * 64)
            self.assertTrue(any("theta candidates" in warning for warning in first["warnings"]))
            with self.assertRaisesRegex(ValueError, "conflicts"):
                predictor.recommend_matrix(
                    matrix_path, (0.2, 0.4), level=4
                )


class AMGIntegrationTests(unittest.TestCase):
    def test_commands_apply_recommended_theta_to_existing_driver(self) -> None:
        problem = AMGProblem(
            mesh_family="simplex",
            level=3,
            pattern="vertical_split",
            epsilon=0.0,
            repeats=2,
            warmup_runs=1,
        )
        assembly = assemble_command(
            Path("driver"), problem, Path("operator.petsc")
        )
        self.assertIn("--assemble-only", assembly)
        solve = solve_command(
            Path("driver"),
            problem,
            0.4,
            Path("records"),
            Path("operator.npz"),
        )
        self.assertEqual(solve[solve.index("--theta-values") + 1], "0.4")
        self.assertIn("--skip-matrix-write", solve)
        self.assertEqual(solve[solve.index("--repeats") + 1], "2")

    def test_problem_scope_and_matrix_identity(self) -> None:
        problem = AMGProblem(
            mesh_family="polygonal",
            level=4,
            pattern="checkerboard_2x2",
            epsilon=1.2,
        )
        problem.validate()
        self.assertEqual(
            problem.matrix_id,
            "poly_l4_checkerboard_2x2_e1p2_high_white",
        )
        with self.assertRaisesRegex(ValueError, "supports"):
            AMGProblem(
                mesh_family="quadrilateral",
                level=3,
                pattern="vertical_split",
                epsilon=0.0,
            ).validate()

    def test_workflow_lock_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "workflow"
            output.mkdir()
            manifest = output / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1}))
            import hashlib

            expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (output / "workflow_lock.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifacts": {"manifest.json": expected},
                    }
                )
            )
            loaded, _ = validate_existing_workflow(output)
            self.assertEqual(loaded["schema_version"], 1)
            manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                validate_existing_workflow(output)


if __name__ == "__main__":
    unittest.main()
