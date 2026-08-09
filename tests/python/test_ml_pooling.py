from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.pooling import (
    PAPER_POOLING_SPEC,
    PoolingSpec,
    balanced_block_boundaries,
    pool_csr_arrays,
    validate_feature_artifact,
    write_feature_artifact_atomic,
)


class PoolingTests(unittest.TestCase):
    def test_balanced_boundaries_put_larger_blocks_first(self) -> None:
        np.testing.assert_array_equal(
            balanced_block_boundaries(11, 4), (0, 3, 6, 9, 11)
        )

    def test_rejects_view_larger_than_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than view size"):
            balanced_block_boundaries(3, 4)

    def test_exact_three_channel_pool_and_normalization(self) -> None:
        # [[ 4, -2,  3,  0],
        #  [-1,  0,  0,  1],
        #  [ 0,  5, -4,  0],
        #  [ 0,  0,  0,  2]]
        indptr = np.asarray((0, 3, 5, 7, 8), dtype=np.int64)
        indices = np.asarray((0, 1, 2, 0, 3, 1, 2, 3), dtype=np.int32)
        data = np.asarray((4, -2, 3, -1, 1, 5, -4, 2), dtype=np.float64)
        spec = PoolingSpec(view_size=2)

        actual = pool_csr_arrays(indptr, indices, data, (4, 4), spec)

        averaged = np.asarray(
            (
                ((4 / 3, 3 / 2), (5, 1)),
                ((2 / 3, 0), (0, 2)),
                ((1 / 3, 2), (5, -1)),
            ),
            dtype=np.float64,
        )
        transformed = np.sign(averaged) * np.log1p(np.abs(averaged))
        for channel in range(3):
            transformed[channel] /= np.max(np.abs(transformed[channel]))
        np.testing.assert_allclose(actual, transformed.astype(np.float32))
        self.assertEqual(actual.dtype, np.dtype("float32"))

    def test_all_zero_sign_channel_stays_finite_and_zero(self) -> None:
        indptr = np.asarray((0, 1, 2, 3, 4), dtype=np.int64)
        indices = np.asarray((0, 1, 2, 3), dtype=np.int32)
        data = np.ones(4, dtype=np.float64)
        view = pool_csr_arrays(
            indptr, indices, data, (4, 4), PoolingSpec(view_size=2)
        )
        self.assertTrue(np.isfinite(view).all())
        np.testing.assert_array_equal(view[1], np.zeros((2, 2)))

    def test_rejects_invalid_csr_and_non_square_matrix(self) -> None:
        indptr = np.asarray((0, 1, 1), dtype=np.int64)
        indices = np.asarray((2,), dtype=np.int32)
        data = np.asarray((1.0,), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "square"):
            pool_csr_arrays(
                indptr, indices, data, (2, 3), PoolingSpec(view_size=2)
            )
        with self.assertRaisesRegex(ValueError, "out of range"):
            pool_csr_arrays(
                indptr, indices, data, (2, 2), PoolingSpec(view_size=2)
            )


class FeatureArtifactTests(unittest.TestCase):
    def test_atomic_round_trip_and_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.npz"
            view = np.zeros((3, 100, 100), dtype=np.float32)
            metadata = {
                "feature_schema_version": 1,
                "source_sha256": "a" * 64,
                "source_path": "matrix.npz",
                "matrix_identity": {"matrix_id": "matrix"},
                "matrix_shape": [100, 100],
                "matrix_nnz": 100,
                "pooling_spec": PAPER_POOLING_SPEC.to_dict(),
                "ordering_contract": "csr_global_dof_order_v1",
            }
            write_feature_artifact_atomic(path, view, metadata)
            loaded = validate_feature_artifact(
                path, expected_source_sha256="a" * 64
            )
            self.assertEqual(loaded["matrix_nnz"], 100)
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                validate_feature_artifact(
                    path, expected_source_sha256="b" * 64
                )

    def test_detects_wrong_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            metadata = {
                "feature_schema_version": 1,
                "source_sha256": "a" * 64,
                "pooling_spec": PAPER_POOLING_SPEC.to_dict(),
            }
            np.savez_compressed(
                path,
                schema_version=np.asarray(1),
                view=np.zeros((1, 2, 2), dtype=np.float32),
                metadata_json=np.frombuffer(
                    json.dumps(metadata).encode(), dtype=np.uint8
                ),
            )
            with self.assertRaisesRegex(ValueError, "invalid feature tensor"):
                validate_feature_artifact(path)


if __name__ == "__main__":
    unittest.main()
