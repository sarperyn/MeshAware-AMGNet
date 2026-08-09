from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.indexing import (
    SPLIT_NAMES,
    assign_grouped_stratified_splits,
    build_canonical_samples,
)
from meshaware_ml.inventory import (
    build_feature_cache,
    capture_snapshot,
)


def write_matrix_npz(
    path: Path,
    *,
    matrix_id: str,
    source_hash: str,
    size: int = 100,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indptr = np.arange(size + 1, dtype=np.int64)
    indices = np.arange(size, dtype=np.int32)
    data = np.ones(size, dtype=np.float64)
    identity = {
        "matrix_id": matrix_id,
        "mesh_family": "simplex",
        "level": 3,
        "pattern": "vertical_split",
        "epsilon": 0.0,
        "high_region": "white",
        "dofs": size,
        "nnz": size,
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        indptr=indptr,
        indices=indices,
        data=data,
        shape=np.asarray((size, size), dtype=np.int64),
        source_sha256=np.frombuffer(source_hash.encode(), dtype=np.uint8),
        identity_json=np.frombuffer(
            json.dumps(identity).encode(), dtype=np.uint8
        ),
    )


def synthetic_samples(groups_per_family_level: int = 30) -> list[dict]:
    samples = []
    for family in ("simplex", "polygonal"):
        for level in (3, 5):
            for group_index in range(groups_per_family_level):
                source_hash = __import__("hashlib").sha256(
                    f"{family}:{level}:{group_index}".encode()
                ).hexdigest()
                pattern = (
                    "vertical_split"
                    if group_index % 2 == 0
                    else "checkerboard_4x4"
                )
                epsilon = (0.0, 0.8, 2.0)[group_index % 3]
                for theta in (0.02, 0.46):
                    samples.append(
                        {
                            "matrix_sha256": source_hash,
                            "matrix_id": f"{family}_{level}_{group_index}",
                            "mesh_family": family,
                            "level": level,
                            "pattern": pattern,
                            "epsilon": epsilon,
                            "theta": theta,
                        }
                    )
    return samples


def write_configs(repo_root: Path, tiers: tuple[str, ...]) -> None:
    configs = repo_root / "configs"
    configs.mkdir(parents=True)
    (configs / "common.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "problem": "heterogeneous_diffusion",
                "domain": [[-1.0, 1.0], [-1.0, 1.0]],
                "finite_element_degree": 1,
                "patterns": ["vertical_split"],
                "epsilons": [0.0],
                "high_region": "white",
                "relative_tolerance": 1e-8,
                "absolute_tolerance": 1e-50,
                "maximum_iterations": 100,
                "warmup_runs": 0,
                "matrix_format": "scipy_csr_npz",
                "save_matrix": True,
            }
        )
    )
    for tier in tiers:
        (configs / f"{tier}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": tier,
                    "mesh_families": ["simplex"],
                    "levels": [3],
                    "theta_values": [0.2, 0.4],
                    "repeats": 1,
                }
            )
        )


def write_record(
    path: Path,
    *,
    matrix_id: str,
    theta: float,
    rho: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "matrix_id": matrix_id,
                "mesh_family": "simplex",
                "level": 3,
                "h_nominal": 0.125,
                "theta": theta,
                "repeat": 0,
                "pattern": "vertical_split",
                "epsilon": 0.0,
                "high_region": "white",
                "convergence_factor": rho,
            }
        )
    )


class SnapshotAndCacheTests(unittest.TestCase):
    def test_snapshot_does_not_admit_files_created_after_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "datasets"
            matrices = dataset / "small" / "simplex" / "matrices"
            records = dataset / "small" / "simplex" / "records"
            records.mkdir(parents=True)
            write_matrix_npz(
                matrices / "matrix_a.npz",
                matrix_id="matrix_a",
                source_hash="a" * 64,
            )
            (records / "record_a.json").write_text("{}")
            snapshot = capture_snapshot(
                dataset, tiers=("small",), mesh_families=("simplex",)
            )

            write_matrix_npz(
                matrices / "matrix_b.npz",
                matrix_id="matrix_b",
                source_hash="b" * 64,
            )
            (records / "record_b.json").write_text("{}")

            self.assertEqual(
                [entry["matrix_id"] for entry in snapshot["matrices"]],
                ["matrix_a"],
            )
            self.assertEqual(len(snapshot["records"]), 1)

    def test_cache_reuses_identical_source_hash_across_matrix_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "datasets"
            matrices = dataset / "small" / "simplex" / "matrices"
            records = dataset / "small" / "simplex" / "records"
            records.mkdir(parents=True)
            write_matrix_npz(
                matrices / "matrix_a.npz",
                matrix_id="matrix_a",
                source_hash="a" * 64,
            )
            write_matrix_npz(
                matrices / "matrix_b.npz",
                matrix_id="matrix_b",
                source_hash="a" * 64,
            )
            snapshot = capture_snapshot(
                dataset, tiers=("small",), mesh_families=("simplex",)
            )
            manifest, _ = build_feature_cache(
                snapshot,
                dataset_root=dataset,
                output_root=dataset / "ml",
            )
            self.assertEqual(manifest["matrix_reference_count"], 2)
            self.assertEqual(manifest["unique_matrix_hash_count"], 1)
            self.assertEqual(manifest["features_created"], 1)
            second, _ = build_feature_cache(
                snapshot,
                dataset_root=dataset,
                output_root=dataset / "ml",
            )
            self.assertEqual(second["features_reused"], 1)


class CanonicalIndexTests(unittest.TestCase):
    def test_excludes_partial_group_without_finalized_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets"
            repo = root / "repo"
            write_configs(repo, ("small",))
            matrix_root = dataset / "small" / "simplex" / "matrices"
            record_root = dataset / "small" / "simplex" / "records"
            write_matrix_npz(
                matrix_root / "complete.npz",
                matrix_id="complete",
                source_hash="a" * 64,
            )
            write_record(
                record_root / "complete_02.json",
                matrix_id="complete",
                theta=0.2,
                rho=0.1,
            )
            write_record(
                record_root / "complete_04.json",
                matrix_id="complete",
                theta=0.4,
                rho=0.2,
            )
            write_record(
                record_root / "partial_02.json",
                matrix_id="partial",
                theta=0.2,
                rho=0.3,
            )
            snapshot = capture_snapshot(
                dataset, tiers=("small",), mesh_families=("simplex",)
            )
            manifest, _ = build_feature_cache(
                snapshot,
                dataset_root=dataset,
                output_root=dataset / "ml",
            )
            samples, audit = build_canonical_samples(
                snapshot,
                manifest,
                dataset_root=dataset,
                repo_root=repo,
            )
            self.assertEqual(len(samples), 2)
            self.assertEqual(
                audit["excluded_record_counts"][
                    "no_finalized_snapshotted_npz"
                ],
                1,
            )

    def test_conflicting_duplicate_across_tiers_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets"
            repo = root / "repo"
            write_configs(repo, ("small", "medium"))
            for tier, rho in (("small", 0.1), ("medium", 0.11)):
                matrix_root = dataset / tier / "simplex" / "matrices"
                record_root = dataset / tier / "simplex" / "records"
                write_matrix_npz(
                    matrix_root / "same.npz",
                    matrix_id="same",
                    source_hash="a" * 64,
                )
                write_record(
                    record_root / "same_02.json",
                    matrix_id="same",
                    theta=0.2,
                    rho=rho,
                )
                write_record(
                    record_root / "same_04.json",
                    matrix_id="same",
                    theta=0.4,
                    rho=0.2,
                )
            snapshot = capture_snapshot(
                dataset,
                tiers=("small", "medium"),
                mesh_families=("simplex",),
            )
            manifest, _ = build_feature_cache(
                snapshot,
                dataset_root=dataset,
                output_root=dataset / "ml",
            )
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                build_canonical_samples(
                    snapshot,
                    manifest,
                    dataset_root=dataset,
                    repo_root=repo,
                )


class GroupedSplitTests(unittest.TestCase):
    def test_grouped_stratified_split_is_deterministic_and_near_target(self) -> None:
        samples = synthetic_samples()
        first, stats = assign_grouped_stratified_splits(samples)
        second, _ = assign_grouped_stratified_splits(samples)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), set(SPLIT_NAMES))
        self.assertAlmostEqual(stats["actual_ratios"]["train"], 0.85, delta=0.01)
        self.assertAlmostEqual(
            stats["actual_ratios"]["validation"], 0.05, delta=0.01
        )
        self.assertAlmostEqual(stats["actual_ratios"]["test"], 0.10, delta=0.01)
        for epsilon in (0.0, 0.8, 2.0):
            present = {
                first[sample["matrix_sha256"]]
                for sample in samples
                if sample["epsilon"] == epsilon
            }
            self.assertEqual(present, set(SPLIT_NAMES))

    def test_incremental_refresh_preserves_existing_assignments(self) -> None:
        initial = synthetic_samples(groups_per_family_level=20)
        existing, _ = assign_grouped_stratified_splits(initial)
        expanded = synthetic_samples(groups_per_family_level=24)
        refreshed, stats = assign_grouped_stratified_splits(
            expanded, previous_assignments=existing
        )
        self.assertTrue(all(refreshed[key] == value for key, value in existing.items()))
        self.assertGreater(stats["new_hash_count"], 0)

    def test_all_rows_for_hash_share_one_split(self) -> None:
        samples = synthetic_samples(groups_per_family_level=30)
        assignments, _ = assign_grouped_stratified_splits(samples)
        observed = {}
        for sample in samples:
            current = assignments[sample["matrix_sha256"]]
            observed.setdefault(sample["matrix_sha256"], current)
            self.assertEqual(observed[sample["matrix_sha256"]], current)


if __name__ == "__main__":
    unittest.main()
