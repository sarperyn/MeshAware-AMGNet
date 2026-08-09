from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.dataset import (
    GroupedRhoDataset,
    HashGroupBatchSampler,
    MatrixViewCache,
    assert_disjoint_hashes,
    collate_matrix_groups,
)
from meshaware_ml.model import PaperCNNConfig, PaperRhoCNN
from meshaware_ml.pooling import (
    FEATURE_SCHEMA_VERSION,
    PAPER_POOLING_SPEC,
    write_feature_artifact_atomic,
)
from meshaware_ml.training import RunConfig, TrainingConfig, train


def write_feature(dataset_root: Path, source_hash: str, value: float) -> str:
    relative = f"ml/features/paper_v1/{source_hash}.npz"
    path = dataset_root / relative
    view = np.full((3, 100, 100), value, dtype=np.float32)
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_sha256": source_hash,
        "source_path": f"synthetic/{source_hash}.npz",
        "matrix_identity": {"matrix_id": source_hash[:8]},
        "matrix_shape": [100, 100],
        "matrix_nnz": 100,
        "pooling_spec": PAPER_POOLING_SPEC.to_dict(),
        "ordering_contract": "csr_global_dof_order_v1",
    }
    write_feature_artifact_atomic(path, view, metadata)
    return relative


def sample(
    *,
    sample_index: int,
    source_hash: str,
    feature_path: str,
    split: str,
    family: str,
    theta: float,
    rho: float,
) -> dict:
    return {
        "schema_version": 1,
        "sample_id": f"sample-{sample_index}",
        "matrix_id": f"matrix-{source_hash[:8]}",
        "matrix_sha256": source_hash,
        "feature_path": feature_path,
        "mesh_family": family,
        "level": 3,
        "h_nominal": 0.125,
        "theta": theta,
        "rho_mean": rho,
        "rho_std": 0.0,
        "repeat_count": 1,
        "pattern": "vertical_split",
        "epsilon": 0.0,
        "high_region": "white",
        "source_tiers": ["small"],
        "source_records": [f"record-{sample_index}.json"],
        "split": split,
    }


def write_synthetic_index(root: Path) -> tuple[Path, Path, Path]:
    dataset_root = root / "datasets"
    index_root = dataset_root / "ml" / "index" / "paper_v1"
    index_root.mkdir(parents=True)
    hashes = {
        "train_a": "a" * 64,
        "train_b": "b" * 64,
        "validation": "c" * 64,
        "test": "d" * 64,
    }
    features = {
        name: write_feature(dataset_root, source_hash, (index + 1) / 10)
        for index, (name, source_hash) in enumerate(hashes.items())
    }
    rows = [
        sample(
            sample_index=0,
            source_hash=hashes["train_a"],
            feature_path=features["train_a"],
            split="train",
            family="simplex",
            theta=0.2,
            rho=0.25,
        ),
        sample(
            sample_index=1,
            source_hash=hashes["train_a"],
            feature_path=features["train_a"],
            split="train",
            family="simplex",
            theta=0.4,
            rho=0.35,
        ),
        sample(
            sample_index=2,
            source_hash=hashes["train_b"],
            feature_path=features["train_b"],
            split="train",
            family="polygonal",
            theta=0.2,
            rho=0.45,
        ),
        sample(
            sample_index=3,
            source_hash=hashes["train_b"],
            feature_path=features["train_b"],
            split="train",
            family="polygonal",
            theta=0.4,
            rho=0.55,
        ),
        sample(
            sample_index=4,
            source_hash=hashes["validation"],
            feature_path=features["validation"],
            split="validation",
            family="simplex",
            theta=0.3,
            rho=0.4,
        ),
        sample(
            sample_index=5,
            source_hash=hashes["test"],
            feature_path=features["test"],
            split="test",
            family="polygonal",
            theta=0.3,
            rho=0.5,
        ),
    ]
    samples_path = index_root / "samples.jsonl"
    samples_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    splits_path = index_root / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_id": "synthetic-snapshot",
                "seed": 2026,
                "ratios": {
                    "train": 0.85,
                    "validation": 0.05,
                    "test": 0.1,
                },
                "assignments": {
                    hashes["train_a"]: "train",
                    hashes["train_b"]: "train",
                    hashes["validation"]: "validation",
                    hashes["test"]: "test",
                },
                "statistics": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset_root, samples_path, splits_path


def tiny_run_config(
    root: Path,
    dataset_root: Path,
    samples_path: Path,
    splits_path: Path,
    *,
    max_epochs: int = 2,
) -> RunConfig:
    return RunConfig(
        samples_path=samples_path,
        splits_path=splits_path,
        dataset_root=dataset_root,
        output_dir=root / "weights" / "test-run",
        report_path=root / "reports" / "phase3.md",
        device="cpu",
        model=PaperCNNConfig(
            conv_width=2,
            embedding_width=4,
            dense_width=4,
            dense_depth=2,
            dropout=0.0,
        ),
        training=TrainingConfig(
            seed=2026,
            batch_size=2,
            max_epochs=max_epochs,
            learning_rate=1.0e-3,
            early_stopping_patience=10,
            num_workers=0,
            cache_capacity=8,
            deterministic=True,
            torch_threads=1,
        ),
    )


class DatasetTests(unittest.TestCase):
    def test_grouped_dataset_loads_scalars_and_reuses_matrix_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, samples_path, _ = write_synthetic_index(root)
            cache = MatrixViewCache(dataset_root, capacity=8)
            dataset = GroupedRhoDataset(
                samples_path,
                dataset_root,
                splits=("train",),
                cache=cache,
            )
            first = dataset[0]
            second = dataset[1]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.sample_count, 4)
            self.assertEqual(tuple(first["view"].shape), (3, 100, 100))
            self.assertTrue(
                torch.equal(
                    first["scalars"],
                    torch.tensor([[3.0, 0.2], [3.0, 0.4]]),
                )
            )
            self.assertIs(first["view"], dataset[0]["view"])
            self.assertIsNot(first["view"], second["view"])
            self.assertEqual(len(cache), 2)

    def test_hash_group_batching_computes_one_view_for_multiple_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, samples_path, _ = write_synthetic_index(root)
            dataset = GroupedRhoDataset(
                samples_path, dataset_root, splits=("train",)
            )
            self.assertEqual(dataset.sample_count, 4)
            self.assertEqual(dataset.group_sizes, [2, 2])
            batch = collate_matrix_groups([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["view"].shape), (2, 3, 100, 100))
            self.assertEqual(tuple(batch["scalars"].shape), (4, 2))
            self.assertTrue(
                torch.equal(
                    batch["view_indices"], torch.tensor([0, 0, 1, 1])
                )
            )
            sampler = HashGroupBatchSampler(
                [2, 2, 2], sample_batch_size=4, shuffle=False
            )
            self.assertEqual(list(sampler), [[0, 1], [2]])

    def test_train_validation_hash_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, samples_path, _ = write_synthetic_index(root)
            rows = [
                json.loads(line)
                for line in samples_path.read_text().splitlines()
            ]
            rows[4]["matrix_sha256"] = rows[0]["matrix_sha256"]
            rows[4]["feature_path"] = rows[0]["feature_path"]
            samples_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            train_data = GroupedRhoDataset(
                samples_path, dataset_root, splits=("train",)
            )
            validation_data = GroupedRhoDataset(
                samples_path, dataset_root, splits=("validation",)
            )
            with self.assertRaisesRegex(ValueError, "leakage"):
                assert_disjoint_hashes(train_data, validation_data)


class ModelTests(unittest.TestCase):
    def test_paper_model_shape_and_backward(self) -> None:
        model = PaperRhoCNN()
        view = torch.randn(2, 3, 100, 100)
        scalars = torch.tensor([[3.0, 0.2], [5.0, 0.4]])
        output = model(view, scalars)
        self.assertEqual(tuple(output.shape), (2,))
        output.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

    def test_grouped_forward_reuses_embeddings(self) -> None:
        model = PaperRhoCNN(
            PaperCNNConfig(
                conv_width=2,
                embedding_width=4,
                dense_width=4,
                dense_depth=2,
                dropout=0.0,
            )
        )
        output = model(
            torch.randn(2, 3, 100, 100),
            torch.tensor(
                [[3.0, 0.2], [3.0, 0.4], [5.0, 0.2], [5.0, 0.4]]
            ),
            torch.tensor([0, 0, 1, 1]),
        )
        self.assertEqual(tuple(output.shape), (4,))


class TrainingTests(unittest.TestCase):
    def test_training_writes_atomic_artifacts_without_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, samples_path, splits_path = write_synthetic_index(root)
            config = tiny_run_config(
                root, dataset_root, samples_path, splits_path
            )
            summary = train(config)
            self.assertEqual(summary["train_samples"], 4)
            self.assertEqual(summary["validation_samples"], 1)
            self.assertEqual(summary["test_samples_evaluated"], 0)
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertEqual(summary["max_train_batch_samples"], 2)
            self.assertEqual(
                summary["validation_diagnostics"]["samples"], 1
            )
            self.assertEqual(
                set(
                    summary["validation_diagnostics"][
                        "by_mesh_family"
                    ]
                ),
                {"simplex"},
            )
            self.assertTrue((config.output_dir / "best.pt").is_file())
            self.assertTrue((config.output_dir / "latest.pt").is_file())
            self.assertTrue(config.report_path.is_file())
            with (config.output_dir / "history.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            leftovers = list(config.output_dir.glob("*.tmp"))
            self.assertEqual(leftovers, [])

    def test_resume_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, samples_path, splits_path = write_synthetic_index(root)
            resumed = tiny_run_config(
                root, dataset_root, samples_path, splits_path, max_epochs=3
            )
            train(resumed, epoch_limit=1)
            resumed_summary = train(resumed, resume=True)

            uninterrupted = RunConfig(
                **{
                    **resumed.__dict__,
                    "output_dir": root / "weights" / "uninterrupted",
                    "report_path": root / "reports" / "uninterrupted.md",
                }
            )
            uninterrupted_summary = train(uninterrupted)
            self.assertEqual(
                resumed_summary["best_epoch"],
                uninterrupted_summary["best_epoch"],
            )
            self.assertTrue(
                math.isclose(
                    resumed_summary["best_metrics"]["validation_mse"],
                    uninterrupted_summary["best_metrics"]["validation_mse"],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
            resumed_checkpoint = torch.load(
                resumed.output_dir / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            uninterrupted_checkpoint = torch.load(
                uninterrupted.output_dir / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, tensor in resumed_checkpoint["model_state"].items():
                self.assertTrue(
                    torch.equal(
                        tensor,
                        uninterrupted_checkpoint["model_state"][name],
                    ),
                    name,
                )

            before = (resumed.output_dir / "latest.pt").read_bytes()
            completed = train(resumed, resume=True)
            after = (resumed.output_dir / "latest.pt").read_bytes()
            self.assertEqual(completed["epochs_completed"], 3)
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
