from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .pooling import PAPER_POOLING_SPEC, validate_feature_artifact

INDEX_SCHEMA_VERSION = 1
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


def load_index_rows(
    path: str | Path,
    *,
    splits: Iterable[str],
) -> list[dict[str, Any]]:
    requested = frozenset(splits)
    if not requested or not requested <= ALLOWED_SPLITS:
        raise ValueError(
            f"splits must be a non-empty subset of {sorted(ALLOWED_SPLITS)}"
        )
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            if split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"invalid split at line {line_number}: {split!r}"
                )
            if split not in requested:
                continue
            if row.get("schema_version") != INDEX_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported sample schema at line {line_number}"
                )
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen_ids:
                raise ValueError(
                    f"missing or duplicate sample_id at line {line_number}"
                )
            seen_ids.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"index contains no samples for {sorted(requested)}")
    return rows


class MatrixViewCache:
    """Bounded process-local cache for repeated theta views of one matrix."""

    def __init__(self, dataset_root: str | Path, *, capacity: int = 1024):
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.dataset_root = Path(dataset_root).resolve()
        self.capacity = capacity
        self._views: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _resolve_feature(self, relative_path: str) -> Path:
        path = (self.dataset_root / relative_path).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"feature path escapes dataset root: {relative_path}"
            ) from exc
        return path

    def get(self, relative_path: str, source_sha256: str) -> torch.Tensor:
        key = f"{source_sha256}:{relative_path}"
        cached = self._views.get(key)
        if cached is not None:
            self._views.move_to_end(key)
            return cached

        path = self._resolve_feature(relative_path)
        metadata = validate_feature_artifact(
            path,
            expected_source_sha256=source_sha256,
            expected_spec=PAPER_POOLING_SPEC,
        )
        if metadata.get("source_sha256") != source_sha256:
            raise ValueError(f"source hash mismatch in feature {path}")
        with np.load(path, allow_pickle=False) as archive:
            array = np.array(archive["view"], dtype=np.float32, copy=True)
        tensor = torch.from_numpy(array)
        self._views[key] = tensor
        if len(self._views) > self.capacity:
            self._views.popitem(last=False)
        return tensor

    def __len__(self) -> int:
        return len(self._views)


def _validate_rows(
    rows: list[dict[str, Any]], splits: frozenset[str]
) -> None:
    required = {
        "sample_id",
        "matrix_id",
        "matrix_sha256",
        "feature_path",
        "mesh_family",
        "h_nominal",
        "theta",
        "rho_mean",
        "split",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"sample {index} is missing fields: {sorted(missing)}"
            )
        if row["split"] not in splits:
            raise ValueError("sample escaped requested split filter")
        if row["mesh_family"] not in {"simplex", "polygonal"}:
            raise ValueError(
                f"unsupported mesh family: {row['mesh_family']!r}"
            )
        source_hash = str(row["matrix_sha256"])
        if len(source_hash) != 64:
            raise ValueError(f"invalid matrix SHA-256: {source_hash!r}")
        values = [
            float(row["h_nominal"]),
            float(row["theta"]),
            float(row["rho_mean"]),
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite sample values for {row['sample_id']}")
        if values[0] <= 0.0:
            raise ValueError(f"h_nominal must be positive for {row['sample_id']}")


class GroupedRhoDataset(Dataset[dict[str, Any]]):
    """One matrix view with every selected theta target for that source hash."""

    def __init__(
        self,
        samples_path: str | Path,
        dataset_root: str | Path,
        *,
        splits: Sequence[str],
        cache: MatrixViewCache | None = None,
    ):
        self.samples_path = Path(samples_path).resolve()
        self.splits = frozenset(splits)
        self.cache = (
            cache if cache is not None else MatrixViewCache(dataset_root)
        )
        rows = load_index_rows(self.samples_path, splits=splits)
        grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for row in rows:
            grouped.setdefault(str(row["matrix_sha256"]), []).append(row)
        self.groups = [
            sorted(group, key=lambda row: (float(row["theta"]), row["sample_id"]))
            for _, group in sorted(grouped.items())
        ]
        self.rows = [row for group in self.groups for row in group]
        _validate_rows(self.rows, self.splits)
        self._validate_groups()

    def _validate_groups(self) -> None:
        for group in self.groups:
            source_hashes = {str(row["matrix_sha256"]) for row in group}
            feature_paths = {str(row["feature_path"]) for row in group}
            if len(source_hashes) != 1 or len(feature_paths) != 1:
                raise ValueError(
                    "one source-hash group refers to multiple matrices or views"
                )

    @property
    def sample_count(self) -> int:
        return len(self.rows)

    @property
    def group_sizes(self) -> list[int]:
        return [len(group) for group in self.groups]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        first = group[0]
        view = self.cache.get(
            str(first["feature_path"]), str(first["matrix_sha256"])
        )
        scalars = torch.tensor(
            [
                [
                    -math.log2(float(row["h_nominal"])),
                    float(row["theta"]),
                ]
                for row in group
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [float(row["rho_mean"]) for row in group], dtype=torch.float32
        )
        return {
            "view": view,
            "scalars": scalars,
            "target": target,
            "sample_id": [str(row["sample_id"]) for row in group],
            "matrix_id": [str(row["matrix_id"]) for row in group],
            "matrix_sha256": str(first["matrix_sha256"]),
            "mesh_family": [str(row["mesh_family"]) for row in group],
        }


def collate_matrix_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        raise ValueError("cannot collate an empty matrix-group batch")
    views = torch.stack([group["view"] for group in groups])
    scalars = torch.cat([group["scalars"] for group in groups])
    target = torch.cat([group["target"] for group in groups])
    view_indices = torch.cat(
        [
            torch.full(
                (len(group["target"]),), index, dtype=torch.int64
            )
            for index, group in enumerate(groups)
        ]
    )
    return {
        "view": views,
        "view_indices": view_indices,
        "scalars": scalars,
        "target": target,
        "sample_id": [
            sample_id
            for group in groups
            for sample_id in group["sample_id"]
        ],
        "matrix_id": [
            matrix_id
            for group in groups
            for matrix_id in group["matrix_id"]
        ],
        "matrix_sha256": [
            str(group["matrix_sha256"]) for group in groups
        ],
        "mesh_family": [
            family for group in groups for family in group["mesh_family"]
        ],
    }


class HashGroupBatchSampler:
    """Shuffle intact hash groups around a target mini-batch size."""

    def __init__(
        self,
        group_sizes: Sequence[int],
        *,
        sample_batch_size: int,
        generator: torch.Generator | None = None,
        shuffle: bool,
    ):
        if sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive")
        if not group_sizes or any(size <= 0 for size in group_sizes):
            raise ValueError("group_sizes must contain positive values")
        self.group_sizes = list(group_sizes)
        self.sample_batch_size = sample_batch_size
        self.generator = generator
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(
                len(self.group_sizes), generator=self.generator
            ).tolist()
        else:
            order = list(range(len(self.group_sizes)))
        batch: list[int] = []
        samples = 0
        for index in order:
            size = self.group_sizes[index]
            # Never split a source hash just to meet the target size. Rare
            # ambiguous hashes can therefore form one oversized batch.
            if batch and samples + size > self.sample_batch_size:
                yield batch
                batch = []
                samples = 0
            batch.append(index)
            samples += size
            if size >= self.sample_batch_size:
                yield batch
                batch = []
                samples = 0
        if batch:
            yield batch

    def __len__(self) -> int:
        return len(self.group_sizes)


def assert_disjoint_hashes(*datasets: GroupedRhoDataset) -> None:
    owners: dict[str, int] = {}
    for dataset_index, dataset in enumerate(datasets):
        for row in dataset.rows:
            source_hash = str(row["matrix_sha256"])
            previous = owners.setdefault(source_hash, dataset_index)
            if previous != dataset_index:
                raise ValueError(
                    "matrix-hash leakage between dataset partitions: "
                    f"{source_hash}"
                )
