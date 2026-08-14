from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PoolingSpec:
    """Versioned description of the paper-derived matrix view."""

    name: str = "paper_v1"
    view_size: int = 100
    operations: tuple[str, ...] = ("positive_max", "negative_max", "sum")
    reduction: str = "count_average"
    normalization: str = "signed_log1p_maxabs"
    output_dtype: str = "float32"

    def validate(self) -> None:
        if self.view_size < 1:
            raise ValueError("view_size must be positive")
        if self.operations != ("positive_max", "negative_max", "sum"):
            raise ValueError("paper_v1 requires positive_max, negative_max, sum")
        if self.reduction != "count_average":
            raise ValueError("paper_v1 requires count_average reduction")
        if self.normalization != "signed_log1p_maxabs":
            raise ValueError("paper_v1 requires signed_log1p_maxabs normalization")
        if self.output_dtype != "float32":
            raise ValueError("paper_v1 feature artifacts must use float32")

    @property
    def channels(self) -> int:
        return len(self.operations)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["operations"] = list(self.operations)
        return value


PAPER_POOLING_SPEC = PoolingSpec()


def balanced_block_boundaries(size: int, blocks: int) -> np.ndarray:
    """Return Algorithm 3 boundaries, with the larger blocks first."""

    if blocks < 1:
        raise ValueError("blocks must be positive")
    if size < blocks:
        raise ValueError(
            f"matrix dimension {size} is smaller than view size {blocks}"
        )
    quotient, remainder = divmod(size, blocks)
    widths = np.full(blocks, quotient, dtype=np.int64)
    widths[:remainder] += 1
    boundaries = np.empty(blocks + 1, dtype=np.int64)
    boundaries[0] = 0
    np.cumsum(widths, dtype=np.int64, out=boundaries[1:])
    return boundaries


def _validate_csr_arrays(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
) -> None:
    rows, columns = shape
    if rows != columns:
        raise ValueError(f"matrix view requires a square matrix, found {shape}")
    if indptr.ndim != 1 or indptr.shape != (rows + 1,):
        raise ValueError("invalid CSR indptr shape")
    if indices.ndim != 1 or data.ndim != 1 or indices.shape != data.shape:
        raise ValueError("CSR indices and data must be equal-length vectors")
    if indptr.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
        raise ValueError("CSR indices must use integer dtypes")
    if data.dtype.kind != "f":
        raise ValueError("CSR values must use a floating-point dtype")
    if int(indptr[0]) != 0 or int(indptr[-1]) != len(data):
        raise ValueError("CSR indptr endpoints do not match nnz")
    if bool((indptr[1:] < indptr[:-1]).any()):
        raise ValueError("CSR indptr must be monotone")
    if len(indices) and (
        int(indices.min()) < 0 or int(indices.max()) >= columns
    ):
        raise ValueError("CSR column index is out of range")
    if not bool(np.isfinite(data).all()):
        raise ValueError("CSR values contain a non-finite entry")


def _column_blocks(columns: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    # searchsorted reproduces Algorithm 3's quotient/remainder mapping while
    # allocating only one source-row-block-sized temporary.
    return np.searchsorted(boundaries[1:], columns, side="right")


def pool_csr_arrays(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
    spec: PoolingSpec = PAPER_POOLING_SPEC,
) -> np.ndarray:
    """Pool CSR arrays without constructing COO row coordinates."""

    spec.validate()
    _validate_csr_arrays(indptr, indices, data, shape)
    boundaries = balanced_block_boundaries(shape[0], spec.view_size)
    size = spec.view_size
    counts = np.zeros((size, size), dtype=np.int64)
    pooled = np.zeros((spec.channels, size, size), dtype=np.float64)

    for output_row in range(size):
        first_row = int(boundaries[output_row])
        last_row = int(boundaries[output_row + 1])
        start = int(indptr[first_row])
        stop = int(indptr[last_row])
        if start == stop:
            continue

        current_indices = indices[start:stop]
        current_values = data[start:stop].astype(np.float64, copy=False)
        output_columns = _column_blocks(current_indices, boundaries)

        counts[output_row] = np.bincount(
            output_columns, minlength=size
        ).astype(np.int64, copy=False)
        pooled[2, output_row] = np.bincount(
            output_columns, weights=current_values, minlength=size
        )
        np.maximum.at(
            pooled[0, output_row],
            output_columns,
            np.maximum(current_values, 0.0),
        )
        np.maximum.at(
            pooled[1, output_row],
            output_columns,
            np.maximum(-current_values, 0.0),
        )

    nonempty = counts > 0
    for channel in range(spec.channels):
        np.divide(
            pooled[channel],
            counts,
            out=pooled[channel],
            where=nonempty,
        )
        pooled[channel, ~nonempty] = 0.0
        pooled[channel] = np.sign(pooled[channel]) * np.log1p(
            np.abs(pooled[channel])
        )
        maximum = float(np.max(np.abs(pooled[channel])))
        if maximum > 0.0:
            pooled[channel] /= maximum

    result = pooled.astype(np.float32)
    if not bool(np.isfinite(result).all()):
        raise ValueError("pooled view contains a non-finite value")
    return result


def load_and_pool_csr_npz(
    matrix_path: str | Path,
    spec: PoolingSpec = PAPER_POOLING_SPEC,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and pool one AMG-ThetaNet CSR NPZ in a single archive pass."""

    matrix_path = Path(matrix_path)
    with np.load(matrix_path, allow_pickle=False) as archive:
        required = {
            "schema_version",
            "indptr",
            "indices",
            "data",
            "shape",
            "source_sha256",
            "identity_json",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{matrix_path} is missing arrays: {sorted(missing)}")
        if int(archive["schema_version"]) != 1:
            raise ValueError(f"unsupported matrix schema in {matrix_path}")
        shape_array = archive["shape"]
        if shape_array.shape != (2,):
            raise ValueError(f"invalid matrix shape in {matrix_path}")
        shape = tuple(int(value) for value in shape_array)
        indptr = archive["indptr"]
        indices = archive["indices"]
        data = archive["data"]
        source_sha256 = bytes(archive["source_sha256"]).decode("ascii")
        identity = json.loads(bytes(archive["identity_json"]).decode("utf-8"))
        view = pool_csr_arrays(indptr, indices, data, shape, spec)
        nnz = len(data)

    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "source_path": str(matrix_path),
        "matrix_identity": identity,
        "matrix_shape": list(shape),
        "matrix_nnz": nnz,
        "pooling_spec": spec.to_dict(),
        "ordering_contract": "csr_global_dof_order_v1",
    }
    return view, metadata


def write_feature_artifact_atomic(
    destination: str | Path,
    view: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        suffix=".npz",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int64),
            view=np.asarray(view, dtype=np.float32),
            metadata_json=np.frombuffer(
                json.dumps(metadata, sort_keys=True).encode("utf-8"),
                dtype=np.uint8,
            ),
        )
        validate_feature_artifact(
            temporary,
            expected_source_sha256=str(metadata["source_sha256"]),
            expected_spec=PoolingSpec(
                name=str(metadata["pooling_spec"]["name"]),
                view_size=int(metadata["pooling_spec"]["view_size"]),
                operations=tuple(metadata["pooling_spec"]["operations"]),
                reduction=str(metadata["pooling_spec"]["reduction"]),
                normalization=str(metadata["pooling_spec"]["normalization"]),
                output_dtype=str(metadata["pooling_spec"]["output_dtype"]),
            ),
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_feature_artifact(
    path: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected_spec: PoolingSpec = PAPER_POOLING_SPEC,
) -> dict[str, Any]:
    path = Path(path)
    expected_spec.validate()
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema_version", "view", "metadata_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        if int(archive["schema_version"]) != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema in {path}")
        view = archive["view"]
        expected_shape = (
            expected_spec.channels,
            expected_spec.view_size,
            expected_spec.view_size,
        )
        if view.shape != expected_shape or view.dtype != np.dtype("float32"):
            raise ValueError(
                f"invalid feature tensor in {path}: {view.shape} {view.dtype}"
            )
        if not bool(np.isfinite(view).all()):
            raise ValueError(f"non-finite feature value in {path}")
        if len(view) and (
            float(view.min()) < -1.000001 or float(view.max()) > 1.000001
        ):
            raise ValueError(f"feature value outside [-1, 1] in {path}")
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))

    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"feature metadata schema mismatch in {path}")
    if metadata.get("pooling_spec") != expected_spec.to_dict():
        raise ValueError(f"pooling specification mismatch in {path}")
    if (
        expected_source_sha256 is not None
        and metadata.get("source_sha256") != expected_source_sha256
    ):
        raise ValueError(f"source hash mismatch in {path}")
    return metadata
