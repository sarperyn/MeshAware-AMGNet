from __future__ import annotations

import json
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import file_sha256

PETSC_MATRIX_CLASS_ID = 1211216
CSR_NPZ_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PetscMatrixHeader:
    rows: int
    columns: int
    nnz: int
    integer_bytes: int
    byte_order: str
    scalar_dtype: str
    header_bytes: int
    row_counts_offset: int
    column_indices_offset: int
    values_offset: int
    expected_file_bytes: int


def _source_integer_dtype(header: PetscMatrixHeader) -> str:
    prefix = ">" if header.byte_order == "big" else "<"
    return f"{prefix}i{header.integer_bytes}"


def _source_scalar_dtype(header: PetscMatrixHeader) -> str:
    prefix = ">" if header.byte_order == "big" else "<"
    if header.scalar_dtype != "float64":
        raise ValueError(f"Unsupported PETSc scalar type: {header.scalar_dtype}")
    return f"{prefix}f8"


def read_petsc_matrix_header(
    source: str | Path, scalar_dtype: str = "float64"
) -> PetscMatrixHeader:
    source = Path(source)
    prefix_by_order = {"big": ">", "little": "<"}
    with source.open("rb") as handle:
        raw = handle.read(32)

    candidates: list[tuple[int, str]] = []
    for integer_bytes, code in ((4, "i"), (8, "q")):
        if len(raw) < integer_bytes:
            continue
        for byte_order, prefix in prefix_by_order.items():
            class_id = struct.unpack(f"{prefix}{code}", raw[:integer_bytes])[0]
            if class_id == PETSC_MATRIX_CLASS_ID:
                candidates.append((integer_bytes, byte_order))
    if len(candidates) != 1:
        raise ValueError(
            f"{source} is not an unambiguous PETSc binary matrix "
            f"(class-id matches={candidates})"
        )

    integer_bytes, byte_order = candidates[0]
    code = "i" if integer_bytes == 4 else "q"
    prefix = prefix_by_order[byte_order]
    header_bytes = 4 * integer_bytes
    if len(raw) < header_bytes:
        raise ValueError(f"Truncated PETSc matrix header in {source}")
    class_id, rows, columns, nnz = struct.unpack(
        f"{prefix}4{code}", raw[:header_bytes]
    )
    if class_id != PETSC_MATRIX_CLASS_ID or min(rows, columns, nnz) < 0:
        raise ValueError(f"Invalid PETSc matrix dimensions in {source}")
    if scalar_dtype != "float64":
        raise ValueError("Only real double PETSc matrices are supported")

    scalar_bytes = 8
    row_counts_offset = header_bytes
    column_indices_offset = row_counts_offset + rows * integer_bytes
    values_offset = column_indices_offset + nnz * integer_bytes
    expected_file_bytes = values_offset + nnz * scalar_bytes
    actual_file_bytes = source.stat().st_size
    if actual_file_bytes != expected_file_bytes:
        raise ValueError(
            f"Unexpected PETSc matrix length for {source}: "
            f"expected {expected_file_bytes}, found {actual_file_bytes}. "
            "The file may be truncated or use a different scalar type."
        )

    return PetscMatrixHeader(
        rows=rows,
        columns=columns,
        nnz=nnz,
        integer_bytes=integer_bytes,
        byte_order=byte_order,
        scalar_dtype=scalar_dtype,
        header_bytes=header_bytes,
        row_counts_offset=row_counts_offset,
        column_indices_offset=column_indices_offset,
        values_offset=values_offset,
        expected_file_bytes=expected_file_bytes,
    )


def _require_numpy():
    try:
        import numpy
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "CSR conversion requires NumPy. Install requirements-data.txt "
            "or use a Python environment that already provides NumPy."
        ) from error
    return numpy


def _copy_memmap_in_chunks(
    source,
    destination,
    *,
    validate_columns: int | None = None,
    validate_finite: bool = False,
    chunk_items: int = 4 * 1024 * 1024,
) -> None:
    np = _require_numpy()
    for start in range(0, len(source), chunk_items):
        stop = min(start + chunk_items, len(source))
        chunk = source[start:stop]
        if validate_columns is not None and len(chunk) and (
            int(chunk.min()) < 0 or int(chunk.max()) >= validate_columns
        ):
            raise ValueError("PETSc matrix contains an out-of-range column index")
        if validate_finite and not bool(np.isfinite(chunk).all()):
            raise ValueError("PETSc matrix contains a non-finite value")
        destination[start:stop] = chunk


def convert_petsc_to_csr_npz(
    source: str | Path,
    destination: str | Path,
    *,
    identity: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one PETSc matrix to a lossless, compressed CSR NPZ file.

    Little-endian temporary memmaps keep conversion memory bounded. The final
    archive replaces its destination only after a complete validation pass.
    """

    np = _require_numpy()
    source = Path(source)
    destination = Path(destination)
    header = read_petsc_matrix_header(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=str(destination.parent)
        )
    )
    try:
        source_integer_dtype = _source_integer_dtype(header)
        row_counts = np.memmap(
            source,
            mode="r",
            dtype=source_integer_dtype,
            offset=header.row_counts_offset,
            shape=(header.rows,),
        )
        source_indices = np.memmap(
            source,
            mode="r",
            dtype=source_integer_dtype,
            offset=header.column_indices_offset,
            shape=(header.nnz,),
        )
        source_data = np.memmap(
            source,
            mode="r",
            dtype=_source_scalar_dtype(header),
            offset=header.values_offset,
            shape=(header.nnz,),
        )

        indptr = np.lib.format.open_memmap(
            temporary / "indptr.npy",
            mode="w+",
            dtype=np.dtype("<i8"),
            shape=(header.rows + 1,),
        )
        indptr[0] = 0
        np.cumsum(row_counts, dtype=np.int64, out=indptr[1:])
        if int(indptr[-1]) != header.nnz:
            raise ValueError(
                "PETSc row nonzero counts do not sum to the header nnz"
            )

        index_dtype = np.dtype("<i4" if header.integer_bytes == 4 else "<i8")
        indices = np.lib.format.open_memmap(
            temporary / "indices.npy",
            mode="w+",
            dtype=index_dtype,
            shape=(header.nnz,),
        )
        _copy_memmap_in_chunks(
            source_indices, indices, validate_columns=header.columns
        )

        data = np.lib.format.open_memmap(
            temporary / "data.npy",
            mode="w+",
            dtype=np.dtype("<f8"),
            shape=(header.nnz,),
        )
        _copy_memmap_in_chunks(source_data, data, validate_finite=True)
        shape = np.asarray((header.rows, header.columns), dtype=np.int64)
        source_sha256 = file_sha256(source, chunk_bytes=8 * 1024 * 1024)
        identity_json = json.dumps(
            identity or {}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        indptr.flush()
        indices.flush()
        data.flush()
        archive_path = temporary / destination.name
        with archive_path.open("wb") as archive:
            np.savez_compressed(
                archive,
                schema_version=np.asarray(
                    CSR_NPZ_SCHEMA_VERSION, dtype=np.int64
                ),
                indptr=indptr,
                indices=indices,
                data=data,
                shape=shape,
                source_sha256=np.frombuffer(
                    source_sha256.encode("ascii"), dtype=np.uint8
                ),
                identity_json=np.frombuffer(identity_json, dtype=np.uint8),
            )

        del indptr, indices, data, row_counts, source_indices, source_data
        metadata = validate_csr_npz(
            archive_path,
            expected_shape=(header.rows, header.columns),
            expected_nnz=header.nnz,
            expected_source_sha256=source_sha256,
        )
        archive_path.replace(destination)
        metadata["path"] = str(destination)
        metadata["bytes"] = destination.stat().st_size
        return metadata
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def validate_csr_npz(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] | None = None,
    expected_nnz: int | None = None,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a compressed CSR artifact and return its embedded metadata."""

    np = _require_numpy()
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
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
            raise ValueError(
                f"Missing arrays in CSR NPZ {path}: {sorted(missing)}"
            )
        schema_version = int(archive["schema_version"])
        if schema_version != CSR_NPZ_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CSR NPZ schema {schema_version} in {path}"
            )

        shape_array = archive["shape"]
        if shape_array.shape != (2,):
            raise ValueError(f"Invalid matrix shape in {path}")
        shape = tuple(int(value) for value in shape_array)
        if min(shape) < 0 or (
            expected_shape is not None and shape != expected_shape
        ):
            raise ValueError(f"Unexpected matrix shape in {path}: {shape}")

        indptr = archive["indptr"]
        if (
            indptr.dtype.kind not in "iu"
            or indptr.shape != (shape[0] + 1,)
            or int(indptr[0]) != 0
        ):
            raise ValueError(f"Invalid CSR row pointer in {path}")
        nnz = int(indptr[-1])
        for start in range(0, len(indptr) - 1, 4 * 1024 * 1024):
            stop = min(start + 4 * 1024 * 1024, len(indptr) - 1)
            if bool((indptr[start + 1 : stop + 1] < indptr[start:stop]).any()):
                raise ValueError(f"Non-monotone CSR row pointer in {path}")
        del indptr
        if expected_nnz is not None and nnz != expected_nnz:
            raise ValueError(f"Unexpected nonzero count in {path}: {nnz}")

        indices = archive["indices"]
        if indices.dtype.kind not in "iu" or indices.shape != (nnz,):
            raise ValueError(f"Invalid CSR column indices in {path}")
        for start in range(0, nnz, 4 * 1024 * 1024):
            chunk = indices[start : start + 4 * 1024 * 1024]
            if len(chunk) and (
                int(chunk.min()) < 0 or int(chunk.max()) >= shape[1]
            ):
                raise ValueError(f"Out-of-range CSR column index in {path}")
        del indices

        data = archive["data"]
        if data.dtype.kind != "f" or data.shape != (nnz,):
            raise ValueError(f"Invalid CSR values in {path}")
        for start in range(0, nnz, 4 * 1024 * 1024):
            if not bool(
                np.isfinite(data[start : start + 4 * 1024 * 1024]).all()
            ):
                raise ValueError(f"Non-finite CSR value in {path}")
        del data

        source_sha256 = bytes(archive["source_sha256"]).decode("ascii")
        if (
            expected_source_sha256 is not None
            and source_sha256 != expected_source_sha256
        ):
            raise ValueError(f"Source checksum mismatch in {path}")
        identity = json.loads(bytes(archive["identity_json"]).decode("utf-8"))

    return {
        "schema_version": schema_version,
        "format": "scipy_csr_npz",
        "shape": list(shape),
        "nnz": nnz,
        "source_sha256": source_sha256,
        "identity": identity,
        "path": str(path),
        "bytes": path.stat().st_size,
    }


def load_scipy_csr_npz(path: str | Path):
    np = _require_numpy()
    try:
        from scipy.sparse import csr_matrix
    except ModuleNotFoundError as error:
        raise RuntimeError("Loading as a SciPy matrix requires SciPy") from error

    path = Path(path)
    validate_csr_npz(path)
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["shape"])
        indptr = archive["indptr"]
        indices = archive["indices"]
        data = archive["data"]
    return csr_matrix((data, indices, indptr), shape=shape, copy=False)
