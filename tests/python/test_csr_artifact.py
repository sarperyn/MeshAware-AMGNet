from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.csr_artifact import (
    PETSC_MATRIX_CLASS_ID,
    convert_petsc_to_csr_directory,
    convert_petsc_to_csr_npz,
    load_scipy_csr_npz,
    read_petsc_matrix_header,
    validate_csr_directory,
    validate_csr_npz,
)


def write_test_matrix(path: Path) -> None:
    row_counts = (2, 1, 2)
    columns = (0, 1, 1, 1, 2)
    values = (4.0, -1.0, 4.0, -1.0, 4.0)
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                ">4i", PETSC_MATRIX_CLASS_ID, 3, 3, len(columns)
            )
        )
        handle.write(struct.pack(">3i", *row_counts))
        handle.write(struct.pack(">5i", *columns))
        handle.write(struct.pack(">5d", *values))


class PetscHeaderTests(unittest.TestCase):
    def test_reads_real_double_32_bit_matrix_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "matrix.petsc"
            write_test_matrix(source)
            header = read_petsc_matrix_header(source)
        self.assertEqual((header.rows, header.columns, header.nnz), (3, 3, 5))
        self.assertEqual(header.integer_bytes, 4)
        self.assertEqual(header.byte_order, "big")
        self.assertEqual(header.expected_file_bytes, 88)

    def test_rejects_truncated_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "matrix.petsc"
            write_test_matrix(source)
            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Unexpected PETSc matrix length"):
                read_petsc_matrix_header(source)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class CsrConversionTests(unittest.TestCase):
    def test_converts_exact_csr_arrays(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "matrix.petsc"
            destination = root / "matrix.csr"
            write_test_matrix(source)
            metadata = convert_petsc_to_csr_directory(
                source,
                destination,
                identity={"matrix_id": "test"},
            )
            validate_csr_directory(destination)
            indptr = np.load(destination / "indptr.npy", allow_pickle=False)
            indices = np.load(destination / "indices.npy", allow_pickle=False)
            data = np.load(destination / "data.npy", allow_pickle=False)

        self.assertEqual(metadata["shape"], [3, 3])
        self.assertEqual(metadata["nnz"], 5)
        np.testing.assert_array_equal(indptr, (0, 2, 3, 5))
        np.testing.assert_array_equal(indices, (0, 1, 1, 1, 2))
        np.testing.assert_allclose(data, (4.0, -1.0, 4.0, -1.0, 4.0))

    def test_converts_exact_compressed_csr_npz(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "matrix.petsc"
            destination = root / "matrix.npz"
            write_test_matrix(source)
            metadata = convert_petsc_to_csr_npz(
                source,
                destination,
                identity={"matrix_id": "test"},
            )
            validated = validate_csr_npz(
                destination, expected_shape=(3, 3), expected_nnz=5
            )
            with np.load(destination, allow_pickle=False) as archive:
                indptr = archive["indptr"]
                indices = archive["indices"]
                data = archive["data"]
            matrix = load_scipy_csr_npz(destination)

        self.assertEqual(metadata["format"], "scipy_csr_npz")
        self.assertEqual(validated["identity"]["matrix_id"], "test")
        np.testing.assert_array_equal(indptr, (0, 2, 3, 5))
        np.testing.assert_array_equal(indices, (0, 1, 1, 1, 2))
        np.testing.assert_allclose(data, (4.0, -1.0, 4.0, -1.0, 4.0))
        np.testing.assert_allclose(
            matrix.toarray(),
            ((4.0, -1.0, 0.0), (0.0, 4.0, 0.0), (0.0, -1.0, 4.0)),
        )


if __name__ == "__main__":
    unittest.main()
