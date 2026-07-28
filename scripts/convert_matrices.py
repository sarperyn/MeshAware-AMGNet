#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.csr_artifact import (
    convert_petsc_to_csr_directory,
    read_petsc_matrix_header,
    validate_csr_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert each unique PETSc matrix in a dataset to memory-mappable "
            "CSR array directories."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_matrix_inputs(dataset_root: Path) -> list[dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {}
    for record_path in sorted(dataset_root.glob("*/records/*.json")):
        with record_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        matrix_id = str(record["matrix_id"])
        family = str(record["mesh_family"])
        canonical_source = (
            dataset_root / family / "matrices" / f"{matrix_id}.petsc"
        )
        recorded_source = Path(record["matrix_path"])
        candidates = (
            canonical_source,
            recorded_source,
            REPO_ROOT / recorded_source,
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"No PETSc matrix found for {matrix_id}; checked "
                + ", ".join(str(path) for path in candidates)
            )
        identity = {
            "matrix_id": matrix_id,
            "mesh_family": family,
            "level": int(record["level"]),
            "pattern": str(record["pattern"]),
            "epsilon": float(record["epsilon"]),
            "high_region": str(record["high_region"]),
            "dofs": int(record["dofs"]),
            "nnz": int(record["nnz"]),
            "ordering_id": (
                "polydeal_agglomerated_dof_order_v1"
                if family == "polygonal"
                else "deal_ii_active_dof_order_v1"
            ),
        }
        previous = matrices.get(matrix_id)
        current = {"source": source, "identity": identity}
        if previous is not None and previous != current:
            raise ValueError(f"Conflicting records for matrix {matrix_id}")
        matrices[matrix_id] = current
    return [matrices[key] for key in sorted(matrices)]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    inputs = load_matrix_inputs(args.dataset_root)
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise SystemExit(f"No matrix records found under {args.dataset_root}")

    entries = []
    converted = 0
    skipped = 0
    for index, item in enumerate(inputs, start=1):
        source: Path = item["source"]
        identity = item["identity"]
        header = read_petsc_matrix_header(source)
        if header.rows != identity["dofs"] or header.nnz != identity["nnz"]:
            raise ValueError(
                f"Record/PETSc mismatch for {identity['matrix_id']}: "
                f"record dofs/nnz={identity['dofs']}/{identity['nnz']}, "
                f"file={header.rows}/{header.nnz}"
            )
        destination = (
            args.dataset_root
            / identity["mesh_family"]
            / "csr"
            / f"{identity['matrix_id']}.csr"
        )
        if destination.exists() and not args.overwrite:
            metadata = validate_csr_directory(destination)
            skipped += 1
            status = "skipped"
        else:
            print(
                f"[{index}/{len(inputs)}] {identity['matrix_id']}",
                flush=True,
            )
            metadata = convert_petsc_to_csr_directory(
                source,
                destination,
                identity=identity,
                overwrite=args.overwrite,
            )
            converted += 1
            status = "converted"
        entries.append(
            {
                "matrix_id": identity["matrix_id"],
                "mesh_family": identity["mesh_family"],
                "source_path": str(source),
                "csr_path": str(destination),
                "shape": metadata["shape"],
                "nnz": metadata["nnz"],
                "source_sha256": metadata["source"]["sha256"],
                "status": status,
            }
        )

    manifest = {
        "schema_version": 1,
        "format": "meshaware_csr_conversion_manifest",
        "dataset_root": str(args.dataset_root),
        "matrix_count": len(entries),
        "converted": converted,
        "skipped": skipped,
        "entries": entries,
    }
    write_manifest(args.dataset_root / "csr_manifest.json", manifest)
    print(
        f"done: converted={converted}, skipped={skipped}, "
        f"manifest={args.dataset_root / 'csr_manifest.json'}"
    )


if __name__ == "__main__":
    main()
