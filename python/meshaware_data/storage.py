from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .schema import ExperimentConfig

PETSC_INDEX_BYTES = 4
SCALAR_BYTES = 8
ESTIMATED_RECORD_BYTES = 1536
# The real small-dataset independent CSR NPZ files used 6.0% of uncompressed
# CSR bytes. Ten percent is a deliberately conservative planning factor until
# a finest-level pilot provides a tier-specific measurement.
CSR_NPZ_COMPRESSION_RATIO_PILOT = 0.10


def _family_size_model(mesh_family: str, level: int) -> dict[str, int | str]:
    subdivisions = 1 << (level + 1)
    background_cells = subdivisions * subdivisions
    if mesh_family == "quadrilateral":
        cells = background_cells
        dofs = (subdivisions + 1) ** 2
        nnz = 9 * dofs
        model = "structured_q1_nine_point_upper"
    elif mesh_family == "simplex":
        cells = 2 * background_cells
        dofs = (subdivisions + 1) ** 2
        nnz = 9 * dofs
        model = "structured_p1_conservative_nine_point_upper"
    elif mesh_family == "simplex-dg":
        cells = 2 * background_cells
        dofs = 3 * cells
        nnz = 12 * dofs
        model = "structured_p1_sipg_conservative_12_entries_per_dof"
    elif mesh_family == "polygonal":
        cells = background_cells // 2
        dofs = 3 * cells
        nnz = 21 * dofs
        model = "polydeal_dg_conservative_21_entries_per_dof"
    else:
        raise ValueError(f"Unknown mesh family: {mesh_family}")
    return {
        "model": model,
        "subdivisions_per_axis": subdivisions,
        "cells": cells,
        "background_cells": background_cells,
        "dofs_per_matrix": dofs,
        "nnz_per_matrix_upper": nnz,
    }


def estimate_experiment_storage(config: ExperimentConfig) -> dict[str, Any]:
    coefficient_cases = len(config.patterns) * len(config.epsilons)
    theta_count = len(config.theta_values)
    breakdown: list[dict[str, Any]] = []
    totals = {
        "matrices": 0,
        "trial_records": 0,
        "petsc_matrix_bytes": 0,
        "csr_matrix_bytes": 0,
        "npz_matrix_bytes_estimate": 0,
        "record_bytes": 0,
        "maximum_solver_ram_per_matrix_heuristic": 0,
        "maximum_conversion_staging_bytes": 0,
    }
    largest_solver_case: dict[str, Any] | None = None

    for mesh_family in config.mesh_families:
        for level in config.levels:
            model = _family_size_model(mesh_family, level)
            matrices = coefficient_cases
            repeats = config.repeats_by_level.get(level, config.repeats)
            trial_records = coefficient_cases * theta_count * repeats
            rows = int(model["dofs_per_matrix"])
            nnz = int(model["nnz_per_matrix_upper"])
            petsc_bytes_per_matrix = (
                4 * PETSC_INDEX_BYTES
                + rows * PETSC_INDEX_BYTES
                + nnz * (PETSC_INDEX_BYTES + SCALAR_BYTES)
            )
            csr_bytes_per_matrix = (
                (rows + 1) * 8
                + nnz * (PETSC_INDEX_BYTES + SCALAR_BYTES)
                + 4096
            )
            npz_bytes_per_matrix_estimate = int(
                csr_bytes_per_matrix * CSR_NPZ_COMPRESSION_RATIO_PILOT
            )
            row = {
                "mesh_family": mesh_family,
                "level": level,
                **model,
                "matrices": matrices,
                "theta_values": theta_count,
                "repeats": repeats,
                "trial_records": trial_records,
                "petsc_bytes_per_matrix_upper": petsc_bytes_per_matrix,
                "csr_bytes_per_matrix_upper": csr_bytes_per_matrix,
                "npz_bytes_per_matrix_estimate": (
                    npz_bytes_per_matrix_estimate
                ),
                "petsc_bytes_all_matrices_upper": matrices
                * petsc_bytes_per_matrix,
                "csr_bytes_all_matrices_upper": matrices * csr_bytes_per_matrix,
                "npz_bytes_all_matrices_estimate": (
                    matrices * npz_bytes_per_matrix_estimate
                ),
                "solver_ram_per_matrix_heuristic": 4 * csr_bytes_per_matrix,
                "conversion_staging_bytes": (
                    petsc_bytes_per_matrix
                    + csr_bytes_per_matrix
                    + npz_bytes_per_matrix_estimate
                ),
            }
            breakdown.append(row)
            totals["matrices"] += matrices
            totals["trial_records"] += trial_records
            totals["petsc_matrix_bytes"] += row[
                "petsc_bytes_all_matrices_upper"
            ]
            totals["csr_matrix_bytes"] += row["csr_bytes_all_matrices_upper"]
            totals["npz_matrix_bytes_estimate"] += row[
                "npz_bytes_all_matrices_estimate"
            ]
            totals["maximum_conversion_staging_bytes"] = max(
                totals["maximum_conversion_staging_bytes"],
                row["conversion_staging_bytes"],
            )

    totals["record_bytes"] = (
        totals["trial_records"] * ESTIMATED_RECORD_BYTES
    )
    totals["petsc_plus_records_bytes"] = (
        totals["petsc_matrix_bytes"] + totals["record_bytes"]
    )
    totals["petsc_plus_csr_plus_records_bytes"] = (
        totals["petsc_matrix_bytes"]
        + totals["csr_matrix_bytes"]
        + totals["record_bytes"]
    )
    if config.matrix_format == "scipy_csr_npz":
        totals["retained_matrix_bytes"] = totals[
            "npz_matrix_bytes_estimate"
        ]
    else:
        totals["retained_matrix_bytes"] = totals["petsc_matrix_bytes"]
    totals["retained_total_bytes"] = (
        totals["retained_matrix_bytes"] + totals["record_bytes"]
    )
    for row in breakdown:
        candidate = int(row["solver_ram_per_matrix_heuristic"])
        if candidate > totals["maximum_solver_ram_per_matrix_heuristic"]:
            totals["maximum_solver_ram_per_matrix_heuristic"] = candidate
            largest_solver_case = {
                "mesh_family": row["mesh_family"],
                "level": row["level"],
                "dofs_per_matrix": row["dofs_per_matrix"],
                "nnz_per_matrix_upper": row["nnz_per_matrix_upper"],
            }
    warnings = []
    if totals["trial_records"] >= 100_000:
        warnings.append(
            "The raw one-JSON-per-trial layout creates at least 100,000 files; "
            "compact completed records before production-scale training."
        )
    return {
        "schema_version": 1,
        "estimate_kind": "conservative_structural_model",
        "config": config.name,
        "assumptions": {
            "petsc_index_bytes": PETSC_INDEX_BYTES,
            "scalar_bytes": SCALAR_BYTES,
            "estimated_json_record_bytes": ESTIMATED_RECORD_BYTES,
            "matrix_format": config.matrix_format,
            "csr_npz_compression_ratio_pilot": (
                CSR_NPZ_COMPRESSION_RATIO_PILOT
            ),
            "petsc_is_temporary_for_npz": (
                config.matrix_format == "scipy_csr_npz"
            ),
            "solver_ram_multiplier_is_heuristic": 4,
        },
        "totals": totals,
        "largest_solver_case": largest_solver_case,
        "breakdown": breakdown,
        "warnings": warnings,
    }


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def inventory_existing_dataset(dataset_root: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    record_paths = list(dataset_root.glob("*/records/*.json"))
    matrix_paths = list(dataset_root.glob("*/matrices/*.petsc"))
    npz_paths = list(dataset_root.glob("*/matrices/*.npz"))
    csr_paths = [path for path in dataset_root.glob("*/csr/*.csr") if path.is_dir()]
    return {
        "dataset_root": str(dataset_root),
        "record_files": len(record_paths),
        "record_bytes": sum(path.stat().st_size for path in record_paths),
        "petsc_matrix_files": len(matrix_paths),
        "petsc_matrix_bytes": sum(path.stat().st_size for path in matrix_paths),
        "npz_matrix_files": len(npz_paths),
        "npz_matrix_bytes": sum(path.stat().st_size for path in npz_paths),
        "csr_artifacts": len(csr_paths),
        "csr_artifact_bytes": sum(_tree_bytes(path) for path in csr_paths),
    }


def add_disk_capacity(
    report: dict[str, Any], target: str | Path
) -> dict[str, Any]:
    target = Path(target)
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    capacity = {
        "probe_path": str(probe),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
    report = json.loads(json.dumps(report))
    report["disk"] = capacity
    required = int(report["totals"]["retained_total_bytes"])
    report["disk"]["estimated_retained_fraction_of_free"] = (
        required / usage.free if usage.free else None
    )
    if required > usage.free:
        report["warnings"].append(
            "Estimated retained matrix + record storage exceeds currently free disk."
        )
    staging = int(report["totals"]["maximum_conversion_staging_bytes"])
    if staging > usage.free:
        report["warnings"].append(
            "One-matrix NPZ conversion staging exceeds currently free disk."
        )
    try:
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        physical_memory = None
    if physical_memory is not None:
        estimated_peak = int(
            report["totals"]["maximum_solver_ram_per_matrix_heuristic"]
        )
        report["memory"] = {
            "physical_bytes": physical_memory,
            "maximum_solver_ram_per_matrix_heuristic": estimated_peak,
            "estimated_peak_fraction_of_physical": (
                estimated_peak / physical_memory if physical_memory else None
            ),
        }
        if estimated_peak > 0.6 * physical_memory:
            report["warnings"].append(
                "The largest matrix's heuristic PETSc+AMG memory exceeds 60% "
                "of physical RAM; run that level only after a measured pilot."
            )
    return report


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")
