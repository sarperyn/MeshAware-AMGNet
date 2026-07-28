#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.reporting import build_family_reports
from meshaware_data.csr_artifact import (
    convert_petsc_to_csr_npz,
    read_petsc_matrix_header,
    validate_csr_npz,
)
from meshaware_data.schema import ExperimentConfig, load_experiment_config


EXECUTABLES = {
    "quadrilateral": "meshaware_diffusion_dealii",
    "simplex": "meshaware_diffusion_dealii",
    "polygonal": "meshaware_diffusion_polydeal",
}


def slug_float(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


def matrix_id(trial: dict[str, Any], high_region: str) -> str:
    family = {
        "quadrilateral": "quad",
        "simplex": "simplex",
        "polygonal": "poly",
    }[trial["mesh_family"]]
    return (
        f"{family}_l{trial['level']}_{trial['pattern']}_"
        f"e{slug_float(trial['epsilon'])}_high_{high_region}"
    )


def sample_id(trial: dict[str, Any], high_region: str) -> str:
    return (
        f"{matrix_id(trial, high_region)}_"
        f"theta_{slug_float(trial['theta'])}_repeat_{trial['repeat']}"
    )


def command_for(
    executable: Path,
    config: ExperimentConfig,
    trial: dict[str, Any],
    record_path: Path,
    matrix_path: Path,
    write_matrix: bool,
) -> list[str]:
    command = [
        str(executable),
        "--mesh-family",
        trial["mesh_family"],
        "--level",
        str(trial["level"]),
        "--epsilon",
        str(trial["epsilon"]),
        "--pattern",
        trial["pattern"],
        "--high-region",
        config.high_region,
        "--theta",
        str(trial["theta"]),
        "--rtol",
        str(config.relative_tolerance),
        "--atol",
        str(config.absolute_tolerance),
        "--max-iterations",
        str(config.maximum_iterations),
        "--repeat",
        str(trial["repeat"]),
        "--record",
        str(record_path),
        "--matrix",
        str(matrix_path),
    ]
    if not write_matrix:
        command.append("--skip-matrix-write")
    return command


def group_trials_by_matrix(
    trials: Iterable[dict[str, Any]], high_region: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        current_matrix_id = matrix_id(trial, high_region)
        groups.setdefault(current_matrix_id, []).append(trial)
    return list(groups.items())


def command_for_batch(
    executable: Path,
    config: ExperimentConfig,
    representative: dict[str, Any],
    record_directory: Path,
    matrix_path: Path,
    theta_values: Iterable[float],
    repeats: int,
    write_matrix: bool,
    skip_existing_records: bool,
) -> list[str]:
    command = [
        str(executable),
        "--mesh-family",
        representative["mesh_family"],
        "--level",
        str(representative["level"]),
        "--epsilon",
        str(representative["epsilon"]),
        "--pattern",
        representative["pattern"],
        "--high-region",
        config.high_region,
        "--theta-values",
        ",".join(str(theta) for theta in theta_values),
        "--rtol",
        str(config.relative_tolerance),
        "--atol",
        str(config.absolute_tolerance),
        "--max-iterations",
        str(config.maximum_iterations),
        "--repeats",
        str(repeats),
        "--warmup-runs",
        str(config.warmup_runs),
        "--record-dir",
        str(record_directory),
        "--matrix",
        str(matrix_path),
    ]
    if not write_matrix:
        command.append("--skip-matrix-write")
    if skip_existing_records:
        command.append("--skip-existing-records")
    return command


def run_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("OMPI_MCA_btl", "self")
    return environment


def require_executable(build_dir: Path, family: str) -> Path:
    executable = build_dir / EXECUTABLES[family]
    if not executable.is_file():
        raise SystemExit(
            f"Missing executable {executable}. Configure/build with deal.II first."
        )
    return executable


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def update_record_matrix_references(
    record_paths: Iterable[Path], matrix_path: Path
) -> None:
    for record_path in record_paths:
        with record_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        record["matrix_format"] = "scipy_csr_npz"
        record["matrix_path"] = str(matrix_path)
        write_json_atomic(record_path, record)


def finalize_npz_matrix(
    petsc_path: Path,
    npz_path: Path,
    record_paths: list[Path],
    *,
    force_from_source: bool,
) -> None:
    if not record_paths:
        raise ValueError("Cannot finalize a matrix without trial records")
    with record_paths[0].open(encoding="utf-8") as handle:
        record = json.load(handle)
    expected_shape = (int(record["dofs"]), int(record["dofs"]))
    expected_nnz = int(record["nnz"])
    identity = {
        "matrix_id": str(record["matrix_id"]),
        "mesh_family": str(record["mesh_family"]),
        "level": int(record["level"]),
        "pattern": str(record["pattern"]),
        "epsilon": float(record["epsilon"]),
        "high_region": str(record["high_region"]),
        "dofs": expected_shape[0],
        "nnz": expected_nnz,
    }

    metadata = None
    if npz_path.exists() and not force_from_source:
        try:
            metadata = validate_csr_npz(
                npz_path,
                expected_shape=expected_shape,
                expected_nnz=expected_nnz,
            )
        except (OSError, ValueError):
            if not petsc_path.is_file():
                raise

    if metadata is None:
        header = read_petsc_matrix_header(petsc_path)
        if (header.rows, header.columns) != expected_shape:
            raise ValueError(
                f"Record/PETSc shape mismatch for {identity['matrix_id']}"
            )
        if header.nnz != expected_nnz:
            raise ValueError(
                f"Record/PETSc nnz mismatch for {identity['matrix_id']}"
            )
        metadata = convert_petsc_to_csr_npz(
            petsc_path,
            npz_path,
            identity=identity,
            overwrite=npz_path.exists(),
        )
    if metadata.get("identity", {}).get("matrix_id") != identity["matrix_id"]:
        raise ValueError(
            f"NPZ identity mismatch for {identity['matrix_id']}: "
            f"{metadata.get('identity', {}).get('matrix_id')}"
        )

    update_record_matrix_references(record_paths, npz_path)
    for record_path in record_paths:
        with record_path.open(encoding="utf-8") as handle:
            updated = json.load(handle)
        if (
            updated.get("matrix_format") != "scipy_csr_npz"
            or updated.get("matrix_path") != str(npz_path)
        ):
            raise RuntimeError(f"Failed to update {record_path}")

    if petsc_path.exists():
        petsc_path.unlink()
    petsc_info_path = Path(str(petsc_path) + ".info")
    if petsc_info_path.exists():
        petsc_info_path.unlink()
    print(
        f"stored {identity['matrix_id']} as compressed CSR "
        f"({metadata['bytes']} bytes)",
        flush=True,
    )


def run_single_trials(
    args: argparse.Namespace,
    config: ExperimentConfig,
    experiment_root: Path,
    trials: list[dict[str, Any]],
) -> tuple[int, int]:
    completed = 0
    skipped = 0
    matrices_written: set[str] = set()
    for index, trial in enumerate(trials, start=1):
        family = trial["mesh_family"]
        executable = require_executable(args.build_dir, family)
        family_root = experiment_root / family
        current_matrix_id = matrix_id(trial, config.high_region)
        matrix_path = family_root / "matrices" / f"{current_matrix_id}.petsc"
        npz_path = family_root / "matrices" / f"{current_matrix_id}.npz"
        record_path = (
            family_root
            / "records"
            / f"{sample_id(trial, config.high_region)}.json"
        )
        if record_path.exists() and not args.overwrite_records:
            if config.save_matrix and config.matrix_format == "scipy_csr_npz":
                finalize_npz_matrix(
                    matrix_path,
                    npz_path,
                    [record_path],
                    force_from_source=False,
                )
            skipped += 1
            continue

        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_matrix = (
            config.save_matrix
            and current_matrix_id not in matrices_written
            and (
                args.overwrite_records
                or (
                    not matrix_path.exists()
                    and not (
                        config.matrix_format == "scipy_csr_npz"
                        and npz_path.exists()
                    )
                )
            )
        )
        command = command_for(
            executable,
            config,
            trial,
            record_path,
            matrix_path,
            write_matrix,
        )
        print(f"[{index}/{len(trials)}] {sample_id(trial, config.high_region)}")
        subprocess.run(command, check=True, env=run_environment())
        if config.save_matrix and config.matrix_format == "scipy_csr_npz":
            finalize_npz_matrix(
                matrix_path,
                npz_path,
                [record_path],
                force_from_source=write_matrix,
            )
        if write_matrix:
            matrices_written.add(current_matrix_id)
        completed += 1
    return completed, skipped


def run_matrix_batches(
    args: argparse.Namespace,
    config: ExperimentConfig,
    experiment_root: Path,
    groups: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[int, int]:
    completed = 0
    skipped = 0
    for index, (current_matrix_id, group) in enumerate(groups, start=1):
        representative = group[0]
        family = representative["mesh_family"]
        executable = require_executable(args.build_dir, family)
        family_root = experiment_root / family
        record_directory = family_root / "records"
        matrix_path = family_root / "matrices" / f"{current_matrix_id}.petsc"
        npz_path = family_root / "matrices" / f"{current_matrix_id}.npz"
        expected_records = [
            record_directory / f"{sample_id(trial, config.high_region)}.json"
            for trial in group
        ]
        existing_before = sum(path.exists() for path in expected_records)
        if existing_before == len(expected_records) and not args.overwrite_records:
            if config.save_matrix and config.matrix_format == "scipy_csr_npz":
                finalize_npz_matrix(
                    matrix_path,
                    npz_path,
                    expected_records,
                    force_from_source=False,
                )
            skipped += existing_before
            continue

        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        record_directory.mkdir(parents=True, exist_ok=True)
        theta_values = list(dict.fromkeys(trial["theta"] for trial in group))
        repeat_values = {trial["repeat"] for trial in group}
        repeats = max(repeat_values) + 1
        if repeat_values != set(range(repeats)):
            raise RuntimeError(
                f"Non-contiguous repeat identifiers for matrix {current_matrix_id}"
            )
        if len(group) != len(theta_values) * repeats:
            raise RuntimeError(
                f"Incomplete theta/repeat grid for matrix {current_matrix_id}"
            )

        if config.matrix_format == "scipy_csr_npz":
            write_matrix = config.save_matrix and (
                args.overwrite_records or not npz_path.exists()
            )
        else:
            write_matrix = config.save_matrix and (
                args.overwrite_records or not matrix_path.exists()
            )
        command = command_for_batch(
            executable,
            config,
            representative,
            record_directory,
            matrix_path,
            theta_values,
            repeats,
            write_matrix,
            skip_existing_records=not args.overwrite_records,
        )
        print(
            f"[matrix {index}/{len(groups)}] {current_matrix_id} "
            f"trials={len(group)}",
            flush=True,
        )
        subprocess.run(command, check=True, env=run_environment())

        existing_after = sum(path.exists() for path in expected_records)
        if existing_after != len(expected_records):
            raise RuntimeError(
                f"Batch {current_matrix_id} produced {existing_after}/"
                f"{len(expected_records)} expected records"
            )
        if config.save_matrix and config.matrix_format == "scipy_csr_npz":
            finalize_npz_matrix(
                matrix_path,
                npz_path,
                expected_records,
                force_from_source=write_matrix,
            )
        if args.overwrite_records:
            completed += len(expected_records)
        else:
            completed += existing_after - existing_before
            skipped += existing_before
    return completed, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a versioned experiment grid and run one FE driver per matrix."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "smoke.json",
    )
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--mesh-family",
        action="append",
        choices=("quadrilateral", "simplex", "polygonal"),
        help="Restrict execution; repeat this option to select multiple families.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run/print at most this many trials (useful for smoke checks).",
    )
    parser.add_argument(
        "--overwrite-records",
        action="store_true",
        help="Rerun trials whose JSON record already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    selected = set(args.mesh_family or config.mesh_families)
    unavailable = sorted(selected.difference(EXECUTABLES))
    if unavailable:
        raise SystemExit(
            "Drivers not implemented yet for: "
            + ", ".join(unavailable)
            + ". Build the corresponding finite-element driver first."
        )

    experiment_root = args.output_root / config.name
    manifest_path = experiment_root / "manifest.json"
    trials = [
        trial for trial in config.iter_trials() if trial["mesh_family"] in selected
    ]
    execution_mode = "matrix_batch"
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        trials = trials[: args.limit]
        execution_mode = "single_trial_debug"
    groups = group_trials_by_matrix(trials, config.high_region)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(args.config),
                    "selected_mesh_families": sorted(selected),
                    "expanded_trials": len(trials),
                    "matrix_count": len(groups),
                    "execution_mode": execution_mode,
                    "matrix_format": config.matrix_format,
                    "first_trial": trials[0] if trials else None,
                },
                indent=2,
            )
        )
        return

    if config.save_matrix and config.matrix_format == "scipy_csr_npz":
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError as error:
            raise SystemExit(
                "Compressed NPZ matrix storage requires NumPy. Create the "
                "data environment with:\n"
                "  python3 -m venv .venv-data\n"
                "  .venv-data/bin/python -m pip install "
                "-r requirements-data.txt\n"
                "Then run this command with .venv-data/bin/python."
            ) from error

    experiment_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_config": str(args.config.resolve()),
        "expanded_config": asdict(config),
        "selected_mesh_families": sorted(selected),
        "trial_count": len(trials),
        "matrix_count": len(groups),
        "execution_mode": execution_mode,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if execution_mode == "single_trial_debug":
        completed, skipped = run_single_trials(
            args, config, experiment_root, trials
        )
    else:
        completed, skipped = run_matrix_batches(
            args, config, experiment_root, groups
        )

    report_records = 0
    for family in sorted(selected):
        report_records += build_family_reports(experiment_root / family, config.name)
    print(
        f"done: completed={completed}, skipped={skipped}, "
        f"reported={report_records}, records={experiment_root}"
    )


if __name__ == "__main__":
    main()
