from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meshaware_data.artifacts import file_sha256, write_json_atomic
from meshaware_data.csr_artifact import (
    convert_petsc_to_csr_npz,
    read_petsc_matrix_header,
)
from meshaware_data.identifiers import matrix_id
from meshaware_data.schema import PATTERNS
from meshaware_data.solver import (
    EXECUTABLE_NAMES,
    base_solver_command,
    solver_environment,
)

from .inference import RhoPredictor, parse_theta_values

WORKFLOW_SCHEMA_VERSION = 1
SUPPORTED_MESH_FAMILIES = frozenset({"simplex", "polygonal"})


@dataclass(frozen=True)
class AMGProblem:
    mesh_family: str
    level: int
    pattern: str
    epsilon: float
    high_region: str = "white"
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 1.0e-50
    maximum_iterations: int = 10000
    repeats: int = 1
    warmup_runs: int = 0

    def validate(self) -> None:
        if self.mesh_family not in SUPPORTED_MESH_FAMILIES:
            raise ValueError(
                f"Phase 5 supports {sorted(SUPPORTED_MESH_FAMILIES)}, "
                f"not {self.mesh_family!r}"
            )
        if self.pattern not in PATTERNS:
            raise ValueError(f"unsupported coefficient pattern: {self.pattern}")
        if self.high_region not in {"white", "gray"}:
            raise ValueError("high_region must be white or gray")
        if self.level < 0 or self.level > 10:
            raise ValueError("level must lie in [0, 10]")
        if not math.isfinite(self.epsilon) or self.epsilon < 0.0:
            raise ValueError("epsilon must be finite and non-negative")
        if (
            self.relative_tolerance <= 0.0
            or self.absolute_tolerance < 0.0
            or self.maximum_iterations <= 0
        ):
            raise ValueError("invalid solver tolerances")
        if self.repeats <= 0 or self.warmup_runs < 0:
            raise ValueError("invalid repeat or warm-up count")

    @property
    def matrix_id(self) -> str:
        return matrix_id(
            mesh_family=self.mesh_family,
            level=self.level,
            pattern=self.pattern,
            epsilon=self.epsilon,
            high_region=self.high_region,
        )

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "mesh_family": self.mesh_family,
            "level": self.level,
            "pattern": self.pattern,
            "epsilon": self.epsilon,
            "high_region": self.high_region,
        }


def _base_command(executable: Path, problem: AMGProblem) -> list[str]:
    return base_solver_command(
        executable,
        mesh_family=problem.mesh_family,
        level=problem.level,
        epsilon=problem.epsilon,
        pattern=problem.pattern,
        high_region=problem.high_region,
        relative_tolerance=problem.relative_tolerance,
        absolute_tolerance=problem.absolute_tolerance,
        maximum_iterations=problem.maximum_iterations,
    )


def assemble_command(
    executable: Path, problem: AMGProblem, matrix_path: Path
) -> list[str]:
    return [
        *_base_command(executable, problem),
        "--matrix",
        str(matrix_path),
        "--assemble-only",
    ]


def solve_command(
    executable: Path,
    problem: AMGProblem,
    theta: float,
    record_directory: Path,
    matrix_reference: Path,
) -> list[str]:
    return [
        *_base_command(executable, problem),
        "--theta-values",
        str(theta),
        "--repeats",
        str(problem.repeats),
        "--warmup-runs",
        str(problem.warmup_runs),
        "--record-dir",
        str(record_directory),
        "--matrix",
        str(matrix_reference),
        "--skip-matrix-write",
    ]


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        check=True,
        env=solver_environment(),
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    return {
        "argv": list(command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def _load_records(record_directory: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(record_directory.glob("*.json"))
    ]
    if not records:
        raise RuntimeError("AMG integration produced no solver records")
    return records


def validate_existing_workflow(
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = Path(output_dir).resolve()
    lock_path = output_dir / "workflow_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("workflow output exists without workflow_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise ValueError("unsupported workflow lock schema")
    for relative, expected_hash in lock["artifacts"].items():
        artifact = output_dir / relative
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise ValueError(f"workflow artifact changed: {relative}")
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest, lock


def run_amg_workflow(
    predictor: RhoPredictor,
    *,
    build_dir: str | Path,
    output_dir: str | Path,
    problem: AMGProblem,
    theta_values: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    problem.validate()
    candidates = parse_theta_values(theta_values)
    build_dir = Path(build_dir).resolve()
    output_dir = Path(output_dir).resolve()
    executable = build_dir / EXECUTABLE_NAMES[problem.mesh_family]
    if not executable.is_file():
        raise FileNotFoundError(f"solver executable does not exist: {executable}")
    if output_dir.exists():
        manifest, lock = validate_existing_workflow(output_dir)
        expected = {
            "problem": asdict(problem),
            "theta_values": list(candidates),
            "checkpoint_sha256": predictor.provenance["checkpoint_sha256"],
        }
        if lock.get("request") != expected:
            raise ValueError("existing workflow lock has a different request")
        return manifest, lock

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.staging-",
        )
    )
    final_matrix = output_dir / "operator.npz"
    try:
        petsc_path = staging / "operator.petsc"
        npz_path = staging / "operator.npz"
        records_path = staging / "records"
        records_path.mkdir()
        assembly = _run_command(
            assemble_command(executable, problem, petsc_path)
        )
        header = read_petsc_matrix_header(petsc_path)
        identity = {
            **problem.identity,
            "dofs": header.rows,
            "nnz": header.nnz,
        }
        conversion = convert_petsc_to_csr_npz(
            petsc_path,
            npz_path,
            identity=identity,
        )
        recommendation = predictor.recommend_matrix(
            npz_path, candidates, level=problem.level
        )
        recommendation["matrix"]["path"] = str(final_matrix)
        write_json_atomic(staging / "recommendation.json", recommendation)
        selected_theta = float(recommendation["recommendation"]["theta"])
        solve = _run_command(
            solve_command(
                executable,
                problem,
                selected_theta,
                records_path,
                final_matrix,
            )
        )
        records = _load_records(records_path)
        if len(records) != problem.repeats:
            raise RuntimeError(
                f"expected {problem.repeats} records, found {len(records)}"
            )
        for path, record in zip(
            sorted(records_path.glob("*.json")), records, strict=True
        ):
            if (
                str(record["matrix_id"]) != problem.matrix_id
                or not math.isclose(
                    float(record["theta"]),
                    selected_theta,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise RuntimeError("solver record does not match recommendation")
            record["matrix_format"] = "scipy_csr_npz"
            record["matrix_path"] = str(final_matrix)
            write_json_atomic(path, record)
        measured_rho = sum(
            float(record["convergence_factor"]) for record in records
        ) / len(records)
        predicted_rho = float(
            recommendation["recommendation"]["predicted_rho"]
        )
        petsc_path.unlink(missing_ok=True)
        Path(str(petsc_path) + ".info").unlink(missing_ok=True)
        manifest = {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "cnn_recommendation_then_boomeramg",
            "problem": asdict(problem),
            "matrix": {
                "path": str(final_matrix),
                "source_sha256": recommendation["matrix"]["source_sha256"],
                "shape": recommendation["matrix"]["shape"],
                "nnz": recommendation["matrix"]["nnz"],
                "compressed_bytes": conversion["bytes"],
            },
            "candidate_theta_values": list(candidates),
            "selected_theta": selected_theta,
            "predicted_rho": predicted_rho,
            "measured_rho_mean": measured_rho,
            "absolute_prediction_error": abs(
                predicted_rho - measured_rho
            ),
            "solver": {
                "records": len(records),
                "cg_iterations": [
                    int(record["cg_iterations"]) for record in records
                ],
                "amg_levels": [
                    int(record["amg_levels"]) for record in records
                ],
                "convergence_factors": [
                    float(record["convergence_factor"])
                    for record in records
                ],
            },
            "provenance": recommendation["provenance"],
            "commands": {
                "assemble": assembly,
                "solve": solve,
            },
            "implementation_note": (
                "The deterministic finite-element operator is assembled once "
                "for inference and reassembled by the existing driver for the "
                "selected solve; PyTorch is not embedded in C++."
            ),
        }
        write_json_atomic(staging / "manifest.json", manifest)
        artifact_paths = [
            staging / "manifest.json",
            staging / "recommendation.json",
            staging / "operator.npz",
            *sorted(records_path.glob("*.json")),
        ]
        request = {
            "problem": asdict(problem),
            "theta_values": list(candidates),
            "checkpoint_sha256": predictor.provenance["checkpoint_sha256"],
        }
        lock = {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "request": request,
            "artifacts": {
                str(path.relative_to(staging)): file_sha256(path)
                for path in artifact_paths
            },
        }
        write_json_atomic(staging / "workflow_lock.json", lock)
        os.replace(staging, output_dir)
        return manifest, lock
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
