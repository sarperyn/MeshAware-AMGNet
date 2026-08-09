from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator


PATTERNS = (
    "vertical_split",
    "checkerboard_2x2",
    "vertical_stripes_4",
    "checkerboard_4x4",
)
MESH_FAMILIES = ("quadrilateral", "simplex", "polygonal")
AMG_BACKENDS = ("boomeramg", "polydeal-agglomeration")
BOOMERAMG_PROFILES = ("default", "polygonal-nodal")
AMG_SMOOTHERS = (
    "chebyshev",
    "damped-jacobi",
    "l1-symmetric-gauss-seidel",
    "symmetric-gauss-seidel",
)


def _linspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("theta.count must be positive")
    if count == 1:
        if start != stop:
            raise ValueError("a one-point theta grid requires start == stop")
        return (float(start),)
    step = (stop - start) / (count - 1)
    return tuple(float(start + i * step) for i in range(count))


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    mesh_families: tuple[str, ...]
    levels: tuple[int, ...]
    patterns: tuple[str, ...]
    epsilons: tuple[float, ...]
    theta_values: tuple[float, ...]
    high_region: str
    relative_tolerance: float
    absolute_tolerance: float
    maximum_iterations: int
    amg_backend: str
    boomeramg_profile: str
    amg_smoother: str
    jacobi_damping: float
    warmup_runs: int
    repeats: int
    repeats_by_level: dict[int, int]
    save_matrix: bool
    matrix_format: str
    family_subdirectories: bool

    def validate(self) -> None:
        if not self.name:
            raise ValueError("experiment name cannot be empty")
        if not self.mesh_families or any(
            family not in MESH_FAMILIES for family in self.mesh_families
        ):
            raise ValueError(f"mesh_families must be chosen from {MESH_FAMILIES}")
        if not self.levels or any(level < 0 for level in self.levels):
            raise ValueError("levels must be non-negative")
        if not self.patterns or any(pattern not in PATTERNS for pattern in self.patterns):
            raise ValueError(f"patterns must be chosen from {PATTERNS}")
        if not self.epsilons or any(epsilon < 0 for epsilon in self.epsilons):
            raise ValueError("epsilons must be non-negative")
        if not self.theta_values or any(
            not 0.0 < theta < 1.0 for theta in self.theta_values
        ):
            raise ValueError("all strong-threshold values must lie strictly in (0,1)")
        if len(set(self.theta_values)) != len(self.theta_values):
            raise ValueError("strong-threshold values must be unique")
        if self.high_region not in ("white", "gray"):
            raise ValueError("high_region must be 'white' or 'gray'")
        if self.relative_tolerance <= 0 or self.absolute_tolerance < 0:
            raise ValueError("solver tolerances must be valid")
        if self.maximum_iterations < 1 or self.repeats < 1:
            raise ValueError("iteration and repeat counts must be positive")
        if self.amg_backend not in AMG_BACKENDS:
            raise ValueError(f"amg_backend must be chosen from {AMG_BACKENDS}")
        if (
            self.amg_backend == "polydeal-agglomeration"
            and self.mesh_families != ("polygonal",)
        ):
            raise ValueError(
                "polydeal-agglomeration requires polygonal-only mesh_families"
            )
        if self.boomeramg_profile not in BOOMERAMG_PROFILES:
            raise ValueError(
                f"boomeramg_profile must be chosen from {BOOMERAMG_PROFILES}"
            )
        if self.boomeramg_profile == "polygonal-nodal" and (
            self.amg_backend != "boomeramg"
            or self.mesh_families != ("polygonal",)
        ):
            raise ValueError(
                "polygonal-nodal requires the boomeramg backend and a "
                "polygonal-only experiment"
            )
        if self.amg_smoother not in AMG_SMOOTHERS:
            raise ValueError(f"amg_smoother must be chosen from {AMG_SMOOTHERS}")
        if (
            self.amg_backend == "polydeal-agglomeration"
            and self.amg_smoother == "l1-symmetric-gauss-seidel"
        ):
            raise ValueError(
                "l1-symmetric-gauss-seidel is available only for BoomerAMG"
            )
        if (
            self.boomeramg_profile == "polygonal-nodal"
            and self.amg_smoother == "l1-symmetric-gauss-seidel"
        ):
            raise ValueError(
                "polygonal-nodal is incompatible with HYPRE's "
                "l1-symmetric-gauss-seidel relaxation"
            )
        if not 0.0 < self.jacobi_damping <= 1.0:
            raise ValueError("jacobi_damping must lie in (0,1]")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if any(level not in self.levels for level in self.repeats_by_level):
            raise ValueError("repeats_by_level contains a level outside levels")
        if any(repeats < 1 for repeats in self.repeats_by_level.values()):
            raise ValueError("repeat counts must be positive")
        if self.matrix_format not in ("petsc_binary", "scipy_csr_npz"):
            raise ValueError(
                "matrix_format must be 'petsc_binary' or 'scipy_csr_npz'"
            )
        if not self.family_subdirectories and len(self.mesh_families) != 1:
            raise ValueError(
                "flat output requires exactly one configured mesh family"
            )

    @property
    def h_values(self) -> tuple[float, ...]:
        return tuple(2.0 ** (-level) for level in self.levels)

    @property
    def trial_count(self) -> int:
        repeats = sum(
            self.repeats_by_level.get(level, self.repeats) for level in self.levels
        )
        return (
            len(self.mesh_families)
            * repeats
            * len(self.patterns)
            * len(self.epsilons)
            * len(self.theta_values)
        )

    def iter_trials(self) -> Iterator[dict[str, Any]]:
        for family, level, pattern, epsilon, theta in product(
            self.mesh_families,
            self.levels,
            self.patterns,
            self.epsilons,
            self.theta_values,
        ):
            repeats = self.repeats_by_level.get(level, self.repeats)
            for repeat in range(repeats):
                yield {
                    "mesh_family": family,
                    "level": level,
                    "h_nominal": 2.0 ** (-level),
                    "pattern": pattern,
                    "epsilon": epsilon,
                    "theta": theta,
                    "repeat": repeat,
                }


@dataclass(frozen=True)
class TrialRecord:
    schema_version: int
    sample_id: str
    matrix_id: str
    problem: str
    mesh_family: str
    level: int
    h_nominal: float
    h_max: float
    pattern: str
    epsilon: float
    high_region: str
    theta: float
    repeat: int
    cells: int
    background_cells: int
    dofs: int
    nnz: int
    cg_iterations: int
    amg_levels: int
    ksp_converged_reason: int
    residual_initial: float
    residual_final: float
    convergence_factor: float
    l2_error: float
    h1_seminorm_error: float
    energy_error: float
    assembly_time_seconds: float
    amg_setup_time_seconds: float
    solve_time_seconds: float
    matrix_format: str
    matrix_path: str
    amg_backend: str = "boomeramg"
    boomeramg_profile: str = "default"
    amg_smoother: str = "symmetric-gauss-seidel"
    amg_relaxation_weight: float = 1.0
    grid_complexity: float = 0.0
    operator_complexity: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def load_experiment_config(
    experiment_path: str | Path,
    common_path: str | Path | None = None,
) -> ExperimentConfig:
    experiment_path = Path(experiment_path)
    if common_path is None:
        common_path = experiment_path.with_name("common.json")

    with Path(common_path).open(encoding="utf-8") as handle:
        merged: dict[str, Any] = json.load(handle)
    with experiment_path.open(encoding="utf-8") as handle:
        merged.update(json.load(handle))

    theta_values = merged.get("theta_values")
    if theta_values is None:
        theta = merged["theta"]
        theta_values = _linspace(
            float(theta["start"]), float(theta["stop"]), int(theta["count"])
        )

    repeats_by_level = {
        int(level): int(repeats)
        for level, repeats in merged.get("repeats_by_level", {}).items()
    }
    config = ExperimentConfig(
        name=str(merged["name"]),
        mesh_families=tuple(merged["mesh_families"]),
        levels=tuple(int(level) for level in merged["levels"]),
        patterns=tuple(merged["patterns"]),
        epsilons=tuple(float(value) for value in merged["epsilons"]),
        theta_values=tuple(float(value) for value in theta_values),
        high_region=str(merged["high_region"]),
        relative_tolerance=float(merged["relative_tolerance"]),
        absolute_tolerance=float(merged["absolute_tolerance"]),
        maximum_iterations=int(merged["maximum_iterations"]),
        amg_backend=str(merged.get("amg_backend", "boomeramg")),
        boomeramg_profile=str(merged.get("boomeramg_profile", "default")),
        amg_smoother=str(
            merged.get("amg_smoother", "symmetric-gauss-seidel")
        ),
        jacobi_damping=float(merged.get("jacobi_damping", 2.0 / 3.0)),
        warmup_runs=int(merged.get("warmup_runs", 1)),
        repeats=int(merged.get("repeats", 1)),
        repeats_by_level=repeats_by_level,
        save_matrix=bool(merged["save_matrix"]),
        matrix_format=str(merged["matrix_format"]),
        family_subdirectories=bool(merged.get("family_subdirectories", True)),
    )
    config.validate()
    return config
