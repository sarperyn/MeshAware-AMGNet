from __future__ import annotations

import os
from pathlib import Path

EXECUTABLE_NAMES = {
    "quadrilateral": "meshaware_diffusion_dealii",
    "simplex": "meshaware_diffusion_dealii",
    "polygonal": "meshaware_diffusion_polydeal",
}


def solver_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("OMPI_MCA_btl", "self")
    return environment


def base_solver_command(
    executable: Path,
    *,
    mesh_family: str,
    level: int,
    epsilon: float,
    pattern: str,
    high_region: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_iterations: int,
) -> list[str]:
    return [
        str(executable),
        "--mesh-family",
        mesh_family,
        "--level",
        str(level),
        "--epsilon",
        str(epsilon),
        "--pattern",
        pattern,
        "--high-region",
        high_region,
        "--rtol",
        str(relative_tolerance),
        "--atol",
        str(absolute_tolerance),
        "--max-iterations",
        str(maximum_iterations),
    ]
