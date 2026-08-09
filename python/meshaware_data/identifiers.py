from __future__ import annotations

MATRIX_PREFIXES = {
    "quadrilateral": "quad",
    "simplex": "simplex",
    "simplex-dg": "simplex_dg",
    "polygonal": "poly",
}


def slug_float(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


def matrix_id(
    *,
    mesh_family: str,
    level: int,
    pattern: str,
    epsilon: float,
    high_region: str,
) -> str:
    try:
        prefix = MATRIX_PREFIXES[mesh_family]
    except KeyError as error:
        raise ValueError(f"unsupported mesh family: {mesh_family!r}") from error
    return (
        f"{prefix}_l{level}_{pattern}_e{slug_float(epsilon)}_high_{high_region}"
    )


def sample_id(matrix: str, *, theta: float, repeat: int) -> str:
    return f"{matrix}_theta_{slug_float(theta)}_repeat_{repeat}"
