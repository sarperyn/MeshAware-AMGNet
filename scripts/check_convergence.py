from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any


def rate(coarse_error: float, fine_error: float, coarse_h: float, fine_h: float) -> float:
    if min(coarse_error, fine_error, coarse_h, fine_h) <= 0:
        raise ValueError("errors and mesh sizes must be positive")
    return math.log(coarse_error / fine_error) / math.log(coarse_h / fine_h)


def load_groups(pattern: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No records matched {pattern}")
    for filename in paths:
        with Path(filename).open(encoding="utf-8") as handle:
            record = json.load(handle)
        key = (
            record["mesh_family"],
            record["pattern"],
            float(record["epsilon"]),
            float(record["theta"]),
        )
        groups[key].append(record)
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check consecutive L2 and H1 convergence rates in solver records."
    )
    parser.add_argument(
        "--records-glob",
        default="datasets/convergence/**/records/*.json",
    )
    parser.add_argument("--minimum-l2-rate", type=float, default=1.7)
    parser.add_argument("--minimum-h1-rate", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for key, records in sorted(load_groups(args.records_glob).items()):
        records.sort(key=lambda record: float(record["h_nominal"]), reverse=True)
        if len(records) < 2:
            failures.append(f"{key}: fewer than two mesh sizes")
            continue
        for coarse, fine in pairwise(records):
            l2_rate = rate(
                float(coarse["l2_error"]),
                float(fine["l2_error"]),
                float(coarse["h_nominal"]),
                float(fine["h_nominal"]),
            )
            h1_rate = rate(
                float(coarse["h1_seminorm_error"]),
                float(fine["h1_seminorm_error"]),
                float(coarse["h_nominal"]),
                float(fine["h_nominal"]),
            )
            print(
                f"{key[0]}: level {coarse['level']}->{fine['level']} "
                f"L2={l2_rate:.4f} H1={h1_rate:.4f}"
            )
            if l2_rate < args.minimum_l2_rate:
                failures.append(f"{key}: L2 rate {l2_rate:.4f}")
            if h1_rate < args.minimum_h1_rate:
                failures.append(f"{key}: H1 rate {h1_rate:.4f}")
    if failures:
        raise SystemExit("convergence check failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
