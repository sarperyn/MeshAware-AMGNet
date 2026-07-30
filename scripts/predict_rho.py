#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.inference import (
    PredictorPaths,
    RhoPredictor,
    parse_theta_values,
    write_json_atomic,
)


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict convergence factors and recommend a BoomerAMG strong "
            "threshold for one finalized MeshAware CSR matrix."
        )
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "ml_cnn_inference.json",
    )
    parser.add_argument("--level", type=int)
    parser.add_argument(
        "--theta",
        type=float,
        action="append",
        help="Candidate theta; repeat the option. Overrides config defaults.",
    )
    parser.add_argument(
        "--theta-values",
        help="Comma-separated candidates. Overrides config defaults.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = json.loads(args.config.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise SystemExit("unsupported inference configuration schema")
    if args.theta and args.theta_values:
        raise SystemExit("--theta and --theta-values are mutually exclusive")
    if args.theta:
        raw_theta = args.theta
    elif args.theta_values:
        raw_theta = [
            float(token) for token in args.theta_values.split(",") if token
        ]
    else:
        raw_theta = value["default_theta_values"]
    theta_values = parse_theta_values(raw_theta)
    predictor = RhoPredictor.from_paths(
        PredictorPaths(
            training_config=_resolve(value["training_config"]),
            checkpoint=_resolve(value["checkpoint"]),
            phase3_summary=_resolve(value["phase3_summary"]),
        ),
        repo_root=REPO_ROOT,
        device=args.device,
    )
    result = predictor.recommend_matrix(
        args.matrix, theta_values, level=args.level
    )
    if args.output:
        write_json_atomic(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    recommendation = result["recommendation"]
    print(
        f"recommended_theta={recommendation['theta']:.10g} "
        f"predicted_rho={recommendation['predicted_rho']:.10g}",
        file=sys.stderr if not args.output else sys.stdout,
        flush=True,
    )


if __name__ == "__main__":
    main()
