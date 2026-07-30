#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.amg_integration import AMGProblem, run_amg_workflow
from meshaware_ml.inference import PredictorPaths, RhoPredictor, parse_theta_values


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a matrix, recommend theta with cnn_paper_v1, and run "
            "the existing BoomerAMG solver at the selected threshold."
        )
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=REPO_ROOT / "configs" / "ml_cnn_inference.json",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=REPO_ROOT / "build-unified"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mesh-family", choices=("simplex", "polygonal"), required=True
    )
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument(
        "--pattern",
        choices=(
            "vertical_split",
            "checkerboard_2x2",
            "vertical_stripes_4",
            "checkerboard_4x4",
        ),
        required=True,
    )
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument(
        "--high-region", choices=("white", "gray"), default="white"
    )
    parser.add_argument("--theta-values")
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-50)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = json.loads(
        args.inference_config.read_text(encoding="utf-8")
    )
    if value.get("schema_version") != 1:
        raise SystemExit("unsupported inference configuration schema")
    raw_theta = (
        [
            float(token)
            for token in args.theta_values.split(",")
            if token
        ]
        if args.theta_values
        else value["default_theta_values"]
    )
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
    problem = AMGProblem(
        mesh_family=args.mesh_family,
        level=args.level,
        pattern=args.pattern,
        epsilon=args.epsilon,
        high_region=args.high_region,
        relative_tolerance=args.rtol,
        absolute_tolerance=args.atol,
        maximum_iterations=args.max_iterations,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
    )
    manifest, _ = run_amg_workflow(
        predictor,
        build_dir=args.build_dir,
        output_dir=args.output_dir,
        problem=problem,
        theta_values=theta_values,
    )
    print(
        f"selected_theta={manifest['selected_theta']:.10g} "
        f"predicted_rho={manifest['predicted_rho']:.10g} "
        f"measured_rho={manifest['measured_rho_mean']:.10g} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
