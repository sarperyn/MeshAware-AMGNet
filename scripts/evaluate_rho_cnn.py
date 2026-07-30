#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.evaluation import evaluate, load_evaluation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or validate the locked paper-v1 held-out test evaluation. "
            "An existing lock is verified and reused without model inference."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "ml_cnn_evaluation.json",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Override the training configuration device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_evaluation_config(args.config, repo_root=REPO_ROOT)
    if args.device:
        config = replace(
            config,
            training_run=replace(
                config.training_run, device=args.device
            ),
        )
    metrics, lock = evaluate(config)
    overall = metrics["overall"]
    decision = metrics["theta_selection"]["overall"]
    print(
        f"evaluation={lock['evaluation_id']} "
        f"test_rmse={overall['rmse']:.7f} "
        f"test_mae={overall['mae']:.7f} "
        f"theta_exact={decision['exact_optimum_rate']:.2%} "
        f"report={config.report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
