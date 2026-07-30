#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.training import load_run_config, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the paper-v1 CNN convergence-factor regressor using only "
            "the canonical train and validation partitions."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "ml_cnn_baseline.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume exactly from output/latest.pt.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Override the configured device.",
    )
    parser.add_argument(
        "--epoch-limit",
        type=int,
        help=(
            "Run at most this many epochs in this invocation. This is intended "
            "for smoke tests or resumable time-bounded runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config, repo_root=REPO_ROOT)
    if args.device:
        config = replace(config, device=args.device)
    summary = train(
        config,
        resume=args.resume,
        epoch_limit=args.epoch_limit,
    )
    print(
        f"done: best_epoch={summary['best_epoch']} "
        f"validation_rmse="
        f"{summary['best_metrics']['validation_rmse']:.7f} "
        f"report={summary['artifacts']['report']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
