from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from meshaware_data.artifacts import file_sha256, write_json_atomic
from torch.utils.data import DataLoader

from .dataset import (
    GroupedRhoDataset,
    HashGroupBatchSampler,
    MatrixViewCache,
    collate_matrix_groups,
    load_index_rows,
)
from .model import PaperRhoCNN
from .training import (
    CHECKPOINT_SCHEMA_VERSION,
    RunConfig,
    load_run_config,
    resolve_device,
    seed_everything,
)

EVALUATION_SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
PREDICTION_FIELDS = (
    "sample_id",
    "matrix_id",
    "matrix_sha256",
    "mesh_family",
    "level",
    "pattern",
    "epsilon",
    "theta",
    "target_rho",
    "predicted_rho",
    "error",
    "absolute_error",
    "squared_error",
)
DECISION_FIELDS = (
    "matrix_id",
    "matrix_sha256",
    "mesh_family",
    "level",
    "pattern",
    "epsilon",
    "theta_count",
    "true_best_theta",
    "predicted_best_theta",
    "true_best_rho",
    "selected_true_rho",
    "regret",
    "absolute_theta_error",
    "exact_optimum",
)


@dataclass(frozen=True)
class EvaluationConfig:
    config_path: Path
    training_run: RunConfig
    checkpoint_path: Path
    phase3_summary_path: Path
    output_dir: Path
    report_path: Path
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    create_plots: bool

    def validate(self) -> None:
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be non-negative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


def _resolve(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_evaluation_config(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> EvaluationConfig:
    repo_root = Path(repo_root).resolve()
    config_path = Path(path).resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation configuration schema")
    required = {
        "training_config",
        "checkpoint",
        "phase3_summary",
        "output",
        "bootstrap",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"evaluation config missing keys: {sorted(missing)}")
    training_config_path = _resolve(repo_root, value["training_config"])
    training_run = load_run_config(training_config_path, repo_root=repo_root)
    output = value["output"]
    bootstrap = value["bootstrap"]
    config = EvaluationConfig(
        config_path=config_path,
        training_run=training_run,
        checkpoint_path=_resolve(repo_root, value["checkpoint"]),
        phase3_summary_path=_resolve(repo_root, value["phase3_summary"]),
        output_dir=_resolve(repo_root, output["directory"]),
        report_path=_resolve(repo_root, output["report"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
        confidence_level=float(bootstrap["confidence_level"]),
        create_plots=bool(output.get("plots", True)),
    )
    config.validate()
    return config


def _safe_metric(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def regression_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot compute regression metrics for no rows")
    target = np.asarray([row["target_rho"] for row in rows], dtype=np.float64)
    prediction = np.asarray(
        [row["predicted_rho"] for row in rows], dtype=np.float64
    )
    residual = prediction - target
    absolute = np.abs(residual)
    squared = np.square(residual)
    mse = float(np.mean(squared))
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    target_variation = float(np.dot(target_centered, target_centered))
    prediction_variation = float(
        np.dot(prediction_centered, prediction_centered)
    )
    r_squared = (
        1.0 - float(np.sum(squared)) / target_variation
        if target_variation > 0.0
        else None
    )
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if len(rows) > 1
        and target_variation > 0.0
        and prediction_variation > 0.0
        else None
    )
    quantiles = {
        f"p{int(level * 100):02d}": float(np.quantile(absolute, level))
        for level in (0.5, 0.9, 0.95, 0.99)
    }
    return {
        "samples": len(rows),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(residual)),
        "r_squared": _safe_metric(r_squared),
        "pearson_correlation": _safe_metric(correlation),
        "max_absolute_error": float(np.max(absolute)),
        "absolute_error_quantiles": quantiles,
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "predictions_below_zero": int(np.sum(prediction < 0.0)),
        "predictions_above_one": int(np.sum(prediction > 1.0)),
    }


def stratified_metrics(
    rows: Sequence[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        value: regression_metrics(group)
        for value, group in sorted(groups.items())
    }


def grouped_bootstrap_intervals(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    group_names = sorted(groups)
    if not group_names:
        raise ValueError("bootstrap needs at least one group")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "rmse": [],
        "mae": [],
        "bias": [],
        "r_squared": [],
    }
    for _ in range(replicates):
        selected = rng.integers(0, len(group_names), size=len(group_names))
        sample = [
            row
            for index in selected
            for row in groups[group_names[int(index)]]
        ]
        metrics = regression_metrics(sample)
        for metric, metric_values in values.items():
            value = metrics[metric]
            if value is not None:
                metric_values.append(float(value))
    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, Any] = {}
    for metric, samples in values.items():
        if not samples:
            intervals[metric] = None
            continue
        intervals[metric] = {
            "lower": float(np.quantile(samples, alpha)),
            "upper": float(np.quantile(samples, 1.0 - alpha)),
        }
    return {
        "method": "source-hash cluster percentile bootstrap",
        "group_key": group_key,
        "groups": len(group_names),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "intervals": intervals,
    }


def theta_decisions(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        matrices[str(row["matrix_id"])].append(row)
    decisions: list[dict[str, Any]] = []
    tolerance = 1.0e-12
    for matrix_id, group in sorted(matrices.items()):
        ordered = sorted(group, key=lambda row: float(row["theta"]))
        identity_values = {
            (
                row["matrix_sha256"],
                row["mesh_family"],
                row["level"],
                row["pattern"],
                row["epsilon"],
            )
            for row in ordered
        }
        if len(identity_values) != 1:
            raise ValueError(f"matrix_id identity conflict: {matrix_id}")
        true_best_rho = min(float(row["target_rho"]) for row in ordered)
        true_best_rows = [
            row
            for row in ordered
            if abs(float(row["target_rho"]) - true_best_rho) <= tolerance
        ]
        predicted = min(
            ordered,
            key=lambda row: (
                float(row["predicted_rho"]),
                float(row["theta"]),
            ),
        )
        predicted_theta = float(predicted["theta"])
        closest_true_theta = min(
            (float(row["theta"]) for row in true_best_rows),
            key=lambda theta: abs(theta - predicted_theta),
        )
        regret = float(predicted["target_rho"]) - true_best_rho
        first = ordered[0]
        decisions.append(
            {
                "matrix_id": matrix_id,
                "matrix_sha256": first["matrix_sha256"],
                "mesh_family": first["mesh_family"],
                "level": first["level"],
                "pattern": first["pattern"],
                "epsilon": first["epsilon"],
                "theta_count": len(ordered),
                "true_best_theta": min(
                    float(row["theta"]) for row in true_best_rows
                ),
                "predicted_best_theta": predicted_theta,
                "true_best_rho": true_best_rho,
                "selected_true_rho": float(predicted["target_rho"]),
                "regret": max(0.0, regret),
                "absolute_theta_error": abs(
                    predicted_theta - closest_true_theta
                ),
                "exact_optimum": any(
                    abs(predicted_theta - float(row["theta"])) <= tolerance
                    for row in true_best_rows
                ),
            }
        )
    return decisions, decision_metrics(decisions)


def decision_metrics(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        raise ValueError("cannot compute decision metrics for no matrices")
    regret = np.asarray(
        [decision["regret"] for decision in decisions], dtype=np.float64
    )
    theta_error = np.asarray(
        [decision["absolute_theta_error"] for decision in decisions],
        dtype=np.float64,
    )
    exact = np.asarray(
        [bool(decision["exact_optimum"]) for decision in decisions]
    )
    return {
        "matrices": len(decisions),
        "exact_optimum_count": int(exact.sum()),
        "exact_optimum_rate": float(exact.mean()),
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "max_regret": float(regret.max()),
        "mean_absolute_theta_error": float(theta_error.mean()),
        "median_absolute_theta_error": float(np.median(theta_error)),
        "max_absolute_theta_error": float(theta_error.max()),
    }


def _decision_strata(
    decisions: Sequence[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        groups[str(decision[key])].append(decision)
    return {
        value: decision_metrics(group)
        for value, group in sorted(groups.items())
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _run_test_inference(
    config: EvaluationConfig,
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    run = config.training_run
    cache = MatrixViewCache(
        run.dataset_root, capacity=run.training.cache_capacity
    )
    dataset = GroupedRhoDataset(
        run.samples_path,
        run.dataset_root,
        splits=("test",),
        cache=cache,
    )
    sampler = HashGroupBatchSampler(
        dataset.group_sizes,
        sample_batch_size=run.training.batch_size,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_matrix_groups,
    )
    model = PaperRhoCNN(run.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    row_by_id = {
        str(row["sample_id"]): row for row in dataset.rows
    }
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            view = batch["view"].to(device=device, dtype=torch.float32)
            scalars = batch["scalars"].to(
                device=device, dtype=torch.float32
            )
            view_indices = batch["view_indices"].to(device=device)
            output = model(view, scalars, view_indices).cpu().tolist()
            for sample_id, predicted in zip(
                batch["sample_id"], output, strict=True
            ):
                source = row_by_id[str(sample_id)]
                target = float(source["rho_mean"])
                error = float(predicted) - target
                predictions.append(
                    {
                        "sample_id": str(sample_id),
                        "matrix_id": str(source["matrix_id"]),
                        "matrix_sha256": str(source["matrix_sha256"]),
                        "mesh_family": str(source["mesh_family"]),
                        "level": int(source["level"]),
                        "pattern": str(source["pattern"]),
                        "epsilon": float(source["epsilon"]),
                        "theta": float(source["theta"]),
                        "target_rho": target,
                        "predicted_rho": float(predicted),
                        "error": error,
                        "absolute_error": abs(error),
                        "squared_error": error * error,
                    }
                )
    predictions.sort(key=lambda row: row["sample_id"])
    if len(predictions) != dataset.sample_count:
        raise RuntimeError("test inference did not produce one prediction per sample")
    if len({row["sample_id"] for row in predictions}) != len(predictions):
        raise RuntimeError("test inference produced duplicate sample IDs")
    return predictions


def _verify_source_contract(
    config: EvaluationConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    run = config.training_run
    phase3 = json.loads(
        config.phase3_summary_path.read_text(encoding="utf-8")
    )
    samples_sha = file_sha256(run.samples_path)
    splits_sha = file_sha256(run.splits_path)
    checkpoint_sha = file_sha256(config.checkpoint_path)
    config_sha = file_sha256(config.config_path)
    if phase3.get("samples_sha256") != samples_sha:
        raise ValueError("Phase 3 sample-index fingerprint changed")
    if phase3.get("splits_sha256") != splits_sha:
        raise ValueError("Phase 3 split fingerprint changed")
    if phase3.get("test_samples_evaluated") != 0:
        raise ValueError("Phase 3 summary does not certify a held-out test split")
    checkpoint = torch.load(
        config.checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 3 checkpoint schema")
    if checkpoint.get("samples_sha256") != samples_sha:
        raise ValueError("checkpoint sample-index fingerprint changed")
    if checkpoint.get("splits_sha256") != splits_sha:
        raise ValueError("checkpoint split fingerprint changed")
    if checkpoint.get("epoch") != phase3.get("best_epoch"):
        raise ValueError("checkpoint is not the Phase 3 selected epoch")
    if checkpoint.get("config_fingerprint") != phase3.get(
        "config_fingerprint"
    ):
        raise ValueError("checkpoint and Phase 3 configuration differ")
    if checkpoint.get("model_config") != run.model.to_dict():
        raise ValueError("checkpoint architecture differs from training config")

    split_hashes: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        split_hashes[split] = {
            str(row["matrix_sha256"])
            for row in load_index_rows(run.samples_path, splits=(split,))
        }
    if (
        split_hashes["train"] & split_hashes["test"]
        or split_hashes["validation"] & split_hashes["test"]
        or split_hashes["train"] & split_hashes["validation"]
    ):
        raise ValueError("source-hash leakage exists between canonical splits")
    source = {
        "checkpoint_sha256": checkpoint_sha,
        "samples_sha256": samples_sha,
        "splits_sha256": splits_sha,
        "evaluation_config_sha256": config_sha,
        "config_fingerprint": str(checkpoint["config_fingerprint"]),
        "snapshot_id": str(checkpoint["snapshot_id"]),
        "best_epoch": int(checkpoint["epoch"]),
    }
    return phase3, checkpoint, source


def _evaluation_id(source: dict[str, Any]) -> str:
    encoded = json.dumps(
        source, sort_keys=True, separators=(",", ":")
    ).encode()
    return "test-v1-" + hashlib.sha256(encoded).hexdigest()[:16]


def _create_plots(
    output_dir: Path,
    predictions: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    stratified: dict[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[str] = []
    colors = {"simplex": "#2563eb", "polygonal": "#dc2626"}
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    for family in ("simplex", "polygonal"):
        rows = [
            row for row in predictions if row["mesh_family"] == family
        ]
        axis.scatter(
            [row["target_rho"] for row in rows],
            [row["predicted_rho"] for row in rows],
            s=20,
            alpha=0.7,
            label=family,
            color=colors[family],
        )
    bounds = [
        min(row["target_rho"] for row in predictions),
        max(row["target_rho"] for row in predictions),
    ]
    axis.plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Measured convergence factor")
    axis.set_ylabel("Predicted convergence factor")
    axis.set_title("Held-out test predictions")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    name = "predicted_vs_measured.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    created.append(name)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for family in ("simplex", "polygonal"):
        residuals = [
            row["error"]
            for row in predictions
            if row["mesh_family"] == family
        ]
        axis.hist(
            residuals,
            bins=24,
            alpha=0.55,
            label=family,
            color=colors[family],
        )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Prediction error")
    axis.set_ylabel("Samples")
    axis.set_title("Held-out residual distribution")
    axis.legend()
    figure.tight_layout()
    name = "residual_histogram.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    created.append(name)

    theta_values = sorted(
        (
            (float(theta), metrics["rmse"])
            for theta, metrics in stratified["theta"].items()
        ),
        key=lambda pair: pair[0],
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        [pair[0] for pair in theta_values],
        [pair[1] for pair in theta_values],
        marker="o",
        color="#7c3aed",
    )
    axis.set_xlabel("AMG strong threshold theta")
    axis.set_ylabel("Test RMSE")
    axis.set_title("Prediction error across theta")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    name = "rmse_by_theta.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    created.append(name)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.hist(
        [decision["regret"] for decision in decisions],
        bins=20,
        color="#059669",
        alpha=0.8,
    )
    axis.set_xlabel("True rho regret at predicted-best theta")
    axis.set_ylabel("Matrices")
    axis.set_title("Theta-selection regret")
    figure.tight_layout()
    name = "theta_selection_regret.png"
    figure.savefig(output_dir / name, dpi=180)
    plt.close(figure)
    created.append(name)
    return created


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.10g}"


def _metric_table(groups: dict[str, dict[str, Any]], label: str) -> list[str]:
    lines = [
        f"| {label} | Samples | RMSE | MAE | Bias | R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for value, metrics in groups.items():
        lines.append(
            f"| {value} | {metrics['samples']} | "
            f"{metrics['rmse']:.8g} | {metrics['mae']:.8g} | "
            f"{metrics['bias']:.8g} | "
            f"{_format_metric(metrics['r_squared'])} |"
        )
    return lines


def _report_markdown(
    metrics: dict[str, Any],
    lock: dict[str, Any],
    config: EvaluationConfig,
) -> str:
    overall = metrics["overall"]
    decision = metrics["theta_selection"]["overall"]
    interval = metrics["bootstrap"]["intervals"]
    family_metrics = metrics["stratified"]["mesh_family"]
    if {"polygonal", "simplex"} <= family_metrics.keys():
        family_comparison = (
            "Polygonal RMSE is "
            f"{family_metrics['polygonal']['rmse'] / family_metrics['simplex']['rmse']:.2f}× "
            "the simplex RMSE."
        )
    else:
        family_comparison = (
            "Only one mesh family is present, so no cross-family comparison "
            "is available."
        )
    plot_lines = [
        f"- `{name}`" for name in metrics["artifacts"]["plots"]
    ] or ["- Plot generation disabled."]
    return "\n".join(
        [
            "# ML Phase 4 Held-Out Test Report",
            "",
            "## Locked evaluation",
            "",
            (
                f"Evaluation `{lock['evaluation_id']}` used the Phase 3 "
                f"checkpoint from epoch **{lock['source']['best_epoch']}** "
                "without retraining or test-driven model selection."
            ),
            "",
            f"- Snapshot: `{lock['source']['snapshot_id']}`",
            (
                "- Checkpoint SHA-256: "
                f"`{lock['source']['checkpoint_sha256']}`"
            ),
            f"- Sample index SHA-256: `{lock['source']['samples_sha256']}`",
            f"- Split file SHA-256: `{lock['source']['splits_sha256']}`",
            "- Source-hash leakage across splits: none",
            f"- Held-out samples: {overall['samples']}",
            f"- Held-out source hashes: {metrics['test_source_hashes']}",
            f"- Held-out named matrices: {decision['matrices']}",
            "",
            "## Test regression result",
            "",
            f"- MSE: {overall['mse']:.10g}",
            f"- RMSE: {overall['rmse']:.10g}",
            (
                f"- RMSE {config.confidence_level:.0%} grouped-bootstrap CI: "
                f"[{interval['rmse']['lower']:.10g}, "
                f"{interval['rmse']['upper']:.10g}]"
            ),
            f"- MAE: {overall['mae']:.10g}",
            f"- Bias: {overall['bias']:.10g}",
            f"- R²: {_format_metric(overall['r_squared'])}",
            (
                "- Pearson correlation: "
                f"{_format_metric(overall['pearson_correlation'])}"
            ),
            f"- Maximum absolute error: {overall['max_absolute_error']:.10g}",
            (
                "- Train-mean baseline RMSE: "
                f"{metrics['train_mean_baseline_rmse']:.10g}"
            ),
            (
                "- Validation-to-test RMSE change: "
                f"{metrics['validation_to_test_rmse_change']:+.10g}"
            ),
            (
                "- Predictions outside [0,1]: "
                f"{overall['predictions_below_zero']} below, "
                f"{overall['predictions_above_one']} above"
            ),
            "",
            "## Mesh-family breakdown",
            "",
            *_metric_table(metrics["stratified"]["mesh_family"], "Family"),
            "",
            "## Level breakdown",
            "",
            *_metric_table(metrics["stratified"]["level"], "Level"),
            "",
            "## Theta selection",
            "",
            (
                f"- Exact best-theta selections: "
                f"{decision['exact_optimum_count']}/{decision['matrices']} "
                f"({decision['exact_optimum_rate']:.2%})"
            ),
            f"- Mean rho regret: {decision['mean_regret']:.10g}",
            f"- Median rho regret: {decision['median_regret']:.10g}",
            f"- 95th-percentile rho regret: {decision['p95_regret']:.10g}",
            f"- Maximum rho regret: {decision['max_regret']:.10g}",
            (
                "- Mean absolute theta error: "
                f"{decision['mean_absolute_theta_error']:.10g}"
            ),
            "",
            "| Family | Matrices | Exact rate | Mean regret | Max regret |",
            "|---|---:|---:|---:|---:|",
            *[
                (
                    f"| {family} | {values['matrices']} | "
                    f"{values['exact_optimum_rate']:.2%} | "
                    f"{values['mean_regret']:.8g} | "
                    f"{values['max_regret']:.8g} |"
                )
                for family, values in metrics["theta_selection"][
                    "by_mesh_family"
                ].items()
            ],
            "",
            "## Conclusion",
            "",
            (
                "The CNN predicts held-out convergence factors substantially "
                "better than the train-mean baseline. "
                f"{family_comparison}"
            ),
            "",
            (
                "Exact theta-grid selection is modest, while the resulting "
                "rho regret is usually small. The model is therefore more "
                "credible as an approximate convergence-factor/ranking model "
                "than as an exact theta classifier."
            ),
            "",
            "## Diagnostic plots",
            "",
            *plot_lines,
            "",
            "## Interpretation boundary",
            "",
            (
                "These numbers are the final locked test assessment for "
                "`cnn_paper_v1`. Any architecture, normalization, training, "
                "or hyperparameter changes must create a new model and a new "
                "evaluation version; this test result must not be reused for "
                "model selection."
            ),
            "",
            ("Detailed predictions, stratified metrics, bootstrap settings, "
            "worst cases, and per-matrix theta decisions are stored under "
            f"`{config.output_dir}`."),
            "",
        ]
    )


def _write_report_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def validate_existing_evaluation(
    config: EvaluationConfig,
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = config.output_dir / "evaluation_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("evaluation output exists without a lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation lock schema")
    if lock.get("source") != source:
        raise ValueError("locked evaluation source fingerprints changed")
    for relative, expected_hash in lock["artifacts"].items():
        path = config.output_dir / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"locked evaluation artifact changed: {relative}")
    metrics = json.loads(
        (config.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    _write_report_atomic(
        config.report_path, _report_markdown(metrics, lock, config)
    )
    return metrics, lock


def evaluate(config: EvaluationConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    config.validate()
    phase3, checkpoint, source = _verify_source_contract(config)
    evaluation_id = _evaluation_id(source)
    if config.output_dir.exists():
        return validate_existing_evaluation(config, source)

    seed_everything(config.training_run.training)
    device = resolve_device(config.training_run.device)
    output_parent = config.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_parent,
            prefix=f".{config.output_dir.name}.staging-",
        )
    )
    try:
        predictions = _run_test_inference(
            config, checkpoint, device=device
        )
        overall = regression_metrics(predictions)
        stratified = {
            key: stratified_metrics(predictions, key)
            for key in (
                "mesh_family",
                "level",
                "pattern",
                "epsilon",
                "theta",
            )
        }
        bootstrap = grouped_bootstrap_intervals(
            predictions,
            group_key="matrix_sha256",
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed,
            confidence_level=config.confidence_level,
        )
        decisions, decision_overall = theta_decisions(predictions)
        decision_by_family = _decision_strata(
            decisions, "mesh_family"
        )
        train_mean = float(
            phase3["validation_diagnostics"]["train_target_mean"]
        )
        baseline_mse = float(
            np.mean(
                [
                    (float(row["target_rho"]) - train_mean) ** 2
                    for row in predictions
                ]
            )
        )
        worst = sorted(
            predictions,
            key=lambda row: (-row["absolute_error"], row["sample_id"]),
        )[:20]
        metrics = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "overall": overall,
            "bootstrap": bootstrap,
            "stratified": stratified,
            "theta_selection": {
                "overall": decision_overall,
                "by_mesh_family": decision_by_family,
            },
            "test_source_hashes": len(
                {row["matrix_sha256"] for row in predictions}
            ),
            "train_mean_baseline_rmse": math.sqrt(baseline_mse),
            "validation_rmse": float(
                phase3["best_metrics"]["validation_rmse"]
            ),
            "validation_to_test_rmse_change": (
                overall["rmse"]
                - float(phase3["best_metrics"]["validation_rmse"])
            ),
            "worst_predictions": worst,
            "artifacts": {"plots": []},
        }
        _write_jsonl(staging / "predictions.jsonl", predictions)
        _write_csv(
            staging / "predictions.csv", predictions, PREDICTION_FIELDS
        )
        _write_jsonl(staging / "theta_decisions.jsonl", decisions)
        _write_csv(
            staging / "theta_decisions.csv", decisions, DECISION_FIELDS
        )
        if config.create_plots:
            metrics["artifacts"]["plots"] = _create_plots(
                staging, predictions, decisions, stratified
            )
        write_json_atomic(staging / "metrics.json", metrics)
        artifact_names = [
            "metrics.json",
            "predictions.jsonl",
            "predictions.csv",
            "theta_decisions.jsonl",
            "theta_decisions.csv",
            *metrics["artifacts"]["plots"],
        ]
        lock = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "test_contract": {
                "samples": len(predictions),
                "source_hashes": metrics["test_source_hashes"],
                "matrix_ids": decision_overall["matrices"],
                "split": "test",
                "model_updates": 0,
            },
            "artifacts": {
                name: file_sha256(staging / name)
                for name in artifact_names
            },
        }
        write_json_atomic(staging / "evaluation_lock.json", lock)
        os.replace(staging, config.output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    _write_report_atomic(
        config.report_path, _report_markdown(metrics, lock, config)
    )
    return metrics, lock
