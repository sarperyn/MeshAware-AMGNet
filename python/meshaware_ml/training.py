from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from meshaware_data.artifacts import file_sha256, write_json_atomic
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import (
    GroupedRhoDataset,
    HashGroupBatchSampler,
    MatrixViewCache,
    assert_disjoint_hashes,
    collate_matrix_groups,
)
from .model import PaperCNNConfig, PaperRhoCNN

TRAINING_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
HISTORY_FIELDS = (
    "epoch",
    "train_mse",
    "train_rmse",
    "train_mae",
    "validation_mse",
    "validation_rmse",
    "validation_mae",
    "learning_rate",
    "epoch_seconds",
)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 2026
    batch_size: int = 32
    max_epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 1.0e-7
    num_workers: int = 0
    cache_capacity: int = 1024
    deterministic: bool = True
    torch_threads: int = 0

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer hyperparameters")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.num_workers != 0:
            raise ValueError(
                "paper_v1 requires num_workers=0 for bounded shared caching"
            )
        if self.cache_capacity <= 0 or self.torch_threads < 0:
            raise ValueError("invalid cache capacity or torch thread count")


@dataclass(frozen=True)
class RunConfig:
    samples_path: Path
    splits_path: Path
    dataset_root: Path
    output_dir: Path
    report_path: Path
    device: str
    model: PaperCNNConfig
    training: TrainingConfig


def _require_keys(value: dict[str, Any], keys: set[str], section: str) -> None:
    missing = keys - value.keys()
    if missing:
        raise ValueError(f"{section} is missing keys: {sorted(missing)}")


def load_run_config(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> RunConfig:
    config_path = Path(path).resolve()
    repo_root = Path(repo_root).resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported training configuration schema")
    _require_keys(
        value,
        {"dataset", "model", "training", "output"},
        "configuration",
    )
    dataset = value["dataset"]
    output = value["output"]
    _require_keys(
        dataset, {"samples", "splits", "root"}, "dataset configuration"
    )
    _require_keys(output, {"directory", "report"}, "output configuration")

    def resolve(raw: str) -> Path:
        candidate = Path(raw)
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (repo_root / candidate).resolve()
        )

    model_config = PaperCNNConfig(**value["model"])
    training_config = TrainingConfig(**value["training"])
    model_config.validate()
    training_config.validate()
    device = str(value.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return RunConfig(
        samples_path=resolve(dataset["samples"]),
        splits_path=resolve(dataset["splits"]),
        dataset_root=resolve(dataset["root"]),
        output_dir=resolve(output["directory"]),
        report_path=resolve(output["report"]),
        device=device,
        model=model_config,
        training=training_config,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def seed_everything(config: TrainingConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.torch_threads:
        torch.set_num_threads(config.torch_threads)
    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _config_fingerprint(config: RunConfig) -> str:
    value = {
        "model": config.model.to_dict(),
        "training": asdict(config.training),
        "samples_sha256": file_sha256(config.samples_path),
        "splits_sha256": file_sha256(config.splits_path),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _torch_save_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_history_atomic(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(history)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }


def _restore_rng_state(value: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    generator.set_state(value["loader_generator"])


def _metrics(total_squared: float, total_absolute: float, count: int) -> dict[str, float]:
    mse = total_squared / count
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": total_absolute / count,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    optimizer: Adam | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    squared = 0.0
    absolute = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            view = batch["view"].to(device=device, dtype=torch.float32)
            scalars = batch["scalars"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            view_indices = batch.get("view_indices")
            if view_indices is not None:
                view_indices = view_indices.to(device=device)
            prediction = model(view, scalars, view_indices)
            difference = prediction - target
            loss = torch.mean(difference.square())
            if training:
                loss.backward()
                optimizer.step()
            squared += float(difference.detach().square().sum().cpu())
            absolute += float(difference.detach().abs().sum().cpu())
            count += int(target.numel())
    if count == 0:
        raise RuntimeError("data loader produced no samples")
    return _metrics(squared, absolute, count)


def _validation_diagnostics(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    train_target_mean: float,
) -> dict[str, Any]:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    families: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            view = batch["view"].to(device=device, dtype=torch.float32)
            scalars = batch["scalars"].to(
                device=device, dtype=torch.float32
            )
            view_indices = batch["view_indices"].to(device=device)
            prediction = model(view, scalars, view_indices)
            predictions.extend(prediction.cpu().tolist())
            targets.extend(batch["target"].tolist())
            families.extend(str(value) for value in batch["mesh_family"])
    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if not len(target_array) or len(target_array) != len(families):
        raise RuntimeError("validation diagnostics received invalid samples")
    residual = prediction_array - target_array
    target_centered = target_array - target_array.mean()
    total_variation = float(np.dot(target_centered, target_centered))
    residual_sum = float(np.dot(residual, residual))
    prediction_variation = float(
        np.dot(
            prediction_array - prediction_array.mean(),
            prediction_array - prediction_array.mean(),
        )
    )
    correlation = (
        float(np.corrcoef(prediction_array, target_array)[0, 1])
        if len(target_array) > 1
        and total_variation > 0.0
        and prediction_variation > 0.0
        else None
    )
    baseline_residual = target_array - train_target_mean

    by_family: dict[str, Any] = {}
    family_array = np.asarray(families)
    for family in sorted(set(families)):
        mask = family_array == family
        family_residual = residual[mask]
        family_mse = float(np.mean(np.square(family_residual)))
        by_family[family] = {
            "samples": int(mask.sum()),
            "mse": family_mse,
            "rmse": math.sqrt(family_mse),
            "mae": float(np.mean(np.abs(family_residual))),
            "max_absolute_error": float(
                np.max(np.abs(family_residual))
            ),
        }
    return {
        "samples": len(targets),
        "r_squared": (
            1.0 - residual_sum / total_variation
            if total_variation > 0.0
            else None
        ),
        "pearson_correlation": correlation,
        "max_absolute_error": float(np.max(np.abs(residual))),
        "train_target_mean": train_target_mean,
        "train_mean_baseline_rmse": float(
            np.sqrt(np.mean(np.square(baseline_residual)))
        ),
        "by_mesh_family": by_family,
    }


def _checkpoint(
    *,
    config: RunConfig,
    fingerprint: str,
    model: PaperRhoCNN,
    optimizer: Adam,
    generator: torch.Generator,
    epoch: int,
    best_epoch: int,
    best_validation_mse: float,
    stale_epochs: int,
    history: list[dict[str, Any]],
    snapshot_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "snapshot_id": snapshot_id,
        "samples_sha256": file_sha256(config.samples_path),
        "splits_sha256": file_sha256(config.splits_path),
        "model_config": config.model.to_dict(),
        "training_config": asdict(config.training),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(generator),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "stale_epochs": stale_epochs,
        "history": history,
    }


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    model: PaperRhoCNN,
    optimizer: Adam,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, int, float, int, list[dict[str, Any]]]:
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if value.get("config_fingerprint") != fingerprint:
        raise ValueError(
            "checkpoint configuration or frozen index does not match this run"
        )
    model.load_state_dict(value["model_state"])
    optimizer.load_state_dict(value["optimizer_state"])
    _restore_rng_state(value["rng_state"], generator)
    return (
        int(value["epoch"]) + 1,
        int(value["best_epoch"]),
        float(value["best_validation_mse"]),
        int(value["stale_epochs"]),
        list(value["history"]),
    )


def _report_markdown(summary: dict[str, Any], config: RunConfig) -> str:
    best = summary["best_metrics"]
    diagnostics = summary["validation_diagnostics"]
    def metric(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.10g}"

    lines = [
        "# ML Phase 3 Training Report",
        "",
        "## Outcome",
        "",
        (
            f"The paper-v1 CNN trained on **{summary['train_samples']}** "
            f"training samples and selected its checkpoint using "
            f"**{summary['validation_samples']}** validation samples. "
            "No held-out test feature or target was loaded or evaluated."
        ),
        "",
        "## Frozen data contract",
        "",
        f"- Snapshot: `{summary['snapshot_id']}`",
        f"- Sample index SHA-256: `{summary['samples_sha256']}`",
        f"- Split file SHA-256: `{summary['splits_sha256']}`",
        "- Training partitions: `train`, `validation`",
        "- Held-out partition: `test` (no feature/target access)",
        "- Matrix-hash leakage between train and validation: none",
        "",
        "## Architecture",
        "",
        "- Input: `3×100×100` paper-v1 pooled matrix view",
        "- Conditioning scalars: `-log2(h)` and AMG threshold `theta`",
        (
            "- Target mini-batch size: "
            f"{config.training.batch_size}; intact source-hash groups are not "
            f"split (observed maximum: {summary['max_train_batch_samples']})"
        ),
        (
            "- Repeated theta targets for one source hash share one CNN view "
            "computation"
        ),
        (
            f"- Convolutions: {config.model.conv_depth} layers, "
            f"width {config.model.conv_width}, 3×3 kernels; only the first "
            "layer is padded"
        ),
        (
            f"- Max pooling: {config.model.pool_size}×"
            f"{config.model.pool_size}; dropout: {config.model.dropout}"
        ),
        (
            f"- Matrix embedding: {config.model.embedding_width}; "
            f"conditioned dense stack: {config.model.dense_depth}×"
            f"{config.model.dense_width}"
        ),
        f"- Trainable parameters: {summary['trainable_parameters']:,}",
        "- Output: one unconstrained convergence-factor prediction",
        "",
        "## Optimization result",
        "",
        f"- Device: `{summary['device']}`",
        f"- Seed: `{config.training.seed}`",
        f"- Epochs completed: {summary['epochs_completed']}",
        f"- Stop reason: `{summary['stop_reason']}`",
        f"- Best epoch: {summary['best_epoch']}",
        f"- Best validation MSE: {best['validation_mse']:.10g}",
        f"- Best validation RMSE: {best['validation_rmse']:.10g}",
        f"- Best validation MAE: {best['validation_mae']:.10g}",
        f"- Corresponding train RMSE: {best['train_rmse']:.10g}",
        f"- Validation R²: {metric(diagnostics['r_squared'])}",
        (
            "- Validation Pearson correlation: "
            f"{metric(diagnostics['pearson_correlation'])}"
        ),
        (
            "- Validation maximum absolute error: "
            f"{diagnostics['max_absolute_error']:.10g}"
        ),
        (
            "- Train-mean validation baseline RMSE: "
            f"{diagnostics['train_mean_baseline_rmse']:.10g}"
        ),
        f"- Epoch compute time: {summary['training_seconds']:.3f} seconds",
        "",
        "## Validation breakdown",
        "",
        "| Mesh family | Samples | RMSE | MAE | Maximum absolute error |",
        "|---|---:|---:|---:|---:|",
        *[
            (
                f"| {family} | {metrics['samples']} | "
                f"{metrics['rmse']:.10g} | {metrics['mae']:.10g} | "
                f"{metrics['max_absolute_error']:.10g} |"
            )
            for family, metrics in diagnostics["by_mesh_family"].items()
        ],
        "",
        (
            "These are validation diagnostics from the checkpoint selected on "
            "that same validation partition; they are not final generalization "
            "estimates. Phase 4 owns the one-time test evaluation."
        ),
        "",
        "## Artifacts",
        "",
        f"- Best checkpoint: `{summary['artifacts']['best_checkpoint']}`",
        f"- Latest checkpoint: `{summary['artifacts']['latest_checkpoint']}`",
        f"- Epoch history: `{summary['artifacts']['history_csv']}`",
        f"- Machine-readable summary: `{summary['artifacts']['summary_json']}`",
        "",
        ("Phase 4 may load `best.pt` once to evaluate the untouched test split, "
        "including per-family and error-distribution analyses."),
        "",
    ]
    return "\n".join(lines)


def train(
    config: RunConfig,
    *,
    resume: bool = False,
    epoch_limit: int | None = None,
) -> dict[str, Any]:
    config.training.validate()
    config.model.validate()
    if epoch_limit is not None and epoch_limit <= 0:
        raise ValueError("epoch_limit must be positive")
    seed_everything(config.training)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    splits_value = json.loads(config.splits_path.read_text(encoding="utf-8"))
    snapshot_id = str(splits_value["snapshot_id"])
    cache = MatrixViewCache(
        config.dataset_root, capacity=config.training.cache_capacity
    )
    train_data = GroupedRhoDataset(
        config.samples_path,
        config.dataset_root,
        splits=("train",),
        cache=cache,
    )
    validation_data = GroupedRhoDataset(
        config.samples_path,
        config.dataset_root,
        splits=("validation",),
        cache=cache,
    )
    assert_disjoint_hashes(train_data, validation_data)

    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    train_sampler = HashGroupBatchSampler(
        train_data.group_sizes,
        sample_batch_size=config.training.batch_size,
        generator=generator,
        shuffle=True,
    )
    validation_sampler = HashGroupBatchSampler(
        validation_data.group_sizes,
        sample_batch_size=config.training.batch_size,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        num_workers=config.training.num_workers,
        collate_fn=collate_matrix_groups,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_sampler=validation_sampler,
        num_workers=config.training.num_workers,
        collate_fn=collate_matrix_groups,
    )

    model = PaperRhoCNN(config.model).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    fingerprint = _config_fingerprint(config)
    latest_path = config.output_dir / "latest.pt"
    best_path = config.output_dir / "best.pt"
    history_path = config.output_dir / "history.csv"
    summary_path = config.output_dir / "summary.json"

    start_epoch = 1
    best_epoch = 0
    best_validation_mse = math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {latest_path}")
        (
            start_epoch,
            best_epoch,
            best_validation_mse,
            stale_epochs,
            history,
        ) = _load_checkpoint(
            latest_path,
            fingerprint=fingerprint,
            model=model,
            optimizer=optimizer,
            generator=generator,
            device=device,
        )
    elif latest_path.exists() or best_path.exists():
        raise FileExistsError(
            f"run artifacts already exist in {config.output_dir}; "
            "use --resume or a new output directory"
        )

    maximum = config.training.max_epochs
    already_stopped = stale_epochs >= config.training.early_stopping_patience
    if already_stopped:
        maximum = start_epoch - 1
    elif epoch_limit is not None:
        maximum = min(maximum, start_epoch + epoch_limit - 1)
    wall_start = time.perf_counter()
    stop_reason = "early_stopping" if already_stopped else "max_epochs"
    for epoch in range(start_epoch, maximum + 1):
        epoch_start = time.perf_counter()
        train_metrics = _run_epoch(
            model, train_loader, device=device, optimizer=optimizer
        )
        validation_metrics = _run_epoch(
            model, validation_loader, device=device, optimizer=None
        )
        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "validation_mse": validation_metrics["mse"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_mae": validation_metrics["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)

        improved = (
            validation_metrics["mse"]
            < best_validation_mse - config.training.early_stopping_min_delta
        )
        if improved:
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = _checkpoint(
            config=config,
            fingerprint=fingerprint,
            model=model,
            optimizer=optimizer,
            generator=generator,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation_mse=best_validation_mse,
            stale_epochs=stale_epochs,
            history=history,
            snapshot_id=snapshot_id,
        )
        _torch_save_atomic(checkpoint, latest_path)
        if improved:
            _torch_save_atomic(checkpoint, best_path)
        _write_history_atomic(history_path, history)
        print(
            f"epoch={epoch:04d} "
            f"train_rmse={train_metrics['rmse']:.7f} "
            f"validation_rmse={validation_metrics['rmse']:.7f} "
            f"best_epoch={best_epoch} seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if stale_epochs >= config.training.early_stopping_patience:
            stop_reason = "early_stopping"
            break
    else:
        if epoch_limit is not None and maximum < config.training.max_epochs:
            stop_reason = "epoch_limit"

    if not history or best_epoch == 0:
        raise RuntimeError("training produced no valid checkpoint")
    best_row = next(row for row in history if row["epoch"] == best_epoch)
    best_checkpoint = torch.load(
        best_path, map_location=device, weights_only=False
    )
    if best_checkpoint.get("config_fingerprint") != fingerprint:
        raise ValueError("best checkpoint fingerprint does not match this run")
    model.load_state_dict(best_checkpoint["model_state"])
    train_target_mean = float(
        np.mean([float(row["rho_mean"]) for row in train_data.rows])
    )
    validation_diagnostics = _validation_diagnostics(
        model,
        validation_loader,
        device=device,
        train_target_mean=train_target_mean,
    )
    summary = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "samples_sha256": file_sha256(config.samples_path),
        "splits_sha256": file_sha256(config.splits_path),
        "config_fingerprint": fingerprint,
        "device": str(device),
        "train_samples": train_data.sample_count,
        "validation_samples": validation_data.sample_count,
        "train_source_hash_groups": len(train_data),
        "validation_source_hash_groups": len(validation_data),
        "max_train_batch_samples": max(
            config.training.batch_size, max(train_data.group_sizes)
        ),
        "test_samples_evaluated": 0,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "best_metrics": {
            key: best_row[key]
            for key in (
                "train_mse",
                "train_rmse",
                "train_mae",
                "validation_mse",
                "validation_rmse",
                "validation_mae",
            )
        },
        "validation_diagnostics": validation_diagnostics,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "training_seconds": sum(
            float(row["epoch_seconds"]) for row in history
        ),
        "invocation_wall_seconds": time.perf_counter() - wall_start,
        "model": config.model.to_dict(),
        "training": asdict(config.training),
        "artifacts": {
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_path),
            "history_csv": str(history_path),
            "summary_json": str(summary_path),
            "report": str(config.report_path),
        },
    }
    write_json_atomic(summary_path, summary)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = _report_markdown(summary, config)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=config.report_path.parent,
        prefix=f".{config.report_path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(report_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, config.report_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return summary
