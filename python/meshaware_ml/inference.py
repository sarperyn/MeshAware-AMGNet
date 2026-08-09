from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from meshaware_data.artifacts import file_sha256

from .dataset import load_index_rows
from .model import PaperRhoCNN
from .pooling import PAPER_POOLING_SPEC, load_and_pool_csr_npz
from .training import (
    CHECKPOINT_SCHEMA_VERSION,
    RunConfig,
    load_run_config,
    resolve_device,
    seed_everything,
)

INFERENCE_SCHEMA_VERSION = 1
SUPPORTED_MESH_FAMILIES = frozenset({"simplex", "polygonal"})


@dataclass(frozen=True)
class PredictorPaths:
    training_config: Path
    checkpoint: Path
    phase3_summary: Path


def parse_theta_values(values: Iterable[float]) -> tuple[float, ...]:
    candidates = tuple(sorted(float(value) for value in values))
    if not candidates:
        raise ValueError("at least one theta candidate is required")
    if any(
        not math.isfinite(theta) or not 0.0 < theta < 1.0
        for theta in candidates
    ):
        raise ValueError("all theta candidates must be finite and in (0, 1)")
    if len(set(candidates)) != len(candidates):
        raise ValueError("theta candidates must be unique")
    return candidates


class RhoPredictor:
    """Verified inference-only wrapper around the selected paper-v1 CNN."""

    def __init__(
        self,
        run: RunConfig,
        checkpoint_path: str | Path,
        phase3_summary_path: str | Path,
        *,
        device: str | None = None,
    ):
        self.run = replace(run, device=device) if device else run
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.phase3_summary_path = Path(phase3_summary_path).resolve()
        self._checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.device = resolve_device(self.run.device)
        seed_everything(self.run.training)
        self.summary = json.loads(
            self.phase3_summary_path.read_text(encoding="utf-8")
        )
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self._verify_model_contract()
        self.model = PaperRhoCNN(self.run.model).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()
        self.training_domain = self._training_domain()

    @classmethod
    def from_paths(
        cls,
        paths: PredictorPaths,
        *,
        repo_root: str | Path,
        device: str | None = None,
    ) -> RhoPredictor:
        run = load_run_config(paths.training_config, repo_root=repo_root)
        return cls(
            run,
            paths.checkpoint,
            paths.phase3_summary,
            device=device,
        )

    def _verify_model_contract(self) -> None:
        samples_sha = file_sha256(self.run.samples_path)
        splits_sha = file_sha256(self.run.splits_path)
        checkpoint = self.checkpoint
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported inference checkpoint schema")
        if checkpoint.get("samples_sha256") != samples_sha:
            raise ValueError("checkpoint sample-index fingerprint changed")
        if checkpoint.get("splits_sha256") != splits_sha:
            raise ValueError("checkpoint split fingerprint changed")
        if self.summary.get("samples_sha256") != samples_sha:
            raise ValueError("Phase 3 sample-index fingerprint changed")
        if self.summary.get("splits_sha256") != splits_sha:
            raise ValueError("Phase 3 split fingerprint changed")
        if checkpoint.get("epoch") != self.summary.get("best_epoch"):
            raise ValueError("checkpoint is not the selected Phase 3 epoch")
        if checkpoint.get("config_fingerprint") != self.summary.get(
            "config_fingerprint"
        ):
            raise ValueError("checkpoint and Phase 3 summary differ")
        if checkpoint.get("model_config") != self.run.model.to_dict():
            raise ValueError("checkpoint architecture differs from config")
        for name, tensor in checkpoint["model_state"].items():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"checkpoint tensor is non-finite: {name}")

    def _training_domain(self) -> dict[str, Any]:
        rows = load_index_rows(
            self.run.samples_path, splits=("train", "validation")
        )
        return {
            "mesh_families": sorted(
                {str(row["mesh_family"]) for row in rows}
            ),
            "level_min": min(int(row["level"]) for row in rows),
            "level_max": max(int(row["level"]) for row in rows),
            "theta_min": min(float(row["theta"]) for row in rows),
            "theta_max": max(float(row["theta"]) for row in rows),
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_epoch": int(self.checkpoint["epoch"]),
            "config_fingerprint": str(
                self.checkpoint["config_fingerprint"]
            ),
            "snapshot_id": str(self.checkpoint["snapshot_id"]),
            "model_config": self.run.model.to_dict(),
            "pooling_spec": PAPER_POOLING_SPEC.to_dict(),
            "device": str(self.device),
        }

    def _resolve_level(
        self, identity: dict[str, Any], requested: int | None
    ) -> int:
        embedded_raw = identity.get("level")
        embedded = int(embedded_raw) if embedded_raw is not None else None
        if requested is None and embedded is None:
            raise ValueError(
                "matrix identity has no level; provide an explicit level"
            )
        if requested is not None and requested < 0:
            raise ValueError("level must be non-negative")
        if requested is not None and embedded is not None and requested != embedded:
            raise ValueError(
                f"requested level {requested} conflicts with matrix "
                f"identity level {embedded}"
            )
        return embedded if requested is None else requested

    def recommend_matrix(
        self,
        matrix_path: str | Path,
        theta_values: Iterable[float],
        *,
        level: int | None = None,
    ) -> dict[str, Any]:
        matrix_path = Path(matrix_path).resolve()
        candidates = parse_theta_values(theta_values)
        feature_start = time.perf_counter()
        view, metadata = load_and_pool_csr_npz(matrix_path)
        feature_seconds = time.perf_counter() - feature_start
        identity = dict(metadata["matrix_identity"])
        family = identity.get("mesh_family")
        if family is not None and family not in SUPPORTED_MESH_FAMILIES:
            raise ValueError(
                f"model does not support mesh family {family!r}; "
                f"supported={sorted(SUPPORTED_MESH_FAMILIES)}"
            )
        resolved_level = self._resolve_level(identity, level)
        warnings: list[str] = []
        if not (
            self.training_domain["level_min"]
            <= resolved_level
            <= self.training_domain["level_max"]
        ):
            warnings.append(
                "level lies outside the train/validation support "
                f"[{self.training_domain['level_min']}, "
                f"{self.training_domain['level_max']}]"
            )
        outside_theta = [
            theta
            for theta in candidates
            if not (
                self.training_domain["theta_min"]
                <= theta
                <= self.training_domain["theta_max"]
            )
        ]
        if outside_theta:
            warnings.append(
                "theta candidates outside train/validation support "
                f"[{self.training_domain['theta_min']}, "
                f"{self.training_domain['theta_max']}]: {outside_theta}"
            )

        view_tensor = torch.from_numpy(
            np.asarray(view, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        scalars = torch.tensor(
            [[float(resolved_level), theta] for theta in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        view_indices = torch.zeros(
            len(candidates), dtype=torch.int64, device=self.device
        )
        inference_start = time.perf_counter()
        with torch.inference_mode():
            predicted = (
                self.model(view_tensor, scalars, view_indices).cpu().tolist()
            )
        inference_seconds = time.perf_counter() - inference_start
        ranked = sorted(
            zip(candidates, predicted, strict=True),
            key=lambda pair: (float(pair[1]), float(pair[0])),
        )
        rank_by_theta = {
            theta: rank
            for rank, (theta, _) in enumerate(ranked, start=1)
        }
        predictions = [
            {
                "theta": theta,
                "predicted_rho": float(prediction),
                "rank": rank_by_theta[theta],
            }
            for theta, prediction in zip(
                candidates, predicted, strict=True
            )
        ]
        recommended_theta, recommended_rho = ranked[0]
        return {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "inference_only",
            "model_updates": 0,
            "matrix": {
                "path": str(matrix_path),
                "source_sha256": metadata["source_sha256"],
                "identity": identity,
                "shape": metadata["matrix_shape"],
                "nnz": metadata["matrix_nnz"],
            },
            "inputs": {
                "level": resolved_level,
                "h_nominal": math.pow(2.0, -resolved_level),
                "theta_values": list(candidates),
            },
            "predictions": predictions,
            "recommendation": {
                "theta": float(recommended_theta),
                "predicted_rho": float(recommended_rho),
            },
            "training_domain": self.training_domain,
            "warnings": warnings,
            "timing_seconds": {
                "feature_construction": feature_seconds,
                "model_inference": inference_seconds,
            },
            "provenance": self.provenance,
        }
