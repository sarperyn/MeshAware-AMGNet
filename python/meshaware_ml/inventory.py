from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meshaware_data.artifacts import write_json_atomic
from meshaware_data.csr_artifact import read_csr_npz_identity

from .pooling import (
    PAPER_POOLING_SPEC,
    load_and_pool_csr_npz,
    validate_feature_artifact,
    write_feature_artifact_atomic,
)

SNAPSHOT_SCHEMA_VERSION = 1
FEATURE_MANIFEST_SCHEMA_VERSION = 1


def _file_entry(path: Path, dataset_root: Path, tier: str, family: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(dataset_root)),
        "tier": tier,
        "mesh_family": family,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _snapshot_digest(
    matrix_entries: Iterable[dict[str, Any]],
    record_entries: Iterable[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for kind, entries in (("matrix", matrix_entries), ("record", record_entries)):
        for entry in entries:
            digest.update(kind.encode("ascii"))
            digest.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def capture_snapshot(
    dataset_root: str | Path,
    *,
    tiers: tuple[str, ...] = ("small", "medium"),
    mesh_families: tuple[str, ...] = ("simplex", "polygonal"),
) -> dict[str, Any]:
    """Freeze filenames and stats before reading any dataset content."""

    dataset_root = Path(dataset_root).resolve()
    matrices: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    staging_files: list[dict[str, Any]] = []

    for tier in tiers:
        for family in mesh_families:
            family_root = dataset_root / tier / family
            for path in sorted((family_root / "matrices").glob("*.npz")):
                entry = _file_entry(path, dataset_root, tier, family)
                entry["matrix_id"] = path.stem
                matrices.append(entry)
            for path in sorted((family_root / "matrices").glob("*.petsc")):
                entry = _file_entry(path, dataset_root, tier, family)
                entry["matrix_id"] = path.stem
                staging_files.append(entry)
            for path in sorted((family_root / "records").glob("*.json")):
                records.append(_file_entry(path, dataset_root, tier, family))

    digest = _snapshot_digest(matrices, records)
    created_at = datetime.now(timezone.utc)
    snapshot_id = (
        created_at.strftime("%Y%m%dT%H%M%SZ") + "-" + digest[:12]
    )
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_at_utc": created_at.isoformat(),
        "dataset_root": str(dataset_root),
        "tiers": list(tiers),
        "mesh_families": list(mesh_families),
        "matrices": matrices,
        "records": records,
        "excluded_staging_files": staging_files,
        "inventory_sha256": digest,
    }


def save_snapshot(
    snapshot: dict[str, Any], output_root: str | Path
) -> Path:
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema")
    path = (
        Path(output_root)
        / "snapshots"
        / f"{snapshot['snapshot_id']}.json"
    )
    write_json_atomic(path, snapshot)
    return path


def load_snapshot(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema in {path}")
    return snapshot


def _verify_frozen_file(path: Path, entry: dict[str, Any]) -> None:
    stat = path.stat()
    if stat.st_size != int(entry["size_bytes"]) or stat.st_mtime_ns != int(
        entry["mtime_ns"]
    ):
        raise RuntimeError(f"snapshotted file changed after capture: {path}")


def build_feature_cache(
    snapshot: dict[str, Any],
    *,
    dataset_root: str | Path,
    output_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Build or reuse one feature per frozen matrix source hash."""

    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    feature_root = output_root / "features" / PAPER_POOLING_SPEC.name
    started = time.monotonic()
    matrix_references: list[dict[str, Any]] = []
    representative_by_hash: dict[str, tuple[Path, dict[str, Any]]] = {}
    hash_by_matrix_id: dict[str, str] = {}

    for entry in snapshot["matrices"]:
        path = dataset_root / entry["path"]
        _verify_frozen_file(path, entry)
        metadata = read_csr_npz_identity(path)
        source_hash = metadata["source_sha256"]
        matrix_id = str(entry["matrix_id"])
        embedded_matrix_id = str(metadata["identity"].get("matrix_id", ""))
        if embedded_matrix_id and embedded_matrix_id != matrix_id:
            raise ValueError(
                f"matrix filename/identity mismatch: {matrix_id} != "
                f"{embedded_matrix_id}"
            )
        previous_hash = hash_by_matrix_id.setdefault(matrix_id, source_hash)
        if previous_hash != source_hash:
            raise ValueError(
                f"matrix_id {matrix_id} has multiple source checksums"
            )
        feature_path = feature_root / f"{source_hash}.npz"
        matrix_references.append(
            {
                **entry,
                "source_sha256": source_hash,
                "feature_path": str(feature_path.relative_to(dataset_root)),
                "matrix_shape": metadata["shape"],
            }
        )
        representative_by_hash.setdefault(source_hash, (path, entry))

    created = 0
    reused = 0
    artifacts: list[dict[str, Any]] = []
    for index, source_hash in enumerate(sorted(representative_by_hash), start=1):
        source_path, _ = representative_by_hash[source_hash]
        destination = feature_root / f"{source_hash}.npz"
        if destination.exists():
            metadata = validate_feature_artifact(
                destination,
                expected_source_sha256=source_hash,
                expected_spec=PAPER_POOLING_SPEC,
            )
            reused += 1
            status = "reused"
        else:
            view, metadata = load_and_pool_csr_npz(
                source_path, PAPER_POOLING_SPEC
            )
            if metadata["source_sha256"] != source_hash:
                raise ValueError(f"source checksum changed while reading {source_path}")
            metadata["snapshot_id"] = snapshot["snapshot_id"]
            write_feature_artifact_atomic(destination, view, metadata)
            metadata = validate_feature_artifact(
                destination,
                expected_source_sha256=source_hash,
                expected_spec=PAPER_POOLING_SPEC,
            )
            created += 1
            status = "created"
        artifacts.append(
            {
                "source_sha256": source_hash,
                "feature_path": str(destination.relative_to(dataset_root)),
                "size_bytes": destination.stat().st_size,
                "mtime_ns": destination.stat().st_mtime_ns,
                "status": status,
                "origin_snapshot_id": metadata.get("snapshot_id"),
                "matrix_shape": metadata["matrix_shape"],
                "matrix_nnz": metadata["matrix_nnz"],
            }
        )
        print(
            f"[feature {index}/{len(representative_by_hash)}] "
            f"{source_hash[:12]} {status}",
            flush=True,
        )

    artifact_mtimes = [int(item["mtime_ns"]) for item in artifacts]
    manifest = {
        "schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pooling_spec": PAPER_POOLING_SPEC.to_dict(),
        "matrix_reference_count": len(matrix_references),
        "unique_matrix_hash_count": len(artifacts),
        "features_created": created,
        "features_reused": reused,
        "features_originating_from_snapshot": sum(
            item["origin_snapshot_id"] == snapshot["snapshot_id"]
            for item in artifacts
        ),
        "artifact_mtime_span_seconds": (
            (max(artifact_mtimes) - min(artifact_mtimes)) / 1e9
            if artifact_mtimes
            else 0.0
        ),
        "elapsed_seconds": time.monotonic() - started,
        "feature_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "matrix_references": matrix_references,
        "artifacts": artifacts,
    }
    manifest_path = (
        output_root
        / "manifests"
        / f"features-{snapshot['snapshot_id']}.json"
    )
    write_json_atomic(manifest_path, manifest)
    return manifest, manifest_path
