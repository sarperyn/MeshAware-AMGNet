from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .inventory import write_json_atomic


def _git_state(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def _environment() -> dict[str, str]:
    import scipy
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "python_executable": str(__import__("sys").executable),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def _load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_table(
    samples: list[dict[str, Any]], field: str
) -> list[tuple[str, int, int, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        counts[str(sample[field])][str(sample["split"])] += 1
    return [
        (
            value,
            split_counts["train"],
            split_counts["validation"],
            split_counts["test"],
        )
        for value, split_counts in sorted(counts.items())
    ]


def _markdown_table(
    headers: tuple[str, ...], rows: list[tuple[Any, ...]]
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def build_audit_and_report(
    *,
    repo_root: str | Path,
    dataset_root: str | Path,
    snapshot: dict[str, Any],
    snapshot_path: Path,
    feature_manifest: dict[str, Any],
    feature_manifest_path: Path,
    index_summary: dict[str, Any],
    index_paths: dict[str, Path],
    report_path: str | Path,
    audit_path: str | Path,
    generation_status: str,
    verification_status: str,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    report_path = Path(report_path)
    audit_path = Path(audit_path)
    samples = _load_samples(index_paths["samples"])
    git = _git_state(repo_root)
    environment = _environment()
    split_counts = Counter(str(sample["split"]) for sample in samples)
    total = len(samples)

    audit = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": {
            "id": snapshot["snapshot_id"],
            "created_at_utc": snapshot["created_at_utc"],
            "path": str(snapshot_path),
            "inventory_sha256": snapshot["inventory_sha256"],
            "finalized_matrix_files": len(snapshot["matrices"]),
            "record_files": len(snapshot["records"]),
            "excluded_staging_files": snapshot["excluded_staging_files"],
            "generation_status": generation_status,
        },
        "features": {
            "manifest_path": str(feature_manifest_path),
            "matrix_references": feature_manifest["matrix_reference_count"],
            "unique_hashes": feature_manifest["unique_matrix_hash_count"],
            "created": feature_manifest["features_created"],
            "reused": feature_manifest["features_reused"],
            "originating_from_snapshot": feature_manifest[
                "features_originating_from_snapshot"
            ],
            "artifact_mtime_span_seconds": feature_manifest[
                "artifact_mtime_span_seconds"
            ],
            "bytes": feature_manifest["feature_bytes"],
            "elapsed_seconds": feature_manifest["elapsed_seconds"],
            "pooling_spec": feature_manifest["pooling_spec"],
        },
        "index": {
            "paths": {key: str(value) for key, value in index_paths.items()},
            **index_summary,
        },
        "verification_status": verification_status,
        "environment": environment,
        "git": git,
    }
    write_json_atomic(audit_path, audit)

    lines = [
        "# MeshAware-AMG ML Phase 1–2 Report",
        "",
        f"Generated: `{audit['created_at_utc']}`",
        "",
        "## Snapshot boundary",
        "",
        f"- Snapshot: `{snapshot['snapshot_id']}`",
        f"- Captured: `{snapshot['created_at_utc']}`",
        f"- Inventory checksum: `{snapshot['inventory_sha256']}`",
        f"- Generator status at execution: `{generation_status}`",
        f"- Finalized NPZ references captured: {len(snapshot['matrices'])}",
        f"- Record files captured: {len(snapshot['records'])}",
        f"- PETSc staging files excluded: {len(snapshot['excluded_staging_files'])}",
        "",
    ]
    for entry in snapshot["excluded_staging_files"]:
        lines.append(f"  - `{entry['path']}`")

    lines.extend(
        [
            "",
            "The inventory was frozen before feature processing. Files finalized "
            "after this boundary were not admitted.",
            "",
            "## Phase 1: pooled matrix features",
            "",
            "- Representation: `pp+np+sum`, `100×100`, count-average, "
            "signed-log1p/max-abs.",
            f"- Matrix references: {feature_manifest['matrix_reference_count']}",
            f"- Unique source hashes: {feature_manifest['unique_matrix_hash_count']}",
            f"- Features created: {feature_manifest['features_created']}",
            f"- Features reused: {feature_manifest['features_reused']}",
            f"- Features originating from this snapshot: "
            f"{feature_manifest['features_originating_from_snapshot']}",
            f"- Feature storage: {feature_manifest['feature_bytes'] / 1024**2:.2f} MiB",
            f"- Current validation/build pass: "
            f"{feature_manifest['elapsed_seconds']:.2f} s",
            f"- Initial artifact materialization span: "
            f"{feature_manifest['artifact_mtime_span_seconds']:.2f} s",
            "",
            "## Phase 2: canonical samples and splits",
            "",
            f"- Aggregated samples: {total}",
            f"- Named matrices: {index_summary['audit']['named_matrices']}",
            f"- Matrix content hashes: {index_summary['audit']['matrix_hashes']}",
            f"- Duplicate tier records removed: "
            f"{index_summary['audit']['duplicate_records_removed']}",
            "",
        ]
    )
    split_rows = []
    for split in ("train", "validation", "test"):
        count = split_counts[split]
        split_rows.append(
            (
                split,
                count,
                f"{100.0 * count / total:.3f}%" if total else "0%",
                f"{100.0 * index_summary['split_statistics']['ratios'][split]:.1f}%",
            )
        )
    lines.extend(
        _markdown_table(
            ("Split", "Samples", "Actual", "Target"), split_rows
        )
    )

    excluded = index_summary["audit"]["excluded_record_counts"]
    lines.extend(["", "### Exclusions", ""])
    if excluded:
        lines.extend(
            _markdown_table(
                ("Reason", "Records"),
                [(reason, count) for reason, count in sorted(excluded.items())],
            )
        )
    else:
        lines.append("No snapshotted records were excluded.")

    for field, title in (
        ("mesh_family", "Mesh family"),
        ("level", "Refinement level"),
        ("pattern", "Coefficient pattern"),
        ("epsilon", "Coefficient epsilon"),
        ("theta", "Strong threshold theta"),
    ):
        lines.extend(["", f"### {title} distribution", ""])
        lines.extend(
            _markdown_table(
                (title, "Train", "Validation", "Test"),
                _split_table(samples, field),
            )
        )

    ambiguity = index_summary["ambiguity"]
    lines.extend(
        [
            "",
            "## Matrix-only target ambiguity",
            "",
            f"- Identical matrix-hash/theta groups with differing rho: "
            f"{ambiguity['identical_hash_theta_groups_with_different_targets']}",
            f"- Maximum rho spread: `{ambiguity['maximum_rho_spread']:.12g}`",
            f"- Mean rho spread: `{ambiguity['mean_rho_spread']:.12g}`",
        ]
    )
    for family, values in ambiguity["by_family"].items():
        lines.append(
            f"- {family}: {values['groups']} groups, maximum "
            f"`{values['maximum_rho_spread']:.12g}`, mean "
            f"`{values['mean_rho_spread']:.12g}`"
        )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- Status: `{verification_status}`",
            f"- Python: `{environment['python']}` at "
            f"`{environment['python_executable']}`",
            f"- NumPy `{environment['numpy']}`, SciPy `{environment['scipy']}`, "
            f"scikit-learn `{environment['scikit_learn']}`, "
            f"PyTorch `{environment['torch']}`",
            f"- Git revision: `{git['revision']}`",
            f"- Worktree dirty before/after this implementation: `{git['dirty']}`",
            "",
            "## Incremental refresh",
            "",
            "Run the same command after more polygonal NPZ matrices are finalized:",
            "",
            "```bash",
            '"$ENV/pytorch/bin/python" scripts/build_ml_pipeline.py \\',
            "  --dataset-root datasets \\",
            "  --output-root datasets/ml \\",
            "  --report reports/ml_phase_1_2_report.md",
            "```",
            "",
            "Validated existing features are reused. Existing matrix hashes retain "
            "their split; only new hashes are assigned against the current "
            "85/5/10 stratification deficits.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return audit
