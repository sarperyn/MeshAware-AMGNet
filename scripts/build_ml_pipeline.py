from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_ml.indexing import build_index
from meshaware_ml.inventory import (
    build_feature_cache,
    capture_snapshot,
    load_snapshot,
    save_snapshot,
)
from meshaware_ml.reporting import build_audit_and_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze finalized matrices, build paper-v1 views, and create the "
            "canonical leakage-safe ML index."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / "datasets"
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "datasets" / "ml"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports" / "ml_phase_1_2_report.md",
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=("small", "medium"),
        help="Dataset tier to include; defaults to small and medium.",
    )
    parser.add_argument(
        "--mesh-family",
        action="append",
        choices=("simplex", "polygonal"),
        help="Mesh family to include; defaults to simplex and polygonal.",
    )
    parser.add_argument(
        "--generation-status",
        choices=("active", "inactive", "unknown"),
        default="unknown",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Reuse an existing frozen snapshot instead of capturing a new one.",
    )
    parser.add_argument(
        "--verification-status",
        default="not-yet-run",
        help="Text recorded in the generated report.",
    )
    parser.add_argument(
        "--reset-splits",
        action="store_true",
        help=(
            "Recompute all hash assignments for a new split version. By "
            "default, existing hash assignments remain stable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if args.snapshot is None:
        snapshot = capture_snapshot(
            dataset_root,
            tiers=tuple(args.tier or ("small", "medium")),
            mesh_families=tuple(args.mesh_family or ("simplex", "polygonal")),
        )
        snapshot_path = save_snapshot(snapshot, output_root)
    else:
        snapshot_path = args.snapshot.resolve()
        snapshot = load_snapshot(snapshot_path)
        if Path(snapshot["dataset_root"]).resolve() != dataset_root:
            raise SystemExit(
                "snapshot dataset_root does not match --dataset-root"
            )
    print(f"snapshot={snapshot_path}", flush=True)

    feature_manifest, feature_manifest_path = build_feature_cache(
        snapshot,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    index_summary, index_paths = build_index(
        snapshot,
        feature_manifest,
        dataset_root=dataset_root,
        output_root=output_root,
        repo_root=REPO_ROOT,
        reset_splits=args.reset_splits,
    )
    audit_path = output_root / "audits" / "phase_1_2.json"
    build_audit_and_report(
        repo_root=REPO_ROOT,
        dataset_root=dataset_root,
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        feature_manifest=feature_manifest,
        feature_manifest_path=feature_manifest_path,
        index_summary=index_summary,
        index_paths=index_paths,
        report_path=args.report,
        audit_path=audit_path,
        generation_status=args.generation_status,
        verification_status=args.verification_status,
    )
    print(
        f"done: features={feature_manifest['unique_matrix_hash_count']} "
        f"samples={index_summary['audit']['aggregated_samples']} "
        f"report={args.report} audit={audit_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
