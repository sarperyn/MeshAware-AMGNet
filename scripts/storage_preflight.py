from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.schema import load_experiment_config
from meshaware_data.storage import (
    add_disk_capacity,
    estimate_experiment_storage,
    human_bytes,
    inventory_existing_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate dataset storage and inventory existing artifacts."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--enforce-free-space",
        action="store_true",
        help="Exit nonzero when retained artifacts exceed free disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    target = args.dataset_root or REPO_ROOT / "datasets" / config.name
    report = add_disk_capacity(estimate_experiment_storage(config), target)
    if args.dataset_root is not None and args.dataset_root.exists():
        report["inventory"] = inventory_existing_dataset(args.dataset_root)

    totals = report["totals"]
    print(
        f"{config.name}: matrices={totals['matrices']:,}, "
        f"trial_records={totals['trial_records']:,}"
    )
    print(
        "matrix storage models: "
        f"PETSc={human_bytes(totals['petsc_matrix_bytes'])}, "
        f"uncompressed-CSR={human_bytes(totals['csr_matrix_bytes'])}, "
        f"compressed-NPZ-estimate="
        f"{human_bytes(totals['npz_matrix_bytes_estimate'])}"
    )
    print(
        "configured retained storage: "
        f"format={config.matrix_format}, "
        f"records={human_bytes(totals['record_bytes'])}, "
        f"total={human_bytes(totals['retained_total_bytes'])}, "
        "maximum-one-matrix-conversion-staging="
        f"{human_bytes(totals['maximum_conversion_staging_bytes'])}"
    )
    print(f"free disk: {human_bytes(report['disk']['free_bytes'])}")
    if "memory" in report:
        print(
            "physical RAM: "
            f"{human_bytes(report['memory']['physical_bytes'])}, "
            "largest solver heuristic="
            f"{human_bytes(report['memory']['maximum_solver_ram_per_matrix_heuristic'])}"
        )
    for warning in report["warnings"]:
        print(f"warning: {warning}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(args.json_output)
        print(f"report: {args.json_output}")

    required = totals["retained_total_bytes"]
    if args.enforce_free_space and required > report["disk"]["free_bytes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
