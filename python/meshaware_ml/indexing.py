from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meshaware_data.artifacts import write_json_atomic
from meshaware_data.schema import ExperimentConfig, load_experiment_config

INDEX_SCHEMA_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
SPLIT_RATIOS = {"train": 0.85, "validation": 0.05, "test": 0.10}
SPLIT_NAMES = tuple(SPLIT_RATIOS)
SPLIT_SEED = 2026


def _float_token(value: float) -> str:
    return format(float(value), ".17g")


def _stable_sample_id(matrix_id: str, theta: float) -> str:
    digest = hashlib.sha256(
        f"{matrix_id}|{_float_token(theta)}".encode()
    ).hexdigest()
    return f"rho-{digest[:24]}"


def _verify_frozen_record(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    if stat.st_size != int(entry["size_bytes"]) or stat.st_mtime_ns != int(
        entry["mtime_ns"]
    ):
        raise RuntimeError(f"snapshotted record changed after capture: {path}")
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema_version") != 1:
        raise ValueError(f"unsupported record schema in {path}")
    record["_snapshot_path"] = str(entry["path"])
    record["_tier"] = str(entry["tier"])
    return record


def _expected_trial_keys(
    config: ExperimentConfig, level: int
) -> set[tuple[str, int]]:
    repeats = config.repeats_by_level.get(level, config.repeats)
    return {
        (_float_token(theta), repeat)
        for theta in config.theta_values
        for repeat in range(repeats)
    }


def _record_trial_key(record: dict[str, Any]) -> tuple[str, int]:
    return _float_token(float(record["theta"])), int(record["repeat"])


def _dedup_comparison(record: dict[str, Any]) -> tuple[Any, ...]:
    fields = (
        "matrix_id",
        "mesh_family",
        "level",
        "h_nominal",
        "theta",
        "repeat",
        "pattern",
        "epsilon",
        "high_region",
        "convergence_factor",
    )
    return tuple(record[field] for field in fields)


def _load_configs(
    repo_root: Path, tiers: Iterable[str]
) -> dict[str, ExperimentConfig]:
    configs = {}
    for tier in tiers:
        path = repo_root / "configs" / f"{tier}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing tier configuration: {path}")
        configs[tier] = load_experiment_config(path)
    return configs


def build_canonical_samples(
    snapshot: dict[str, Any],
    feature_manifest: dict[str, Any],
    *,
    dataset_root: str | Path,
    repo_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = Path(dataset_root).resolve()
    repo_root = Path(repo_root).resolve()
    configs = _load_configs(repo_root, snapshot["tiers"])

    matrix_reference_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for reference in feature_manifest["matrix_references"]:
        key = (
            str(reference["tier"]),
            str(reference["mesh_family"]),
            str(reference["matrix_id"]),
        )
        matrix_reference_by_key[key] = reference

    grouped_records: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    raw_record_count = 0
    for entry in snapshot["records"]:
        path = dataset_root / entry["path"]
        record = _verify_frozen_record(path, entry)
        raw_record_count += 1
        key = (
            str(entry["tier"]),
            str(entry["mesh_family"]),
            str(record["matrix_id"]),
        )
        grouped_records[key].append(record)

    excluded_by_reason: Counter[str] = Counter()
    excluded_matrix_ids: dict[str, set[str]] = defaultdict(set)
    complete_records: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for key, records in sorted(grouped_records.items()):
        tier, family, matrix_id = key
        reference = matrix_reference_by_key.get(key)
        if reference is None:
            excluded_by_reason["no_finalized_snapshotted_npz"] += len(records)
            excluded_matrix_ids["no_finalized_snapshotted_npz"].add(matrix_id)
            continue
        levels = {int(record["level"]) for record in records}
        if len(levels) != 1:
            raise ValueError(f"records disagree on level for {key}")
        level = next(iter(levels))
        actual_keys = [_record_trial_key(record) for record in records]
        if len(set(actual_keys)) != len(actual_keys):
            raise ValueError(f"duplicate trial record inside tier group {key}")
        expected_keys = _expected_trial_keys(configs[tier], level)
        if set(actual_keys) != expected_keys:
            excluded_by_reason["incomplete_theta_repeat_grid"] += len(records)
            excluded_matrix_ids["incomplete_theta_repeat_grid"].add(matrix_id)
            continue
        for record in records:
            if record["mesh_family"] != family:
                raise ValueError(f"record family/path mismatch for {record}")
            complete_records.append((record, reference))

    for key, reference in matrix_reference_by_key.items():
        if key not in grouped_records:
            excluded_by_reason["finalized_matrix_without_records"] += 1
            excluded_matrix_ids["finalized_matrix_without_records"].add(
                str(reference["matrix_id"])
            )

    deduplicated: dict[
        tuple[str, str, int], tuple[dict[str, Any], dict[str, Any], set[str], list[str]]
    ] = {}
    duplicate_record_count = 0
    for record, reference in complete_records:
        key = (
            str(record["matrix_id"]),
            _float_token(float(record["theta"])),
            int(record["repeat"]),
        )
        current = deduplicated.get(key)
        if current is None:
            deduplicated[key] = (
                record,
                reference,
                {str(record["_tier"])},
                [str(record["_snapshot_path"])],
            )
            continue
        previous, previous_reference, tiers, paths = current
        if (
            _dedup_comparison(previous) != _dedup_comparison(record)
            or previous_reference["source_sha256"] != reference["source_sha256"]
        ):
            raise ValueError(f"conflicting duplicate trial {key}")
        duplicate_record_count += 1
        tiers.add(str(record["_tier"]))
        paths.append(str(record["_snapshot_path"]))

    by_matrix_theta: dict[
        tuple[str, str], list[
            tuple[dict[str, Any], dict[str, Any], set[str], list[str]]
        ]
    ] = defaultdict(list)
    for (matrix_id, theta_token, _), value in deduplicated.items():
        by_matrix_theta[(matrix_id, theta_token)].append(value)

    samples: list[dict[str, Any]] = []
    for (matrix_id, theta_token), values in sorted(by_matrix_theta.items()):
        values.sort(key=lambda value: int(value[0]["repeat"]))
        first_record, reference, _, _ = values[0]
        source_hashes = {
            str(value[1]["source_sha256"]) for value in values
        }
        if len(source_hashes) != 1:
            raise ValueError(f"repeat records use different matrices: {matrix_id}")
        metadata_fields = (
            "mesh_family",
            "level",
            "h_nominal",
            "pattern",
            "epsilon",
            "high_region",
        )
        for record, _, _, _ in values[1:]:
            if any(record[field] != first_record[field] for field in metadata_fields):
                raise ValueError(f"repeat metadata conflict for {matrix_id}")
        rhos = [float(value[0]["convergence_factor"]) for value in values]
        source_tiers = sorted(
            {tier for _, _, tiers, _ in values for tier in tiers}
        )
        source_records = sorted(
            {path for _, _, _, paths in values for path in paths}
        )
        theta = float(theta_token)
        samples.append(
            {
                "schema_version": INDEX_SCHEMA_VERSION,
                "sample_id": _stable_sample_id(matrix_id, theta),
                "matrix_id": matrix_id,
                "matrix_sha256": next(iter(source_hashes)),
                "feature_path": str(reference["feature_path"]),
                "mesh_family": str(first_record["mesh_family"]),
                "level": int(first_record["level"]),
                "h_nominal": float(first_record["h_nominal"]),
                "theta": theta,
                "rho_mean": statistics.mean(rhos),
                "rho_std": statistics.pstdev(rhos) if len(rhos) > 1 else 0.0,
                "repeat_count": len(rhos),
                "pattern": str(first_record["pattern"]),
                "epsilon": float(first_record["epsilon"]),
                "high_region": str(first_record["high_region"]),
                "source_tiers": source_tiers,
                "source_records": source_records,
            }
        )

    audit = {
        "raw_snapshot_records": raw_record_count,
        "records_in_complete_tier_groups": len(complete_records),
        "duplicate_records_removed": duplicate_record_count,
        "deduplicated_repeat_records": len(deduplicated),
        "aggregated_samples": len(samples),
        "named_matrices": len({sample["matrix_id"] for sample in samples}),
        "matrix_hashes": len({sample["matrix_sha256"] for sample in samples}),
        "excluded_record_counts": dict(sorted(excluded_by_reason.items())),
        "excluded_matrix_ids": {
            reason: sorted(matrix_ids)
            for reason, matrix_ids in sorted(excluded_matrix_ids.items())
        },
    }
    return samples, audit


def _group_labels(samples: list[dict[str, Any]]) -> Counter[str]:
    labels: Counter[str] = Counter()
    for sample in samples:
        family = str(sample["mesh_family"])
        level = int(sample["level"])
        labels[f"family={family}"] += 1
        labels[f"level={level}"] += 1
        labels[f"family_level={family}:{level}"] += 1
        labels[f"pattern={sample['pattern']}"] += 1
        labels[f"epsilon={_float_token(sample['epsilon'])}"] += 1
        labels[f"theta={_float_token(sample['theta'])}"] += 1
    return labels


def _split_objective(
    split: str,
    group_size: int,
    group_labels: Counter[str],
    current_totals: Counter[str],
    current_labels: dict[str, Counter[str]],
    target_totals: dict[str, float],
    target_labels: dict[str, dict[str, float]],
) -> float:
    total_deficit = (
        target_totals[split] - current_totals[split]
    ) / max(target_totals[split], 1.0)
    label_score = 0.0
    label_weight = 0
    for label, count in group_labels.items():
        target = target_labels[split][label]
        deficit = (target - current_labels[split][label]) / max(target, 1.0)
        label_score += deficit * count
        label_weight += count
    if label_weight:
        label_score /= label_weight
    projected_overflow = max(
        0.0,
        current_totals[split] + group_size - target_totals[split],
    ) / max(target_totals[split], 1.0)
    return 3.0 * total_deficit + label_score - 8.0 * projected_overflow


def assign_grouped_stratified_splits(
    samples: list[dict[str, Any]],
    *,
    previous_assignments: dict[str, str] | None = None,
    seed: int = SPLIT_SEED,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["matrix_sha256"])].append(sample)
    if not groups:
        raise ValueError("cannot split an empty dataset")

    labels_by_group = {
        source_hash: _group_labels(group) for source_hash, group in groups.items()
    }
    global_labels: Counter[str] = Counter()
    for labels in labels_by_group.values():
        global_labels.update(labels)
    total_samples = len(samples)
    target_totals = {
        split: total_samples * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    target_labels = {
        split: {
            label: count * SPLIT_RATIOS[split]
            for label, count in global_labels.items()
        }
        for split in SPLIT_NAMES
    }
    assignments: dict[str, str] = {}
    current_totals: Counter[str] = Counter()
    current_labels = {split: Counter() for split in SPLIT_NAMES}
    fixed_hashes: set[str] = set()

    for source_hash, split in (previous_assignments or {}).items():
        if source_hash not in groups:
            continue
        if split not in SPLIT_NAMES:
            raise ValueError(f"invalid previous split {split}")
        assignments[source_hash] = split
        fixed_hashes.add(source_hash)
        current_totals[split] += len(groups[source_hash])
        current_labels[split].update(labels_by_group[source_hash])

    def rarity(source_hash: str) -> float:
        labels = labels_by_group[source_hash]
        return sum(
            count / global_labels[label] for label, count in labels.items()
        )

    def tie_key(source_hash: str) -> str:
        return hashlib.sha256(f"{seed}:{source_hash}".encode("ascii")).hexdigest()

    pending_set = {
        source_hash for source_hash in groups if source_hash not in assignments
    }
    new_hashes = set(pending_set)

    def assign(source_hash: str, split: str) -> None:
        assignments[source_hash] = split
        current_totals[split] += len(groups[source_hash])
        current_labels[split].update(labels_by_group[source_hash])
        pending_set.discard(source_hash)

    # Explicitly seed coverage for every sufficiently represented
    # family/refinement stratum. A 5% validation target can otherwise round to
    # zero for a small stratum even though the global split is well balanced.
    hashes_by_family_level: dict[tuple[str, int], set[str]] = defaultdict(set)
    for source_hash, group in groups.items():
        for sample in group:
            hashes_by_family_level[
                (str(sample["mesh_family"]), int(sample["level"]))
            ].add(source_hash)
    for _, stratum_hashes in sorted(hashes_by_family_level.items()):
        if len(stratum_hashes) < len(SPLIT_NAMES):
            continue
        present = {
            assignments[source_hash]
            for source_hash in stratum_hashes
            if source_hash in assignments
        }
        for missing_split in (
            split for split in SPLIT_NAMES if split not in present
        ):
            candidates = sorted(
                stratum_hashes.intersection(pending_set),
                key=lambda source_hash: (
                    len(groups[source_hash]),
                    tie_key(source_hash),
                ),
            )
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda source_hash: _split_objective(
                    missing_split,
                    len(groups[source_hash]),
                    labels_by_group[source_hash],
                    current_totals,
                    current_labels,
                    target_totals,
                    target_labels,
                ),
            )
            assign(best, missing_split)
            present.add(missing_split)

    # Preserve categorical coverage as well as marginal proportions. This is
    # particularly important for epsilon=0, whose identical operators form
    # larger hash groups than the heterogeneous cases.
    coverage_prefixes = ("family=", "pattern=", "epsilon=")
    hashes_by_label: dict[str, set[str]] = defaultdict(set)
    for source_hash, labels in labels_by_group.items():
        for label in labels:
            if label.startswith(coverage_prefixes):
                hashes_by_label[label].add(source_hash)
    for _, label_hashes in sorted(hashes_by_label.items()):
        if len(label_hashes) < len(SPLIT_NAMES):
            continue
        present = {
            assignments[source_hash]
            for source_hash in label_hashes
            if source_hash in assignments
        }
        for missing_split in (
            split for split in SPLIT_NAMES if split not in present
        ):
            candidates = sorted(
                label_hashes.intersection(pending_set),
                key=lambda source_hash: (
                    len(groups[source_hash]),
                    tie_key(source_hash),
                ),
            )
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda source_hash: _split_objective(
                    missing_split,
                    len(groups[source_hash]),
                    labels_by_group[source_hash],
                    current_totals,
                    current_labels,
                    target_totals,
                    target_labels,
                ),
            )
            assign(best, missing_split)
            present.add(missing_split)

    pending = sorted(
        pending_set,
        key=lambda source_hash: (
            -rarity(source_hash),
            -len(groups[source_hash]),
            tie_key(source_hash),
        ),
    )
    split_priority = {name: -index for index, name in enumerate(SPLIT_NAMES)}
    for source_hash in pending:
        group_size = len(groups[source_hash])
        labels = labels_by_group[source_hash]
        split = max(
            SPLIT_NAMES,
            key=lambda candidate: (
                _split_objective(
                    candidate,
                    group_size,
                    labels,
                    current_totals,
                    current_labels,
                    target_totals,
                    target_labels,
                ),
                split_priority[candidate],
            ),
        )
        assign(source_hash, split)

    actual_ratios = {
        split: current_totals[split] / total_samples for split in SPLIT_NAMES
    }
    maximum_group_size = max(len(group) for group in groups.values())
    for split in SPLIT_NAMES:
        if (
            abs(current_totals[split] - target_totals[split])
            > maximum_group_size
        ):
            raise RuntimeError(
                f"{split} sample count is outside one-group tolerance: "
                f"{current_totals[split]} vs {target_totals[split]:.1f}"
            )

    for family in sorted({sample["mesh_family"] for sample in samples}):
        present = {
            assignments[str(sample["matrix_sha256"])]
            for sample in samples
            if sample["mesh_family"] == family
        }
        if present != set(SPLIT_NAMES):
            raise RuntimeError(f"family {family} is absent from a split")

    family_level_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    family_level_splits: dict[tuple[str, int], set[str]] = defaultdict(set)
    for sample in samples:
        key = (str(sample["mesh_family"]), int(sample["level"]))
        source_hash = str(sample["matrix_sha256"])
        family_level_hashes[key].add(source_hash)
        family_level_splits[key].add(assignments[source_hash])
    for key, hashes in family_level_hashes.items():
        if len(hashes) >= 3 and family_level_splits[key] != set(SPLIT_NAMES):
            raise RuntimeError(f"family/level {key} is absent from a split")

    label_distributions: dict[str, dict[str, dict[str, float | int]]] = {}
    for label in sorted(global_labels):
        label_distributions[label] = {}
        for split in SPLIT_NAMES:
            actual = current_labels[split][label]
            target = target_labels[split][label]
            label_distributions[label][split] = {
                "count": actual,
                "target": target,
                "fraction_of_label": actual / global_labels[label],
                "deviation_from_target": actual - target,
            }

    stats = {
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "sample_counts": dict(current_totals),
        "actual_ratios": actual_ratios,
        "target_sample_counts": target_totals,
        "matrix_hash_counts": dict(
            Counter(assignments[source_hash] for source_hash in groups)
        ),
        "maximum_group_size": maximum_group_size,
        "fixed_existing_hash_count": len(fixed_hashes),
        "new_hash_count": len(new_hashes),
        "label_distributions": label_distributions,
    }
    return assignments, stats


def _load_previous_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported split schema in {path}")
    if value.get("seed") != SPLIT_SEED or value.get("ratios") != SPLIT_RATIOS:
        raise ValueError("existing split manifest uses a different policy")
    return {
        str(source_hash): str(split)
        for source_hash, split in value["assignments"].items()
    }


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ambiguity_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[
            (
                str(sample["matrix_sha256"]),
                _float_token(float(sample["theta"])),
            )
        ].append(sample)
    by_family: dict[str, list[float]] = defaultdict(list)
    all_spreads: list[float] = []
    ambiguous_groups = 0
    for group in grouped.values():
        matrix_ids = {sample["matrix_id"] for sample in group}
        if len(matrix_ids) < 2:
            continue
        rhos = [float(sample["rho_mean"]) for sample in group]
        spread = max(rhos) - min(rhos)
        if spread > 0.0:
            ambiguous_groups += 1
            all_spreads.append(spread)
            for family in {str(sample["mesh_family"]) for sample in group}:
                by_family[family].append(spread)
    return {
        "identical_hash_theta_groups_with_different_targets": ambiguous_groups,
        "maximum_rho_spread": max(all_spreads, default=0.0),
        "mean_rho_spread": statistics.mean(all_spreads) if all_spreads else 0.0,
        "by_family": {
            family: {
                "groups": len(spreads),
                "maximum_rho_spread": max(spreads, default=0.0),
                "mean_rho_spread": statistics.mean(spreads) if spreads else 0.0,
            }
            for family, spreads in sorted(by_family.items())
        },
    }


def build_index(
    snapshot: dict[str, Any],
    feature_manifest: dict[str, Any],
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    reset_splits: bool = False,
) -> tuple[dict[str, Any], dict[str, Path]]:
    output_root = Path(output_root).resolve()
    index_root = output_root / "index" / "paper_v1"
    samples, audit = build_canonical_samples(
        snapshot,
        feature_manifest,
        dataset_root=dataset_root,
        repo_root=repo_root,
    )
    splits_path = index_root / "splits.json"
    previous_assignments = (
        {} if reset_splits else _load_previous_assignments(splits_path)
    )
    assignments, split_stats = assign_grouped_stratified_splits(
        samples, previous_assignments=previous_assignments
    )
    for sample in samples:
        sample["split"] = assignments[str(sample["matrix_sha256"])]
    samples.sort(key=lambda row: (row["split"], row["matrix_id"], row["theta"]))

    ambiguity = _ambiguity_summary(samples)
    summary = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_manifest_snapshot_id": feature_manifest["snapshot_id"],
        "audit": audit,
        "split_statistics": split_stats,
        "ambiguity": ambiguity,
    }
    splits = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "assignments": dict(sorted(assignments.items())),
        "statistics": split_stats,
    }
    samples_path = index_root / "samples.jsonl"
    summary_path = index_root / "summary.json"
    _write_jsonl_atomic(samples_path, samples)
    write_json_atomic(splits_path, splits)
    write_json_atomic(summary_path, summary)
    return summary, {
        "samples": samples_path,
        "splits": splits_path,
        "summary": summary_path,
    }
