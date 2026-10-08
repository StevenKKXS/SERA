#!/usr/bin/env python3
"""Dataset-adapted, outcome-blind episode sampling for rollout campaigns.

The module deliberately separates three concerns:

1. a dataset loader normalizes source metadata into safe episode cards;
2. a named sampler selects cards without seeing rollout outcomes;
3. the caller materializes the selected episode ids for its executor.

Dataset-specific fields such as seeds are excluded unless explicitly listed as
safe sampling attributes. They are never needed by the rollout planner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


EpisodeCard = dict[str, Any]
Sampler = Callable[[list[EpisodeCard], int, dict[str, Any]], list[EpisodeCard]]


def evenly_spaced(rows: list[EpisodeCard], count: int) -> list[EpisodeCard]:
    if count < 0:
        raise ValueError("sample count cannot be negative")
    if count > len(rows):
        raise ValueError(f"requested {count} episodes from only {len(rows)} records")
    if count == 0:
        return []
    if count == 1:
        return [rows[0]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("even spacing produced duplicate source indices")
    return [rows[index] for index in indices]


def uniform_even(
    records: list[EpisodeCard], limit: int, spec: dict[str, Any]
) -> list[EpisodeCard]:
    del spec
    rows = sorted(records, key=lambda row: int(row["episode_id"]))
    return evenly_spaced(rows, limit)


def allocate_strata(
    limit: int, order: list[str], weights: dict[str, float]
) -> dict[str, int]:
    if limit < 1:
        raise ValueError("sample limit must be positive")
    if not order or set(order) != set(weights):
        raise ValueError("stratum_order must contain each configured weight exactly once")
    total = sum(float(weights[name]) for name in order)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"stratum weights must sum to one, got {total}")
    # Ordered remainder preserves the campaign's historical 30 -> 15/7/8 rule.
    allocation = {name: int(limit * float(weights[name])) for name in order}
    for name in reversed(order):
        if sum(allocation.values()) >= limit:
            break
        allocation[name] += 1
    if sum(allocation.values()) != limit:
        raise ValueError("unable to allocate requested sample count")
    return allocation


def stratified_even(
    records: list[EpisodeCard], limit: int, spec: dict[str, Any]
) -> list[EpisodeCard]:
    field = str(spec["stratum_field"])
    order = [str(value) for value in spec["stratum_order"]]
    weights = {str(key): float(value) for key, value in spec["stratum_weights"].items()}
    allocation = allocate_strata(limit, order, weights)
    groups: dict[str, list[EpisodeCard]] = {name: [] for name in order}
    for record in sorted(records, key=lambda row: int(row["episode_id"])):
        value = str(record.get("attributes", {}).get(field))
        if value not in groups:
            raise ValueError(f"unknown {field} stratum {value!r}")
        groups[value].append(record)
    selected: list[EpisodeCard] = []
    for name in order:
        selected.extend(evenly_spaced(groups[name], allocation[name]))
    return sorted(selected, key=lambda row: int(row["episode_id"]))


def explicit_ids(
    records: list[EpisodeCard], limit: int, spec: dict[str, Any]
) -> list[EpisodeCard]:
    requested = [int(value) for value in spec["episode_ids"]]
    if len(requested) != limit:
        raise ValueError("explicit episode_ids length must equal the split sample limit")
    if len(set(requested)) != len(requested):
        raise ValueError("explicit episode_ids must be unique")
    by_id = {int(record["episode_id"]): record for record in records}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"explicit episode ids are absent from metadata: {missing}")
    return [by_id[episode_id] for episode_id in requested]


SAMPLERS: dict[str, Sampler] = {
    "uniform_even": uniform_even,
    "stratified_even": stratified_even,
    "explicit_ids": explicit_ids,
}


def load_json_records(
    source_root: Path, dataset_spec: dict[str, Any], split: str, task: str
) -> list[EpisodeCard]:
    loader = dataset_spec["loader"]
    relative = str(loader["path_template"]).format(split=split, task=task)
    path = (source_root / relative).resolve()
    try:
        path.relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError("dataset metadata path escapes source root") from exc
    value = json.loads(path.read_text(encoding="utf-8"))
    rows: Any = value
    for key in loader.get("records_path", []):
        rows = rows[key]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No metadata records in {path}")
    id_field = str(loader["episode_id_field"])
    safe_fields = [str(field) for field in loader.get("safe_sampling_fields", [])]
    cards = []
    for row in rows:
        cards.append(
            {
                "episode_id": int(row[id_field]),
                "attributes": {field: row[field] for field in safe_fields},
            }
        )
    if len({card["episode_id"] for card in cards}) != len(cards):
        raise ValueError(f"Duplicate episode ids in {path}")
    return cards


LOADERS = {"json_records": load_json_records}


def select_dataset_episodes(
    *, config: dict[str, Any], source_root: Path, split: str, task: str, limit: int
) -> tuple[list[EpisodeCard], dict[str, Any]]:
    sampling = config["episode_sampling"]
    dataset_id = str(sampling["dataset_id"])
    dataset_spec = sampling["datasets"].get(dataset_id)
    if dataset_spec is None:
        raise ValueError(f"No sampling specification for dataset {dataset_id!r}")
    loader_name = str(dataset_spec["loader"]["name"])
    algorithm_name = str(dataset_spec["algorithm"]["name"])
    if loader_name not in LOADERS:
        raise ValueError(f"Unknown dataset loader {loader_name!r}")
    if algorithm_name not in SAMPLERS:
        raise ValueError(f"Unknown episode sampling algorithm {algorithm_name!r}")
    records = LOADERS[loader_name](source_root, dataset_spec, split, task)
    selected = SAMPLERS[algorithm_name](records, limit, dataset_spec["algorithm"])
    provenance = {
        "dataset_id": dataset_id,
        "loader": loader_name,
        "algorithm": algorithm_name,
        "source_record_count": len(records),
        "selected_count": len(selected),
        "outcome_blind": True,
    }
    return selected, provenance
