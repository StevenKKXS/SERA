#!/usr/bin/env python3
"""Auto-trigger and select a deterministic train12/val8 hard extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from episode_sampling import LOADERS, SAMPLERS
    from prepare_evolution_dataset import atomic_write, load_json, sha256_file
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.episode_sampling import LOADERS, SAMPLERS
    from scripts.codex_rollout.prepare_evolution_dataset import (
        atomic_write,
        load_json,
        sha256_file,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def episode_id(record: dict[str, Any]) -> int:
    value = record.get("episode_id")
    if value is None:
        value = record.get("episode")
    if value is None:
        raise ValueError("Episode record has no episode identity")
    return int(value)


def decide_trigger(gate: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    if gate.get("complete") is not True:
        raise ValueError("Hard exploration requires a complete Evolution50 gate")
    attempts = int(gate["evolution_episode_count"])
    if attempts < 1:
        raise ValueError("Evolution dataset is empty")
    successes = int(gate["successes"])
    success_rate = successes / attempts
    mode = str(settings["mode"])
    if mode not in {"auto", "disabled", "always"}:
        raise ValueError(f"Unsupported hard exploration mode {mode!r}")
    lower = float(settings["trigger_min_success_rate"])
    upper = float(settings["trigger_max_success_rate"])
    in_trigger_range = lower <= success_rate <= upper
    triggered = mode == "always" or (mode == "auto" and in_trigger_range)
    reason = (
        "success_rate_in_auto_range"
        if triggered and mode == "auto"
        else (
            "forced"
            if triggered
            else "disabled" if mode == "disabled" else "success_rate_outside_auto_range"
        )
    )
    return {
        "schema_version": "self-evolve-v3-hard-trigger-v1",
        "task": gate["task"],
        "mode": mode,
        "evolution_attempts": attempts,
        "evolution_successes": successes,
        "evolution_failures": int(gate["failures"]),
        "success_rate": success_rate,
        "trigger_min_success_rate": lower,
        "trigger_max_success_rate": upper,
        "decision": "trigger" if triggered else "skip",
        "reason": reason,
        "train_count": int(settings["train_count"]) if triggered else 0,
        "val_count": int(settings["val_count"]) if triggered else 0,
    }


def exclusion_ids(
    evolution_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
) -> dict[str, set[int]]:
    excluded = {"train": set(), "val": set()}
    for row in evolution_rows + extra_rows:
        split = str(row.get("dataset_split"))
        if split not in excluded:
            raise ValueError(f"Hard extension rejects split {split!r}")
        value = episode_id(row)
        if value in excluded[split]:
            raise ValueError(f"Duplicate excluded episode: {split}:{value}")
        excluded[split].add(value)
    return excluded


def select_split(
    *,
    config: dict[str, Any],
    metadata_root: Path,
    task: str,
    split: str,
    count: int,
    excluded: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampling = config["episode_sampling"]
    dataset_id = str(sampling["dataset_id"])
    dataset_spec = sampling["datasets"][dataset_id]
    loader_name = str(dataset_spec["loader"]["name"])
    base_algorithm = str(dataset_spec["algorithm"]["name"])
    if loader_name not in LOADERS:
        raise ValueError(f"Unknown dataset loader {loader_name!r}")
    if base_algorithm not in SAMPLERS:
        raise ValueError(f"Unknown base sampler {base_algorithm!r}")
    source_records = LOADERS[loader_name](
        metadata_root.resolve(), dataset_spec, split, task
    )
    available = [
        row for row in source_records if int(row["episode_id"]) not in excluded
    ]
    selected = SAMPLERS[base_algorithm](available, count, dataset_spec["algorithm"])
    records = [
        {
            "schema_version": "self-evolve-v3-hard-extension-selection-v1",
            "task": task,
            "dataset_id": dataset_id,
            "dataset_split": split,
            "episode_id": int(row["episode_id"]),
            "attributes": dict(row.get("attributes") or {}),
            "selection_role": "hard_extension",
            "selection_algorithm": (
                "stratified_even_excluding_consumed_v1"
                if base_algorithm == "stratified_even"
                else f"{base_algorithm}_excluding_consumed_v1"
            ),
            "base_sampling_algorithm": base_algorithm,
            "outcome_used_for_selection": False,
        }
        for row in selected
    ]
    provenance = {
        "dataset_split": split,
        "source_record_count": len(source_records),
        "excluded_count": len(excluded),
        "available_count": len(available),
        "selected_count": len(records),
        "base_sampling_algorithm": base_algorithm,
        "outcome_blind": True,
    }
    return records, provenance


def prepare(
    *,
    config_path: Path,
    gate_report_path: Path,
    evolution_manifest_path: Path,
    metadata_root: Path,
    output_dir: Path,
    extra_exclusion_paths: list[Path] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(config_path)
    settings = config["hard_exploration"]
    if settings.get("selection_algorithm") != "stratified_even_excluding_consumed_v1":
        raise ValueError("v3 Hard20 requires stratified_even_excluding_consumed_v1")
    gate = load_json(gate_report_path)
    trigger = decide_trigger(gate, settings)
    evolution_rows = read_jsonl(evolution_manifest_path)
    if len(evolution_rows) != int(gate["evolution_episode_count"]):
        raise ValueError("Evolution manifest and gate episode counts differ")
    extra_rows = []
    for path in extra_exclusion_paths or []:
        extra_rows.extend(read_jsonl(path))
    excluded = exclusion_ids(evolution_rows, extra_rows)
    selections: list[dict[str, Any]] = []
    split_provenance = []
    if trigger["decision"] == "trigger":
        for split, count_key in (("train", "train_count"), ("val", "val_count")):
            rows, provenance = select_split(
                config=config,
                metadata_root=metadata_root,
                task=trigger["task"],
                split=split,
                count=int(trigger[count_key]),
                excluded=excluded[split],
            )
            selections.extend(rows)
            split_provenance.append(provenance)
    trigger.update(
        {
            "selection_algorithm": settings["selection_algorithm"],
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "gate_report_sha256": sha256_file(gate_report_path),
            "evolution_manifest_sha256": sha256_file(evolution_manifest_path),
            "split_provenance": split_provenance,
            "selected_count": len(selections),
            "run_at_most_once": bool(settings.get("run_at_most_once", True)),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output_dir / "trigger_report.json", json.dumps(trigger, indent=2) + "\n"
    )
    atomic_write(
        output_dir / "selection_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selections),
    )
    return trigger, selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--evolution-manifest", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    trigger, selections = prepare(
        config_path=args.config.resolve(),
        gate_report_path=args.gate_report.resolve(),
        evolution_manifest_path=args.evolution_manifest.resolve(),
        metadata_root=args.metadata_root.resolve(),
        output_dir=args.output_dir.resolve(),
        extra_exclusion_paths=[path.resolve() for path in args.exclude_manifest],
    )
    print(
        json.dumps(
            {
                "decision": trigger["decision"],
                "selected_count": len(selections),
                "train_episode_ids": [
                    row["episode_id"]
                    for row in selections
                    if row["dataset_split"] == "train"
                ],
                "val_episode_ids": [
                    row["episode_id"]
                    for row in selections
                    if row["dataset_split"] == "val"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
