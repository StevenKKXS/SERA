#!/usr/bin/env python3
"""Gate and report Hard Extension20 independently from Evolution50."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from prepare_evolution_dataset import (
        atomic_write,
        collect_split,
        load_json,
        sha256_file,
    )
    from prepare_hard_extension import read_jsonl
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.prepare_evolution_dataset import (
        atomic_write,
        collect_split,
        load_json,
        sha256_file,
    )
    from scripts.codex_rollout.prepare_hard_extension import read_jsonl


def prepare(
    *,
    config_path: Path,
    selection_manifest_path: Path,
    train_run_root: Path | list[Path] | tuple[Path, ...],
    val_run_root: Path | list[Path] | tuple[Path, ...],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(config_path)
    settings = config["hard_exploration"]
    selected = read_jsonl(selection_manifest_path)
    if not selected:
        raise ValueError("Hard extension selection is empty")
    tasks = {str(row["task"]) for row in selected}
    if len(tasks) != 1:
        raise ValueError("Hard extension must contain exactly one task")
    task = next(iter(tasks))
    expected = {
        split: [
            int(row["episode_id"]) for row in selected if row["dataset_split"] == split
        ]
        for split in ("train", "val")
    }
    if len(expected["train"]) != int(settings["train_count"]):
        raise ValueError("Hard extension train count mismatch")
    if len(expected["val"]) != int(settings["val_count"]):
        raise ValueError("Hard extension val count mismatch")
    profile = str(settings["baseline_profile"])
    dataset_gate = config.get("evolution_dataset_gate") or {}
    required_behavior_id = dataset_gate.get("required_baseline_behavior_id")
    compatible_profiles = frozenset(
        str(value)
        for value in dataset_gate.get("compatible_baseline_profiles", [])
    )
    require_episode_config_audit = bool(
        dataset_gate.get("require_episode_config_behavior_audit", False)
    )
    rows = collect_split(
        run_root=train_run_root,
        task=task,
        split="train",
        expected_ids=expected["train"],
        profile=profile,
        required_behavior_id=required_behavior_id,
        compatible_profiles=compatible_profiles,
        require_episode_config_audit=require_episode_config_audit,
    ) + collect_split(
        run_root=val_run_root,
        task=task,
        split="val",
        expected_ids=expected["val"],
        profile=profile,
        required_behavior_id=required_behavior_id,
        compatible_profiles=compatible_profiles,
        require_episode_config_audit=require_episode_config_audit,
    )
    for row in rows:
        row["source_pool"] = "hard_extension20"
        row["representative_benchmark"] = False
    by_split = Counter((row["dataset_split"], row["success"]) for row in rows)
    report = {
        "schema_version": "self-evolve-v3-hard-extension-report-v1",
        "task": task,
        "complete": True,
        "attempts": len(rows),
        "successes": sum(row["success"] for row in rows),
        "failures": sum(not row["success"] for row in rows),
        "split_summary": {
            split: {
                "attempts": len(expected[split]),
                "successes": by_split[(split, True)],
                "failures": by_split[(split, False)],
            }
            for split in ("train", "val")
        },
        "representative_benchmark": False,
        "must_report_separately_from_evolution50": True,
        "baseline_behavior_id": required_behavior_id,
        "source_baseline_profiles": sorted({
            str(row["source_baseline_profile"]) for row in rows
        }),
        "cross_campaign_cache_reused": any(
            row["cache_compatibility_mode"]
            == "behavior_equivalent_legacy_profile"
            for row in rows
        ),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "config_sha256": sha256_file(config_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output_dir / "hard_extension20_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    atomic_write(
        output_dir / "extension_report.json", json.dumps(report, indent=2) + "\n"
    )
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--train-run-root", type=Path, action="append", required=True)
    parser.add_argument("--val-run-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, _ = prepare(
        config_path=args.config.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        train_run_root=[path.resolve() for path in args.train_run_root],
        val_run_root=[path.resolve() for path in args.val_run_root],
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
