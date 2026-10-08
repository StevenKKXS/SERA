#!/usr/bin/env python3
"""Combine Evolution50 and Hard Extension20 failures for problem discovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from prepare_evolution_dataset import atomic_write, sha256_file
    from prepare_hard_extension import episode_id, read_jsonl
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.prepare_evolution_dataset import (
        atomic_write,
        sha256_file,
    )
    from scripts.codex_rollout.prepare_hard_extension import episode_id, read_jsonl


def build(
    *,
    evolution_manifest_path: Path,
    hard_manifest_path: Path | None,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [("evolution50", read_jsonl(evolution_manifest_path))]
    if hard_manifest_path is not None:
        sources.append(("hard_extension20", read_jsonl(hard_manifest_path)))
    all_rows = []
    seen: set[tuple[str, int]] = set()
    tasks = set()
    for source_pool, rows in sources:
        for source in rows:
            split = str(source["dataset_split"])
            if split not in {"train", "val"}:
                raise ValueError(f"Discovery pool rejects split {split!r}")
            key = (split, episode_id(source))
            if key in seen:
                raise ValueError(f"Discovery pools overlap at {split}:{key[1]}")
            seen.add(key)
            tasks.add(str(source["task"]))
            all_rows.append(
                {
                    **source,
                    "episode_id": key[1],
                    "source_pool": source_pool,
                }
            )
    if len(tasks) != 1:
        raise ValueError("Discovery inputs must contain exactly one task")
    failures = [
        {
            **row,
            "analysis_role": "failure_discovery",
            "selected_for_analysis": True,
            "consumed_for_analysis": False,
        }
        for row in all_rows
        if not row["success"]
    ]
    successes = [
        {
            **row,
            "analysis_role": "success_counterexample_candidate",
            "selected_for_analysis": False,
            "consumed_for_analysis": False,
        }
        for row in all_rows
        if row["success"]
    ]
    failure_counts = Counter(row["source_pool"] for row in failures)
    report = {
        "schema_version": "self-evolve-v3-failure-discovery-pool-v1",
        "task": next(iter(tasks)),
        "source_attempts": {source_pool: len(rows) for source_pool, rows in sources},
        "failure_counts": {
            source_pool: failure_counts[source_pool] for source_pool, _ in sources
        },
        "combined_failure_count": len(failures),
        "success_counterexample_candidate_count": len(successes),
        "success_rates_merged_for_reporting": False,
        "evolution_manifest_sha256": sha256_file(evolution_manifest_path),
        "hard_manifest_sha256": (
            sha256_file(hard_manifest_path) if hard_manifest_path is not None else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        output_dir / "failure_discovery_pool.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
    )
    atomic_write(
        output_dir / "success_counterexample_pool.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in successes),
    )
    atomic_write(output_dir / "pool_report.json", json.dumps(report, indent=2) + "\n")
    return report, failures, successes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution-manifest", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, _, _ = build(
        evolution_manifest_path=args.evolution_manifest.resolve(),
        hard_manifest_path=args.hard_manifest.resolve() if args.hard_manifest else None,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
