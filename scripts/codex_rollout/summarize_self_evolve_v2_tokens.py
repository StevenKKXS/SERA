#!/usr/bin/env python3
"""Summarize actual Codex token usage for one self-evolve v2 task directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from planner import write_json


TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def add_usage(total: Counter[str], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in TOKEN_KEYS:
        total[key] += int(usage.get(key) or 0)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def component(paths: list[Path], usage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    for usage in usage_rows:
        add_usage(total, usage)
    return {
        "call_count": len(usage_rows),
        "artifact_count": len(paths),
        "usage": {key: total[key] for key in TOKEN_KEYS},
    }


def summarize(task_root: Path) -> dict[str, Any]:
    experiment_paths = sorted(task_root.glob("round_*/experiment_plan/plan.json"))
    experiment_usages = [read_json(path).get("usage", {}) for path in experiment_paths]

    init_paths = sorted(task_root.glob("round_*/**/cases/*/case_summary.json"))
    init_usages = []
    for path in init_paths:
        value = read_json(path)
        if value.get("inference_cache_hit") is not True:
            init_usages.append(value.get("usage", {}))

    rollout_paths = sorted(task_root.glob("round_*/**/codex_planner/rounds/*/record.json"))
    rollout_usages = []
    for path in rollout_paths:
        usage = read_json(path).get("codex_usage")
        if isinstance(usage, dict):
            rollout_usages.append(usage)

    analysis_paths = sorted(task_root.glob("round_*/analysis/episode_*/analysis.json"))
    analysis_usages = []
    seen_attempts: set[str] = set()
    for path in analysis_paths:
        value = read_json(path)
        attempts = value.get("model_attempts") or []
        if not attempts and value.get("model_metadata"):
            attempts = [value["model_metadata"]]
        for attempt in attempts:
            identity = json.dumps(
                {
                    "thread_id": attempt.get("thread_id"),
                    "started_at": attempt.get("started_at"),
                    "usage": attempt.get("usage"),
                },
                sort_keys=True,
            )
            if identity in seen_attempts:
                continue
            seen_attempts.add(identity)
            analysis_usages.append(attempt.get("usage", {}))

    components = {
        "experiment_planner": component(experiment_paths, experiment_usages),
        "init_analyzer": component(init_paths, init_usages),
        "rollout_planner": component(rollout_paths, rollout_usages),
        "failure_analyzer": component(analysis_paths, analysis_usages),
    }
    grand: Counter[str] = Counter()
    for value in components.values():
        add_usage(grand, value["usage"])
    return {
        "schema_version": "self-evolve-v2-token-report-v1",
        "task": task_root.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accounting_rule": (
            "Count every recorded model call, including interrupted rollout attempts; "
            "exclude init inference cache hits and deduplicate analyzer attempts."
        ),
        "components": components,
        "total": {key: grand[key] for key in TOKEN_KEYS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    report = summarize(task_root)
    output = args.output.resolve() if args.output else task_root / "token_report.json"
    write_json(output, report)
    print(output)


if __name__ == "__main__":
    main()
