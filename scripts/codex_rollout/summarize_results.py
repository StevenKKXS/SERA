#!/usr/bin/env python3
"""Verify and summarize Codex-grounded RoboMME rollout results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


METHOD_BY_MEMORY_MODE = {
    "keyframes": "codex_planner_plus_grounded_level_ft_pi05",
    "text_progress": "codex_text_progress_planner_plus_grounded_level_ft_pi05",
}
RECOVERY_STATUSES = {"progressing", "completed"}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_memer_rates(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["task"]: float(row["success_rate"])
            for row in csv.DictReader(handle)
            if row.get("variant") == "memer"
        }


def load_timings(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid timing JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Timing row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def rethinking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("recent_status")
        if isinstance(status, str):
            status_counts[status] = status_counts.get(status, 0) + 1
    trigger_indices = [
        index for index, row in enumerate(rows) if row.get("rethinking_required_this_turn") is True
    ]
    observed = [index for index in trigger_indices if index + 1 < len(rows)]
    recovered = [
        index
        for index in observed
        if rows[index + 1].get("recent_status") in RECOVERY_STATUSES
    ]
    no_progress_values = [
        int(value)
        for row in rows
        for value in (row.get("consecutive_no_progress_rounds_after"),)
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    return {
        "recent_status_counts": dict(sorted(status_counts.items())),
        "rethinking_triggers": len(trigger_indices),
        "post_rethinking_observed": len(observed),
        "post_rethinking_recovered": len(recovered),
        "post_rethinking_recovery_rate": len(recovered) / len(observed) if observed else None,
        "max_consecutive_no_progress_rounds": max(no_progress_values, default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--memer-summary-csv", type=Path)
    parser.add_argument(
        "--planner-memory-mode",
        choices=sorted(METHOD_BY_MEMORY_MODE),
        help="When set, require this memory mode and its corresponding method name.",
    )
    parser.add_argument(
        "--planner-prompt-profile",
        help="When set, require every episode to record this prompt profile.",
    )
    parser.add_argument(
        "--rethinking-threshold",
        type=int,
        help="When set, require every episode to record this rethinking threshold.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    memer_rates = load_memer_rates(
        args.memer_summary_csv.resolve() if args.memer_summary_csv else None
    )
    missing: list[str] = []
    invalid: list[str] = []
    episodes: list[dict[str, Any]] = []
    for task in tasks:
        for episode in range(args.num_episodes):
            path = run_dir / task / f"episode_{episode:02d}" / "summary.json"
            if not path.is_file():
                missing.append(f"{task}/episode_{episode:02d}")
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid.append(str(path))
                continue
            memory_mode = row.get("planner_memory_mode", "keyframes")
            expected_method = METHOD_BY_MEMORY_MODE.get(memory_mode)
            if (
                row.get("complete") is not True
                or row.get("task") != task
                or int(row.get("episode", -1)) != episode
                or expected_method is None
                or row.get("method") != expected_method
                or (args.planner_memory_mode and memory_mode != args.planner_memory_mode)
                or (
                    args.planner_prompt_profile
                    and row.get("planner_prompt_profile") != args.planner_prompt_profile
                )
                or (
                    args.rethinking_threshold is not None
                    and row.get("rethinking_threshold") != args.rethinking_threshold
                )
            ):
                invalid.append(str(path))
                continue
            try:
                row["rethinking_metrics"] = rethinking_metrics(
                    load_timings(path.parent / "timings.jsonl")
                )
            except ValueError:
                invalid.append(str(path.parent / "timings.jsonl"))
                continue
            episodes.append(row)

    verification = {
        "complete": not missing and not invalid and len(episodes) == len(tasks) * args.num_episodes,
        "tasks": tasks,
        "num_episodes": args.num_episodes,
        "expected": len(tasks) * args.num_episodes,
        "valid": len(episodes),
        "missing": missing,
        "invalid": invalid,
    }
    write_json(run_dir / "verification.json", verification)

    episode_fields = [
        "task",
        "episode",
        "result",
        "success",
        "steps_executed",
        "rounds",
        "episode_wall_seconds",
        "planner_total_seconds",
        "planner_mean_seconds_per_round",
        "vla_total_seconds",
        "env_step_total_seconds",
        "keyframe_count",
        "video_saved",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "rethinking_triggers",
        "post_rethinking_observed",
        "post_rethinking_recovered",
        "post_rethinking_recovery_rate",
        "max_consecutive_no_progress_rounds",
    ]
    episode_rows = []
    for row in episodes:
        usage = row.get("codex_usage_total") or {}
        metrics = row.get("rethinking_metrics") or {}
        episode_rows.append(
            {
                **{field: row.get(field) for field in episode_fields},
                "input_tokens": usage.get("input_tokens", 0),
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
                "rethinking_triggers": metrics.get("rethinking_triggers", 0),
                "post_rethinking_observed": metrics.get("post_rethinking_observed", 0),
                "post_rethinking_recovered": metrics.get("post_rethinking_recovered", 0),
                "post_rethinking_recovery_rate": metrics.get(
                    "post_rethinking_recovery_rate"
                ),
                "max_consecutive_no_progress_rounds": metrics.get(
                    "max_consecutive_no_progress_rounds", 0
                ),
            }
        )
    write_csv(run_dir / "episode_results.csv", episode_rows, episode_fields)

    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        values = [row for row in episodes if row["task"] == task]
        successes = sum(bool(row["success"]) for row in values)
        rate = successes / len(values) if values else None
        memer_rate = memer_rates.get(task)
        task_rows.append(
            {
                "task": task,
                "episodes": len(values),
                "successes": successes,
                "failures": sum(row["result"] == "fail" for row in values),
                "timeouts": sum(row["result"] == "timeout" for row in values),
                "errors": sum(row["result"] == "error" for row in values),
                "success_rate": rate,
                "memer_reproduction_success_rate": memer_rate,
                "delta_vs_memer": rate - memer_rate if rate is not None and memer_rate is not None else None,
                "mean_steps": mean(row["steps_executed"] for row in values) if values else None,
                "mean_episode_wall_seconds": mean(row["episode_wall_seconds"] for row in values) if values else None,
                "mean_planner_seconds": mean(row["planner_total_seconds"] for row in values) if values else None,
                "planner_seconds_per_round": (
                    sum(row["planner_total_seconds"] for row in values)
                    / sum(row["rounds"] for row in values)
                    if values and sum(row["rounds"] for row in values)
                    else None
                ),
                "total_input_tokens": sum((row.get("codex_usage_total") or {}).get("input_tokens", 0) for row in values),
                "total_cached_input_tokens": sum((row.get("codex_usage_total") or {}).get("cached_input_tokens", 0) for row in values),
                "total_output_tokens": sum((row.get("codex_usage_total") or {}).get("output_tokens", 0) for row in values),
                "total_keyframes": sum(row["keyframe_count"] for row in values),
                "rethinking_triggers": sum(
                    (row.get("rethinking_metrics") or {}).get("rethinking_triggers", 0)
                    for row in values
                ),
                "post_rethinking_observed": sum(
                    (row.get("rethinking_metrics") or {}).get("post_rethinking_observed", 0)
                    for row in values
                ),
                "post_rethinking_recovered": sum(
                    (row.get("rethinking_metrics") or {}).get("post_rethinking_recovered", 0)
                    for row in values
                ),
            }
        )
        observed = task_rows[-1]["post_rethinking_observed"]
        task_rows[-1]["post_rethinking_recovery_rate"] = (
            task_rows[-1]["post_rethinking_recovered"] / observed if observed else None
        )
    task_fields = list(task_rows[0]) if task_rows else []
    write_csv(run_dir / "summary.csv", task_rows, task_fields)

    overall_successes = sum(row["successes"] for row in task_rows)
    overall_episodes = sum(row["episodes"] for row in task_rows)
    lines = [
        "# Codex planner + grounded-level FT pi0.5",
        "",
        f"Completeness: {len(episodes)}/{len(tasks) * args.num_episodes}; verified={verification['complete']}.",
        "",
        "| Task | S/F/T/E | Success | MemER reproduction | Delta | Mean steps | Planner s/round | Rethink recovery | Input tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task_rows:
        memer = row["memer_reproduction_success_rate"]
        delta = row["delta_vs_memer"]
        lines.append(
            f"| {row['task']} | {row['successes']}/{row['failures']}/{row['timeouts']}/{row['errors']} | "
            f"{row['success_rate']:.3f} | {memer:.3f} | {delta:+.3f} | "
            f"{row['mean_steps']:.1f} | {row['planner_seconds_per_round']:.2f} | "
            f"{row['post_rethinking_recovered']}/{row['post_rethinking_observed']} | "
            f"{row['total_input_tokens']} |"
            if memer is not None and delta is not None and row["success_rate"] is not None
            else f"| {row['task']} | incomplete | | | | | | | |"
        )
    if args.planner_memory_mode == "text_progress":
        trace_description = (
            "Every episode directory contains per-round prompt, structured progress update, "
            "grounded subtask, point conversion, recent-frame manifest, loop/rethinking state, "
            "Codex usage/timing, and stderr. "
            "Requested episode videos are under each episode's `videos/` directory."
        )
    else:
        trace_description = (
            "Every episode directory contains per-round prompt, structured stage summary, "
            "grounded subtask, point conversion, keyframe descriptions/reasons, "
            "Codex usage/timing, and stderr. "
            "Requested episode videos are under each episode's `videos/` directory."
        )
    lines.extend(
        [
            "",
            f"Overall success: {overall_successes}/{overall_episodes} "
            f"({overall_successes / overall_episodes:.3f})."
            if overall_episodes
            else "Overall success: unavailable.",
            "",
            trace_description,
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False))
    if not verification["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
