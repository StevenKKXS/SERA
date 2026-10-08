#!/usr/bin/env python3
"""State controller for bounded, multi-problem train/val self-evolution."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

try:
    from episode_sampling import select_dataset_episodes
    from self_evolve_v2 import (
        assert_no_privileged_identity,
        behavior_cache_key,
        canonical_hash,
        git_commit,
        load_json,
        sha256_file,
        utc_now,
        write_json,
    )
    from verify_self_evolve_v2_experience import validate_candidate
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.episode_sampling import select_dataset_episodes
    from scripts.codex_rollout.self_evolve_v2 import (
        assert_no_privileged_identity,
        behavior_cache_key,
        canonical_hash,
        git_commit,
        load_json,
        sha256_file,
        utc_now,
        write_json,
    )
    from scripts.codex_rollout.verify_self_evolve_v2_experience import (
        validate_candidate,
    )


PROBLEM_STATES = {
    "active",
    "deferred",
    "resolved",
    "unactionable",
    "hypothesis_rejected",
    "exhausted",
    "singleton",
    "ambiguous",
}
SELECTABLE_PROBLEM_STATES = {"active", "deferred", "ambiguous"}
TERMINAL_PROBLEM_STATES = {
    "resolved",
    "unactionable",
    "hypothesis_rejected",
    "exhausted",
}
RUNTIME_FIELDS = (
    "experience_id",
    "revision",
    "task_scope",
    "condition",
    "lesson",
    "recommended_skill_or_policy",
    "success_evidence",
    "forbidden_behavior",
)


def normalize_label(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text:
        raise ValueError("Problem labels must be non-empty")
    return text


def validate_episode_problem_record(record: dict[str, Any]) -> None:
    split = record.get("dataset_split")
    if split not in {"train", "val"}:
        raise ValueError("Problem discovery accepts only train/val records")
    if "episode_id" not in record:
        raise ValueError("Problem record requires episode_id")
    outcome = record.get("outcome", "failure")
    if outcome not in {"success", "failure"}:
        raise ValueError("Problem record outcome must be success or failure")
    if outcome == "success":
        if not isinstance(record.get("counterexample_for", []), list):
            raise ValueError("Success counterexample_for must be a list")
        return
    required = {
        "task_variant",
        "failure_stage",
        "primary_problem",
        "first_divergence",
        "secondary_symptoms",
        "responsible_module",
        "runtime_observable",
        "planner_actionability",
        "confidence",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Problem record is missing fields: {missing}")
    if not isinstance(record["primary_problem"], str):
        raise ValueError("An episode must have exactly one scalar primary_problem")
    if not isinstance(record["secondary_symptoms"], list):
        raise ValueError("secondary_symptoms must be a list")
    confidence = float(record["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between zero and one")


def problem_signature(record: dict[str, Any]) -> tuple[str, ...]:
    """Build a stable root-cause signature without episode identity."""
    return tuple(
        normalize_label(record.get(field))
        for field in (
            "task_variant",
            "failure_stage",
            "primary_problem",
            "responsible_module",
        )
    )


def actionability_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return {
        "none": 0.0,
        "low": 0.25,
        "partial": 0.6,
        "high": 0.85,
        "full": 1.0,
    }.get(normalize_label(value), 0.0)


def runtime_bundle_rows(experiences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exactly what the planner receives, excluding offline provenance."""
    return [
        {**{field: row[field] for field in RUNTIME_FIELDS}, "status": "approved"}
        for row in experiences
    ]


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def render_problem_report(discovery: dict[str, Any]) -> str:
    lines = [
        f"# {discovery['task']} problem discovery",
        "",
        f"Episodes: {discovery['episode_record_count']}",
        f"Failures: {discovery['failure_record_count']}",
        f"Success counterexamples: {discovery['success_counterexample_record_count']}",
        "",
        "| problem | status | support | train | val | priority |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cluster in discovery["clusters"]:
        lines.append(
            "| {problem_id} | {status} | {support} | {train} | {val} | {priority:.3f} |".format(
                problem_id=cluster["problem_id"],
                status=cluster["status"],
                support=cluster["support"],
                train=cluster["split_support"]["train"],
                val=cluster["split_support"]["val"],
                priority=float(cluster["priority_score"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_problem_discovery(
    records: list[dict[str, Any]], config: dict[str, Any], task: str
) -> dict[str, Any]:
    settings = config["problem_discovery"]
    failures: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    success_records = []
    seen_episode_keys: set[tuple[str, int]] = set()
    for record in records:
        validate_episode_problem_record(record)
        episode_key = (str(record["dataset_split"]), int(record["episode_id"]))
        if episode_key in seen_episode_keys:
            raise ValueError(f"Duplicate problem record for {episode_key}")
        seen_episode_keys.add(episode_key)
        if record.get("outcome", "failure") == "success":
            success_records.append(record)
            continue
        failures.setdefault(problem_signature(record), []).append(record)

    clusters = []
    for signature, members in failures.items():
        problem_id = f"{task}-P{canonical_hash(signature)[:8]}"
        splits = sorted({str(row["dataset_split"]) for row in members})
        observable_ratio = sum(
            bool(row["runtime_observable"]) for row in members
        ) / len(members)
        actionability = sum(
            actionability_score(row["planner_actionability"]) for row in members
        ) / len(members)
        confidence = sum(float(row["confidence"]) for row in members) / len(members)
        aliases = {signature[2], problem_id.lower()}
        counterexamples = []
        if settings.get("use_success_counterexamples", False):
            normalized_variant = normalize_label(signature[0])
            for row in success_records:
                targets = {
                    normalize_label(value)
                    for value in row.get("counterexample_for", [])
                }
                same_variant = (
                    normalize_label(row.get("task_variant")) == normalized_variant
                )
                if same_variant and ("*" in targets or aliases.intersection(targets)):
                    counterexamples.append(row)
        support = len(members)
        if observable_ratio == 0.0 or actionability == 0.0:
            status = "unactionable"
        elif support < int(settings["min_cluster_support"]):
            status = "singleton"
        elif confidence < 0.5:
            status = "ambiguous"
        else:
            status = "active"
        estimated_gain = max(
            float(row.get("estimated_episode_gain", 0.0)) for row in members
        )
        regression_risk = (
            sum(float(row.get("regression_risk", 0.0)) for row in members) / support
        )
        experiment_cost = (
            sum(float(row.get("experiment_cost", 0.0)) for row in members) / support
        )
        priority = (
            support
            + (2.0 if len(splits) == 2 else 0.0)
            + observable_ratio
            + actionability
            + estimated_gain
            + 0.25 * len(counterexamples)
            - regression_risk
            - experiment_cost
        )
        clusters.append(
            {
                "problem_id": problem_id,
                "signature": {
                    "task_variant": signature[0],
                    "failure_stage": signature[1],
                    "primary_problem": signature[2],
                    "responsible_module": signature[3],
                },
                "status": status,
                "support": support,
                "split_support": {
                    split: sum(row["dataset_split"] == split for row in members)
                    for split in ("train", "val")
                },
                "first_divergence_examples": sorted(
                    {str(row["first_divergence"]) for row in members}
                ),
                "secondary_symptoms": sorted(
                    {
                        str(symptom)
                        for row in members
                        for symptom in row["secondary_symptoms"]
                    }
                ),
                "runtime_observable_ratio": observable_ratio,
                "planner_actionability_score": actionability,
                "mean_confidence": confidence,
                "success_counterexample_count": len(counterexamples),
                "priority_score": priority,
                "episode_refs": [
                    {
                        "dataset_split": row["dataset_split"],
                        "episode_id": int(row["episode_id"]),
                    }
                    for row in members
                ],
                "selected_for_deep_review": False,
            }
        )

    clusters.sort(
        key=lambda row: (
            float(row["priority_score"]),
            int(row["support"]),
            row["problem_id"],
        ),
        reverse=True,
    )
    eligible = [row for row in clusters if row["status"] in {"active", "ambiguous"}]
    for row in eligible[: int(settings["max_problem_clusters"])]:
        row["selected_for_deep_review"] = True
    if not settings.get("keep_singletons", True):
        clusters = [row for row in clusters if row["status"] != "singleton"]
    return {
        "schema_version": "self-evolve-v3-problem-discovery-v1",
        "task": task,
        "episode_record_count": len(records),
        "failure_record_count": sum(len(rows) for rows in failures.values()),
        "success_counterexample_record_count": len(success_records),
        "clusters": clusters,
    }


def append_event(state: dict[str, Any], event: str, payload: dict[str, Any]) -> None:
    state.setdefault("events", []).append(
        {"event": event, "at": utc_now(), "payload": payload}
    )
    state["updated_at"] = utc_now()


def add_usage(state: dict[str, Any], component: str, usage: dict[str, Any]) -> None:
    totals = state.setdefault("token_usage", {}).setdefault(component, {})
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            totals[key] = totals.get(key, 0) + value


def create_state(
    config: dict[str, Any], task: str, source_commit: str
) -> dict[str, Any]:
    if task not in config["task_order"]:
        raise ValueError(f"Task {task!r} is absent from configured task_order")
    loop = config["evolution_loop"]
    state = {
        "schema_version": "self-evolve-v3-multiproblem-task-state-v1",
        "experiment_id": config["experiment_id"],
        "task": task,
        "base_profile": config["base_profile"],
        "source_commit": source_commit,
        "max_total_evolution_rounds": int(loop["max_total_evolution_rounds"]),
        "max_attempts_per_problem": int(loop["max_attempts_per_problem"]),
        "total_evolution_rounds": 0,
        "remaining_problem_budget": int(loop["max_total_evolution_rounds"]),
        "active_problem_id": None,
        "problem_attempts": {},
        "problems": {},
        "current_candidate": None,
        "accepted_experiences": [],
        "working_bundle": [],
        "bundle_revision": 0,
        "round_history": [],
        "rollout_cache": {},
        "analysis_cache": {},
        "analysis_consumed_samples": [],
        "hard_exploration": {
            "mode": str((config.get("hard_exploration") or {}).get("mode", "disabled")),
            "trigger": None,
            "selection": None,
            "results": None,
        },
        "token_usage": {},
        "stage": "initialized",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "events": [],
    }
    append_event(
        state,
        "initialized",
        {
            "max_total_evolution_rounds": state["max_total_evolution_rounds"],
            "max_attempts_per_problem": state["max_attempts_per_problem"],
        },
    )
    return state


def record_hard_trigger(
    state: dict[str, Any], report: dict[str, Any], report_path: Path
) -> None:
    if report.get("task") != state["task"]:
        raise ValueError("Hard trigger task does not match state")
    if report.get("decision") not in {"trigger", "skip"}:
        raise ValueError("Hard trigger decision must be trigger or skip")
    hard = state.setdefault("hard_exploration", {})
    if hard.get("trigger") is not None:
        raise ValueError("Hard exploration trigger is already recorded")
    hard["trigger"] = {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "decision": report["decision"],
        "success_rate": float(report["success_rate"]),
    }
    state["stage"] = (
        "hard_extension_required"
        if report["decision"] == "trigger"
        else "problem_discovery_ready"
    )
    append_event(state, "hard_trigger_recorded", hard["trigger"])


def record_hard_selection(
    state: dict[str, Any], rows: list[dict[str, Any]], manifest_path: Path
) -> None:
    hard = state.setdefault("hard_exploration", {})
    trigger = hard.get("trigger") or {}
    if trigger.get("decision") != "trigger":
        raise ValueError("Hard selection requires a trigger decision")
    if hard.get("selection") is not None:
        raise ValueError("Hard exploration selection is already recorded")
    if any(row.get("task") != state["task"] for row in rows):
        raise ValueError("Hard selection task does not match state")
    split_counts = {
        split: sum(row.get("dataset_split") == split for row in rows)
        for split in ("train", "val")
    }
    if split_counts != {"train": 12, "val": 8}:
        raise ValueError("v3 Hard20 selection must contain train12 and val8")
    identities = [(str(row["dataset_split"]), int(row["episode_id"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Hard selection contains duplicate episodes")
    if any(row.get("outcome_used_for_selection") is not False for row in rows):
        raise ValueError("Hard selection must be outcome-blind")
    hard["selection"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "selected_count": len(rows),
        "split_counts": split_counts,
    }
    state["stage"] = "hard_extension_selected"
    append_event(state, "hard_selection_recorded", hard["selection"])


def record_hard_results(
    state: dict[str, Any],
    report: dict[str, Any],
    report_path: Path,
    manifest_path: Path,
) -> None:
    hard = state.setdefault("hard_exploration", {})
    if hard.get("selection") is None:
        raise ValueError("Hard results require a recorded selection")
    if hard.get("results") is not None:
        raise ValueError("Hard exploration results are already recorded")
    if report.get("task") != state["task"] or report.get("complete") is not True:
        raise ValueError(
            "Hard extension report is incomplete or belongs to another task"
        )
    if int(report.get("attempts", 0)) != 20:
        raise ValueError("v3 Hard Extension must contain 20 completed attempts")
    if report.get("representative_benchmark") is not False:
        raise ValueError("Hard extension must not be reported as a benchmark")
    hard["results"] = {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "attempts": int(report["attempts"]),
        "successes": int(report["successes"]),
        "failures": int(report["failures"]),
    }
    state["stage"] = "problem_discovery_ready"
    append_event(state, "hard_results_recorded", hard["results"])


def record_rollout(
    state: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
    *,
    planner_commit: str,
    planner_config_sha256: str,
    executor_version: str,
    experience_bundle_sha256: str,
    metadata_sha256: str,
) -> str:
    split = str(summary.get("dataset_split"))
    if split not in {"train", "val"}:
        raise ValueError("Evolution rollout cache accepts only train/val summaries")
    if summary.get("task") != state["task"]:
        raise ValueError("Rollout task does not match task state")
    episode_value = summary.get("episode")
    if episode_value is None:
        episode_value = summary.get("episode_id")
    episode = int(episode_value)
    key = behavior_cache_key(
        task=state["task"],
        split=split,
        episode=episode,
        planner_commit=planner_commit,
        planner_config_sha256=planner_config_sha256,
        executor_version=executor_version,
        experience_bundle_sha256=experience_bundle_sha256,
        metadata_sha256=metadata_sha256,
    )
    state["rollout_cache"][key] = {
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "dataset_split": split,
        "episode_id": episode,
        "success": bool(summary.get("success")),
        "codex_usage": summary.get("codex_usage_total", {}),
    }
    add_usage(state, "planner", summary.get("codex_usage_total", {}))
    append_event(
        state,
        "rollout_recorded",
        {"cache_key": key, "dataset_split": split, "episode_id": episode},
    )
    return key


def record_analysis(
    state: dict[str, Any],
    analysis: dict[str, Any],
    analysis_path: Path,
    *,
    trajectory_sha256: str,
    ground_truth_reference_sha256: str,
    analyzer_prompt_sha256: str,
    model: str,
) -> str:
    split = str(analysis.get("dataset_split"))
    if split not in {"train", "val"}:
        raise ValueError("Analysis cache rejects test or unaudited trajectories")
    if analysis.get("task") != state["task"]:
        raise ValueError("Analysis task does not match task state")
    episode_value = analysis.get("episode")
    if episode_value is None:
        episode_value = analysis.get("episode_id")
    episode = int(episode_value)
    key = canonical_hash(
        {
            "schema_version": "self-evolve-v3-analysis-cache-key-v1",
            "trajectory_sha256": trajectory_sha256,
            "ground_truth_reference_sha256": ground_truth_reference_sha256,
            "analyzer_version": analysis.get("analyzer_version"),
            "analyzer_prompt_sha256": analyzer_prompt_sha256,
            "model": model,
        }
    )
    usage = (analysis.get("model_metadata") or {}).get("usage", {})
    state["analysis_cache"][key] = {
        "analysis_path": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "dataset_split": split,
        "episode_id": episode,
        "usage": usage,
    }
    consumed = {"dataset_split": split, "episode_id": episode}
    if consumed not in state["analysis_consumed_samples"]:
        state["analysis_consumed_samples"].append(consumed)
    add_usage(state, "analyzer", usage)
    append_event(state, "analysis_recorded", {"cache_key": key, **consumed})
    return key


def install_problem_discovery(state: dict[str, Any], discovery: dict[str, Any]) -> None:
    if int(state.get("total_evolution_rounds", 0)) > 0:
        raise ValueError("Problem discovery cannot be replaced after evolution starts")
    if discovery.get("task") != state["task"]:
        raise ValueError("Problem discovery task does not match state")
    hard = state.get("hard_exploration") or {}
    if hard.get("mode") == "auto" and hard.get("trigger") is None:
        raise ValueError("Auto hard exploration must be decided before discovery")
    if (hard.get("trigger") or {}).get("decision") == "trigger" and hard.get(
        "results"
    ) is None:
        raise ValueError("Triggered Hard20 must complete before problem discovery")
    problems = {}
    for cluster in discovery.get("clusters", []):
        status = cluster.get("status")
        if status not in PROBLEM_STATES:
            raise ValueError(f"Unsupported problem status {status!r}")
        problems[cluster["problem_id"]] = dict(cluster)
    state["problems"] = problems
    state["problem_attempts"] = {problem_id: 0 for problem_id in problems}
    state["stage"] = "problems_discovered"
    append_event(state, "problem_discovery_recorded", {"problem_count": len(problems)})


def next_problem_id(
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    exclude_problem_id: str | None = None,
) -> str:
    allow_deferred = config["evolution_loop"].get(
        "allow_revisit_deferred_problem", False
    )
    candidates = [
        problem
        for problem in state["problems"].values()
        if problem["status"] in SELECTABLE_PROBLEM_STATES
        and problem["problem_id"] != exclude_problem_id
        and (problem["status"] != "deferred" or allow_deferred)
        and int(state["problem_attempts"].get(problem["problem_id"], 0))
        < int(state["max_attempts_per_problem"])
    ]
    if not candidates:
        raise ValueError("No selectable problem remains")
    return max(
        candidates,
        key=lambda problem: (
            float(problem.get("priority_score", 0.0)),
            int(problem.get("support", 0)),
            problem["problem_id"],
        ),
    )["problem_id"]


def select_problem(
    state: dict[str, Any], problem_id: str, config: dict[str, Any]
) -> None:
    if state.get("current_candidate") is not None:
        raise ValueError("Decide the current candidate before switching problems")
    if problem_id not in state["problems"]:
        raise ValueError(f"Unknown problem {problem_id!r}")
    problem = state["problems"][problem_id]
    if problem["status"] not in SELECTABLE_PROBLEM_STATES:
        raise ValueError(
            f"Problem {problem_id} is not selectable from {problem['status']}"
        )
    if problem["status"] == "deferred" and not config["evolution_loop"].get(
        "allow_revisit_deferred_problem", False
    ):
        raise ValueError("Configuration forbids revisiting deferred problems")
    previous = state.get("active_problem_id")
    if previous and previous != problem_id:
        if not config["evolution_loop"].get("allow_active_problem_switch", False):
            raise ValueError("Configuration forbids active problem switching")
        if state["problems"][previous]["status"] == "active":
            state["problems"][previous]["status"] = "deferred"
    problem["status"] = "active"
    state["active_problem_id"] = problem_id
    state["stage"] = "problem_selected"
    append_event(
        state, "problem_selected", {"problem_id": problem_id, "previous": previous}
    )


def start_candidate(
    state: dict[str, Any], candidate: dict[str, Any], candidate_sha256: str
) -> dict[str, Any]:
    runtime_candidate = {
        key: candidate[key] for key in (*RUNTIME_FIELDS, "status") if key in candidate
    }
    validate_candidate(runtime_candidate)
    assert_no_privileged_identity(candidate)
    if candidate.get("task_scope") != [state["task"]]:
        raise ValueError("Candidate task scope does not match state")
    if state.get("current_candidate") is not None:
        raise ValueError("A candidate is already active")
    problem_id = candidate.get("problem_id")
    if problem_id != state.get("active_problem_id"):
        raise ValueError("Candidate problem_id must match the active problem")
    if state["total_evolution_rounds"] >= state["max_total_evolution_rounds"]:
        raise ValueError("Task evolution round budget is exhausted")
    attempts = int(state["problem_attempts"].get(problem_id, 0))
    if attempts >= state["max_attempts_per_problem"]:
        state["problems"][problem_id]["status"] = "exhausted"
        raise ValueError("Problem attempt budget is exhausted")

    round_number = int(state["total_evolution_rounds"]) + 1
    base_bundle = list(state["working_bundle"])
    base_runtime_bundle = runtime_bundle_rows(base_bundle)
    candidate_runtime_bundle = runtime_bundle_rows(base_bundle + [candidate])
    record = {
        "round": round_number,
        "problem_id": problem_id,
        "problem_attempt": attempts + 1,
        "experience_id": candidate["experience_id"],
        "candidate_revision": int(candidate["revision"]),
        "candidate_sha256": candidate_sha256,
        "candidate": candidate,
        "base_bundle_revision": int(state["bundle_revision"]),
        "base_bundle_experience_ids": [row["experience_id"] for row in base_bundle],
        "base_bundle_sha256": canonical_hash(base_runtime_bundle),
        "candidate_bundle_sha256": canonical_hash(candidate_runtime_bundle),
        "status": "verifier_pending",
    }
    state["total_evolution_rounds"] = round_number
    state["remaining_problem_budget"] = (
        state["max_total_evolution_rounds"] - round_number
    )
    state["problem_attempts"][problem_id] = attempts + 1
    state["current_candidate"] = record
    state["stage"] = "candidate_verification"
    append_event(
        state,
        "candidate_started",
        {
            "round": round_number,
            "problem_id": problem_id,
            "problem_attempt": attempts + 1,
            "experience_id": candidate["experience_id"],
        },
    )
    return record


def decide_candidate(
    state: dict[str, Any], report: dict[str, Any], decision: str, reason: str
) -> dict[str, Any]:
    current = state.get("current_candidate")
    if current is None:
        raise ValueError("No candidate is awaiting a decision")
    if report.get("schema_version") != "self-evolve-v3-pilot-verifier-report-v3":
        raise ValueError(
            "New multi-problem runs require a v3 incremental verifier report"
        )
    if report.get("experience_id") != current["experience_id"]:
        raise ValueError("Verifier report experience does not match current candidate")
    if report.get("task") != state["task"]:
        raise ValueError("Verifier report task does not match state")
    if report.get("problem_id") != current["problem_id"]:
        raise ValueError("Verifier report problem does not match current candidate")
    pilot = report.get("pilot_verifier")
    if not isinstance(pilot, dict):
        raise ValueError("v3 decisions require a bounded pilot verifier report")
    if decision not in {
        "accept",
        "retry",
        "defer",
        "unactionable",
        "hypothesis_rejected",
    }:
        raise ValueError(f"Unsupported candidate decision {decision!r}")
    if decision == "accept":
        if not report.get("test_release_eligible"):
            raise ValueError("Only a verifier-eligible candidate can be accepted")
        if int(pilot.get("regressed", 0)) != 0:
            raise ValueError("A regressing candidate cannot be accepted")
    comparison = report.get("bundle_comparison") or {}
    if comparison.get("base_bundle_sha256") != current["base_bundle_sha256"]:
        raise ValueError("Verifier did not compare against the recorded base bundle")
    if comparison.get("candidate_bundle_sha256") != current["candidate_bundle_sha256"]:
        raise ValueError("Verifier did not run the recorded candidate bundle")
    consumed = {
        (row["dataset_split"], int(row["episode_id"]))
        for row in state.get("analysis_consumed_samples", [])
    }
    reused = [
        row
        for row in report.get("sample_provenance", [])
        if (row.get("dataset_split"), int(row.get("episode_id", -1))) in consumed
    ]
    if reused:
        raise ValueError(
            "Verifier report reuses samples consumed for candidate analysis"
        )

    problem_id = current["problem_id"]
    problem = state["problems"][problem_id]
    round_record = {
        **{key: value for key, value in current.items() if key != "candidate"},
        "decision": decision,
        "decision_reason": reason,
        "verifier_report": report,
    }
    if decision == "accept":
        candidate = current["candidate"]
        accepted = {
            **candidate,
            "status": "approved",
            "accepted_round": current["round"],
            "source_problem_id": problem_id,
            "verifier_acceptance_tier": report["acceptance_tier"],
        }
        state["accepted_experiences"].append(accepted)
        state["working_bundle"].append(accepted)
        state["bundle_revision"] = int(state["bundle_revision"]) + 1
        problem["status"] = "resolved"
        state["active_problem_id"] = None
        round_record["result_bundle_revision"] = state["bundle_revision"]
        round_record["result_bundle_sha256"] = canonical_hash(
            runtime_bundle_rows(state["working_bundle"])
        )
    else:
        target_status = {
            "retry": "active",
            "defer": "deferred",
            "unactionable": "unactionable",
            "hypothesis_rejected": "hypothesis_rejected",
        }[decision]
        if (
            decision == "retry"
            and state["problem_attempts"][problem_id]
            >= state["max_attempts_per_problem"]
        ):
            target_status = "exhausted"
        problem["status"] = target_status
        if target_status != "active":
            state["active_problem_id"] = None

    state["round_history"].append(round_record)
    state["current_candidate"] = None
    state["stage"] = (
        "evolution_budget_exhausted"
        if state["remaining_problem_budget"] == 0
        else "candidate_decided"
    )
    append_event(
        state,
        "candidate_decided",
        {
            "round": current["round"],
            "problem_id": problem_id,
            "experience_id": current["experience_id"],
            "decision": decision,
            "problem_status": problem["status"],
        },
    )
    return round_record


def build_test_candidate_snapshot(
    state: dict[str, Any],
    config: dict[str, Any],
    source_commit: str,
    executor_checkpoint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state.get("current_candidate") is not None:
        raise ValueError("Cannot record a test candidate with an undecided candidate")
    if not state["working_bundle"]:
        raise ValueError("Cannot record an empty test candidate bundle")
    bundle_sha256 = canonical_hash(runtime_bundle_rows(state["working_bundle"]))
    snapshot_id = (
        f"{state['task']}-bundle-r{state['bundle_revision']}-{bundle_sha256[:12]}"
    )
    runtime_memory = {
        "schema_version": "self-evolve-runtime-lessons-v2",
        "memory_id": snapshot_id,
        "revision": int(state["bundle_revision"]),
        "base_profile": state["base_profile"],
        "experiences": runtime_bundle_rows(state["working_bundle"]),
    }
    assert_no_privileged_identity(runtime_memory["experiences"])
    manifest = {
        "schema_version": "self-evolve-v3-test-candidate-snapshot-v1",
        "snapshot_id": snapshot_id,
        "task": state["task"],
        "base_profile": state["base_profile"],
        "source_commit": source_commit,
        "evolution_start_commit": state["source_commit"],
        "experiment_config_sha256": state.get("config_sha256"),
        "bundle_revision": int(state["bundle_revision"]),
        "bundle_sha256": bundle_sha256,
        "accepted_experience_ids": [
            row["experience_id"] for row in state["working_bundle"]
        ],
        "models": config["models"],
        "executor_checkpoint": executor_checkpoint,
        "total_evolution_rounds": int(state["total_evolution_rounds"]),
        "final_bundle_rollout_performed": False,
        "test_policy": "one_shot_feedback_forbidden",
        "test_feedback_used": False,
        "recorded_at": utc_now(),
    }
    return manifest, runtime_memory


def state_path(run_root: Path) -> Path:
    return run_root / "state.v3.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_state(run_root: Path, state: dict[str, Any]) -> None:
    write_json(state_path(run_root), state)


def init_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    path = state_path(run_root)
    if path.exists():
        state = load_json(path)
        if state["task"] != args.task:
            raise ValueError("Run root already belongs to another task")
        print(path)
        return
    config = load_json(args.config.resolve())
    state = create_state(config, args.task, git_commit(args.repo_root.resolve()))
    state["config_path"] = str(args.config.resolve())
    state["config_sha256"] = sha256_file(args.config.resolve())
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.v3.snapshot.json", config)
    save_state(run_root, state)
    print(path)


def candidate_pool_command(args: argparse.Namespace) -> None:
    """Materialize the outcome-blind Evolution50 split without v2 state."""
    run_root = args.run_root.resolve()
    state_file = state_path(run_root)
    state = load_json(state_file)
    config = load_json(Path(state["config_path"]))
    split = args.dataset_split
    if split not in {"train", "val"}:
        raise ValueError("Evolution50 candidate pools permit only train or val")
    gate = config["evolution_dataset_gate"]
    limit = int(
        gate[
            "required_train_episodes"
            if split == "train"
            else "required_val_episodes"
        ]
    )
    selected, provenance = select_dataset_episodes(
        config=config,
        source_root=args.metadata_root.resolve(),
        split=split,
        task=state["task"],
        limit=limit,
    )
    candidates = [
        {
            "task": state["task"],
            "episode": int(row["episode_id"]),
            **dict(row.get("attributes") or {}),
        }
        for row in selected
    ]
    output = {
        "schema_version": "self-evolve-v3-evolution50-candidate-pool-v1",
        "task": state["task"],
        "dataset_split": split,
        "sampling": provenance,
        "candidates": candidates,
    }
    output_dir = run_root / "profiles" / split
    pool_path = output_dir / "candidate_pool.json"
    ids_path = output_dir / "episode_ids.txt"
    write_json(pool_path, output)
    write_text_atomic(
        ids_path, ",".join(str(row["episode"]) for row in candidates) + "\n"
    )
    selection = state.setdefault("evolution_dataset_selection", {})
    selection[split] = {
        "pool_path": str(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "episode_ids_path": str(ids_path),
        "selected_count": len(candidates),
        "sampling": provenance,
    }
    append_event(
        state,
        "evolution50_candidate_pool_recorded",
        {"dataset_split": split, "selected_count": len(candidates)},
    )
    save_state(run_root, state)
    print(ids_path.read_text(encoding="utf-8").strip())


def discover_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    records = read_jsonl(args.episode_records.resolve())
    discovery = build_problem_discovery(records, config, state["task"])
    output_dir = run_root / "problem_discovery"
    write_jsonl(output_dir / "episode_problem_records.jsonl", records)
    write_json(output_dir / "problem_clusters.json", discovery)
    write_text_atomic(
        output_dir / "problem_report.md", render_problem_report(discovery)
    )
    install_problem_discovery(state, discovery)
    save_state(run_root, state)
    print(output_dir / "problem_clusters.json")


def record_hard_trigger_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    report_path = args.trigger_report.resolve()
    record_hard_trigger(state, load_json(report_path), report_path)
    save_state(run_root, state)
    print(state["hard_exploration"]["trigger"]["decision"])


def record_hard_selection_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    manifest_path = args.selection_manifest.resolve()
    record_hard_selection(state, read_jsonl(manifest_path), manifest_path)
    save_state(run_root, state)
    print(state["hard_exploration"]["selection"]["selected_count"])


def record_hard_results_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    report_path = args.extension_report.resolve()
    manifest_path = args.extension_manifest.resolve()
    record_hard_results(state, load_json(report_path), report_path, manifest_path)
    save_state(run_root, state)
    print(state["hard_exploration"]["results"]["failures"])


def record_rollout_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    summary_path = args.summary.resolve()
    key = record_rollout(
        state,
        load_json(summary_path),
        summary_path,
        planner_commit=args.planner_commit,
        planner_config_sha256=args.planner_config_sha256,
        executor_version=args.executor_version,
        experience_bundle_sha256=args.experience_bundle_sha256,
        metadata_sha256=args.metadata_sha256,
    )
    save_state(run_root, state)
    print(key)


def record_analysis_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    analysis_path = args.analysis.resolve()
    key = record_analysis(
        state,
        load_json(analysis_path),
        analysis_path,
        trajectory_sha256=args.trajectory_sha256,
        ground_truth_reference_sha256=args.ground_truth_reference_sha256,
        analyzer_prompt_sha256=args.analyzer_prompt_sha256,
        model=args.model,
    )
    save_state(run_root, state)
    print(key)


def select_problem_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    problem_id = args.problem_id or next_problem_id(state, config)
    select_problem(state, problem_id, config)
    save_state(run_root, state)
    print(problem_id)


def start_candidate_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    candidate_path = args.candidate.resolve()
    record = start_candidate(
        state, load_json(candidate_path), sha256_file(candidate_path)
    )
    round_dir = run_root / f"round_{record['round']:02d}"
    write_json(
        round_dir / "selected_problem.json", state["problems"][record["problem_id"]]
    )
    write_json(
        round_dir / "base_bundle.json",
        {
            "bundle_revision": record["base_bundle_revision"],
            "experience_ids": record["base_bundle_experience_ids"],
            "base_bundle_sha256": record["base_bundle_sha256"],
            "candidate_bundle_sha256": record["candidate_bundle_sha256"],
        },
    )
    write_json(round_dir / "candidate_experience.json", record["candidate"])
    save_state(run_root, state)
    print(record["round"])


def decide_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    report = load_json(args.verifier_report.resolve())
    record = decide_candidate(state, report, args.decision, args.reason)
    round_dir = run_root / f"round_{record['round']:02d}"
    write_json(round_dir / "verifier_report.json", report)
    write_json(round_dir / "decision.json", record)
    if args.decision == "accept":
        write_json(
            run_root
            / "bundle_history"
            / f"bundle_revision_{state['bundle_revision']:03d}.json",
            {
                "bundle_revision": state["bundle_revision"],
                "bundle_sha256": canonical_hash(
                    runtime_bundle_rows(state["working_bundle"])
                ),
                "experiences": state["working_bundle"],
            },
        )
    save_state(run_root, state)
    print(args.decision)


def snapshot_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Repository must be clean before recording a test candidate")
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    manifest, runtime_memory = build_test_candidate_snapshot(
        state, config, git_commit(repo_root), args.executor_checkpoint
    )
    output_dir = run_root / "test_candidate_snapshot"
    write_json(output_dir / "reproducibility_manifest.json", manifest)
    write_json(
        output_dir / "experience_bundle.json", {"experiences": state["working_bundle"]}
    )
    write_json(output_dir / "runtime_lesson_memory.json", runtime_memory)
    state["stage"] = "test_candidate_recorded"
    state["test_candidate_snapshot"] = manifest
    append_event(
        state, "test_candidate_recorded", {"snapshot_id": manifest["snapshot_id"]}
    )
    save_state(run_root, state)
    print(manifest["snapshot_id"])


def status_command(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            load_json(state_path(args.run_root.resolve())), ensure_ascii=False, indent=2
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--repo-root", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--task", required=True)
    init.set_defaults(func=init_command)

    pool = subparsers.add_parser("candidate-pool")
    pool.add_argument("--run-root", type=Path, required=True)
    pool.add_argument("--metadata-root", type=Path, required=True)
    pool.add_argument("--dataset-split", choices=("train", "val"), required=True)
    pool.set_defaults(func=candidate_pool_command)

    discover = subparsers.add_parser("discover-problems")
    discover.add_argument("--run-root", type=Path, required=True)
    discover.add_argument("--episode-records", type=Path, required=True)
    discover.set_defaults(func=discover_command)

    hard_trigger = subparsers.add_parser("record-hard-trigger")
    hard_trigger.add_argument("--run-root", type=Path, required=True)
    hard_trigger.add_argument("--trigger-report", type=Path, required=True)
    hard_trigger.set_defaults(func=record_hard_trigger_command)

    hard_selection = subparsers.add_parser("record-hard-selection")
    hard_selection.add_argument("--run-root", type=Path, required=True)
    hard_selection.add_argument("--selection-manifest", type=Path, required=True)
    hard_selection.set_defaults(func=record_hard_selection_command)

    hard_results = subparsers.add_parser("record-hard-results")
    hard_results.add_argument("--run-root", type=Path, required=True)
    hard_results.add_argument("--extension-report", type=Path, required=True)
    hard_results.add_argument("--extension-manifest", type=Path, required=True)
    hard_results.set_defaults(func=record_hard_results_command)

    rollout = subparsers.add_parser("record-rollout")
    rollout.add_argument("--run-root", type=Path, required=True)
    rollout.add_argument("--summary", type=Path, required=True)
    rollout.add_argument("--planner-commit", required=True)
    rollout.add_argument("--planner-config-sha256", required=True)
    rollout.add_argument("--executor-version", required=True)
    rollout.add_argument("--experience-bundle-sha256", required=True)
    rollout.add_argument("--metadata-sha256", required=True)
    rollout.set_defaults(func=record_rollout_command)

    analysis = subparsers.add_parser("record-analysis")
    analysis.add_argument("--run-root", type=Path, required=True)
    analysis.add_argument("--analysis", type=Path, required=True)
    analysis.add_argument("--trajectory-sha256", required=True)
    analysis.add_argument("--ground-truth-reference-sha256", required=True)
    analysis.add_argument("--analyzer-prompt-sha256", required=True)
    analysis.add_argument("--model", required=True)
    analysis.set_defaults(func=record_analysis_command)

    choose = subparsers.add_parser("select-problem")
    choose.add_argument("--run-root", type=Path, required=True)
    choose.add_argument("--problem-id")
    choose.set_defaults(func=select_problem_command)

    candidate = subparsers.add_parser("start-candidate")
    candidate.add_argument("--run-root", type=Path, required=True)
    candidate.add_argument("--candidate", type=Path, required=True)
    candidate.set_defaults(func=start_candidate_command)

    decision = subparsers.add_parser("decide-candidate")
    decision.add_argument("--run-root", type=Path, required=True)
    decision.add_argument("--verifier-report", type=Path, required=True)
    decision.add_argument(
        "--decision",
        choices=("accept", "retry", "defer", "unactionable", "hypothesis_rejected"),
        required=True,
    )
    decision.add_argument("--reason", required=True)
    decision.set_defaults(func=decide_command)

    snapshot = subparsers.add_parser("record-test-candidate")
    snapshot.add_argument("--run-root", type=Path, required=True)
    snapshot.add_argument("--repo-root", type=Path, required=True)
    snapshot.add_argument("--executor-checkpoint", required=True)
    snapshot.set_defaults(func=snapshot_command)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.set_defaults(func=status_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
