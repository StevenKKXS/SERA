#!/usr/bin/env python3
"""Auditable state and cache controller for train-active self-evolution v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from episode_sampling import select_dataset_episodes
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.episode_sampling import select_dataset_episodes


FORBIDDEN_EXPERIENCE_KEYS = {
    "episode",
    "episode_id",
    "seed",
    "dataset_split",
    "source_episode_refs",
    "supporting_episode_refs",
    "oracle_state",
    "future_frames",
    "absolute_coordinates",
}
FORBIDDEN_EXPERIENCE_TEXT = (
    re.compile(r"(?i)(?:^|[^a-z0-9])(?:episode(?:_id)?|ep)[\s:=/_-]*0*[0-9]+(?:[^0-9]|$)"),
    re.compile(r"(?i)\b(?:environment_)?seed\s*[:=]\s*[0-9]+\b"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_no_privileged_identity(value: Any, source: str = "experience") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_EXPERIENCE_KEYS:
                raise ValueError(f"{source} contains forbidden runtime key {key!r}")
            assert_no_privileged_identity(nested, f"{source}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_privileged_identity(nested, f"{source}[{index}]")
    elif isinstance(value, str):
        for pattern in FORBIDDEN_EXPERIENCE_TEXT:
            match = pattern.search(value)
            if match is not None:
                raise ValueError(
                    f"{source} contains forbidden runtime identity text {match.group(0)!r}"
                )


def resolve_round_limit(config: dict[str, Any], task: str, override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("--max-evolution-rounds must be positive")
        return override
    evolution = config["evolution"]
    return int(evolution.get("task_max_rounds", {}).get(task, evolution["max_rounds"]))


def metadata_records(metadata_root: Path, split: str, task: str) -> list[dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError("Active candidate construction permits only train or val")
    path = metadata_root / split / f"record_dataset_{task}_metadata.json"
    value = load_json(path)
    records = list(value.get("records") or [])
    if not records:
        raise ValueError(f"No metadata records in {path}")
    # Seeds stay inside the resolver and are never included in returned cards.
    return [
        {"task": task, "episode": int(row["episode"]), "difficulty": row["difficulty"]}
        for row in records
    ]


def stratified_pool(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups = {
        difficulty: [row for row in records if row["difficulty"] == difficulty]
        for difficulty in ("easy", "medium", "hard")
    }
    allocation = {"easy": limit // 2, "medium": limit // 4, "hard": limit // 4}
    while sum(allocation.values()) < limit:
        allocation["hard"] += 1
    selected = []
    for difficulty in ("easy", "medium", "hard"):
        rows = groups[difficulty]
        count = min(allocation[difficulty], len(rows))
        if count == 1:
            selected.append(rows[0])
        elif count > 1:
            indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
            selected.extend(rows[index] for index in indices)
    return [
        {
            "task": row["task"],
            "episode": int(row["episode"]),
            "difficulty": row["difficulty"],
        }
        for row in sorted(selected, key=lambda row: int(row["episode"]))
    ]


def resolve_pool_limit(config: dict[str, Any], split: str) -> int:
    """Resolve a split-specific pool size while preserving v2 configs."""
    if split not in {"train", "val"}:
        raise ValueError("Candidate pools permit only train or val")
    evolution = config["evolution"]
    split_key = f"{split}_pool_size"
    limit = int(evolution.get(split_key, evolution["candidate_pool_size"]))
    if limit < 1:
        raise ValueError(f"{split_key} must be positive")
    return limit


def state_path(run_root: Path) -> Path:
    return run_root / "state.json"


def append_event(state: dict[str, Any], event: str, payload: dict[str, Any]) -> None:
    state.setdefault("events", []).append(
        {"event": event, "at": utc_now(), "payload": payload}
    )
    state["updated_at"] = utc_now()


def init_command(args: argparse.Namespace) -> None:
    config = load_json(args.config.resolve())
    if args.task not in config["task_order"]:
        raise ValueError(f"Task {args.task!r} is absent from configured task_order")
    run_root = args.run_root.resolve()
    if state_path(run_root).exists():
        existing = load_json(state_path(run_root))
        if existing.get("task") != args.task:
            raise ValueError(f"Run root already belongs to {existing.get('task')}")
        print(state_path(run_root))
        return
    run_root.mkdir(parents=True, exist_ok=True)
    limit = resolve_round_limit(config, args.task, args.max_evolution_rounds)
    state = {
        "schema_version": "self-evolve-v2-task-state-v1",
        "experiment_id": config["experiment_id"],
        "task": args.task,
        "base_profile": config["base_profile"],
        "source_commit": git_commit(args.repo_root.resolve()),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "max_evolution_rounds": limit,
        "current_round": 0,
        "stage": "initialized",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "rollout_cache": {},
        "analysis_cache": {},
        "experiences": [],
        "token_usage": {},
        "events": [],
    }
    append_event(state, "initialized", {"max_evolution_rounds": limit})
    write_json(state_path(run_root), state)
    write_json(run_root / "config.snapshot.json", config)
    print(state_path(run_root))


def pool_command(args: argparse.Namespace) -> None:
    state = load_json(state_path(args.run_root.resolve()))
    config = load_json(Path(state["config_path"]))
    limit = resolve_pool_limit(config, args.dataset_split)
    if "episode_sampling" in config:
        selected, sampling_provenance = select_dataset_episodes(
            config=config,
            source_root=args.metadata_root.resolve(),
            split=args.dataset_split,
            task=state["task"],
            limit=limit,
        )
        pool = [
            {
                "task": state["task"],
                "episode": int(row["episode_id"]),
                **dict(row.get("attributes") or {}),
            }
            for row in selected
        ]
    else:
        # Backward-compatible path for frozen v2 campaign configurations.
        records = metadata_records(
            args.metadata_root.resolve(), args.dataset_split, state["task"]
        )
        pool = stratified_pool(records, limit)
        sampling_provenance = {
            "dataset_id": "robomme_legacy",
            "loader": "legacy_robomme_metadata",
            "algorithm": "legacy_difficulty_stratified_even",
            "source_record_count": len(records),
            "selected_count": len(pool),
            "outcome_blind": True,
        }
    # This file is offline scheduling provenance. It is never a runtime experience.
    output = {
        "schema_version": "self-evolve-v2-candidate-pool-v1",
        "task": state["task"],
        "dataset_split": args.dataset_split,
        "sampling": sampling_provenance,
        "candidates": pool,
    }
    path = args.run_root.resolve() / "profiles" / args.dataset_split / "candidate_pool.json"
    write_json(path, output)
    print(",".join(str(row["episode"]) for row in pool))


def advance_command(args: argparse.Namespace) -> None:
    path = state_path(args.run_root.resolve())
    state = load_json(path)
    next_round = int(state["current_round"]) + 1
    if next_round > int(state["max_evolution_rounds"]):
        raise ValueError("Configured max_evolution_rounds has been reached")
    state["current_round"] = next_round
    state["stage"] = "exploring"
    append_event(state, "round_started", {"round": next_round})
    write_json(path, state)
    print(next_round)


def behavior_cache_key(
    *, task: str, split: str, episode: int, planner_commit: str, planner_config_sha256: str,
    executor_version: str, experience_bundle_sha256: str, metadata_sha256: str
) -> str:
    return canonical_hash(
        {
            "schema_version": "self-evolve-v2-rollout-cache-key-v1",
            "task": task,
            "dataset_split": split,
            "episode": episode,
            "planner_commit": planner_commit,
            "planner_config_sha256": planner_config_sha256,
            "executor_version": executor_version,
            "experience_bundle_sha256": experience_bundle_sha256,
            "metadata_sha256": metadata_sha256,
        }
    )


def record_rollout_command(args: argparse.Namespace) -> None:
    path = state_path(args.run_root.resolve())
    state = load_json(path)
    summary_path = args.summary.resolve()
    summary = load_json(summary_path)
    split = str(summary.get("dataset_split"))
    if split not in {"train", "val"}:
        raise ValueError("Evolution rollout cache accepts only audited train/val summaries")
    if summary.get("task") != state["task"]:
        raise ValueError("Rollout task does not match task state")
    key = behavior_cache_key(
        task=state["task"],
        split=split,
        episode=int(summary["episode"]),
        planner_commit=args.planner_commit,
        planner_config_sha256=args.planner_config_sha256,
        executor_version=args.executor_version,
        experience_bundle_sha256=args.experience_bundle_sha256,
        metadata_sha256=args.metadata_sha256,
    )
    state["rollout_cache"][key] = {
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "dataset_split": split,
        "episode": int(summary["episode"]),
        "success": bool(summary.get("success")),
        "codex_usage": summary.get("codex_usage_total", {}),
    }
    append_event(state, "rollout_recorded", {"cache_key": key, "split": split})
    write_json(path, state)
    print(key)


def record_analysis_command(args: argparse.Namespace) -> None:
    path = state_path(args.run_root.resolve())
    state = load_json(path)
    analysis_path = args.analysis.resolve()
    analysis = load_json(analysis_path)
    split = analysis.get("dataset_split")
    if split not in {"train", "val"}:
        raise ValueError("Analysis cache rejects test or unaudited trajectories")
    if analysis.get("task") != state["task"]:
        raise ValueError("Analysis task does not match task state")
    key = canonical_hash(
        {
            "schema_version": "self-evolve-v2-analysis-cache-key-v1",
            "trajectory_sha256": args.trajectory_sha256,
            "ground_truth_reference_sha256": args.ground_truth_reference_sha256,
            "analyzer_version": analysis.get("analyzer_version"),
            "analyzer_prompt_sha256": args.analyzer_prompt_sha256,
            "model": args.model,
        }
    )
    state["analysis_cache"][key] = {
        "analysis_path": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "dataset_split": split,
        "episode": int(analysis["episode"]),
        "usage": (analysis.get("model_metadata") or {}).get("usage", {}),
    }
    append_event(state, "analysis_recorded", {"cache_key": key, "split": split})
    write_json(path, state)
    print(key)


def add_experience_command(args: argparse.Namespace) -> None:
    path = state_path(args.run_root.resolve())
    state = load_json(path)
    experience = load_json(args.experience.resolve())
    assert_no_privileged_identity(experience)
    if experience.get("task_scope") != [state["task"]]:
        raise ValueError("v2 experiences must initially be scoped to exactly the active task")
    if experience.get("status") not in {"candidate", "train_verified", "val_supported"}:
        raise ValueError("Unsupported experience status")
    required_runtime_fields = {
        "experience_id",
        "revision",
        "task_scope",
        "condition",
        "lesson",
        "recommended_skill_or_policy",
        "success_evidence",
        "forbidden_behavior",
    }
    missing = sorted(required_runtime_fields.difference(experience))
    if missing:
        raise ValueError(f"Experience is missing runtime fields: {missing}")
    experience_id = str(experience.get("experience_id", ""))
    if not experience_id:
        raise ValueError("Experience requires experience_id")
    rows = [row for row in state["experiences"] if row["experience_id"] != experience_id]
    rows.append(
        {
            "experience_id": experience_id,
            "status": experience["status"],
            "path": str(args.experience.resolve()),
            "sha256": sha256_file(args.experience.resolve()),
            "round": int(state["current_round"]),
        }
    )
    state["experiences"] = rows
    append_event(state, "experience_recorded", {"experience_id": experience_id})
    write_json(path, state)
    print(experience_id)


def freeze_command(args: argparse.Namespace) -> None:
    path = state_path(args.run_root.resolve())
    state = load_json(path)
    accepted = [row for row in state["experiences"] if row["status"] == "val_supported"]
    bundle = {
        "schema_version": "self-evolve-v2-frozen-bundle-v1",
        "bundle_id": f"{state['task']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "task": state["task"],
        "base_profile": state["base_profile"],
        "source_splits": ["train", "val"],
        "experiences": [load_json(Path(row["path"])) for row in accepted],
        "source_commit": state["source_commit"],
        "frozen_at": utc_now(),
    }
    assert_no_privileged_identity(bundle["experiences"])
    bundle_path = args.run_root.resolve() / "frozen" / "experience_bundle.json"
    write_json(bundle_path, bundle)
    runtime_fields = (
        "experience_id",
        "revision",
        "task_scope",
        "condition",
        "lesson",
        "recommended_skill_or_policy",
        "success_evidence",
        "forbidden_behavior",
    )
    runtime_memory = {
        "schema_version": "self-evolve-runtime-lessons-v2",
        "memory_id": bundle["bundle_id"],
        "revision": 1,
        "base_profile": "retrieval_v8_5_refine_then_lock",
        "experiences": [
            {
                **{field: experience[field] for field in runtime_fields},
                "status": "approved",
            }
            for experience in bundle["experiences"]
        ],
    }
    runtime_path = args.run_root.resolve() / "frozen" / "runtime_lesson_memory.json"
    write_json(runtime_path, runtime_memory)
    state["stage"] = "frozen"
    state["frozen_bundle"] = {
        "path": str(bundle_path),
        "sha256": sha256_file(bundle_path),
        "runtime_path": str(runtime_path),
        "runtime_sha256": sha256_file(runtime_path),
        "accepted_experience_count": len(accepted),
    }
    append_event(state, "frozen", state["frozen_bundle"])
    write_json(path, state)
    print(bundle_path)


def status_command(args: argparse.Namespace) -> None:
    print(json.dumps(load_json(state_path(args.run_root.resolve())), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--repo-root", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--task", required=True)
    init.add_argument("--max-evolution-rounds", type=int)
    init.set_defaults(func=init_command)

    pool = subparsers.add_parser("candidate-pool")
    pool.add_argument("--run-root", type=Path, required=True)
    pool.add_argument("--metadata-root", type=Path, required=True)
    pool.add_argument("--dataset-split", choices=("train", "val"), default="train")
    pool.set_defaults(func=pool_command)

    advance = subparsers.add_parser("advance-round")
    advance.add_argument("--run-root", type=Path, required=True)
    advance.set_defaults(func=advance_command)

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

    experience = subparsers.add_parser("add-experience")
    experience.add_argument("--run-root", type=Path, required=True)
    experience.add_argument("--experience", type=Path, required=True)
    experience.set_defaults(func=add_experience_command)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.set_defaults(func=freeze_command)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.set_defaults(func=status_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
