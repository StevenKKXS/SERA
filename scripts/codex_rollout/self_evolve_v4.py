#!/usr/bin/env python3
"""Full-stack train/val self-evolution controller built on the v3 workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    import self_evolve_v3 as v3
    from build_self_evolve_v4_candidate import compose_candidates
    from interventions import (
        canonical_hash,
        compose_candidate_manifests,
        validate_candidate_manifest,
    )
    from planner import write_json
    from prepare_self_evolve_v4_repair_request import route_components
    from self_evolve_v2 import git_commit, load_json, sha256_file
except ModuleNotFoundError:  # Repository-root unittest imports.
    from scripts.codex_rollout import self_evolve_v3 as v3
    from scripts.codex_rollout.build_self_evolve_v4_candidate import compose_candidates
    from scripts.codex_rollout.interventions import (
        canonical_hash,
        compose_candidate_manifests,
        validate_candidate_manifest,
    )
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.prepare_self_evolve_v4_repair_request import (
        route_components,
    )
    from scripts.codex_rollout.self_evolve_v2 import (
        git_commit,
        load_json,
        sha256_file,
    )


STATE_SCHEMA = "self-evolve-v4-fullstack-task-state-v1"
FULLSTACK_BEHAVIOR_COMPONENTS = frozenset({
    "criterion_overlay",
    "bounded_transition_gate",
    "swing_visual_binding_gate",
    "counting_event_ledger",
    "video_repick_event_ledger",
    "video_repick_grounding_recovery_gate",
    "unmask_pickup_grounding_recovery_gate",
    "button_swap_press_grounding_recovery_gate",
    "pickhighlight_pickup_grounding_recovery_gate",
    "retrieval_overlay",
    "action_canonicalizer",
    "reinforcement_overlay",
})


def state_path(run_root: Path) -> Path:
    return run_root / "state.v4.json"


def save_state(run_root: Path, state: dict[str, Any]) -> None:
    write_json(state_path(run_root), state)


def create_state(
    config: dict[str, Any], task: str, source_commit: str
) -> dict[str, Any]:
    state = v3.create_state(config, task, source_commit)
    state["schema_version"] = STATE_SCHEMA
    state["working_bundle"] = []
    state["accepted_experiences"] = []
    state["fallback_candidates"] = []
    state["task_evolution_status"] = "active"
    state["task_terminal_reason"] = None
    state["selected_fallback_candidate_id"] = None
    state["candidate_format"] = "typed_intervention_bundle"
    state["intervention_policy"] = dict(config.get("intervention_policy") or {})
    state["base_behavior_bundle_sha256"] = candidate_bundle_hash([])
    v3.append_event(
        state,
        "v4_fullstack_initialized",
        {"base_behavior_bundle_sha256": state["base_behavior_bundle_sha256"]},
    )
    return state


def candidate_bundle_projection(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    components = []
    candidate_ids = []
    task_scope: list[str] = []
    for record in candidates:
        candidate = record.get("candidate", record)
        if not task_scope:
            task_scope = list(candidate["task_scope"])
        elif task_scope != candidate["task_scope"]:
            raise ValueError("A v4 working bundle cannot mix task scopes")
        candidate_ids.append(candidate["candidate_id"])
        for component in candidate["components"]:
            projected = {
                key: value for key, value in component.items() if key != "artifacts"
            }
            projected["artifacts"] = {
                name: {"sha256": descriptor["sha256"]}
                for name, descriptor in component["artifacts"].items()
            }
            components.append(projected)
    return {
        "schema_version": "self-evolve-v4-behavior-bundle-projection-v1",
        "base_profile": "retrieval_v8_5_refine_then_lock",
        "task_scope": task_scope,
        "candidate_ids": candidate_ids,
        "components": sorted(
            components, key=lambda row: (int(row["priority"]), row["component_id"])
        ),
    }


def candidate_bundle_hash(candidates: list[dict[str, Any]]) -> str:
    return canonical_hash(candidate_bundle_projection(candidates))


def reassess_fullstack_actionability(
    discovery: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Reopen only v3 prompt-unactionable problems covered by a built-in hook."""
    result = json.loads(json.dumps(discovery))
    policy = config.get("intervention_policy") or {}
    for cluster in result.get("clusters", []):
        if cluster.get("status") != "unactionable":
            continue
        if int(cluster.get("support", 0)) < 2:
            continue
        if float(cluster.get("runtime_observable_ratio", 0.0)) <= 0.0:
            continue
        suggestions = route_components(cluster)
        available = sorted(FULLSTACK_BEHAVIOR_COMPONENTS.intersection(suggestions))
        if not available:
            continue
        policy_keys = {
            "criterion_overlay": "allow_runtime_criterion",
            "bounded_transition_gate": "allow_runtime_gate",
            "swing_visual_binding_gate": "allow_runtime_gate",
            "counting_event_ledger": "allow_structured_ledger",
            "video_repick_event_ledger": "allow_structured_ledger",
            "video_repick_grounding_recovery_gate": "allow_runtime_gate",
            "unmask_pickup_grounding_recovery_gate": "allow_runtime_gate",
            "button_swap_press_grounding_recovery_gate": "allow_runtime_gate",
            "pickhighlight_pickup_grounding_recovery_gate": "allow_runtime_gate",
            "retrieval_overlay": "allow_retrieval_overlay",
            "action_canonicalizer": "allow_action_canonicalizer",
            "reinforcement_overlay": "allow_reinforcement_overlay",
        }
        enabled = [
            component
            for component in available
            if not policy or policy.get(policy_keys[component], False)
        ]
        if not enabled:
            continue
        cluster["v3_status_before_fullstack_reassessment"] = "unactionable"
        cluster["v4_actionability_reassessment"] = {
            "decision": "reopen_as_active",
            "reason": "registered_fullstack_component_can_express_runtime_repair",
            "suggested_component_types": enabled,
        }
        cluster["status"] = "active"
    return result


def _candidate_objects(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [record["candidate"] for record in state["working_bundle"]]


def consume_verifier_samples(
    state: dict[str, Any], report: dict[str, Any], *, round_number: int
) -> int:
    """Prevent verifier feedback from becoming reusable holdout data."""
    consumed = state.setdefault("analysis_consumed_samples", [])
    provenance = state.setdefault("verifier_consumed_samples", [])
    added = 0
    for row in report.get("sample_provenance", []):
        split = str(row.get("dataset_split"))
        if split not in {"train", "val"}:
            raise ValueError("Verifier-consumed samples must be train/val")
        sample = {"dataset_split": split, "episode_id": int(row["episode_id"])}
        if sample not in consumed:
            consumed.append(sample)
            added += 1
        audit = {
            **sample,
            "candidate_id": str(report["candidate_id"]),
            "round": int(round_number),
        }
        if audit not in provenance:
            provenance.append(audit)
    v3.append_event(
        state,
        "v4_verifier_samples_consumed",
        {
            "candidate_id": report["candidate_id"],
            "round": int(round_number),
            "new_sample_count": added,
        },
    )
    return added


def start_candidate(state: dict[str, Any], candidate_path: Path) -> dict[str, Any]:
    candidate_path = candidate_path.resolve()
    candidate = validate_candidate_manifest(
        load_json(candidate_path), bundle_dir=candidate_path.parent
    )
    if candidate["status"] != "candidate":
        raise ValueError("A new v4 round requires candidate status")
    if not candidate["authoring"]["analysis_inputs"]:
        raise ValueError(
            "A v4 evolution candidate requires at least one train/val analysis input"
        )
    policy_keys = {
        "prompt_lesson": "allow_prompt_lesson",
        "criterion_overlay": "allow_runtime_criterion",
        "bounded_transition_gate": "allow_runtime_gate",
        "swing_visual_binding_gate": "allow_runtime_gate",
        "counting_event_ledger": "allow_structured_ledger",
        "video_repick_event_ledger": "allow_structured_ledger",
        "video_repick_grounding_recovery_gate": "allow_runtime_gate",
        "unmask_pickup_grounding_recovery_gate": "allow_runtime_gate",
        "button_swap_press_grounding_recovery_gate": "allow_runtime_gate",
        "pickhighlight_pickup_grounding_recovery_gate": "allow_runtime_gate",
        "retrieval_overlay": "allow_retrieval_overlay",
        "action_canonicalizer": "allow_action_canonicalizer",
        "reinforcement_overlay": "allow_reinforcement_overlay",
        "python_plugin": "allow_new_python_plugin_after_static_validation",
        "core_harness_patch": "allow_core_harness_patch_after_materialized_commit",
    }
    intervention_policy = state.get("intervention_policy") or {}
    disabled = [
        component["type"]
        for component in candidate["components"]
        if intervention_policy
        and not intervention_policy.get(policy_keys[component["type"]], False)
    ]
    if disabled:
        raise ValueError(
            f"Candidate uses intervention types disabled by experiment policy: {disabled}"
        )
    if candidate["task_scope"] != [state["task"]]:
        raise ValueError("Candidate task scope does not match state")
    if state.get("current_candidate") is not None:
        raise ValueError("A candidate is already active")
    problem_id = candidate["problem_id"]
    if problem_id != state.get("active_problem_id"):
        raise ValueError("Candidate problem_id must match the active problem")
    if state["total_evolution_rounds"] >= state["max_total_evolution_rounds"]:
        raise ValueError("Task evolution round budget is exhausted")
    attempts = int(state["problem_attempts"].get(problem_id, 0))
    if attempts >= state["max_attempts_per_problem"]:
        state["problems"][problem_id]["status"] = "exhausted"
        raise ValueError("Problem attempt budget is exhausted")

    base_records = list(state["working_bundle"])
    candidate_record = {
        "candidate": candidate,
        "candidate_manifest_path": str(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
    }
    # Conflict validation is performed before a round can consume rollout time.
    compose_candidate_manifests(
        [*(_candidate_objects(state)), candidate],
        bundle_id="static-conflict-check",
        revision=max(1, int(state["bundle_revision"]) + 1),
    )
    round_number = int(state["total_evolution_rounds"]) + 1
    current = {
        "round": round_number,
        "problem_id": problem_id,
        "problem_attempt": attempts + 1,
        "candidate_id": candidate["candidate_id"],
        "candidate_revision": candidate["revision"],
        **candidate_record,
        "base_bundle_revision": int(state["bundle_revision"]),
        "base_bundle_candidate_ids": [
            record["candidate"]["candidate_id"] for record in base_records
        ],
        "base_bundle_sha256": candidate_bundle_hash(base_records),
        "candidate_bundle_sha256": candidate_bundle_hash(
            [*base_records, candidate_record]
        ),
        "status": "verifier_pending",
    }
    state["total_evolution_rounds"] = round_number
    state["remaining_problem_budget"] = (
        state["max_total_evolution_rounds"] - round_number
    )
    state["problem_attempts"][problem_id] = attempts + 1
    state["current_candidate"] = current
    state["stage"] = "candidate_verification"
    v3.append_event(
        state,
        "v4_candidate_started",
        {
            "round": round_number,
            "problem_id": problem_id,
            "candidate_id": candidate["candidate_id"],
            "component_types": [row["type"] for row in candidate["components"]],
        },
    )
    return current


def decide_candidate(
    state: dict[str, Any], report: dict[str, Any], decision: str, reason: str
) -> dict[str, Any]:
    current = state.get("current_candidate")
    if current is None:
        raise ValueError("No candidate is awaiting a decision")
    if report.get("schema_version") != "self-evolve-v4-paired-verifier-report-v1":
        raise ValueError("V4 decisions require a v4 paired verifier report")
    if report.get("candidate_id") != current["candidate_id"]:
        raise ValueError("Verifier report candidate does not match current candidate")
    if report.get("problem_id") != current["problem_id"]:
        raise ValueError("Verifier report problem does not match current candidate")
    if report.get("task") != state["task"]:
        raise ValueError("Verifier report task does not match state")
    if decision not in {
        "accept",
        "retain_fallback",
        "retry",
        "defer",
        "unactionable",
        "hypothesis_rejected",
    }:
        raise ValueError(f"Unsupported candidate decision {decision!r}")
    pilot = report.get("pilot_verifier")
    if not isinstance(pilot, dict):
        raise ValueError("V4 decision requires a paired pilot verifier")
    if decision == "accept":
        if not report.get("early_task_exit_eligible"):
            raise ValueError(
                "Only a strict/perfect candidate can terminate task evolution"
            )
        if int(pilot.get("regressed", 0)) != 0:
            raise ValueError("A regressing candidate cannot be accepted")
    if decision == "retain_fallback":
        if not report.get("fallback_pool_eligible"):
            raise ValueError("Only a +1/zero-regression candidate can be retained")
        if report.get("early_task_exit_eligible"):
            raise ValueError("A strict/perfect candidate should be accepted")
    comparison = report.get("bundle_comparison") or {}
    if comparison.get("base_bundle_sha256") != current["base_bundle_sha256"]:
        raise ValueError("Verifier did not run the recorded full base bundle")
    if comparison.get("candidate_bundle_sha256") != current["candidate_bundle_sha256"]:
        raise ValueError("Verifier did not run the recorded full candidate bundle")
    consumed = {
        (row["dataset_split"], int(row["episode_id"]))
        for row in state.get("analysis_consumed_samples", [])
    }
    if any(
        (row.get("dataset_split"), int(row.get("episode_id", -1))) in consumed
        for row in report.get("sample_provenance", [])
    ):
        raise ValueError("Verifier reuses samples consumed for candidate analysis")

    problem_id = current["problem_id"]
    problem = state["problems"][problem_id]
    round_record = {
        key: value for key, value in current.items() if key not in {"candidate"}
    }
    round_record.update(
        {
            "decision": decision,
            "decision_reason": reason,
            "verifier_report": report,
        }
    )
    if decision == "accept":
        approved = json.loads(json.dumps(current["candidate"]))
        approved["status"] = "approved"
        accepted = {
            "candidate": approved,
            "candidate_manifest_path": current["candidate_manifest_path"],
            "candidate_manifest_sha256": current["candidate_manifest_sha256"],
            "accepted_round": current["round"],
            "source_problem_id": problem_id,
            "verifier_acceptance_tier": report["acceptance_tier"],
        }
        state["accepted_experiences"].append(accepted)
        state["working_bundle"].append(accepted)
        state["bundle_revision"] = int(state["bundle_revision"]) + 1
        problem["status"] = "resolved"
        state["active_problem_id"] = None
        state["task_evolution_status"] = "early_exit_eligible"
        state["task_terminal_reason"] = (
            "strict_paired_improvement"
            if report.get("strict_pass")
            else "perfect_paired_candidate"
        )
        round_record["result_bundle_revision"] = state["bundle_revision"]
        round_record["result_bundle_sha256"] = candidate_bundle_hash(
            state["working_bundle"]
        )
    elif decision == "retain_fallback":
        retained = {
            "candidate": json.loads(json.dumps(current["candidate"])),
            "candidate_manifest_path": current["candidate_manifest_path"],
            "candidate_manifest_sha256": current["candidate_manifest_sha256"],
            "retained_round": current["round"],
            "source_problem_id": problem_id,
            "verifier_acceptance_tier": report["acceptance_tier"],
            "net_improvements": int(pilot["net_improvements"]),
            "regressions": int(pilot["regressed"]),
            "candidate_successes": int(report.get("candidate_successes", 0)),
            "verifier_report": report,
        }
        state.setdefault("fallback_candidates", []).append(retained)
        problem["status"] = "deferred"
        state["active_problem_id"] = None
        state["task_evolution_status"] = "active"
        round_record["fallback_pool_size"] = len(state["fallback_candidates"])
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
    consume_verifier_samples(state, report, round_number=int(current["round"]))
    state["current_candidate"] = None
    state["stage"] = (
        "evolution_terminal"
        if state.get("task_evolution_status") == "early_exit_eligible"
        else "evolution_budget_exhausted"
        if state["remaining_problem_budget"] == 0
        else "candidate_decided"
    )
    v3.append_event(
        state,
        "v4_candidate_decided",
        {
            "round": current["round"],
            "candidate_id": current["candidate_id"],
            "decision": decision,
            "problem_status": problem["status"],
        },
    )
    return round_record


def select_final_fallback(state: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Promote one retained fallback only after all evolution rounds."""
    if int(state["remaining_problem_budget"]) != 0:
        raise ValueError("Fallback selection is allowed only after round exhaustion")
    if state.get("task_evolution_status") == "early_exit_eligible":
        raise ValueError("Strict/perfect task terminal cannot select a fallback")
    matches = [
        row
        for row in state.get("fallback_candidates", [])
        if row["candidate"]["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("Fallback candidate_id must identify one retained candidate")
    selected = json.loads(json.dumps(matches[0]))
    selected["candidate"]["status"] = "approved"
    selected["selected_after_round_budget"] = True
    state["working_bundle"] = [selected]
    state["accepted_experiences"] = [selected]
    state["bundle_revision"] = int(state.get("bundle_revision", 0)) + 1
    state["selected_fallback_candidate_id"] = candidate_id
    state["task_evolution_status"] = "budget_fallback_selected"
    state["task_terminal_reason"] = "best_fallback_after_round_budget"
    state["stage"] = "evolution_terminal"
    v3.append_event(
        state,
        "v4_final_fallback_selected",
        {"candidate_id": candidate_id, "bundle_revision": state["bundle_revision"]},
    )
    return selected


def assert_test_snapshot_eligible(state: dict[str, Any]) -> None:
    if state.get("task_evolution_status") not in {
        "early_exit_eligible",
        "budget_fallback_selected",
    }:
        raise ValueError(
            "Test snapshot requires strict/perfect early exit or a fallback "
            "selected after all evolution rounds"
        )


def reclassify_accepted_fallback(
    state: dict[str, Any], candidate_id: str, reason: str
) -> dict[str, Any]:
    """Explicitly migrate an old v4 +1 acceptance into the fallback pool."""
    accepted = [
        row
        for row in state.get("accepted_experiences", [])
        if row["candidate"]["candidate_id"] == candidate_id
    ]
    if len(accepted) != 1:
        raise ValueError("Expected exactly one accepted candidate to reclassify")
    record = accepted[0]
    matching_rounds = [
        row
        for row in state.get("round_history", [])
        if row.get("candidate_id") == candidate_id
    ]
    if len(matching_rounds) != 1:
        raise ValueError("Expected exactly one historical candidate round")
    report = matching_rounds[0].get("verifier_report") or {}
    if report.get("acceptance_tier") != "fallback":
        raise ValueError("Only a historical fallback acceptance can be reclassified")
    pilot = report.get("pilot_verifier") or {}
    retained = json.loads(json.dumps(record))
    retained["candidate"]["status"] = "candidate"
    retained["retained_round"] = retained.pop("accepted_round")
    retained["reclassified_from_accept"] = True
    retained["reclassification_reason"] = reason
    retained["net_improvements"] = int(pilot.get("net_improvements", 0))
    retained["regressions"] = int(pilot.get("regressed", 0))
    retained["candidate_successes"] = int(
        report.get(
            "candidate_successes",
            int(pilot.get("improved", 0)) + int(pilot.get("unchanged_success", 0)),
        )
    )
    retained["verifier_report"] = report
    state.setdefault("fallback_candidates", []).append(retained)
    state["accepted_experiences"] = [
        row
        for row in state.get("accepted_experiences", [])
        if row["candidate"]["candidate_id"] != candidate_id
    ]
    state["working_bundle"] = [
        row
        for row in state.get("working_bundle", [])
        if row["candidate"]["candidate_id"] != candidate_id
    ]
    state["bundle_revision"] = len(state["working_bundle"])
    problem_id = record["source_problem_id"]
    state["problems"][problem_id]["status"] = "deferred"
    state["active_problem_id"] = None
    state["task_evolution_status"] = "active"
    state["task_terminal_reason"] = None
    state["selected_fallback_candidate_id"] = None
    if state.get("test_candidate_snapshot"):
        state["revoked_test_candidate_snapshot"] = state.pop(
            "test_candidate_snapshot"
        )
    state["stage"] = (
        "evolution_budget_exhausted"
        if int(state["remaining_problem_budget"]) == 0
        else "candidate_decided"
    )
    matching_rounds[0]["decision"] = "retain_fallback"
    matching_rounds[0]["decision_reason"] = reason
    matching_rounds[0]["reclassified_from_accept"] = True
    v3.append_event(
        state,
        "v4_accepted_fallback_reclassified",
        {"candidate_id": candidate_id, "reason": reason},
    )
    return retained


def refresh_retained_fallback_report(
    state: dict[str, Any], candidate_id: str, report: dict[str, Any]
) -> None:
    if report.get("candidate_id") != candidate_id:
        raise ValueError("Corrected verifier report candidate_id mismatch")
    if report.get("acceptance_tier") != "fallback":
        raise ValueError("Corrected retained report must remain fallback tier")
    if not report.get("fallback_pool_eligible"):
        raise ValueError("Corrected report must be fallback-pool eligible")
    if report.get("early_task_exit_eligible") or report.get(
        "test_release_eligible"
    ):
        raise ValueError("Retained fallback cannot authorize early test release")
    fallbacks = [
        row
        for row in state.get("fallback_candidates", [])
        if row["candidate"]["candidate_id"] == candidate_id
    ]
    rounds = [
        row
        for row in state.get("round_history", [])
        if row.get("candidate_id") == candidate_id
    ]
    if len(fallbacks) != 1 or len(rounds) != 1:
        raise ValueError("Expected one retained fallback and one matching round")
    pilot = report["pilot_verifier"]
    fallbacks[0]["verifier_report"] = report
    fallbacks[0]["net_improvements"] = int(pilot["net_improvements"])
    fallbacks[0]["regressions"] = int(pilot["regressed"])
    fallbacks[0]["candidate_successes"] = int(report["candidate_successes"])
    rounds[0]["verifier_report"] = report
    v3.append_event(
        state,
        "v4_retained_fallback_report_refreshed",
        {"candidate_id": candidate_id},
    )


def init_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    path = state_path(run_root)
    if path.exists():
        state = load_json(path)
        if state["task"] != args.task:
            raise ValueError("Run root already belongs to another task")
        print(path)
        return
    config_path = args.config.resolve()
    config = load_json(config_path)
    state = create_state(config, args.task, git_commit(args.repo_root.resolve()))
    state["config_path"] = str(config_path)
    state["config_sha256"] = sha256_file(config_path)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.v4.snapshot.json", config)
    save_state(run_root, state)
    print(path)


def candidate_pool_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    split = args.dataset_split
    gate = config["evolution_dataset_gate"]
    limit = int(
        gate["required_train_episodes" if split == "train" else "required_val_episodes"]
    )
    selected, provenance = v3.select_dataset_episodes(
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
        "schema_version": "self-evolve-v4-evolution50-candidate-pool-v1",
        "task": state["task"],
        "dataset_split": split,
        "sampling": provenance,
        "candidates": candidates,
    }
    output_dir = run_root / "profiles" / split
    pool_path = output_dir / "candidate_pool.json"
    ids_path = output_dir / "episode_ids.txt"
    write_json(pool_path, output)
    v3.write_text_atomic(
        ids_path, ",".join(str(row["episode"]) for row in candidates) + "\n"
    )
    state.setdefault("evolution_dataset_selection", {})[split] = {
        "pool_path": str(pool_path),
        "pool_sha256": sha256_file(pool_path),
        "episode_ids_path": str(ids_path),
        "selected_count": len(candidates),
        "sampling": provenance,
    }
    v3.append_event(
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
    records = v3.read_jsonl(args.episode_records.resolve())
    discovery = reassess_fullstack_actionability(
        v3.build_problem_discovery(records, config, state["task"]), config
    )
    output_dir = run_root / "problem_discovery"
    v3.write_jsonl(output_dir / "episode_problem_records.jsonl", records)
    write_json(output_dir / "problem_clusters.json", discovery)
    v3.write_text_atomic(
        output_dir / "problem_report.md", v3.render_problem_report(discovery)
    )
    v3.install_problem_discovery(state, discovery)
    save_state(run_root, state)
    print(output_dir / "problem_clusters.json")


def record_hard_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    if args.command == "record-hard-trigger":
        path = args.trigger_report.resolve()
        v3.record_hard_trigger(state, load_json(path), path)
        result = state["hard_exploration"]["trigger"]["decision"]
    elif args.command == "record-hard-selection":
        path = args.selection_manifest.resolve()
        v3.record_hard_selection(state, v3.read_jsonl(path), path)
        result = state["hard_exploration"]["selection"]["selected_count"]
    else:
        report_path = args.extension_report.resolve()
        manifest_path = args.extension_manifest.resolve()
        v3.record_hard_results(
            state, load_json(report_path), report_path, manifest_path
        )
        result = state["hard_exploration"]["results"]["failures"]
    save_state(run_root, state)
    print(result)


def record_rollout_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    summary_path = args.summary.resolve()
    key = v3.record_rollout(
        state,
        load_json(summary_path),
        summary_path,
        planner_commit=args.planner_commit,
        planner_config_sha256=args.planner_config_sha256,
        executor_version=args.executor_version,
        experience_bundle_sha256=args.intervention_bundle_sha256,
        metadata_sha256=args.metadata_sha256,
    )
    save_state(run_root, state)
    print(key)


def record_analysis_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    path = args.analysis.resolve()
    key = v3.record_analysis(
        state,
        load_json(path),
        path,
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
    problem_id = args.problem_id or v3.next_problem_id(state, config)
    v3.select_problem(state, problem_id, config)
    save_state(run_root, state)
    print(problem_id)


def start_candidate_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    current = start_candidate(state, args.candidate.resolve())
    round_dir = run_root / f"round_{current['round']:02d}"
    write_json(
        round_dir / "selected_problem.json", state["problems"][current["problem_id"]]
    )
    write_json(
        round_dir / "base_bundle.json",
        {
            "bundle_revision": current["base_bundle_revision"],
            "candidate_ids": current["base_bundle_candidate_ids"],
            "base_bundle_sha256": current["base_bundle_sha256"],
            "candidate_bundle_sha256": current["candidate_bundle_sha256"],
        },
    )
    write_json(round_dir / "candidate_intervention.json", current["candidate"])
    save_state(run_root, state)
    print(current["round"])


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
                "behavior_bundle_sha256": candidate_bundle_hash(
                    state["working_bundle"]
                ),
                "candidates": state["working_bundle"],
            },
        )
    save_state(run_root, state)
    print(args.decision)


def select_final_fallback_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    selected = select_final_fallback(state, args.candidate_id)
    write_json(
        run_root
        / "bundle_history"
        / f"bundle_revision_{state['bundle_revision']:03d}.json",
        {
            "bundle_revision": state["bundle_revision"],
            "behavior_bundle_sha256": candidate_bundle_hash(state["working_bundle"]),
            "candidates": state["working_bundle"],
            "selection_policy": "best_fallback_after_round_budget",
        },
    )
    save_state(run_root, state)
    print(selected["candidate"]["candidate_id"])


def reclassify_fallback_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    retained = reclassify_accepted_fallback(state, args.candidate_id, args.reason)
    write_json(
        run_root / "migration" / f"{args.candidate_id}.fallback_reclassification.json",
        {
            "schema_version": "self-evolve-v4-fallback-reclassification-v1",
            "candidate_id": args.candidate_id,
            "reason": args.reason,
            "reportable_test_snapshot": False,
            "retained_candidate": retained,
        },
    )
    save_state(run_root, state)
    print(args.candidate_id)


def refresh_fallback_report_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    report = load_json(args.verifier_report.resolve())
    refresh_retained_fallback_report(state, args.candidate_id, report)
    save_state(run_root, state)
    print(args.candidate_id)


def consume_verifier_report_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    report = load_json(args.verifier_report.resolve())
    added = consume_verifier_samples(
        state, report, round_number=int(args.round_number)
    )
    save_state(run_root, state)
    print(added)


def snapshot_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Repository must be clean before recording a test candidate")
    state = load_json(state_path(run_root))
    if state.get("current_candidate") is not None:
        raise ValueError("Cannot snapshot an undecided candidate")
    assert_test_snapshot_eligible(state)
    if not state["working_bundle"]:
        raise ValueError("Cannot snapshot an empty intervention bundle")
    output_dir = run_root / "test_candidate_snapshot"
    candidate_paths = [
        Path(record["candidate_manifest_path"]) for record in state["working_bundle"]
    ]
    bundle_path = compose_candidates(
        candidate_paths,
        output_dir,
        bundle_id=f"{state['task']}-bundle-r{state['bundle_revision']}",
        revision=int(state["bundle_revision"]),
    )
    runtime_bundle = load_json(bundle_path)
    manifest = {
        "schema_version": "self-evolve-v4-test-candidate-snapshot-v1",
        "snapshot_id": runtime_bundle["bundle_id"],
        "task": state["task"],
        "base_profile": state["base_profile"],
        "source_commit": git_commit(repo_root),
        "evolution_start_commit": state["source_commit"],
        "experiment_config_sha256": state["config_sha256"],
        "bundle_revision": state["bundle_revision"],
        "runtime_bundle_sha256": runtime_bundle["bundle_sha256"],
        "behavior_bundle_sha256": candidate_bundle_hash(state["working_bundle"]),
        "accepted_candidate_ids": runtime_bundle["candidate_ids"],
        "executor_checkpoint": args.executor_checkpoint,
        "total_evolution_rounds": state["total_evolution_rounds"],
        "test_policy": "one_shot_feedback_forbidden",
        "test_feedback_used": False,
        "final_train_val50_required": False,
    }
    write_json(output_dir / "reproducibility_manifest.json", manifest)
    state["stage"] = "test_candidate_recorded"
    state["test_candidate_snapshot"] = manifest
    v3.append_event(state, "v4_test_candidate_recorded", manifest)
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
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--repo-root", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--task", required=True)
    init.set_defaults(func=init_command)

    pool = sub.add_parser("candidate-pool")
    pool.add_argument("--run-root", type=Path, required=True)
    pool.add_argument("--metadata-root", type=Path, required=True)
    pool.add_argument("--dataset-split", choices=("train", "val"), required=True)
    pool.set_defaults(func=candidate_pool_command)

    discover = sub.add_parser("discover-problems")
    discover.add_argument("--run-root", type=Path, required=True)
    discover.add_argument("--episode-records", type=Path, required=True)
    discover.set_defaults(func=discover_command)

    hard_trigger = sub.add_parser("record-hard-trigger")
    hard_trigger.add_argument("--run-root", type=Path, required=True)
    hard_trigger.add_argument("--trigger-report", type=Path, required=True)
    hard_trigger.set_defaults(func=record_hard_command)
    hard_selection = sub.add_parser("record-hard-selection")
    hard_selection.add_argument("--run-root", type=Path, required=True)
    hard_selection.add_argument("--selection-manifest", type=Path, required=True)
    hard_selection.set_defaults(func=record_hard_command)
    hard_results = sub.add_parser("record-hard-results")
    hard_results.add_argument("--run-root", type=Path, required=True)
    hard_results.add_argument("--extension-report", type=Path, required=True)
    hard_results.add_argument("--extension-manifest", type=Path, required=True)
    hard_results.set_defaults(func=record_hard_command)

    rollout = sub.add_parser("record-rollout")
    rollout.add_argument("--run-root", type=Path, required=True)
    rollout.add_argument("--summary", type=Path, required=True)
    rollout.add_argument("--planner-commit", required=True)
    rollout.add_argument("--planner-config-sha256", required=True)
    rollout.add_argument("--executor-version", required=True)
    rollout.add_argument("--intervention-bundle-sha256", required=True)
    rollout.add_argument("--metadata-sha256", required=True)
    rollout.set_defaults(func=record_rollout_command)

    analysis = sub.add_parser("record-analysis")
    analysis.add_argument("--run-root", type=Path, required=True)
    analysis.add_argument("--analysis", type=Path, required=True)
    analysis.add_argument("--trajectory-sha256", required=True)
    analysis.add_argument("--ground-truth-reference-sha256", required=True)
    analysis.add_argument("--analyzer-prompt-sha256", required=True)
    analysis.add_argument("--model", required=True)
    analysis.set_defaults(func=record_analysis_command)

    choose = sub.add_parser("select-problem")
    choose.add_argument("--run-root", type=Path, required=True)
    choose.add_argument("--problem-id")
    choose.set_defaults(func=select_problem_command)
    candidate = sub.add_parser("start-candidate")
    candidate.add_argument("--run-root", type=Path, required=True)
    candidate.add_argument("--candidate", type=Path, required=True)
    candidate.set_defaults(func=start_candidate_command)
    decision = sub.add_parser("decide-candidate")
    decision.add_argument("--run-root", type=Path, required=True)
    decision.add_argument("--verifier-report", type=Path, required=True)
    decision.add_argument(
        "--decision",
        choices=(
            "accept",
            "retain_fallback",
            "retry",
            "defer",
            "unactionable",
            "hypothesis_rejected",
        ),
        required=True,
    )
    decision.add_argument("--reason", required=True)
    decision.set_defaults(func=decide_command)
    fallback = sub.add_parser("select-final-fallback")
    fallback.add_argument("--run-root", type=Path, required=True)
    fallback.add_argument("--candidate-id", required=True)
    fallback.set_defaults(func=select_final_fallback_command)
    reclassify = sub.add_parser("reclassify-accepted-fallback")
    reclassify.add_argument("--run-root", type=Path, required=True)
    reclassify.add_argument("--candidate-id", required=True)
    reclassify.add_argument("--reason", required=True)
    reclassify.set_defaults(func=reclassify_fallback_command)
    refresh = sub.add_parser("refresh-retained-fallback-report")
    refresh.add_argument("--run-root", type=Path, required=True)
    refresh.add_argument("--candidate-id", required=True)
    refresh.add_argument("--verifier-report", type=Path, required=True)
    refresh.set_defaults(func=refresh_fallback_report_command)
    consume = sub.add_parser("consume-verifier-report")
    consume.add_argument("--run-root", type=Path, required=True)
    consume.add_argument("--verifier-report", type=Path, required=True)
    consume.add_argument("--round-number", type=int, required=True)
    consume.set_defaults(func=consume_verifier_report_command)
    snapshot = sub.add_parser("record-test-candidate")
    snapshot.add_argument("--run-root", type=Path, required=True)
    snapshot.add_argument("--repo-root", type=Path, required=True)
    snapshot.add_argument("--executor-checkpoint", required=True)
    snapshot.set_defaults(func=snapshot_command)
    status = sub.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    status.set_defaults(func=status_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
