#!/usr/bin/env python3
"""Deep, revision-driven train/val self-evolution controller (v5)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    import self_evolve_v3 as v3
    import self_evolve_v4 as v4
    from build_self_evolve_v4_candidate import compose_candidates
    from interventions import canonical_hash
    from planner import write_json
    from self_evolve_v2 import git_commit, load_json, sha256_file
    from verify_self_evolve_v5_candidate import (
        PANEL_SCHEMA,
        REPORT_SCHEMA,
        panel_hash,
        validate_panel,
    )
except ModuleNotFoundError:  # Repository-root unittest imports.
    from scripts.codex_rollout import self_evolve_v3 as v3
    from scripts.codex_rollout import self_evolve_v4 as v4
    from scripts.codex_rollout.build_self_evolve_v4_candidate import (
        compose_candidates,
    )
    from scripts.codex_rollout.interventions import canonical_hash
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.self_evolve_v2 import (
        git_commit,
        load_json,
        sha256_file,
    )
    from scripts.codex_rollout.verify_self_evolve_v5_candidate import (
        PANEL_SCHEMA,
        REPORT_SCHEMA,
        panel_hash,
        validate_panel,
    )


STATE_SCHEMA = "self-evolve-v5-deep-pilot-task-state-v1"
LINEAGE_SCHEMA = "self-evolve-v5-candidate-lineage-v1"
RETROSPECTIVE_SCHEMA = "self-evolve-v5-candidate-retrospective-v1"

FAILURE_TYPES = {
    "none",
    "wiring_failure",
    "underpowered_repair",
    "side_effect_regression",
    "hypothesis_rejected",
    "capability_unavailable",
    "inconclusive",
}
RETROSPECTIVE_ACTIONS = {
    "promote_to_generalization",
    "accept",
    "revise_same_problem",
    "switch_problem",
    "retain_fallback",
    "unactionable",
    "hypothesis_rejected",
}
SWITCH_REASON_CODES = {
    "hypothesis_disproved",
    "runtime_unobservable",
    "capability_unavailable",
    "behaviorally_equivalent_revisions",
    "activated_no_effect_limit",
    "problem_attempt_budget_exhausted",
    "higher_expected_value_problem",
}


def state_path(run_root: Path) -> Path:
    return run_root / "state.v5.json"


def save_state(run_root: Path, state: dict[str, Any]) -> None:
    write_json(state_path(run_root), state)


def create_state(
    config: dict[str, Any], task: str, source_commit: str
) -> dict[str, Any]:
    state = v4.create_state(config, task, source_commit)
    state["schema_version"] = STATE_SCHEMA
    state["panel_registry"] = {}
    state["panel_history"] = []
    state["problem_candidate_lineage"] = {}
    state["retrospectives"] = []
    state["development_samples_are_reusable"] = True
    state["generalization_samples_must_be_fresh"] = True
    state["candidate_format"] = "v4_typed_intervention_plus_v5_lineage"
    v3.append_event(
        state,
        "v5_initialized",
        {
            "development_panel_reuse": True,
            "perfect_verifier_denominator": "registered_panel_size",
        },
    )
    return state


def _deep_failure_record(record: dict[str, Any]) -> dict[str, Any]:
    v3.validate_episode_problem_record(record)
    if record.get("outcome", "failure") == "success":
        return json.loads(json.dumps(record))
    causal = record.get("causal_chain")
    required = {
        "physical_first_divergence",
        "task_progress_divergence",
        "terminal_failure",
        "causal_link_evidence",
    }
    if not isinstance(causal, dict) or set(causal) != required:
        raise ValueError(
            "V5 failure analysis requires physical, task-progress, and terminal "
            "causal-chain fields"
        )
    for field in required - {"causal_link_evidence"}:
        if not isinstance(causal[field], str) or not causal[field].strip():
            raise ValueError(f"causal_chain.{field} must be non-empty")
    if (
        not isinstance(causal["causal_link_evidence"], list)
        or not causal["causal_link_evidence"]
    ):
        raise ValueError("causal_chain.causal_link_evidence must be non-empty")
    enriched = json.loads(json.dumps(record))
    # Match the proven v1 grouping boundary: controlled failure type,
    # responsible module, and task/method scope determine the group.  The deep
    # natural-language causal chain is supporting evidence only; using it as
    # an exact cluster key fragments equivalent failures by wording.
    enriched["cluster_policy"] = "v1_categorical"
    return enriched


def build_deep_problem_discovery(
    records: list[dict[str, Any]], config: dict[str, Any], task: str
) -> dict[str, Any]:
    cluster_policy = str(
        config.get("problem_discovery", {}).get("cluster_policy", "v1_categorical")
    )
    if cluster_policy != "v1_categorical":
        raise ValueError(f"Unsupported v5 cluster policy: {cluster_policy}")
    enriched = [_deep_failure_record(record) for record in records]
    discovery = v4.reassess_fullstack_actionability(
        v3.build_problem_discovery(enriched, config, task), config
    )
    by_identity = {
        (str(row["dataset_split"]), int(row["episode_id"])): row for row in records
    }
    for cluster in discovery["clusters"]:
        causal_examples = []
        candidate_support_refs = []
        for ref in cluster.get("episode_refs", []):
            source = by_identity[(str(ref["dataset_split"]), int(ref["episode_id"]))]
            if source.get("outcome", "failure") == "failure":
                causal_examples.append(
                    {
                        "dataset_split": ref["dataset_split"],
                        "episode_id": int(ref["episode_id"]),
                        **source["causal_chain"],
                    }
                )
                if source.get("candidate_support_allowed", True):
                    candidate_support_refs.append(
                        {
                            "dataset_split": ref["dataset_split"],
                            "episode_id": int(ref["episode_id"]),
                        }
                    )
        cluster["causal_chain_examples"] = causal_examples
        cluster["task_progress_divergence_examples"] = sorted(
            {row["task_progress_divergence"] for row in causal_examples}
        )
        cluster["terminal_failure_examples"] = sorted(
            {row["terminal_failure"] for row in causal_examples}
        )
        cluster["analysis_depth"] = "physical_to_task_progress_to_terminal"
        cluster["candidate_support_count"] = len(candidate_support_refs)
        cluster["candidate_support_refs"] = candidate_support_refs
        if not candidate_support_refs:
            cluster["status"] = "unactionable"
            cluster["selected_for_deep_review"] = False
            cluster["candidate_exclusion_reason"] = (
                "v1 requires adjudicated candidate support; disputed model "
                "hypotheses remain offline evidence only"
            )
    discovery["schema_version"] = "self-evolve-v5-deep-problem-discovery-v1"
    discovery["analysis_depth_required"] = True
    discovery["cluster_policy"] = cluster_policy
    discovery["causal_layers"] = [
        "physical_first_divergence",
        "task_progress_divergence",
        "terminal_failure",
    ]
    return discovery


def _panel_record(panel: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    return {
        "panel": panel,
        "panel_path": str(path.resolve()) if path else None,
        "panel_sha256": panel_hash(panel),
        "sample_count": len(panel["samples"]),
    }


def register_panel(
    state: dict[str, Any], panel: dict[str, Any], config: dict[str, Any], path: Path
) -> dict[str, Any]:
    panel = validate_panel(panel, config)
    if panel["task"] != state["task"]:
        raise ValueError("Panel task does not match state")
    problem_id = panel["problem_id"]
    if problem_id not in state["problems"]:
        raise ValueError("Panel problem_id is not installed in state")
    role = panel["panel_role"]
    panels = state.setdefault("panel_registry", {}).setdefault(problem_id, {})
    record = _panel_record(panel, path)
    existing = panels.get(role)
    if existing:
        if existing["panel_sha256"] == record["panel_sha256"]:
            return existing
        current = state.get("current_candidate") or {}
        existing_was_used = any(
            row.get("problem_id") == problem_id
            and (row.get("verifier_reports") or {}).get("generalization")
            and row.get("generalization_panel_sha256")
            == existing["panel_sha256"]
            for row in state.get("round_history", [])
        )
        may_refresh_generalization = (
            role == "generalization"
            and current.get("problem_id") == problem_id
            and current.get("status") == "development_decision_pending"
            and existing_was_used
        )
        if not may_refresh_generalization:
            raise ValueError(
                f"The fixed {role} panel for {problem_id} cannot be replaced"
            )

    identities = {
        (row["dataset_split"], int(row["episode_id"])) for row in panel["samples"]
    }
    if role == "generalization":
        development = panels.get("development")
        if development is None:
            raise ValueError("Register the development panel first")
        development_ids = {
            (row["dataset_split"], int(row["episode_id"]))
            for row in development["panel"]["samples"]
        }
        if identities.intersection(development_ids):
            raise ValueError("Generalization panel must be disjoint from development")
        analyzed = {
            (row["dataset_split"], int(row["episode_id"]))
            for row in state.get("analysis_consumed_samples", [])
        }
        if identities.intersection(analyzed):
            raise ValueError(
                "Generalization panel must use unanalysed train/val samples"
            )
        prior_generalization = [
            item["record"]
            for item in state.get("panel_history", [])
            if item.get("problem_id") == problem_id
            and item.get("panel_role") == "generalization"
        ]
        if existing is not None:
            prior_generalization.append(existing)
        prior_ids = {
            (row["dataset_split"], int(row["episode_id"]))
            for prior in prior_generalization
            for row in prior["panel"]["samples"]
        }
        if identities.intersection(prior_ids):
            raise ValueError(
                "Fresh generalization panel must be disjoint from all prior "
                "generalization panels for this problem"
            )
    if existing is not None:
        state.setdefault("panel_history", []).append(
            {
                "problem_id": problem_id,
                "panel_role": role,
                "record": json.loads(json.dumps(existing)),
                "replaced_by_candidate_id": (
                    state.get("current_candidate") or {}
                ).get("candidate_id"),
            }
        )
    panels[role] = record
    v3.append_event(
        state,
        "v5_panel_registered",
        {
            "problem_id": problem_id,
            "panel_id": panel["panel_id"],
            "panel_role": role,
            "sample_count": len(panel["samples"]),
            "replaced_prior_panel": existing is not None,
        },
    )
    return record


def validate_lineage(
    lineage: dict[str, Any], candidate: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "candidate_id",
        "problem_id",
        "revision",
        "parent_candidate_id",
        "inherited_component_ids",
        "modified_component_ids",
        "preserved_invariants",
        "behavioral_delta",
        "revision_reason",
    }
    if not isinstance(lineage, dict) or set(lineage) != required:
        raise ValueError(f"Candidate lineage fields mismatch: {sorted(required)}")
    if lineage["schema_version"] != LINEAGE_SCHEMA:
        raise ValueError("Unsupported v5 lineage schema")
    for field in ("candidate_id", "problem_id", "behavioral_delta", "revision_reason"):
        if not isinstance(lineage[field], str) or not lineage[field].strip():
            raise ValueError(f"Lineage {field} must be non-empty")
    if lineage["candidate_id"] != candidate["candidate_id"]:
        raise ValueError("Lineage candidate_id mismatch")
    if lineage["problem_id"] != candidate["problem_id"]:
        raise ValueError("Lineage problem_id mismatch")
    if int(lineage["revision"]) != int(candidate["revision"]):
        raise ValueError("Lineage revision mismatch")
    for field in (
        "inherited_component_ids",
        "modified_component_ids",
        "preserved_invariants",
    ):
        if not isinstance(lineage[field], list):
            raise ValueError(f"Lineage {field} must be a list")

    history = state.setdefault("problem_candidate_lineage", {}).setdefault(
        candidate["problem_id"], []
    )
    parent = lineage["parent_candidate_id"]
    if not history:
        if parent is not None:
            raise ValueError("The first candidate for a problem cannot have a parent")
    else:
        expected_parent = history[-1]["candidate_id"]
        if parent != expected_parent:
            raise ValueError(
                "A candidate revision must inherit from the most recent candidate "
                f"for the same problem ({expected_parent})"
            )
        if int(lineage["revision"]) <= int(history[-1]["revision"]):
            raise ValueError("Candidate revision must increase within a problem")
    return lineage


def candidate_behavior_hash(candidate: dict[str, Any]) -> str:
    projection = []
    for component in candidate["components"]:
        projection.append(
            {
                "type": component["type"],
                "priority": component["priority"],
                "state_namespace": component["state_namespace"],
                "claims": sorted(component["claims"]),
                "config": component["config"],
                "artifact_hashes": sorted(
                    descriptor["sha256"]
                    for descriptor in component["artifacts"].values()
                ),
            }
        )
    return canonical_hash(sorted(projection, key=lambda row: canonical_hash(row)))


def start_candidate(
    state: dict[str, Any], candidate_path: Path, lineage: dict[str, Any]
) -> dict[str, Any]:
    candidate = load_json(candidate_path)
    lineage = validate_lineage(lineage, candidate, state)
    problem_id = candidate["problem_id"]
    development = (
        (state.get("panel_registry") or {}).get(problem_id, {}).get("development")
    )
    if development is None:
        raise ValueError("A fixed development panel is required before a candidate")
    behavior_hash = candidate_behavior_hash(candidate)
    prior = state["problem_candidate_lineage"].get(problem_id, [])
    if prior and behavior_hash == prior[-1].get("behavior_sha256"):
        raise ValueError("Candidate revision is behaviorally identical to its parent")

    current = v4.start_candidate(state, candidate_path)
    current["lineage"] = lineage
    current["behavior_sha256"] = behavior_hash
    current["development_panel_id"] = development["panel"]["panel_id"]
    current["development_panel_sha256"] = development["panel_sha256"]
    current["status"] = "development_verifier_pending"
    current["verifier_reports"] = {}
    current["retrospectives"] = {}
    state["problem_candidate_lineage"][problem_id].append(
        {**json.loads(json.dumps(lineage)), "behavior_sha256": behavior_hash}
    )
    state["stage"] = "development_verification"
    v3.append_event(
        state,
        "v5_candidate_started",
        {
            "candidate_id": candidate["candidate_id"],
            "problem_id": problem_id,
            "parent_candidate_id": lineage["parent_candidate_id"],
            "development_panel_id": current["development_panel_id"],
        },
    )
    return current


def record_verifier(state: dict[str, Any], report: dict[str, Any]) -> None:
    current = state.get("current_candidate")
    if current is None:
        raise ValueError("No candidate is awaiting verification")
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("V5 requires a v5 paired-panel verifier report")
    for key in ("candidate_id", "problem_id"):
        if report.get(key) != current[key]:
            raise ValueError(f"Verifier {key} does not match current candidate")
    if report.get("task") != state["task"]:
        raise ValueError("Verifier task does not match state")
    comparison = report.get("bundle_comparison") or {}
    if comparison.get("base_bundle_sha256") != current["base_bundle_sha256"]:
        raise ValueError("Verifier did not run the recorded base bundle")
    if comparison.get("candidate_bundle_sha256") != current["candidate_bundle_sha256"]:
        raise ValueError("Verifier did not run the recorded candidate bundle")
    role = report.get("panel_role")
    expected_status = {
        "development": "development_verifier_pending",
        "generalization": "generalization_verifier_pending",
    }.get(role)
    if expected_status is None or current["status"] != expected_status:
        raise ValueError("Verifier panel role is invalid for candidate stage")
    registered = state["panel_registry"][current["problem_id"]][role]
    if (
        report.get("panel_id") != registered["panel"]["panel_id"]
        or report.get("panel_sha256") != registered["panel_sha256"]
    ):
        raise ValueError("Verifier did not use the fixed registered panel")
    expected_samples = {
        (str(row["dataset_split"]), int(row["episode_id"]))
        for row in registered["panel"]["samples"]
    }
    reported_samples = {
        (str(row["dataset_split"]), int(row["episode_id"]))
        for row in report.get("sample_provenance", [])
    }
    if reported_samples != expected_samples or int(report.get("panel_size", -1)) != len(
        expected_samples
    ):
        raise ValueError("Verifier provenance does not exactly cover the fixed panel")
    current["verifier_reports"][role] = report
    current["status"] = f"{role}_retrospective_pending"
    state["stage"] = current["status"]
    v3.append_event(
        state,
        "v5_verifier_recorded",
        {
            "candidate_id": current["candidate_id"],
            "panel_role": role,
            "panel_size": report["panel_size"],
            "acceptance_tier": report["acceptance_tier"],
        },
    )


def validate_retrospective(
    retrospective: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "candidate_id",
        "problem_id",
        "panel_id",
        "panel_role",
        "outcome_summary",
        "activation_audit",
        "hypothesis_assessment",
        "implementation_assessment",
        "failure_type",
        "knowledge_gained",
        "recommended_action",
        "next_revision",
        "switch_justification",
    }
    if not isinstance(retrospective, dict) or set(retrospective) != required:
        raise ValueError(f"Retrospective fields mismatch: {sorted(required)}")
    if retrospective["schema_version"] != RETROSPECTIVE_SCHEMA:
        raise ValueError("Unsupported v5 retrospective schema")
    current = state.get("current_candidate")
    if current is None:
        raise ValueError("No current candidate for retrospective")
    for field in ("candidate_id", "problem_id"):
        if retrospective[field] != current[field]:
            raise ValueError(f"Retrospective {field} mismatch")
    role = retrospective["panel_role"]
    if current["status"] != f"{role}_retrospective_pending":
        raise ValueError("Retrospective does not match current verifier stage")
    report = current["verifier_reports"].get(role)
    if report is None or retrospective["panel_id"] != report["panel_id"]:
        raise ValueError("Retrospective panel does not match verifier report")
    if retrospective["failure_type"] not in FAILURE_TYPES:
        raise ValueError("Unsupported retrospective failure_type")
    if retrospective["recommended_action"] not in RETROSPECTIVE_ACTIONS:
        raise ValueError("Unsupported retrospective recommended_action")
    if (
        not isinstance(retrospective["knowledge_gained"], list)
        or not retrospective["knowledge_gained"]
    ):
        raise ValueError("Retrospective must state what the experiment taught us")
    audit = retrospective["activation_audit"]
    if not isinstance(audit, dict) or audit.get("status") not in {
        "activated",
        "partially_activated",
        "not_activated",
        "not_applicable",
        "unknown",
    }:
        raise ValueError("Retrospective requires a structured activation audit")
    if not isinstance(audit.get("evidence"), list):
        raise ValueError("Activation audit evidence must be a list")
    if retrospective["recommended_action"] == "revise_same_problem":
        next_revision = retrospective["next_revision"]
        if (
            not isinstance(next_revision, dict)
            or not str(next_revision.get("behavioral_delta", "")).strip()
        ):
            raise ValueError("Same-problem revision requires a behavioral delta")
    if retrospective["recommended_action"] == "switch_problem":
        switch = retrospective["switch_justification"]
        if (
            not isinstance(switch, dict)
            or switch.get("reason_code") not in SWITCH_REASON_CODES
            or not isinstance(switch.get("evidence"), list)
            or not switch["evidence"]
        ):
            raise ValueError("Problem switching requires a supported reason code")
    return retrospective


def record_retrospective(state: dict[str, Any], retrospective: dict[str, Any]) -> None:
    retrospective = validate_retrospective(retrospective, state)
    current = state["current_candidate"]
    role = retrospective["panel_role"]
    current["retrospectives"][role] = retrospective
    current["status"] = f"{role}_decision_pending"
    state["retrospectives"].append(json.loads(json.dumps(retrospective)))
    state["stage"] = current["status"]
    v3.append_event(
        state,
        "v5_retrospective_recorded",
        {
            "candidate_id": current["candidate_id"],
            "panel_role": role,
            "recommended_action": retrospective["recommended_action"],
            "failure_type": retrospective["failure_type"],
        },
    )


def _finalize_budget_exhaustion(state: dict[str, Any]) -> None:
    """Record an explicit task-level outcome when the last round is closed.

    ``remaining_problem_budget`` is consumed when a candidate is created, so
    the final verifier decision cannot rely on a later ``advance`` call to
    finalize the task.  Keep retained fallbacks selectable, but never leave an
    exhausted task looking active.
    """
    if int(state.get("remaining_problem_budget", 0)) != 0:
        return
    if state.get("task_evolution_status") in {
        "early_exit_eligible",
        "budget_fallback_selected",
    }:
        return

    state["active_problem_id"] = None
    state["stage"] = "evolution_budget_exhausted"
    if state.get("fallback_candidates"):
        status = "fallback_selection_pending"
        reason = "round_budget_exhausted_fallback_available"
    else:
        status = "no_improvement"
        reason = "round_budget_exhausted_no_candidate"

    already_recorded = (
        state.get("task_evolution_status") == status
        and state.get("task_terminal_reason") == reason
    )
    state["task_evolution_status"] = status
    state["task_terminal_reason"] = reason
    if not already_recorded:
        v3.append_event(
            state,
            "v5_evolution_budget_exhausted",
            {
                "reason": reason,
                "fallback_candidate_count": len(
                    state.get("fallback_candidates", [])
                ),
            },
        )


def _close_round(
    state: dict[str, Any], decision: str, reason: str, problem_status: str
) -> dict[str, Any]:
    current = state["current_candidate"]
    problem_id = current["problem_id"]
    record = {key: value for key, value in current.items() if key != "candidate"}
    record.update({"decision": decision, "decision_reason": reason})
    state["problems"][problem_id]["status"] = problem_status
    if problem_status != "active":
        state["active_problem_id"] = None
    state["round_history"].append(record)
    state["current_candidate"] = None
    state["stage"] = (
        "evolution_budget_exhausted"
        if int(state["remaining_problem_budget"]) == 0
        else "candidate_decided"
    )
    _finalize_budget_exhaustion(state)
    return record


def _fallback_record(
    *,
    candidate: dict[str, Any],
    candidate_manifest_path: str,
    candidate_manifest_sha256: str,
    round_number: int,
    problem_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    paired = report["paired_verifier"]
    return {
        "candidate": json.loads(json.dumps(candidate)),
        "candidate_manifest_path": candidate_manifest_path,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "retained_round": round_number,
        "source_problem_id": problem_id,
        "verifier_acceptance_tier": report["acceptance_tier"],
        "net_improvements": int(paired["net_improvements"]),
        "regressions": int(paired["regressed"]),
        "candidate_successes": int(report.get("candidate_successes", 0)),
        "verifier_report": json.loads(json.dumps(report)),
    }


def _retain_current_fallback(
    state: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any] | None:
    """Keep a +1/zero-regression candidate without ending the evolution loop."""
    if not report.get("fallback_eligible"):
        return None
    current = state["current_candidate"]
    candidate_id = current["candidate_id"]
    existing = [
        row
        for row in state.get("fallback_candidates", [])
        if row["candidate"]["candidate_id"] == candidate_id
    ]
    if existing:
        return existing[0]
    retained = _fallback_record(
        candidate=current["candidate"],
        candidate_manifest_path=current["candidate_manifest_path"],
        candidate_manifest_sha256=current["candidate_manifest_sha256"],
        round_number=int(current["round"]),
        problem_id=current["problem_id"],
        report=report,
    )
    state.setdefault("fallback_candidates", []).append(retained)
    v3.append_event(
        state,
        "v5_fallback_candidate_retained",
        {
            "candidate_id": candidate_id,
            "round": int(current["round"]),
            "continued_evolution": True,
        },
    )
    return retained


def reconcile_historical_fallbacks(state: dict[str, Any]) -> list[str]:
    """Repair runs created before fallback retention was independent of decisions."""
    existing_ids = {
        row["candidate"]["candidate_id"]
        for row in state.get("fallback_candidates", [])
    }
    added: list[str] = []
    for row in state.get("round_history", []):
        candidate_id = row.get("candidate_id")
        if not candidate_id or candidate_id in existing_ids:
            continue
        report = (row.get("verifier_reports") or {}).get("development")
        if not isinstance(report, dict) or not report.get("fallback_eligible"):
            continue
        manifest_path = Path(row["candidate_manifest_path"])
        candidate = load_json(manifest_path)
        if candidate.get("candidate_id") != candidate_id:
            raise ValueError("Historical fallback manifest candidate_id mismatch")
        retained = _fallback_record(
            candidate=candidate,
            candidate_manifest_path=str(manifest_path),
            candidate_manifest_sha256=row["candidate_manifest_sha256"],
            round_number=int(row["round"]),
            problem_id=row["problem_id"],
            report=report,
        )
        retained["reconciled_from_historical_round"] = True
        state.setdefault("fallback_candidates", []).append(retained)
        existing_ids.add(candidate_id)
        added.append(candidate_id)
    if added:
        v3.append_event(
            state,
            "v5_historical_fallbacks_reconciled",
            {"candidate_ids": added},
        )
        if (
            int(state.get("remaining_problem_budget", 0)) == 0
            and state.get("task_evolution_status") == "no_improvement"
        ):
            state["stage"] = "evolution_budget_exhausted"
            state["task_evolution_status"] = "fallback_selection_pending"
            state["task_terminal_reason"] = (
                "round_budget_exhausted_fallback_available"
            )
    return added


def advance_evolution(
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    exclude_problem_id: str | None = None,
) -> str | None:
    """Advance to another discovered problem without spending a round.

    A rejected candidate closes only its current problem attempt.  It must not
    implicitly terminate the task while other problem clusters and round
    budget remain.  The just-closed problem is excluded when alternatives
    exist, preventing an immediate accidental revisit of the same hypothesis.
    """
    if state.get("current_candidate") is not None:
        raise ValueError("Cannot advance while a candidate is active")
    if state.get("task_evolution_status") in {
        "early_exit_eligible",
        "budget_fallback_selected",
    }:
        return None
    if int(state.get("remaining_problem_budget", 0)) <= 0:
        _finalize_budget_exhaustion(state)
        return None

    try:
        problem_id = v3.next_problem_id(
            state, config, exclude_problem_id=exclude_problem_id
        )
    except ValueError:
        # No discovered problem can consume the remaining budget.  This is a
        # legitimate task-level terminal condition, unlike switch_problem.
        state["active_problem_id"] = None
        state["stage"] = "evolution_terminal"
        state["task_evolution_status"] = "no_improvement"
        state["task_terminal_reason"] = "no_selectable_problem_remaining"
        v3.append_event(state, "v5_evolution_terminal", {
            "reason": state["task_terminal_reason"],
            "remaining_problem_budget": int(state["remaining_problem_budget"]),
        })
        return None

    v3.select_problem(state, problem_id, config)
    state["task_evolution_status"] = "active"
    state["task_terminal_reason"] = None
    v3.append_event(
        state,
        "v5_evolution_advanced_after_decision",
        {
            "problem_id": problem_id,
            "excluded_problem_id": exclude_problem_id,
            "remaining_problem_budget": int(state["remaining_problem_budget"]),
        },
    )
    return problem_id


def decide_candidate(
    state: dict[str, Any],
    decision: str,
    reason: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = state.get("current_candidate")
    if current is None:
        raise ValueError("No candidate is awaiting a decision")
    status = current["status"]
    if not status.endswith("_decision_pending"):
        raise ValueError("A mandatory retrospective is required before decision")
    role = status.removesuffix("_decision_pending")
    report = current["verifier_reports"][role]
    retrospective = current["retrospectives"][role]
    problem_id = current["problem_id"]
    if retrospective["recommended_action"] != decision:
        raise ValueError("Decision must match the evidence-backed retrospective")
    if role == "development":
        _retain_current_fallback(state, report)

    if decision == "promote_to_generalization":
        if role != "development" or not report.get("development_ready"):
            raise ValueError("Only a development-ready candidate can generalize")
        panels = state["panel_registry"][current["problem_id"]]
        if "generalization" not in panels:
            raise ValueError("Register a fresh generalization panel first")
        current["generalization_panel_id"] = panels["generalization"]["panel"][
            "panel_id"
        ]
        current["generalization_panel_sha256"] = panels["generalization"][
            "panel_sha256"
        ]
        current["status"] = "generalization_verifier_pending"
        state["stage"] = "generalization_verification"
        v3.append_event(
            state,
            "v5_candidate_promoted_to_generalization",
            {"candidate_id": current["candidate_id"]},
        )
        return {"decision": decision, "candidate_id": current["candidate_id"]}

    if decision == "accept":
        require_generalization = bool(
            state.get("v5_policy", {}).get("require_generalization_before_accept", True)
        )
        if require_generalization:
            if role != "generalization" or not report.get("generalization_pass"):
                raise ValueError("Acceptance requires a passing generalization panel")
        elif not (report.get("development_ready") or report.get("generalization_pass")):
            raise ValueError("Candidate is not acceptance eligible")
        approved = json.loads(json.dumps(current["candidate"]))
        approved["status"] = "approved"
        accepted = {
            "candidate": approved,
            "candidate_manifest_path": current["candidate_manifest_path"],
            "candidate_manifest_sha256": current["candidate_manifest_sha256"],
            "accepted_round": current["round"],
            "source_problem_id": current["problem_id"],
            "development_report": current["verifier_reports"].get("development"),
            "generalization_report": current["verifier_reports"].get("generalization"),
        }
        state["accepted_experiences"].append(accepted)
        state["working_bundle"].append(accepted)
        state["bundle_revision"] = int(state["bundle_revision"]) + 1
        state["task_evolution_status"] = "early_exit_eligible"
        state["task_terminal_reason"] = "paired_development_and_generalization_pass"
        record = _close_round(state, decision, reason, "resolved")
        record["result_bundle_revision"] = state["bundle_revision"]
        record["result_bundle_sha256"] = v4.candidate_bundle_hash(
            state["working_bundle"]
        )
        state["stage"] = "evolution_terminal"
        return record

    if decision == "retain_fallback":
        if role != "development" or not report.get("fallback_eligible"):
            raise ValueError("Only a development fallback can be retained")
        record = _close_round(state, decision, reason, "deferred")
        if config is not None and int(state["remaining_problem_budget"]) > 0:
            record["next_problem_id"] = advance_evolution(
                state, config, exclude_problem_id=problem_id
            )
        return record

    if decision == "revise_same_problem":
        attempts = int(state["problem_attempts"][current["problem_id"]])
        status_after = (
            "exhausted"
            if attempts >= int(state["max_attempts_per_problem"])
            else "active"
        )
        return _close_round(state, decision, reason, status_after)

    if decision == "switch_problem":
        attempts = int(state["problem_attempts"][current["problem_id"]])
        status_after = (
            "exhausted"
            if attempts >= int(state["max_attempts_per_problem"])
            else "deferred"
        )
        record = _close_round(state, decision, reason, status_after)
        if config is not None and int(state["remaining_problem_budget"]) > 0:
            record["next_problem_id"] = advance_evolution(
                state, config, exclude_problem_id=problem_id
            )
        return record
    if decision == "unactionable":
        record = _close_round(state, decision, reason, "unactionable")
        if config is not None and int(state["remaining_problem_budget"]) > 0:
            record["next_problem_id"] = advance_evolution(
                state, config, exclude_problem_id=problem_id
            )
        return record
    if decision == "hypothesis_rejected":
        record = _close_round(state, decision, reason, "hypothesis_rejected")
        if config is not None and int(state["remaining_problem_budget"]) > 0:
            record["next_problem_id"] = advance_evolution(
                state, config, exclude_problem_id=problem_id
            )
        return record
    raise ValueError(f"Unsupported v5 decision {decision!r}")


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
    state["v5_policy"] = dict(config.get("v5_policy") or {})
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.v5.snapshot.json", config)
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
        "schema_version": "self-evolve-v5-evolution50-candidate-pool-v1",
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
        "v5_evolution50_candidate_pool_recorded",
        {"dataset_split": split, "selected_count": len(candidates)},
    )
    save_state(run_root, state)
    print(ids_path.read_text(encoding="utf-8").strip())


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


def record_hard_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    if args.command == "record-hard-trigger":
        path = args.trigger_report.resolve()
        v3.record_hard_trigger(state, load_json(path), path)
        result: Any = state["hard_exploration"]["trigger"]["decision"]
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


def discover_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    records = v3.read_jsonl(args.episode_records.resolve())
    discovery = build_deep_problem_discovery(records, config, state["task"])
    output_dir = run_root / "problem_discovery"
    v3.write_jsonl(output_dir / "episode_problem_records.v5.jsonl", records)
    write_json(output_dir / "problem_clusters.v5.json", discovery)
    v3.write_text_atomic(
        output_dir / "problem_report.v5.md", v3.render_problem_report(discovery)
    )
    v3.install_problem_discovery(state, discovery)
    save_state(run_root, state)
    print(output_dir / "problem_clusters.v5.json")


def select_problem_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    problem_id = args.problem_id or v3.next_problem_id(state, config)
    v3.select_problem(state, problem_id, config)
    save_state(run_root, state)
    print(problem_id)


def register_panel_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    path = args.panel.resolve()
    record = register_panel(state, load_json(path), config, path)
    output = run_root / "panels" / f"{record['panel']['panel_id']}.json"
    write_json(output, record["panel"])
    save_state(run_root, state)
    print(output)


def start_candidate_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    current = start_candidate(
        state, args.candidate.resolve(), load_json(args.lineage.resolve())
    )
    round_dir = run_root / f"round_{current['round']:02d}"
    write_json(
        round_dir / "selected_problem.json", state["problems"][current["problem_id"]]
    )
    write_json(round_dir / "candidate_intervention.json", current["candidate"])
    write_json(round_dir / "candidate_lineage.json", current["lineage"])
    write_json(
        round_dir / "development_panel.json",
        state["panel_registry"][current["problem_id"]]["development"]["panel"],
    )
    save_state(run_root, state)
    print(current["round"])


def record_verifier_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    report = load_json(args.verifier_report.resolve())
    record_verifier(state, report)
    round_dir = run_root / f"round_{state['current_candidate']['round']:02d}"
    write_json(round_dir / f"{report['panel_role']}_verifier_report.json", report)
    save_state(run_root, state)
    print(report["acceptance_tier"])


def record_retrospective_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    retrospective = load_json(args.retrospective.resolve())
    record_retrospective(state, retrospective)
    round_dir = run_root / f"round_{state['current_candidate']['round']:02d}"
    write_json(
        round_dir / f"{retrospective['panel_role']}_retrospective.json",
        retrospective,
    )
    save_state(run_root, state)
    print(retrospective["recommended_action"])


def decide_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    round_number = state["current_candidate"]["round"]
    record = decide_candidate(state, args.decision, args.reason, config)
    round_dir = run_root / f"round_{round_number:02d}"
    write_json(round_dir / f"decision_{args.decision}.json", record)
    save_state(run_root, state)
    print(args.decision)


def advance_command(args: argparse.Namespace) -> None:
    """Resume a run after a previously recorded problem-switch decision."""
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    config = load_json(Path(state["config_path"]))
    problem_id = advance_evolution(
        state, config, exclude_problem_id=args.exclude_problem_id
    )
    save_state(run_root, state)
    print(problem_id or state.get("task_terminal_reason") or state.get("stage"))


def select_final_fallback_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    selected = v4.select_final_fallback(state, args.candidate_id)
    write_json(
        run_root
        / "bundle_history"
        / f"bundle_revision_{state['bundle_revision']:03d}.json",
        {
            "bundle_revision": state["bundle_revision"],
            "behavior_bundle_sha256": v4.candidate_bundle_hash(state["working_bundle"]),
            "candidates": state["working_bundle"],
            "selection_policy": "best_fallback_after_round_budget",
        },
    )
    save_state(run_root, state)
    print(selected["candidate"]["candidate_id"])


def reconcile_fallback_pool_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    state = load_json(state_path(run_root))
    added = reconcile_historical_fallbacks(state)
    report = {
        "schema_version": "self-evolve-v5-fallback-reconciliation-v1",
        "added_candidate_ids": added,
        "fallback_candidate_ids": [
            row["candidate"]["candidate_id"]
            for row in state.get("fallback_candidates", [])
        ],
        "task_evolution_status": state.get("task_evolution_status"),
        "task_terminal_reason": state.get("task_terminal_reason"),
    }
    write_json(run_root / "fallback_reconciliation.json", report)
    save_state(run_root, state)
    print(json.dumps(report, sort_keys=True))


def snapshot_command(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Repository must be clean before recording a test candidate")
    state = load_json(state_path(run_root))
    if state.get("task_evolution_status") not in {
        "early_exit_eligible",
        "budget_fallback_selected",
    }:
        raise ValueError("V5 test snapshot requires an accepted final bundle")
    output_dir = run_root / "test_candidate_snapshot"
    bundle_path = compose_candidates(
        [Path(row["candidate_manifest_path"]) for row in state["working_bundle"]],
        output_dir,
        bundle_id=f"{state['task']}-v5-bundle-r{state['bundle_revision']}",
        revision=int(state["bundle_revision"]),
    )
    runtime = load_json(bundle_path)
    manifest = {
        "schema_version": "self-evolve-v5-test-candidate-snapshot-v1",
        "snapshot_id": runtime["bundle_id"],
        "task": state["task"],
        "source_commit": git_commit(repo_root),
        "evolution_start_commit": state["source_commit"],
        "experiment_config_sha256": state["config_sha256"],
        "runtime_bundle_sha256": runtime["bundle_sha256"],
        "behavior_bundle_sha256": v4.candidate_bundle_hash(state["working_bundle"]),
        "accepted_candidate_ids": runtime["candidate_ids"],
        "executor_checkpoint": args.executor_checkpoint,
        "total_evolution_rounds": state["total_evolution_rounds"],
        "development_panel_reused_across_revisions": True,
        "generalization_panel_required": bool(
            state.get("v5_policy", {}).get("require_generalization_before_accept", True)
        ),
        "test_policy": "one_shot_feedback_forbidden",
        "test_feedback_used": False,
        "final_train_val50_required": False,
    }
    write_json(output_dir / "reproducibility_manifest.json", manifest)
    state["test_candidate_snapshot"] = manifest
    state["stage"] = "test_candidate_recorded"
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
    discover = sub.add_parser("discover-problems")
    discover.add_argument("--run-root", type=Path, required=True)
    discover.add_argument("--episode-records", type=Path, required=True)
    discover.set_defaults(func=discover_command)
    select = sub.add_parser("select-problem")
    select.add_argument("--run-root", type=Path, required=True)
    select.add_argument("--problem-id")
    select.set_defaults(func=select_problem_command)
    panel = sub.add_parser("register-panel")
    panel.add_argument("--run-root", type=Path, required=True)
    panel.add_argument("--panel", type=Path, required=True)
    panel.set_defaults(func=register_panel_command)
    candidate = sub.add_parser("start-candidate")
    candidate.add_argument("--run-root", type=Path, required=True)
    candidate.add_argument("--candidate", type=Path, required=True)
    candidate.add_argument("--lineage", type=Path, required=True)
    candidate.set_defaults(func=start_candidate_command)
    verifier = sub.add_parser("record-verifier")
    verifier.add_argument("--run-root", type=Path, required=True)
    verifier.add_argument("--verifier-report", type=Path, required=True)
    verifier.set_defaults(func=record_verifier_command)
    retrospective = sub.add_parser("record-retrospective")
    retrospective.add_argument("--run-root", type=Path, required=True)
    retrospective.add_argument("--retrospective", type=Path, required=True)
    retrospective.set_defaults(func=record_retrospective_command)
    decision = sub.add_parser("decide-candidate")
    decision.add_argument("--run-root", type=Path, required=True)
    decision.add_argument(
        "--decision", choices=sorted(RETROSPECTIVE_ACTIONS), required=True
    )
    decision.add_argument("--reason", required=True)
    decision.set_defaults(func=decide_command)
    advance = sub.add_parser("advance-evolution")
    advance.add_argument("--run-root", type=Path, required=True)
    advance.add_argument("--exclude-problem-id")
    advance.set_defaults(func=advance_command)
    fallback = sub.add_parser("select-final-fallback")
    fallback.add_argument("--run-root", type=Path, required=True)
    fallback.add_argument("--candidate-id", required=True)
    fallback.set_defaults(func=select_final_fallback_command)
    reconcile_fallback = sub.add_parser("reconcile-fallback-pool")
    reconcile_fallback.add_argument("--run-root", type=Path, required=True)
    reconcile_fallback.set_defaults(func=reconcile_fallback_pool_command)
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
