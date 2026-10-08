#!/usr/bin/env python3
"""Prepare initial or revision repair-authoring input for self-evolve v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from planner import write_json
    from prepare_self_evolve_v4_repair_request import route_components
except ModuleNotFoundError:
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.prepare_self_evolve_v4_repair_request import (
        route_components,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def build_request(
    state: dict[str, Any],
    problem_id: str,
    *,
    previous_candidate: dict[str, Any] | None = None,
    verifier_report: dict[str, Any] | None = None,
    retrospective: dict[str, Any] | None = None,
) -> dict[str, Any]:
    problem = (state.get("problems") or {}).get(problem_id)
    if problem is None:
        raise ValueError(f"Unknown problem {problem_id!r}")
    supplied = [previous_candidate, verifier_report, retrospective]
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError(
            "A revision request requires previous candidate, verifier, and retrospective"
        )
    mode = "revision" if previous_candidate is not None else "initial"
    revision_context = None
    if mode == "revision":
        assert previous_candidate is not None
        assert verifier_report is not None
        assert retrospective is not None
        if previous_candidate.get("problem_id") != problem_id:
            raise ValueError("Previous candidate belongs to another problem")
        if verifier_report.get("candidate_id") != previous_candidate.get(
            "candidate_id"
        ):
            raise ValueError("Verifier does not match previous candidate")
        if retrospective.get("candidate_id") != previous_candidate.get("candidate_id"):
            raise ValueError("Retrospective does not match previous candidate")
        if retrospective.get("recommended_action") != "revise_same_problem":
            raise ValueError("Revision authoring requires revise_same_problem evidence")
        revision_context = {
            "parent_candidate": previous_candidate,
            "paired_verifier": verifier_report,
            "retrospective": retrospective,
            "required_parent_candidate_id": previous_candidate["candidate_id"],
            "required_next_revision_minimum": int(previous_candidate["revision"]) + 1,
            "required_behavioral_delta": retrospective["next_revision"][
                "behavioral_delta"
            ],
            "preserve_invariants": retrospective["next_revision"].get(
                "preserve_invariants", []
            ),
        }
    development_panel = (
        (state.get("panel_registry") or {}).get(problem_id, {}).get("development")
    )
    return {
        "schema_version": "self-evolve-v5-repair-authoring-request-v1",
        "skill_name": "self-evolve-intervention-author",
        "mode": mode,
        "task": state["task"],
        "problem_id": problem_id,
        "problem_cluster": problem,
        "causal_analysis": {
            "required_layers": [
                "physical_first_divergence",
                "task_progress_divergence",
                "terminal_failure",
            ],
            "examples": problem.get("causal_chain_examples", []),
        },
        "development_panel": development_panel,
        "revision_context": revision_context,
        "suggested_component_types": route_components(problem, state["task"]),
        "authoring_policy": {
            "candidate_format": "v4_typed_intervention_bundle",
            "lineage_sidecar_required": True,
            "choose_weakest_effective_fullstack_repair": True,
            "runtime_episode_identity_forbidden": True,
            "offline_verifier_mutation_forbidden": True,
            "do_not_switch_problem_inside_authoring": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--previous-candidate", type=Path)
    parser.add_argument("--verifier-report", type=Path)
    parser.add_argument("--retrospective", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = build_request(
        load_json(args.state.resolve()),
        args.problem_id,
        previous_candidate=(
            load_json(args.previous_candidate.resolve())
            if args.previous_candidate
            else None
        ),
        verifier_report=(
            load_json(args.verifier_report.resolve()) if args.verifier_report else None
        ),
        retrospective=(
            load_json(args.retrospective.resolve()) if args.retrospective else None
        ),
    )
    write_json(args.output.resolve(), request)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
