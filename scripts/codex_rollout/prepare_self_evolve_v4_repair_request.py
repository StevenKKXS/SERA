#!/usr/bin/env python3
"""Route one discovered train/val problem to the v4 repair-authoring skill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from planner import write_json
except ModuleNotFoundError:
    from scripts.codex_rollout.planner import write_json


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def route_components(
    cluster: dict[str, Any], task_name: str | None = None
) -> list[str]:
    signature = cluster.get("signature") or {}
    text = " ".join(
        [
            str(signature.get("task_variant", "")),
            str(signature.get("failure_stage", "")),
            str(signature.get("primary_problem", "")),
            str(signature.get("responsible_module", "")),
            *[str(value) for value in cluster.get("first_divergence_examples", [])],
            *[str(value) for value in cluster.get("secondary_symptoms", [])],
        ]
    ).lower()
    task = str(task_name or signature.get("task_variant", "")).replace(
        " ", ""
    ).lower()
    suggestions: list[str] = []
    if task == "buttonunmaskswap" and any(
        token in text for token in ("recover", "ground", "contact", "button")
    ):
        suggestions.append("button_swap_press_grounding_recovery_gate")
    if any(token in text for token in ("count", "ordinal", "repetition", "quota")):
        suggestions.extend(
            ["counting_event_ledger", "retrieval_overlay", "action_canonicalizer"]
        )
    if "swing" in task and any(
        token in text for token in ("side", "swap", "target", "ground")
    ):
        suggestions.extend(["criterion_overlay", "swing_visual_binding_gate"])
    if any(
        token in text
        for token in (
            "completion",
            "advance",
            "transition",
            "held",
            "pickup",
            "release",
        )
    ):
        suggestions.extend(["criterion_overlay", "bounded_transition_gate"])
    if any(token in text for token in ("retriev", "skill missing", "top-k", "top k")):
        suggestions.append("retrieval_overlay")
    if any(token in text for token in ("canonical", "parameter", "wrong color")):
        suggestions.append("action_canonicalizer")
    if not suggestions:
        suggestions.append("prompt_lesson")
    return list(dict.fromkeys(suggestions))


def build_request(
    discovery: dict[str, Any], records: list[dict[str, Any]], problem_id: str
) -> dict[str, Any]:
    matching = [
        cluster
        for cluster in discovery.get("clusters", [])
        if cluster.get("problem_id") == problem_id
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one problem cluster {problem_id!r}")
    cluster = matching[0]
    refs = {
        (str(row["dataset_split"]), int(row["episode_id"]))
        for row in cluster.get("episode_refs", [])
    }
    evidence = [
        row
        for row in records
        if (str(row.get("dataset_split")), int(row.get("episode_id", -1))) in refs
    ]
    if len(evidence) != len(refs):
        raise ValueError("Problem evidence records are incomplete")
    if any(row.get("dataset_split") not in {"train", "val"} for row in evidence):
        raise ValueError("V4 repair authoring accepts only train/val evidence")
    return {
        "schema_version": "self-evolve-v4-repair-authoring-request-v1",
        "skill_name": "self-evolve-intervention-author",
        "task": discovery["task"],
        "problem_id": problem_id,
        "problem_cluster": cluster,
        "analysis_inputs": [
            {
                "dataset_split": row["dataset_split"],
                "episode_id": int(row["episode_id"]),
            }
            for row in evidence
        ],
        "first_divergence_evidence": [
            {
                "dataset_split": row["dataset_split"],
                "episode_id": int(row["episode_id"]),
                "first_divergence": row.get("first_divergence"),
                "responsible_module": row.get("responsible_module"),
                "runtime_observable": row.get("runtime_observable"),
            }
            for row in evidence
        ],
        "suggested_component_types": route_components(
            cluster, str(discovery.get("task", ""))
        ),
        "selection_policy": (
            "The authoring skill must independently choose the weakest component "
            "or component combination that can change the observed failure."
        ),
        "runtime_identity_policy": "task_scoped_episode_identity_forbidden",
        "offline_acceptance_verifier_mutable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-clusters", type=Path, required=True)
    parser.add_argument("--episode-records", type=Path, required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = build_request(
        load_json(args.problem_clusters.resolve()),
        load_jsonl(args.episode_records.resolve()),
        args.problem_id,
    )
    write_json(args.output.resolve(), request)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
