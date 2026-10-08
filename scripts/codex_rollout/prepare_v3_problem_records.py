#!/usr/bin/env python3
"""Convert adjudicated train/val failure analyses into v3 problem records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


FAILURE_STAGES = {
    "task_understanding_error": "task_method_selection",
    "planning_error": "action_planning",
    "grounding_error": "target_grounding",
    "completion_error": "completion_judgment",
    "executor_lapse": "action_execution",
    "recovery_failure": "failure_recovery",
    "observation_insufficient": "observation",
    "capability_limit": "capability",
    "unknown": "ambiguous",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_variant(analysis: dict[str, Any]) -> str:
    derived = (analysis.get("heuristic_analysis") or {}).get("derived_context") or {}
    return str(
        derived.get("selected_method")
        or derived.get("guidance_method")
        or analysis["task"]
    )


def runtime_observable(final: dict[str, Any]) -> bool:
    return any(
        row.get("source") in {"planner_trace", "rollout_video", "episode_summary"}
        for row in final.get("causal_evidence", [])
    )


def actionability(analysis: dict[str, Any], final: dict[str, Any]) -> str:
    eligibility = str(final.get("experience_eligibility"))
    candidate_allowed = bool((analysis.get("adjudication") or {}).get("candidate_allowed"))
    if eligibility == "eligible_candidate" and candidate_allowed:
        return "high"
    if eligibility == "needs_more_evidence":
        return "partial"
    return "none"


def convert(path: Path) -> dict[str, Any]:
    analysis = load_json(path)
    split = str(analysis.get("dataset_split"))
    if split not in {"train", "val"}:
        raise ValueError(f"Problem records reject split {split!r}: {path}")
    final = analysis.get("final_analysis")
    if not isinstance(final, dict):
        raise ValueError(f"Analysis has no final_analysis: {path}")
    source_result = str(analysis.get("source_result") or "failure")
    outcome = "success" if source_result == "success" else "failure"
    failure_type = str(final.get("failure_type"))
    if outcome == "failure" and failure_type not in FAILURE_STAGES:
        raise ValueError(f"Unknown failure type {failure_type!r}: {path}")
    alternatives = [str(value) for value in final.get("alternative_causes", [])]
    secondary = list(dict.fromkeys([f"terminal_result:{source_result}", *alternatives]))
    observable = runtime_observable(final)
    planner_actionability = (
        actionability(analysis, final) if outcome == "failure" else "none"
    )
    adjudication_status = str((analysis.get("adjudication") or {}).get("status"))
    record = {
        "dataset_split": split,
        "episode_id": int(analysis["episode"]),
        "outcome": outcome,
        "task_variant": task_variant(analysis),
        "failure_stage": (
            FAILURE_STAGES[failure_type]
            if outcome == "failure"
            else "successful_execution"
        ),
        "primary_problem": (
            failure_type if outcome == "failure" else "successful_execution"
        ),
        "first_divergence": str(final["analysis_summary"]),
        "secondary_symptoms": secondary,
        "responsible_module": (
            str(final["responsible_module"])
            if outcome == "failure"
            else "none"
        ),
        "runtime_observable": observable,
        "planner_actionability": planner_actionability,
        "confidence": float(final["confidence"]),
        "estimated_episode_gain": 1.0 if planner_actionability == "high" else 0.0,
        "regression_risk": 0.25 if adjudication_status == "disputed" else 0.1,
        "experiment_cost": 0.1,
        "analysis_path": str(path.resolve()),
        "analysis_sha256": sha256_file(path),
        "adjudication_status": adjudication_status,
        "candidate_experience": final.get("candidate_experience"),
        "offline_only": True,
    }
    if outcome == "success":
        # A successful trajectory is a counterexample only for failures from
        # the same task variant.  The discovery stage enforces that variant
        # match before interpreting this wildcard.
        record["counterexample_for"] = ["*"]
    return record


def prepare(analysis_dirs: list[Path], output: Path) -> list[dict[str, Any]]:
    paths = sorted(
        {path.resolve() for root in analysis_dirs for path in root.glob("episode_*/analysis.json")},
        key=lambda path: (
            str(load_json(path).get("dataset_split")),
            int(load_json(path).get("episode", -1)),
        ),
    )
    if not paths:
        raise FileNotFoundError("No episode analysis files were found")
    rows = [convert(path) for path in paths]
    identities = [(row["dataset_split"], row["episode_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate split-qualified episode analyses")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(
        [path.resolve() for path in args.analysis_dir], args.output.resolve()
    )
    print(json.dumps({"record_count": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
