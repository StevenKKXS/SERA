#!/usr/bin/env python3
"""Convert adjudicated v5 train/val analyses into deep problem records."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from prepare_v3_problem_records import (
        FAILURE_STAGES,
        convert as convert_v3,
        runtime_observable,
    )
except ModuleNotFoundError:
    from scripts.codex_rollout.prepare_v3_problem_records import (
        FAILURE_STAGES,
        convert as convert_v3,
        runtime_observable,
    )


CAUSAL_FIELDS = {
    "physical_first_divergence",
    "task_progress_divergence",
    "terminal_failure",
    "causal_link_evidence",
}

MODEL_CONSENSUS_REPAIRABLE_FAILURES = {
    "task_understanding_error",
    "planning_error",
    "grounding_error",
    "completion_error",
    "recovery_failure",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def convert(path: Path) -> dict[str, Any]:
    analysis = load_json(path)
    record = convert_v3(path)
    record["candidate_support_allowed"] = bool(
        (analysis.get("adjudication") or {}).get("candidate_allowed", False)
    )
    if record["outcome"] == "success":
        return record
    final = analysis.get("final_analysis") or {}
    causal = final.get("causal_chain")
    hypothesis_source = "adjudicated_final"
    if not isinstance(causal, dict) or set(causal) != CAUSAL_FIELDS:
        # The legacy deterministic analyzer is deliberately coarse and often
        # returns unknown for failures that the v5 visual review can explain.
        # Preserve that review as an explicitly unverified hypothesis rather
        # than discarding the causal evidence or treating it as accepted truth.
        model = analysis.get("model_analysis") or {}
        model_causal = model.get("causal_chain")
        if not isinstance(model_causal, dict) or set(model_causal) != CAUSAL_FIELDS:
            selected = str((analysis.get("adjudication") or {}).get("selected_source"))
            raise ValueError(f"V5 requires a deep causal chain ({selected}): {path}")
        causal = model_causal
        hypothesis_source = "disputed_model_hypothesis"
        failure_type = str(model.get("failure_type"))
        if failure_type not in FAILURE_STAGES:
            raise ValueError(f"Unknown v5 model failure type {failure_type!r}: {path}")
        # Keep v1's adjudicated categorical diagnosis as the clustering and
        # promotion boundary.  A disputed model review remains useful offline
        # evidence, but must not replace the adjudicated failure type/module or
        # become candidate support.
        record["candidate_experience"] = None
        record["candidate_support_allowed"] = False
        record["requires_cross_episode_confirmation"] = True
    if any(
        not isinstance(causal[field], str) or not causal[field].strip()
        for field in CAUSAL_FIELDS - {"causal_link_evidence"}
    ):
        raise ValueError(f"V5 causal chain contains an empty layer: {path}")
    if (
        not isinstance(causal["causal_link_evidence"], list)
        or not causal["causal_link_evidence"]
    ):
        raise ValueError(f"V5 causal chain requires chronological evidence: {path}")
    record["causal_chain"] = causal
    record["analysis_depth"] = "physical_to_task_progress_to_terminal"
    record["causal_hypothesis_source"] = hypothesis_source
    return record


def promote_cross_episode_model_consensus(
    rows: list[dict[str, Any]], *, minimum_support: int = 2
) -> list[dict[str, Any]]:
    """Promote repeated disputed v5 diagnoses only after cross-split confirmation.

    The per-episode deterministic adjudicator remains the first boundary.  This
    second boundary implements the cross-episode confirmation requested by the
    bounded-hypothesis record: it groups only by v1's controlled categorical
    fields and requires eligible model reviews in both train and val.  Physical
    executor lapses and observation/capability diagnoses never become repair
    support through this path.
    """

    hypotheses: dict[
        tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for record in rows:
        if not record.get("requires_cross_episode_confirmation"):
            continue
        analysis = load_json(Path(record["analysis_path"]))
        model = analysis.get("model_analysis") or {}
        failure_type = str(model.get("failure_type"))
        responsible_module = str(model.get("responsible_module"))
        if failure_type not in MODEL_CONSENSUS_REPAIRABLE_FAILURES:
            continue
        if model.get("experience_eligibility") != "eligible_candidate":
            continue
        candidate = model.get("candidate_experience")
        if not isinstance(candidate, dict) or candidate.get("scope") == "none":
            continue
        key = (str(record["task_variant"]), failure_type, responsible_module)
        hypotheses[key].append((record, model))

    promoted_ids: set[tuple[str, int]] = set()
    for members in hypotheses.values():
        splits = {str(record["dataset_split"]) for record, _ in members}
        if len(members) < minimum_support or not {"train", "val"}.issubset(splits):
            continue
        for record, model in members:
            failure_type = str(model["failure_type"])
            record["failure_stage"] = FAILURE_STAGES[failure_type]
            record["primary_problem"] = failure_type
            record["first_divergence"] = str(model["analysis_summary"])
            record["responsible_module"] = str(model["responsible_module"])
            record["runtime_observable"] = runtime_observable(model)
            record["planner_actionability"] = "high"
            record["confidence"] = float(model["confidence"])
            record["estimated_episode_gain"] = 1.0
            record["regression_risk"] = 0.2
            record["adjudication_status"] = "cross_episode_consensus"
            record["candidate_experience"] = model["candidate_experience"]
            record["candidate_support_allowed"] = True
            record["requires_cross_episode_confirmation"] = False
            record["causal_hypothesis_source"] = "cross_episode_model_consensus"
            promoted_ids.add((str(record["dataset_split"]), int(record["episode_id"])))

    return rows


def prepare(analysis_dirs: list[Path], output: Path) -> list[dict[str, Any]]:
    paths = sorted(
        {
            path.resolve()
            for root in analysis_dirs
            for path in root.glob("episode_*/analysis.json")
        },
        key=lambda path: (
            str(load_json(path).get("dataset_split")),
            int(load_json(path).get("episode", -1)),
        ),
    )
    if not paths:
        raise FileNotFoundError("No episode analysis files were found")
    rows = promote_cross_episode_model_consensus([convert(path) for path in paths])
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
