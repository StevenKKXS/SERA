#!/usr/bin/env python3
"""Evaluate bounded v3 paired pilots before a one-shot test release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from planner import write_json
    from self_evolve_v3 import RUNTIME_FIELDS
    from verify_self_evolve_v2_experience import (
        load_jsonl,
        summarize,
        validate_candidate,
    )
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.self_evolve_v3 import RUNTIME_FIELDS
    from scripts.codex_rollout.verify_self_evolve_v2_experience import (
        load_jsonl,
        summarize,
        validate_candidate,
    )


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    verification = value.get("verification") or {}
    ladder = verification.get("evolution_acceptance_ladder") or []
    if [row.get("tier") for row in ladder] != ["strict", "fallback"]:
        raise ValueError("v3 requires strict then fallback evolution thresholds")
    if not ladder[1].get("only_if_no_strict_candidate"):
        raise ValueError("fallback must be conditional on no strict candidate")
    return value


def threshold_pass(
    summary: dict[str, Any], threshold: dict[str, Any], attempts: int
) -> bool:
    return (
        summary["attempts"] >= attempts
        and summary["net_improvements"] >= int(threshold["min_net_improvements"])
        and summary["regressed"] <= int(threshold["max_regressions"])
    )


def combine_summaries(train: dict[str, Any], val: dict[str, Any]) -> dict[str, Any]:
    outcome_fields = ("improved", "regressed", "unchanged_success", "unchanged_failure")
    combined = {
        "dataset": "robomme_paired_pilot_verifier",
        "attempts": int(train["attempts"]) + int(val["attempts"]),
        **{field: int(train[field]) + int(val[field]) for field in outcome_fields},
        "target_failures_corrected": int(train["target_failures_corrected"])
        + int(val["target_failures_corrected"]),
        "records": list(train["records"]) + list(val["records"]),
    }
    combined["net_improvements"] = combined["improved"] - combined["regressed"]
    return combined


def evaluate(
    candidate: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    runtime_candidate = {
        key: candidate[key] for key in (*RUNTIME_FIELDS, "status") if key in candidate
    }
    validate_candidate(runtime_candidate)
    task = candidate["task_scope"][0]
    if any(row.get("task") != task for row in rows):
        raise ValueError("Validation task does not match candidate task scope")
    invalid_splits = sorted(
        {str(row.get("dataset_split")) for row in rows} - {"train", "val"}
    )
    if invalid_splits:
        raise ValueError(f"Verifier rejects non-train/val rows: {invalid_splits}")

    verification = config["verification"]
    if verification.get("require_verifier_episode_ids") and any(
        "episode_id" not in row for row in rows
    ):
        raise ValueError("Verifier rows require episode_id provenance")
    if verification.get("reject_analysis_consumed_samples") and any(
        row.get("consumed_for_analysis", False) for row in rows
    ):
        raise ValueError(
            "Verifier cannot reuse samples consumed for candidate analysis"
        )
    bundle_comparison = None
    if verification.get("require_incremental_bundle_provenance"):
        base_hashes = {str(row.get("base_bundle_sha256", "")) for row in rows}
        candidate_hashes = {str(row.get("candidate_bundle_sha256", "")) for row in rows}
        if len(base_hashes) != 1 or "" in base_hashes:
            raise ValueError("Verifier rows require one stable base bundle hash")
        if len(candidate_hashes) != 1 or "" in candidate_hashes:
            raise ValueError("Verifier rows require one stable candidate bundle hash")
        bundle_comparison = {
            "base_bundle_sha256": next(iter(base_hashes)),
            "candidate_bundle_sha256": next(iter(candidate_hashes)),
        }
    train = summarize(rows, "train")
    val = summarize(rows, "val")
    pilot = combine_summaries(train, val)
    minimum_total = int(verification["minimum_total_attempts"])
    maximum_total = int(verification["maximum_total_attempts"])
    coverage_pass = (
        train["attempts"] >= int(verification["required_train_attempts"])
        and val["attempts"] >= int(verification["required_val_attempts"])
        and pilot["attempts"] >= minimum_total
    )
    within_budget = pilot["attempts"] <= maximum_total
    ladder = verification["evolution_acceptance_ladder"]
    strict_pass = (
        coverage_pass
        and within_budget
        and threshold_pass(pilot, ladder[0], minimum_total)
    )
    fallback_eligible = (
        not strict_pass
        and coverage_pass
        and within_budget
        and threshold_pass(pilot, ladder[1], minimum_total)
    )
    acceptance_tier = (
        "strict" if strict_pass else "fallback" if fallback_eligible else "rejected"
    )
    extension_eligible = (
        acceptance_tier == "rejected"
        and coverage_pass
        and within_budget
        and pilot["attempts"] < maximum_total
        and pilot["regressed"] == 0
    )
    return {
        "schema_version": "self-evolve-v3-pilot-verifier-report-v3",
        "experience_id": candidate["experience_id"],
        "problem_id": candidate.get("problem_id"),
        "task": task,
        "bundle_comparison": bundle_comparison,
        "sample_provenance": [
            {
                "record_id": str(row["record_id"]),
                "dataset_split": row["dataset_split"],
                "episode_id": int(row["episode_id"]),
                "consumed_for_analysis": bool(row.get("consumed_for_analysis", False)),
            }
            for row in rows
        ],
        "static_identity_isolation": "passed",
        "train": train,
        "val": val,
        "pilot_verifier": pilot,
        "coverage_pass": coverage_pass,
        "within_budget": within_budget,
        "strict_pass": strict_pass,
        "fallback_eligible": fallback_eligible,
        "extension_eligible": extension_eligible,
        "acceptance_tier": acceptance_tier,
        "test_release_eligible": acceptance_tier
        in verification.get("test_eligible_tiers", []),
        "full_train_val_candidate_rollout_required": bool(
            verification.get("full_train_val_candidate_rollout_required", False)
        ),
        "thresholds": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validations", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = evaluate(
        candidate, load_jsonl(args.validations), load_config(args.experiment_config)
    )
    write_json(args.output_report.resolve(), report)
    print(report["acceptance_tier"])


if __name__ == "__main__":
    main()
