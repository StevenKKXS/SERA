#!/usr/bin/env python3
"""Verify a typed v4 intervention bundle on paired, unconsumed train/val samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from interventions import validate_candidate_manifest
    from planner import write_json
    from verify_self_evolve_v2_experience import load_jsonl, summarize
    from verify_self_evolve_v3_experience import (
        combine_summaries,
        load_config,
        threshold_pass,
    )
except ModuleNotFoundError:  # Repository-root unittest imports.
    from scripts.codex_rollout.interventions import validate_candidate_manifest
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.verify_self_evolve_v2_experience import (
        load_jsonl,
        summarize,
    )
    from scripts.codex_rollout.verify_self_evolve_v3_experience import (
        combine_summaries,
        load_config,
        threshold_pass,
    )


def evaluate(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    candidate = validate_candidate_manifest(candidate, bundle_dir=candidate_dir)
    task = candidate["task_scope"][0]
    if any(row.get("task") != task for row in rows):
        raise ValueError("Validation task does not match candidate task scope")
    invalid_splits = sorted(
        {str(row.get("dataset_split")) for row in rows} - {"train", "val"}
    )
    if invalid_splits:
        raise ValueError(f"V4 verifier rejects non-train/val rows: {invalid_splits}")
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

    base_hashes = {str(row.get("base_bundle_sha256", "")) for row in rows}
    candidate_hashes = {str(row.get("candidate_bundle_sha256", "")) for row in rows}
    if len(base_hashes) != 1 or "" in base_hashes:
        raise ValueError("Verifier rows require one stable full base bundle hash")
    if len(candidate_hashes) != 1 or "" in candidate_hashes:
        raise ValueError("Verifier rows require one stable full candidate bundle hash")

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
    candidate_successes = int(pilot["improved"]) + int(
        pilot["unchanged_success"]
    )
    perfect_pass = (
        coverage_pass
        and within_budget
        and pilot["attempts"] >= minimum_total
        and candidate_successes == pilot["attempts"]
        and pilot["regressed"] == 0
    )
    fallback_eligible = (
        not (strict_pass or perfect_pass)
        and coverage_pass
        and within_budget
        and threshold_pass(pilot, ladder[1], minimum_total)
    )
    acceptance_tier = (
        "strict"
        if strict_pass
        else "perfect"
        if perfect_pass
        else "fallback"
        if fallback_eligible
        else "rejected"
    )
    extension_eligible = (
        acceptance_tier == "rejected"
        and coverage_pass
        and within_budget
        and pilot["attempts"] < maximum_total
        and pilot["regressed"] == 0
    )
    return {
        "schema_version": "self-evolve-v4-paired-verifier-report-v1",
        "candidate_id": candidate["candidate_id"],
        "candidate_revision": candidate["revision"],
        "problem_id": candidate["problem_id"],
        "task": task,
        "component_summary": [
            {
                "component_id": component["component_id"],
                "type": component["type"],
                "claims": component["claims"],
            }
            for component in candidate["components"]
        ],
        "bundle_comparison": {
            "base_bundle_sha256": next(iter(base_hashes)),
            "candidate_bundle_sha256": next(iter(candidate_hashes)),
        },
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
        "offline_acceptance_verifier_mutable_by_candidate": False,
        "train": train,
        "val": val,
        "pilot_verifier": pilot,
        "coverage_pass": coverage_pass,
        "within_budget": within_budget,
        "strict_pass": strict_pass,
        "perfect_pass": perfect_pass,
        "candidate_successes": candidate_successes,
        "fallback_eligible": fallback_eligible,
        "fallback_pool_eligible": fallback_eligible,
        "extension_eligible": extension_eligible,
        "acceptance_tier": acceptance_tier,
        "early_task_exit_eligible": strict_pass or perfect_pass,
        # Compatibility alias: this now means immediate release only.
        "test_release_eligible": strict_pass or perfect_pass,
        "thresholds": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validations", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    candidate_path = args.candidate.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    report = evaluate(
        candidate,
        load_jsonl(args.validations.resolve()),
        load_config(args.experiment_config.resolve()),
        candidate_dir=candidate_path.parent,
    )
    write_json(args.output_report.resolve(), report)
    print(report["acceptance_tier"])


if __name__ == "__main__":
    main()
