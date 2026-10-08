#!/usr/bin/env python3
"""Verify a v4 intervention candidate on a named, reusable v5 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from interventions import canonical_hash, validate_candidate_manifest
    from planner import write_json
    from verify_self_evolve_v2_experience import load_jsonl, summarize
    from verify_self_evolve_v3_experience import combine_summaries, load_config
except ModuleNotFoundError:  # Repository-root unittest imports.
    from scripts.codex_rollout.interventions import (
        canonical_hash,
        validate_candidate_manifest,
    )
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.verify_self_evolve_v2_experience import (
        load_jsonl,
        summarize,
    )
    from scripts.codex_rollout.verify_self_evolve_v3_experience import (
        combine_summaries,
        load_config,
    )


REPORT_SCHEMA = "self-evolve-v5-paired-panel-verifier-report-v1"
PANEL_SCHEMA = "self-evolve-v5-paired-panel-v1"


def sample_identity(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["dataset_split"]), int(row["episode_id"])


def validate_panel(panel: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "panel_id",
        "panel_role",
        "task",
        "problem_id",
        "samples",
    }
    if not isinstance(panel, dict) or set(panel) != required:
        raise ValueError(f"Panel fields mismatch: expected {sorted(required)}")
    if panel["schema_version"] != PANEL_SCHEMA:
        raise ValueError("Unsupported v5 panel schema")
    role = panel["panel_role"]
    if role not in {"development", "generalization"}:
        raise ValueError("Panel role must be development or generalization")
    if not isinstance(panel["panel_id"], str) or not panel["panel_id"]:
        raise ValueError("Panel requires a non-empty panel_id")
    if not isinstance(panel["problem_id"], str) or not panel["problem_id"]:
        raise ValueError("Panel requires a non-empty problem_id")
    if not isinstance(panel["task"], str) or not panel["task"]:
        raise ValueError("Panel requires a non-empty task")
    samples = panel["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("Panel requires samples")
    identities: list[tuple[str, int]] = []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {
            "dataset_split",
            "episode_id",
            "baseline_success",
        }:
            raise ValueError(
                "Panel samples require dataset_split, episode_id, baseline_success"
            )
        if sample["dataset_split"] not in {"train", "val"}:
            raise ValueError("V5 panels accept only train/val samples")
        if not isinstance(sample["episode_id"], int):
            raise ValueError("Panel episode_id must be an integer")
        if not isinstance(sample["baseline_success"], bool):
            raise ValueError("Panel baseline_success must be boolean")
        identities.append(sample_identity(sample))
    if len(identities) != len(set(identities)):
        raise ValueError("Panel contains duplicate samples")

    settings = config["verification"][f"{role}_panel"]
    count = len(samples)
    if (
        not int(settings["minimum_total_attempts"])
        <= count
        <= int(settings["maximum_total_attempts"])
    ):
        raise ValueError(f"{role} panel size is outside configured bounds")
    present_splits = {split for split, _ in identities}
    required_splits = set(settings.get("required_splits", []))
    if not required_splits.issubset(present_splits):
        raise ValueError(f"{role} panel does not cover required splits")
    return panel


def panel_hash(panel: dict[str, Any]) -> str:
    return canonical_hash(panel)


def evaluate(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    panel: dict[str, Any],
    config: dict[str, Any],
    *,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    candidate = validate_candidate_manifest(candidate, bundle_dir=candidate_dir)
    panel = validate_panel(panel, config)
    task = candidate["task_scope"][0]
    if panel["task"] != task or candidate["problem_id"] != panel["problem_id"]:
        raise ValueError("Candidate scope does not match verifier panel")
    if any(row.get("task") != task for row in rows):
        raise ValueError("Validation task does not match candidate task scope")
    invalid_splits = sorted(
        {str(row.get("dataset_split")) for row in rows} - {"train", "val"}
    )
    if invalid_splits:
        raise ValueError(f"V5 verifier rejects non-train/val rows: {invalid_splits}")
    if any("episode_id" not in row for row in rows):
        raise ValueError("Verifier rows require episode_id provenance")

    expected = {sample_identity(row): row for row in panel["samples"]}
    actual = {sample_identity(row): row for row in rows}
    if len(actual) != len(rows):
        raise ValueError("Verifier rows contain duplicate samples")
    if set(actual) != set(expected):
        raise ValueError("Verifier rows must exactly match the registered panel")
    for identity, row in actual.items():
        if bool(row.get("baseline_success")) != bool(
            expected[identity]["baseline_success"]
        ):
            raise ValueError("Verifier baseline differs from the cached panel baseline")

    base_hashes = {str(row.get("base_bundle_sha256", "")) for row in rows}
    candidate_hashes = {str(row.get("candidate_bundle_sha256", "")) for row in rows}
    if len(base_hashes) != 1 or "" in base_hashes:
        raise ValueError("Verifier rows require one stable full base bundle hash")
    if len(candidate_hashes) != 1 or "" in candidate_hashes:
        raise ValueError("Verifier rows require one stable full candidate bundle hash")

    train = summarize(rows, "train")
    val = summarize(rows, "val")
    combined = combine_summaries(train, val)
    attempts = int(combined["attempts"])
    candidate_successes = int(combined["improved"]) + int(combined["unchanged_success"])
    candidate_success_rate = candidate_successes / attempts
    perfect_verifier = candidate_successes == attempts and combined["regressed"] == 0

    verification = config["verification"]
    strict_threshold = verification["evolution_acceptance_ladder"][0]
    fallback_threshold = verification["evolution_acceptance_ladder"][1]
    strict_pass = combined["net_improvements"] >= int(
        strict_threshold["min_net_improvements"]
    ) and combined["regressed"] <= int(strict_threshold["max_regressions"])
    fallback_eligible = (
        not (strict_pass or perfect_verifier)
        and combined["net_improvements"]
        >= int(fallback_threshold["min_net_improvements"])
        and combined["regressed"] <= int(fallback_threshold["max_regressions"])
    )
    role = panel["panel_role"]
    generalization_pass = False
    if role == "generalization":
        policy = verification["generalization_acceptance"]
        generalization_pass = (
            candidate_success_rate >= float(policy["min_candidate_success_rate"])
            and combined["regressed"] <= int(policy["max_regressions"])
            and combined["net_improvements"]
            >= int(policy.get("min_net_improvements", 0))
        )
    development_ready = role == "development" and (strict_pass or perfect_verifier)
    acceptance_tier = (
        "generalization_pass"
        if generalization_pass
        else (
            "strict"
            if strict_pass
            else (
                "perfect"
                if perfect_verifier
                else "fallback" if fallback_eligible else "rejected"
            )
        )
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "candidate_revision": int(candidate["revision"]),
        "problem_id": candidate["problem_id"],
        "task": task,
        "panel_id": panel["panel_id"],
        "panel_role": role,
        "panel_sha256": panel_hash(panel),
        "panel_size": attempts,
        "sample_provenance": [
            {
                "record_id": str(row["record_id"]),
                "dataset_split": row["dataset_split"],
                "episode_id": int(row["episode_id"]),
                # Informational only in v5. Development samples are reusable.
                "consumed_for_analysis": bool(row.get("consumed_for_analysis", False)),
            }
            for row in rows
        ],
        "bundle_comparison": {
            "base_bundle_sha256": next(iter(base_hashes)),
            "candidate_bundle_sha256": next(iter(candidate_hashes)),
        },
        "static_identity_isolation": "passed",
        "offline_acceptance_verifier_mutable_by_candidate": False,
        "train": train,
        "val": val,
        "paired_verifier": combined,
        # Compatibility alias for reporting tools.
        "pilot_verifier": combined,
        "candidate_successes": candidate_successes,
        "candidate_success_rate": candidate_success_rate,
        "perfect_verifier": perfect_verifier,
        "perfect_verifier_fraction": f"{candidate_successes}/{attempts}",
        "strict_pass": strict_pass,
        "fallback_eligible": fallback_eligible,
        "development_ready": development_ready,
        "generalization_pass": generalization_pass,
        "acceptance_tier": acceptance_tier,
        "thresholds": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validations", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    candidate_path = args.candidate.resolve()
    report = evaluate(
        json.loads(candidate_path.read_text(encoding="utf-8")),
        load_jsonl(args.validations.resolve()),
        json.loads(args.panel.resolve().read_text(encoding="utf-8")),
        load_config(args.experiment_config.resolve()),
        candidate_dir=candidate_path.parent,
    )
    write_json(args.output_report.resolve(), report)
    print(report["acceptance_tier"])


if __name__ == "__main__":
    main()
