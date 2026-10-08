#!/usr/bin/env python3
"""Promote a sanitized v2 lesson using paired train/val simulator outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from planner import write_json
    from self_evolve_v2 import assert_no_privileged_identity
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.self_evolve_v2 import assert_no_privileged_identity


REQUIRED_EXPERIENCE_FIELDS = {
    "experience_id",
    "revision",
    "status",
    "task_scope",
    "condition",
    "lesson",
    "recommended_skill_or_policy",
    "success_evidence",
    "forbidden_behavior",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_candidate(value: dict[str, Any]) -> None:
    assert_no_privileged_identity(value)
    if set(value) != REQUIRED_EXPERIENCE_FIELDS:
        raise ValueError(
            f"Candidate fields mismatch: expected {sorted(REQUIRED_EXPERIENCE_FIELDS)}, "
            f"got {sorted(value)}"
        )
    if value["status"] not in {"candidate", "train_verified", "val_supported"}:
        raise ValueError("Invalid candidate status")
    if not value["task_scope"] or len(value["task_scope"]) != 1:
        raise ValueError("v2 candidate must be scoped to exactly one task")


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("dataset_split") == split]
    outcomes = Counter()
    records = []
    for row in selected:
        baseline = bool(row["baseline_success"])
        evolved = bool(row["experience_success"])
        outcome = (
            "improved"
            if not baseline and evolved
            else "regressed"
            if baseline and not evolved
            else "unchanged_success"
            if baseline
            else "unchanged_failure"
        )
        outcomes[outcome] += 1
        records.append(
            {
                "record_id": str(row["record_id"]),
                "outcome": outcome,
                "target_failure_corrected": bool(
                    row.get("target_failure_corrected", False)
                ),
            }
        )
    return {
        "dataset_split": split,
        "attempts": len(selected),
        **{name: outcomes[name] for name in (
            "improved", "regressed", "unchanged_success", "unchanged_failure"
        )},
        "net_improvements": outcomes["improved"] - outcomes["regressed"],
        "target_failures_corrected": sum(
            record["target_failure_corrected"] for record in records
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validations", type=Path, required=True)
    parser.add_argument("--output-experience", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--min-train-attempts", type=int, default=4)
    parser.add_argument("--min-train-net", type=int, default=1)
    parser.add_argument("--min-val-attempts", type=int, default=4)
    parser.add_argument("--min-val-net", type=int, default=0)
    parser.add_argument("--max-regressions", type=int, default=0)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    validate_candidate(candidate)
    rows = load_jsonl(args.validations)
    invalid_splits = sorted(
        {str(row.get("dataset_split")) for row in rows}
        - {"train", "val"}
    )
    if invalid_splits:
        raise ValueError(f"Verifier rejects non-train/val rows: {invalid_splits}")
    if any(row.get("task") != candidate["task_scope"][0] for row in rows):
        raise ValueError("Validation task does not match candidate task scope")
    train = summarize(rows, "train")
    val = summarize(rows, "val")
    train_pass = (
        train["attempts"] >= args.min_train_attempts
        and train["net_improvements"] >= args.min_train_net
        and train["regressed"] <= args.max_regressions
    )
    val_pass = (
        val["attempts"] >= args.min_val_attempts
        and val["net_improvements"] >= args.min_val_net
        and val["regressed"] <= args.max_regressions
    )
    status = "val_supported" if train_pass and val_pass else "train_verified" if train_pass else "candidate"
    promoted = dict(candidate)
    promoted["revision"] = int(candidate["revision"]) + (status != candidate["status"])
    promoted["status"] = status
    report = {
        "schema_version": "self-evolve-v2-verifier-report-v1",
        "experience_id": candidate["experience_id"],
        "task": candidate["task_scope"][0],
        "static_identity_isolation": "passed",
        "train": train,
        "val": val,
        "thresholds": {
            "min_train_attempts": args.min_train_attempts,
            "min_train_net": args.min_train_net,
            "min_val_attempts": args.min_val_attempts,
            "min_val_net": args.min_val_net,
            "max_regressions": args.max_regressions,
        },
        "decision": status,
    }
    write_json(args.output_experience.resolve(), promoted)
    write_json(args.output_report.resolve(), report)
    print(status)


if __name__ == "__main__":
    main()
