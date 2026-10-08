#!/usr/bin/env python3
"""Update candidate experience revisions from held-out paired validations.

Each JSONL validation row must contain ``experience_id``, ``task``, ``episode``,
``baseline_success`` and ``experience_success``.  The updater is idempotent by
``task:episode`` and never promotes experiences with observed regressions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from planner import write_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def outcome_name(baseline_success: bool, experience_success: bool) -> str:
    if not baseline_success and experience_success:
        return "improved"
    if baseline_success and not experience_success:
        return "regressed"
    return "unchanged_success" if baseline_success else "unchanged_failure"


def update_memory(
    memory: dict[str, Any],
    validations: list[dict[str, Any]],
    *,
    min_attempts: int,
    min_net_improvements: int,
) -> dict[str, Any]:
    experiences = {
        str(item["experience_id"]): item for item in memory.get("experiences", [])
    }
    applied = 0
    for row in validations:
        experience_id = str(row["experience_id"])
        if experience_id not in experiences:
            raise KeyError(f"Unknown experience_id: {experience_id}")
        experience = experiences[experience_id]
        validation = experience["validation"]
        record_id = f"{row['task']}:{int(row['episode']):02d}"
        if any(item["record_id"] == record_id for item in validation["records"]):
            continue
        outcome = outcome_name(
            bool(row["baseline_success"]), bool(row["experience_success"])
        )
        record = {
            "record_id": record_id,
            "outcome": outcome,
            "baseline_success": bool(row["baseline_success"]),
            "experience_success": bool(row["experience_success"]),
            "notes": str(row.get("notes", "")),
        }
        validation["records"].append(record)
        applied += 1
        validation[outcome] += 1
        validation["attempts"] += 1
        net = validation["improved"] - validation["regressed"]
        if validation["regressed"] > 0:
            experience["status"] = "needs_revision"
        elif validation["attempts"] >= min_attempts and net >= min_net_improvements:
            experience["status"] = "validated"
        else:
            experience["status"] = "candidate"
        experience["revision"] = int(experience.get("revision", 1)) + 1
        experience["promotion_note"] = (
            f"Validation attempts={validation['attempts']}, improved={validation['improved']}, "
            f"regressed={validation['regressed']}, net={net}."
        )
    if applied == 0:
        return memory
    memory["revision"] = int(memory.get("revision", 1)) + 1
    statuses = {item["status"] for item in experiences.values()}
    memory["status"] = "contains_validated" if "validated" in statuses else "candidate_only"
    return memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--validations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-attempts", type=int, default=5)
    parser.add_argument("--min-net-improvements", type=int, default=2)
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    updated = update_memory(
        memory,
        load_jsonl(args.validations),
        min_attempts=args.min_attempts,
        min_net_improvements=args.min_net_improvements,
    )
    write_json(args.output, updated)
    print(f"Wrote experience-memory revision {updated['revision']} to {args.output}")


if __name__ == "__main__":
    main()
