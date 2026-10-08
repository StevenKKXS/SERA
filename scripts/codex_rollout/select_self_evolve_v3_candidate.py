#!/usr/bin/env python3
"""Select strict evidence first, falling back only when no strict report exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from planner import write_json
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.planner import write_json


def rank(report: dict[str, Any]) -> tuple[int, int, int, str]:
    evidence = report.get(
        "pilot_verifier", report.get("evolution50", report["train"])
    )
    return (
        int(evidence["net_improvements"]),
        int(evidence["improved"]),
        -int(evidence["regressed"]),
        str(report["experience_id"]),
    )


def select(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one verifier report is required")
    tasks = {report["task"] for report in reports}
    if len(tasks) != 1:
        raise ValueError("Candidate reports must belong to exactly one task")
    strict = [report for report in reports if report.get("strict_pass")]
    fallback = [report for report in reports if report.get("fallback_eligible")]
    eligible = strict if strict else fallback
    selected = max(eligible, key=rank) if eligible else None
    return {
        "schema_version": "self-evolve-v3-candidate-selection-v1",
        "task": next(iter(tasks)),
        "selection_policy": "strict_first_then_fallback",
        "selected_experience_id": selected["experience_id"] if selected else None,
        "selected_tier": selected["acceptance_tier"] if selected else "none",
        "selected_report": selected,
        "candidate_reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = select(reports)
    write_json(args.output.resolve(), result)
    print(result["selected_experience_id"] or "none")


if __name__ == "__main__":
    main()
