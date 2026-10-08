#!/usr/bin/env python3
"""Record a verifier-approved lesson as the immutable input to one test50 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from planner import write_json
    from verify_self_evolve_v2_experience import validate_candidate
except ModuleNotFoundError:  # Support repository-root unittest imports.
    from scripts.codex_rollout.planner import write_json
    from scripts.codex_rollout.verify_self_evolve_v2_experience import validate_candidate


RUNTIME_FIELDS = (
    "experience_id",
    "revision",
    "task_scope",
    "condition",
    "lesson",
    "recommended_skill_or_policy",
    "success_evidence",
    "forbidden_behavior",
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def build_release(
    *,
    candidate: dict[str, Any],
    report: dict[str, Any],
    config: dict[str, Any],
    source_commit: str,
    candidate_sha256: str,
    report_sha256: str,
    config_sha256: str,
    executor_checkpoint: str,
    released_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_candidate(candidate)
    if candidate.get("status") != "val_supported":
        raise ValueError("Test release requires a val_supported candidate")
    if report.get("experience_id") != candidate.get("experience_id"):
        raise ValueError("Verifier report and candidate experience differ")
    task = candidate["task_scope"][0]
    if report.get("task") != task:
        raise ValueError("Verifier report and candidate task differ")
    eligible_tiers = set(config["verification"].get("test_eligible_tiers", []))
    tier = report.get("acceptance_tier")
    if tier not in eligible_tiers or not report.get("test_release_eligible"):
        raise ValueError("Verifier report is not eligible for test release")
    if config["verification"].get("full_train_val_candidate_rollout_required"):
        raise ValueError("Configuration still requires a full train/val candidate rollout")

    release_id = (
        f"{task}-r{int(candidate['revision'])}-"
        f"{candidate_sha256[:12]}"
    )
    runtime_memory = {
        "schema_version": "self-evolve-runtime-lessons-v2",
        "memory_id": release_id,
        "revision": int(candidate["revision"]),
        "base_profile": config["base_profile"],
        "experiences": [
            {
                **{field: candidate[field] for field in RUNTIME_FIELDS},
                "status": "approved",
            }
        ],
    }
    manifest = {
        "schema_version": "self-evolve-v3-test-release-v1",
        "release_id": release_id,
        "task": task,
        "experience_id": candidate["experience_id"],
        "experience_revision": int(candidate["revision"]),
        "acceptance_tier": tier,
        "candidate_sha256": candidate_sha256,
        "verifier_report_sha256": report_sha256,
        "experiment_config_sha256": config_sha256,
        "source_commit": source_commit,
        "base_profile": config["base_profile"],
        "models": config["models"],
        "executor_checkpoint": executor_checkpoint,
        "released_at": released_at,
        "test_policy": "one_shot_feedback_forbidden",
        "test_feedback_used": False,
        "historical_test_experience_used": False,
        "full_train_val_candidate_rollout_used_for_acceptance": False,
    }
    return manifest, runtime_memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--verifier-report", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--executor-checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    report_path = args.verifier_report.resolve()
    config_path = args.experiment_config.resolve()
    repo_root = args.repo_root.resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Repository must be clean before creating a test release")
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    released_at = datetime.now(timezone.utc).isoformat()
    manifest, runtime_memory = build_release(
        candidate=load_json(candidate_path),
        report=load_json(report_path),
        config=load_json(config_path),
        source_commit=source_commit,
        candidate_sha256=sha256_path(candidate_path),
        report_sha256=sha256_path(report_path),
        config_sha256=sha256_path(config_path),
        executor_checkpoint=args.executor_checkpoint,
        released_at=released_at,
    )
    output_dir = args.output_dir.resolve()
    write_json(output_dir / "test_release_manifest.json", manifest)
    write_json(output_dir / "runtime_lesson_memory.json", runtime_memory)
    print(manifest["release_id"])


if __name__ == "__main__":
    main()
