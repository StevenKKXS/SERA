from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "release_self_evolve_v3_candidate",
    SCRIPT_DIR / "release_self_evolve_v3_candidate.py",
)
assert SPEC is not None and SPEC.loader is not None
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


def candidate() -> dict:
    return {
        "experience_id": "lesson-a",
        "revision": 2,
        "status": "val_supported",
        "task_scope": ["PatternLock"],
        "condition": "When the next target is visible.",
        "lesson": "Use the stable grid.",
        "recommended_skill_or_policy": "Ground the next transition.",
        "success_evidence": ["The intended target activates."],
        "forbidden_behavior": ["Do not use test feedback."],
    }


def config() -> dict:
    return {
        "base_profile": "retrieval_v8_5_refine_then_lock",
        "models": {"experiment_planner": "gpt-5.6-sol"},
        "verification": {
            "test_eligible_tiers": ["strict", "fallback"],
            "full_train_val_candidate_rollout_required": False,
        },
    }


class ReleaseSelfEvolveV3CandidateTests(unittest.TestCase):
    def test_builds_lightweight_test_boundary(self) -> None:
        manifest, runtime = RELEASE.build_release(
            candidate=candidate(),
            report={
                "experience_id": "lesson-a",
                "task": "PatternLock",
                "acceptance_tier": "strict",
                "test_release_eligible": True,
            },
            config=config(),
            source_commit="abc123",
            candidate_sha256="1" * 64,
            report_sha256="2" * 64,
            config_sha256="3" * 64,
            executor_checkpoint="models/pi05/79999",
            released_at="2026-08-26T00:00:00+00:00",
        )
        self.assertEqual(manifest["test_policy"], "one_shot_feedback_forbidden")
        self.assertFalse(manifest["full_train_val_candidate_rollout_used_for_acceptance"])
        self.assertEqual(runtime["experiences"][0]["status"], "approved")

    def test_rejects_noneligible_report(self) -> None:
        with self.assertRaises(ValueError):
            RELEASE.build_release(
                candidate=candidate(),
                report={
                    "experience_id": "lesson-a",
                    "task": "PatternLock",
                    "acceptance_tier": "rejected",
                    "test_release_eligible": False,
                },
                config=config(),
                source_commit="abc123",
                candidate_sha256="1" * 64,
                report_sha256="2" * 64,
                config_sha256="3" * 64,
                executor_checkpoint="models/pi05/79999",
                released_at="2026-08-26T00:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
