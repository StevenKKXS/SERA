from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = load_module(
    "verify_self_evolve_v3_experience", "verify_self_evolve_v3_experience.py"
)
SELECT = load_module(
    "select_self_evolve_v3_candidate", "select_self_evolve_v3_candidate.py"
)


def candidate() -> dict:
    return {
        "experience_id": "lesson-a",
        "revision": 1,
        "status": "candidate",
        "problem_id": "MoveCube-Pgrounding",
        "task_scope": ["MoveCube"],
        "condition": "When visible progress stalls.",
        "lesson": "Reobserve before repeating.",
        "recommended_skill_or_policy": "Use current images and action history.",
        "success_evidence": "Visible task progress resumes.",
        "forbidden_behavior": "Do not use hidden identity information.",
    }


def rows(improved: int, regressed: int = 0, attempts: int = 6) -> list[dict]:
    values = []
    for index in range(attempts):
        baseline = index >= improved and index < improved + regressed
        evolved = index < improved or (baseline and index >= improved + regressed)
        split = "train" if index < attempts // 2 else "val"
        values.append(
            {
                "record_id": f"{split}-{index}",
                "episode_id": index,
                "dataset_split": split,
                "task": "MoveCube",
                "baseline_success": baseline,
                "experience_success": evolved,
                "consumed_for_analysis": False,
                "base_bundle_sha256": "base-sha",
                "candidate_bundle_sha256": "candidate-sha",
            }
        )
    return values


CONFIG = {
    "verification": {
        "required_train_attempts": 3,
        "required_val_attempts": 3,
        "minimum_total_attempts": 6,
        "maximum_total_attempts": 12,
        "full_train_val_candidate_rollout_required": False,
        "require_incremental_bundle_provenance": True,
        "require_verifier_episode_ids": True,
        "reject_analysis_consumed_samples": True,
        "test_eligible_tiers": ["strict", "fallback"],
        "evolution_acceptance_ladder": [
            {"tier": "strict", "min_net_improvements": 2, "max_regressions": 0},
            {
                "tier": "fallback",
                "min_net_improvements": 1,
                "max_regressions": 0,
                "only_if_no_strict_candidate": True,
            },
        ],
    }
}


class VerifySelfEvolveV3Tests(unittest.TestCase):
    def test_strict_and_fallback_are_distinct(self) -> None:
        strict = VERIFY.evaluate(candidate(), rows(2), CONFIG)
        fallback = VERIFY.evaluate(candidate(), rows(1), CONFIG)
        self.assertEqual(strict["acceptance_tier"], "strict")
        self.assertEqual(fallback["acceptance_tier"], "fallback")
        self.assertEqual(strict["pilot_verifier"]["attempts"], 6)
        self.assertTrue(strict["test_release_eligible"])
        self.assertFalse(strict["full_train_val_candidate_rollout_required"])

    def test_incomplete_val_coverage_is_rejected(self) -> None:
        report = VERIFY.evaluate(candidate(), rows(2)[:5], CONFIG)
        self.assertFalse(report["coverage_pass"])
        self.assertEqual(report["acceptance_tier"], "rejected")

    def test_boundary_result_can_extend_but_not_past_twelve(self) -> None:
        boundary = VERIFY.evaluate(candidate(), rows(0), CONFIG)
        over_budget = VERIFY.evaluate(candidate(), rows(2, attempts=14), CONFIG)
        self.assertTrue(boundary["extension_eligible"])
        self.assertFalse(over_budget["within_budget"])
        self.assertEqual(over_budget["acceptance_tier"], "rejected")

    def test_analysis_consumed_sample_is_rejected(self) -> None:
        consumed = rows(2)
        consumed[0]["consumed_for_analysis"] = True
        with self.assertRaisesRegex(ValueError, "consumed"):
            VERIFY.evaluate(candidate(), consumed, CONFIG)

    def test_selection_never_uses_fallback_when_strict_exists(self) -> None:
        strict = VERIFY.evaluate(candidate(), rows(2), CONFIG)
        fallback_candidate = candidate()
        fallback_candidate["experience_id"] = "lesson-b"
        fallback = VERIFY.evaluate(fallback_candidate, rows(1), CONFIG)
        selected = SELECT.select([fallback, strict])
        self.assertEqual(selected["selected_experience_id"], "lesson-a")
        self.assertEqual(selected["selected_tier"], "strict")

    def test_selection_ranks_with_combined_pilot_evidence(self) -> None:
        first = VERIFY.evaluate(candidate(), rows(2), CONFIG)
        second_candidate = candidate()
        second_candidate["experience_id"] = "lesson-b"
        second = VERIFY.evaluate(second_candidate, rows(3), CONFIG)
        selected = SELECT.select([first, second])
        self.assertEqual(selected["selected_experience_id"], "lesson-b")


if __name__ == "__main__":
    unittest.main()
