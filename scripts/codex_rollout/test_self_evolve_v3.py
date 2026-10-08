from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "self_evolve_v3", SCRIPT_DIR / "self_evolve_v3.py"
)
assert SPEC is not None and SPEC.loader is not None
V3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V3)


def config() -> dict:
    return {
        "experiment_id": "test-v3",
        "base_profile": "retrieval_v8_5_refine_then_lock",
        "task_order": ["MoveCube"],
        "models": {"experiment_planner": "gpt-5.6-sol"},
        "problem_discovery": {
            "min_cluster_support": 2,
            "max_problem_clusters": 8,
            "keep_singletons": True,
            "use_success_counterexamples": True,
            "primary_problem_by_first_divergence": True,
        },
        "evolution_loop": {
            "max_attempts_per_problem": 3,
            "max_total_evolution_rounds": 5,
            "allow_active_problem_switch": True,
            "allow_revisit_deferred_problem": True,
        },
    }


def failure(
    episode_id: int,
    *,
    split: str = "train",
    primary: str = "wrong target grounding",
    observable: bool = True,
    actionability: str = "partial",
) -> dict:
    return {
        "dataset_split": split,
        "episode_id": episode_id,
        "outcome": "failure",
        "task_variant": "move cube",
        "failure_stage": "target selection",
        "primary_problem": primary,
        "first_divergence": "planner selects a distractor",
        "secondary_symptoms": ["repeated similar commands", "timeout"],
        "responsible_module": "grounding_and_recovery",
        "runtime_observable": observable,
        "planner_actionability": actionability,
        "confidence": 0.87,
    }


def candidate(problem_id: str, experience_id: str, revision: int = 1) -> dict:
    return {
        "experience_id": experience_id,
        "revision": revision,
        "status": "candidate",
        "problem_id": problem_id,
        "task_scope": ["MoveCube"],
        "condition": "When target grounding is ambiguous.",
        "lesson": "Confirm object identity before moving.",
        "recommended_skill_or_policy": "Use visible attributes and recheck contact.",
        "success_evidence": ["The intended object moves."],
        "forbidden_behavior": ["Do not use episode identity."],
    }


def success_counterexample(
    episode_id: int, *, split: str = "val", variant: str = "move cube"
) -> dict:
    return {
        "dataset_split": split,
        "episode_id": episode_id,
        "outcome": "success",
        "task_variant": variant,
        "failure_stage": "successful_execution",
        "primary_problem": "successful_execution",
        "first_divergence": "no divergence",
        "secondary_symptoms": [],
        "responsible_module": "none",
        "runtime_observable": True,
        "planner_actionability": "none",
        "confidence": 0.95,
        "counterexample_for": ["*"],
    }


def verifier(
    experience_id: str,
    problem_id: str,
    base_bundle_sha256: str,
    candidate_bundle_sha256: str,
    improved: int = 2,
    regressed: int = 0,
) -> dict:
    return {
        "schema_version": "self-evolve-v3-pilot-verifier-report-v3",
        "experience_id": experience_id,
        "problem_id": problem_id,
        "task": "MoveCube",
        "bundle_comparison": {
            "base_bundle_sha256": base_bundle_sha256,
            "candidate_bundle_sha256": candidate_bundle_sha256,
        },
        "sample_provenance": [
            {
                "record_id": f"{split}-{index}",
                "dataset_split": split,
                "episode_id": index,
                "consumed_for_analysis": False,
            }
            for split in ("train", "val")
            for index in range(3)
        ],
        "acceptance_tier": "strict" if improved >= 2 else "fallback",
        "test_release_eligible": improved >= 1 and regressed == 0,
        "pilot_verifier": {
            "attempts": 6,
            "improved": improved,
            "regressed": regressed,
            "net_improvements": improved - regressed,
        },
    }


def verifier_for_current(state: dict, *, improved: int = 2, regressed: int = 0) -> dict:
    current = state["current_candidate"]
    return verifier(
        current["experience_id"],
        current["problem_id"],
        current["base_bundle_sha256"],
        current["candidate_bundle_sha256"],
        improved,
        regressed,
    )


class SelfEvolveV3Tests(unittest.TestCase):
    def test_success_counterexamples_match_only_same_task_variant(self) -> None:
        records = [
            failure(1),
            failure(2, split="val"),
            success_counterexample(3),
            success_counterexample(4, variant="peg push"),
        ]
        discovery = V3.build_problem_discovery(records, config(), "MoveCube")
        cluster = discovery["clusters"][0]
        self.assertEqual(discovery["success_counterexample_record_count"], 2)
        self.assertEqual(cluster["success_counterexample_count"], 1)

    def test_problem_discovery_retains_multiple_roots_and_singletons(self) -> None:
        records = [
            failure(1),
            failure(2, split="val"),
            failure(3, primary="contact never established"),
            failure(
                4,
                primary="hidden simulator state required",
                observable=False,
                actionability="none",
            ),
        ]
        discovery = V3.build_problem_discovery(records, config(), "MoveCube")
        self.assertEqual(len(discovery["clusters"]), 3)
        statuses = {
            row["signature"]["primary_problem"]: row["status"]
            for row in discovery["clusters"]
        }
        self.assertEqual(statuses["wrong target grounding"], "active")
        self.assertEqual(statuses["contact never established"], "singleton")
        self.assertEqual(statuses["hidden simulator state required"], "unactionable")
        active = next(row for row in discovery["clusters"] if row["status"] == "active")
        self.assertEqual(active["split_support"], {"train": 1, "val": 1})

    def test_next_problem_uses_priority_and_skips_unactionable(self) -> None:
        state = V3.create_state(config(), "MoveCube", "abc")
        discovery = V3.build_problem_discovery(
            [
                failure(1),
                failure(2, split="val"),
                failure(
                    3,
                    primary="hidden simulator state required",
                    observable=False,
                    actionability="none",
                ),
                failure(
                    4,
                    split="val",
                    primary="hidden simulator state required",
                    observable=False,
                    actionability="none",
                ),
            ],
            config(),
            "MoveCube",
        )
        V3.install_problem_discovery(state, discovery)
        selected = V3.next_problem_id(state, config())
        self.assertEqual(state["problems"][selected]["status"], "active")

    def test_test_split_problem_record_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "train/val"):
            V3.validate_episode_problem_record(failure(1, split="test"))

    def test_candidate_round_budget_is_separate_from_problem_count(self) -> None:
        state = V3.create_state(config(), "MoveCube", "abc")
        discovery = V3.build_problem_discovery(
            [
                failure(1),
                failure(2, split="val"),
                failure(3, primary="contact failure"),
                failure(4, split="val", primary="contact failure"),
            ],
            config(),
            "MoveCube",
        )
        V3.install_problem_discovery(state, discovery)
        problem_ids = [row["problem_id"] for row in discovery["clusters"]]
        first, second = problem_ids
        V3.select_problem(state, first, config())
        V3.start_candidate(state, candidate(first, "lesson-a"), "a" * 64)
        V3.decide_candidate(
            state, verifier_for_current(state, improved=0), "defer", "No gain"
        )
        self.assertEqual(state["total_evolution_rounds"], 1)
        self.assertEqual(state["problems"][first]["status"], "deferred")

        V3.select_problem(state, second, config())
        round_two = V3.start_candidate(state, candidate(second, "lesson-b"), "b" * 64)
        self.assertEqual(round_two["base_bundle_experience_ids"], [])
        V3.decide_candidate(state, verifier_for_current(state), "accept", "Strict gain")
        self.assertEqual(state["total_evolution_rounds"], 2)
        self.assertEqual(state["bundle_revision"], 1)
        self.assertEqual(state["working_bundle"][0]["experience_id"], "lesson-b")

        V3.select_problem(state, first, config())
        round_three = V3.start_candidate(
            state, candidate(first, "lesson-a", revision=2), "c" * 64
        )
        self.assertEqual(round_three["base_bundle_experience_ids"], ["lesson-b"])
        self.assertEqual(state["problem_attempts"][first], 2)

    def test_three_failed_attempts_exhaust_one_problem_without_ending_early(
        self,
    ) -> None:
        state = V3.create_state(config(), "MoveCube", "abc")
        discovery = V3.build_problem_discovery(
            [failure(1), failure(2, split="val")], config(), "MoveCube"
        )
        V3.install_problem_discovery(state, discovery)
        problem_id = discovery["clusters"][0]["problem_id"]
        for attempt in range(1, 4):
            V3.select_problem(state, problem_id, config())
            experience_id = f"lesson-{attempt}"
            V3.start_candidate(
                state,
                candidate(problem_id, experience_id, revision=attempt),
                str(attempt) * 64,
            )
            V3.decide_candidate(
                state,
                verifier_for_current(state, improved=0),
                "retry",
                "Still ineffective",
            )
        self.assertEqual(state["problems"][problem_id]["status"], "exhausted")
        self.assertEqual(state["total_evolution_rounds"], 3)
        self.assertEqual(state["remaining_problem_budget"], 2)

    def test_snapshot_records_bundle_without_final_rollout(self) -> None:
        state = V3.create_state(config(), "MoveCube", "abc")
        approved = {**candidate("MoveCube-Px", "lesson-a"), "status": "approved"}
        state["working_bundle"] = [approved]
        state["accepted_experiences"] = [approved]
        state["bundle_revision"] = 1
        state["total_evolution_rounds"] = 2
        manifest, runtime = V3.build_test_candidate_snapshot(
            state, config(), "def", "models/pi05/79999"
        )
        self.assertFalse(manifest["final_bundle_rollout_performed"])
        self.assertEqual(manifest["accepted_experience_ids"], ["lesson-a"])
        self.assertEqual(runtime["experiences"][0]["status"], "approved")

    def test_state_rejects_verifier_sample_previously_consumed_by_analysis(
        self,
    ) -> None:
        state = V3.create_state(config(), "MoveCube", "abc")
        discovery = V3.build_problem_discovery(
            [failure(1), failure(2, split="val")], config(), "MoveCube"
        )
        V3.install_problem_discovery(state, discovery)
        problem_id = discovery["clusters"][0]["problem_id"]
        V3.select_problem(state, problem_id, config())
        V3.start_candidate(state, candidate(problem_id, "lesson-a"), "a" * 64)
        state["analysis_consumed_samples"] = [
            {"dataset_split": "train", "episode_id": 0}
        ]
        with self.assertRaisesRegex(ValueError, "consumed"):
            V3.decide_candidate(
                state, verifier_for_current(state), "accept", "Should be rejected"
            )

    def test_auto_hard20_must_complete_before_problem_discovery(self) -> None:
        auto_config = config()
        auto_config["hard_exploration"] = {"mode": "auto"}
        state = V3.create_state(auto_config, "MoveCube", "abc")
        discovery = V3.build_problem_discovery(
            [failure(1), failure(2, split="val")], auto_config, "MoveCube"
        )
        with self.assertRaisesRegex(ValueError, "must be decided"):
            V3.install_problem_discovery(state, discovery)

        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            trigger_path = root / "trigger.json"
            trigger = {
                "task": "MoveCube",
                "decision": "trigger",
                "success_rate": 0.8,
            }
            trigger_path.write_text(json.dumps(trigger), encoding="utf-8")
            V3.record_hard_trigger(state, trigger, trigger_path)
            selected = [
                {
                    "task": "MoveCube",
                    "dataset_split": split,
                    "episode_id": index,
                    "outcome_used_for_selection": False,
                }
                for split, count in (("train", 12), ("val", 8))
                for index in range(count)
            ]
            selection_path = root / "selection.jsonl"
            selection_path.write_text(
                "".join(json.dumps(row) + "\n" for row in selected),
                encoding="utf-8",
            )
            V3.record_hard_selection(state, selected, selection_path)
            with self.assertRaisesRegex(ValueError, "must complete"):
                V3.install_problem_discovery(state, discovery)

            report_path = root / "report.json"
            manifest_path = root / "manifest.jsonl"
            report = {
                "task": "MoveCube",
                "complete": True,
                "attempts": 20,
                "successes": 16,
                "failures": 4,
                "representative_benchmark": False,
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            manifest_path.write_text("{}\n", encoding="utf-8")
            V3.record_hard_results(state, report, report_path, manifest_path)
            V3.install_problem_discovery(state, discovery)
            self.assertEqual(state["stage"], "problems_discovered")
            self.assertEqual(state["total_evolution_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
