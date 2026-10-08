from __future__ import annotations

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import self_evolve_v3 as V3
import self_evolve_v4 as V4
import self_evolve_v5 as V5
from build_self_evolve_v4_candidate import materialize_candidate
from prepare_v5_problem_records import (
    convert as convert_v5_problem_record,
    promote_cross_episode_model_consensus,
)
from verify_self_evolve_v5_candidate import evaluate


CONFIG_PATH = SCRIPT_DIR.parents[1] / "configs" / "self_evolution.json"
EXAMPLE = (
    SCRIPT_DIR
    / "examples"
    / "self_evolve_v4"
    / "movecube_v1_compatible"
    / "candidate.json"
)


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def panel(problem_id: str, role: str, episodes: tuple[int, ...]) -> dict:
    samples = []
    for index, episode in enumerate(episodes):
        samples.append(
            {
                "dataset_split": "train" if index % 2 == 0 else "val",
                "episode_id": episode,
                "baseline_success": index >= 2,
            }
        )
    return {
        "schema_version": "self-evolve-v5-paired-panel-v1",
        "panel_id": f"{problem_id}-{role}",
        "panel_role": role,
        "task": "MoveCube",
        "problem_id": problem_id,
        "samples": samples,
    }


def lineage(candidate: dict, parent: str | None = None) -> dict:
    return {
        "schema_version": "self-evolve-v5-candidate-lineage-v1",
        "candidate_id": candidate["candidate_id"],
        "problem_id": candidate["problem_id"],
        "revision": candidate["revision"],
        "parent_candidate_id": parent,
        "inherited_component_ids": [],
        "modified_component_ids": [
            component["component_id"] for component in candidate["components"]
        ],
        "preserved_invariants": ["runtime episode identity remains forbidden"],
        "behavioral_delta": "Require verified held motion before transition.",
        "revision_reason": (
            "Initial causal-chain repair."
            if parent is None
            else "Pilot evidence requires a stronger gate."
        ),
    }


def retrospective(
    candidate: dict, panel_value: dict, action: str, *, switch: dict | None = None
) -> dict:
    return {
        "schema_version": "self-evolve-v5-candidate-retrospective-v1",
        "candidate_id": candidate["candidate_id"],
        "problem_id": candidate["problem_id"],
        "panel_id": panel_value["panel_id"],
        "panel_role": panel_value["panel_role"],
        "outcome_summary": {"summary": "Paired outcomes inspected."},
        "activation_audit": {
            "status": "activated",
            "evidence": ["gate trace present"],
        },
        "hypothesis_assessment": "supported",
        "implementation_assessment": "correct",
        "failure_type": (
            "none"
            if action in {"accept", "promote_to_generalization"}
            else "underpowered_repair"
        ),
        "knowledge_gained": ["The target transition is runtime observable."],
        "recommended_action": action,
        "next_revision": (
            {
                "behavioral_delta": "Add a second visual confirmation.",
                "preserve_invariants": ["held-motion criterion"],
            }
            if action == "revise_same_problem"
            else None
        ),
        "switch_justification": switch,
    }


class SelfEvolveV5DeepPilotTests(unittest.TestCase):
    def materialized_candidate(self, root: Path, problem_id: str) -> Path:
        path = materialize_candidate(EXAMPLE, root / "candidate")
        candidate = json.loads(path.read_text(encoding="utf-8"))
        candidate["problem_id"] = problem_id
        candidate["authoring"]["analysis_inputs"] = [
            {"dataset_split": "train", "episode_id": 1}
        ]
        path.write_text(json.dumps(candidate), encoding="utf-8")
        return path

    def state_with_problem(self, problem_id: str) -> dict:
        state = V5.create_state(config(), "MoveCube", "abc123")
        state["v5_policy"] = config()["v5_policy"]
        state["problems"] = {
            problem_id: {
                "problem_id": problem_id,
                "status": "active",
                "support": 3,
                "priority_score": 1.0,
            }
        }
        state["problem_attempts"] = {problem_id: 0}
        state["active_problem_id"] = problem_id
        return state

    def rows(
        self,
        candidate: dict,
        panel_value: dict,
        base_hash: str,
        candidate_hash: str,
        *,
        consumed: bool = False,
    ) -> list[dict]:
        return [
            {
                "record_id": f"{sample['dataset_split']}-{sample['episode_id']}",
                "dataset_split": sample["dataset_split"],
                "episode_id": sample["episode_id"],
                "task": "MoveCube",
                "baseline_success": sample["baseline_success"],
                "experience_success": True,
                "target_failure_corrected": not sample["baseline_success"],
                "consumed_for_analysis": consumed,
                "base_bundle_sha256": base_hash,
                "candidate_bundle_sha256": candidate_hash,
            }
            for sample in panel_value["samples"]
        ]

    def test_deep_discovery_requires_three_layer_causal_chain(self) -> None:
        record = {
            "dataset_split": "train",
            "episode_id": 1,
            "outcome": "failure",
            "task_variant": "peg_push",
            "failure_stage": "contact",
            "primary_problem": "generic recovery failure",
            "first_divergence": "generic divergence",
            "secondary_symptoms": ["timeout"],
            "responsible_module": "grounding_and_recovery",
            "runtime_observable": True,
            "planner_actionability": "partial",
            "confidence": 0.9,
        }
        with self.assertRaisesRegex(ValueError, "causal-chain"):
            V5.build_deep_problem_discovery([record], config(), "MoveCube")
        record["causal_chain"] = {
            "physical_first_divergence": "peg moves without cube contact",
            "task_progress_divergence": "cube-contact state never becomes true",
            "terminal_failure": "timeout after repeated similar commands",
            "causal_link_evidence": ["cube remains static across three actions"],
        }
        discovery = V5.build_deep_problem_discovery([record], config(), "MoveCube")
        cluster = discovery["clusters"][0]
        self.assertEqual(
            cluster["signature"]["primary_problem"],
            "generic recovery failure",
        )
        self.assertEqual(len(cluster["causal_chain_examples"]), 1)
        self.assertEqual(
            cluster["task_progress_divergence_examples"],
            ["cube-contact state never becomes true"],
        )

    def test_v1_categorical_grouping_ignores_causal_wording(self) -> None:
        first = {
            "dataset_split": "train",
            "episode_id": 1,
            "outcome": "failure",
            "task_variant": "PickXtimes",
            "failure_stage": "action_planning",
            "primary_problem": "planning_error",
            "first_divergence": "place selected before pickup",
            "secondary_symptoms": ["timeout"],
            "responsible_module": "planner",
            "runtime_observable": True,
            "planner_actionability": "partial",
            "confidence": 0.8,
            "causal_chain": {
                "physical_first_divergence": "empty-handed place",
                "task_progress_divergence": "second pickup named too early",
                "terminal_failure": "premature stop",
                "causal_link_evidence": ["no first release was observed"],
            },
        }
        second = copy.deepcopy(first)
        second["dataset_split"] = "val"
        second["episode_id"] = 2
        second["causal_chain"][
            "task_progress_divergence"
        ] = "ordinal advanced without a completed repetition"
        discovery = V5.build_deep_problem_discovery(
            [first, second], config(), "PickXtimes"
        )
        self.assertEqual(len(discovery["clusters"]), 1)
        self.assertEqual(discovery["clusters"][0]["support"], 2)
        self.assertEqual(
            discovery["clusters"][0]["signature"]["primary_problem"],
            "planning_error",
        )

    def test_v5_converter_rejects_shallow_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.json"
            source = {
                "task": "MoveCube",
                "dataset_split": "train",
                "episode": 7,
                "source_result": "stagnation_early_stop",
                "adjudication": {
                    "status": "consensus",
                    "selected_source": "model",
                    "candidate_allowed": True,
                },
                "heuristic_analysis": {
                    "derived_context": {"selected_method": "peg_push"}
                },
                "final_analysis": {
                    "failure_type": "recovery_failure",
                    "responsible_module": "recovery_policy",
                    "analysis_summary": "Repeated no-contact commands.",
                    "causal_evidence": [
                        {"source": "rollout_video", "observation": "No contact."}
                    ],
                    "alternative_causes": [],
                    "experience_eligibility": "eligible_candidate",
                    "candidate_experience": {"lesson": "Change contact geometry."},
                    "confidence": 0.9,
                },
            }
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deep causal chain"):
                convert_v5_problem_record(path)
            source["final_analysis"]["causal_chain"] = {
                "physical_first_divergence": "peg misses cube",
                "task_progress_divergence": "contact remains unverified",
                "terminal_failure": "stagnation early stop",
                "causal_link_evidence": ["cube stays static after peg motion"],
            }
            path.write_text(json.dumps(source), encoding="utf-8")
            record = convert_v5_problem_record(path)
            self.assertEqual(
                record["causal_chain"]["task_progress_divergence"],
                "contact remains unverified",
            )

    def test_v5_converter_retains_disputed_model_as_bounded_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.json"
            causal = {
                "physical_first_divergence": "place requested while nothing held",
                "task_progress_divergence": "ordinal advanced before first release",
                "terminal_failure": "premature stop",
                "causal_link_evidence": ["round 2 labels pickup as second"],
            }
            source = {
                "task": "PickXtimes",
                "dataset_split": "val",
                "episode": 42,
                "source_result": "fail",
                "adjudication": {
                    "status": "disputed",
                    "selected_source": "heuristic_pending_review",
                    "candidate_allowed": False,
                },
                "heuristic_analysis": {
                    "derived_context": {"selected_method": "unknown"}
                },
                "model_analysis": {
                    "failure_type": "planning_error",
                    "responsible_module": "planner",
                    "analysis_summary": "The ordinal advanced without a verified event.",
                    "causal_chain": causal,
                    "causal_evidence": [
                        {"source": "planner_trace", "observation": "second pickup"}
                    ],
                    "confidence": 0.88,
                },
                "final_analysis": {
                    "failure_type": "unknown",
                    "responsible_module": "unknown",
                    "analysis_summary": "Rules cannot isolate one cause.",
                    "causal_evidence": [],
                    "alternative_causes": [],
                    "experience_eligibility": "needs_more_evidence",
                    "candidate_experience": None,
                    "confidence": 0.2,
                },
            }
            path.write_text(json.dumps(source), encoding="utf-8")
            record = convert_v5_problem_record(path)
            self.assertEqual(
                record["causal_hypothesis_source"], "disputed_model_hypothesis"
            )
            self.assertTrue(record["requires_cross_episode_confirmation"])
            self.assertFalse(record["candidate_support_allowed"])
            self.assertEqual(record["planner_actionability"], "partial")
            self.assertLessEqual(record["confidence"], 0.69)
            self.assertIsNone(record["candidate_experience"])

    def test_v5_promotes_only_cross_split_repairable_model_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for split, episode in (("train", 1), ("val", 2)):
                path = root / f"{split}_{episode}.json"
                model = {
                    "failure_type": "completion_error",
                    "responsible_module": "completion_judge",
                    "analysis_summary": "The repetition advanced without release.",
                    "causal_evidence": [
                        {"source": "planner_trace", "observation": "early advance"}
                    ],
                    "experience_eligibility": "eligible_candidate",
                    "candidate_experience": {
                        "scope": "task",
                        "lesson": "Count only a verified release.",
                    },
                    "confidence": 0.9,
                }
                path.write_text(json.dumps({"model_analysis": model}), encoding="utf-8")
                rows.append(
                    {
                        "dataset_split": split,
                        "episode_id": episode,
                        "outcome": "failure",
                        "task_variant": "repeated_pick_putdown",
                        "failure_stage": "ambiguous",
                        "primary_problem": "unknown",
                        "responsible_module": "unknown",
                        "analysis_path": str(path),
                        "requires_cross_episode_confirmation": True,
                        "candidate_support_allowed": False,
                        "candidate_experience": None,
                    }
                )
            promoted = promote_cross_episode_model_consensus(rows)
            self.assertTrue(all(row["candidate_support_allowed"] for row in promoted))
            self.assertTrue(
                all(row["primary_problem"] == "completion_error" for row in promoted)
            )
            self.assertTrue(
                all(
                    row["causal_hypothesis_source"] == "cross_episode_model_consensus"
                    for row in promoted
                )
            )

            # The same evidence from one split is not enough to promote.
            for row in rows:
                row["dataset_split"] = "train"
                row["candidate_support_allowed"] = False
                row["requires_cross_episode_confirmation"] = True
            bounded = promote_cross_episode_model_consensus(rows)
            self.assertFalse(any(row["candidate_support_allowed"] for row in bounded))

    def test_perfect_verifier_uses_actual_panel_size_and_reuses_analysed_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            problem_id = "MoveCube-P-contact"
            candidate_path = self.materialized_candidate(Path(temporary), problem_id)
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            panel_value = panel(problem_id, "development", (1, 2, 3, 4, 5, 6, 7, 8))
            base_hash = V4.candidate_bundle_hash([])
            candidate_hash = V4.candidate_bundle_hash([candidate])
            report = evaluate(
                candidate,
                self.rows(
                    candidate,
                    panel_value,
                    base_hash,
                    candidate_hash,
                    consumed=True,
                ),
                panel_value,
                config(),
                candidate_dir=candidate_path.parent,
            )
            self.assertTrue(report["perfect_verifier"])
            self.assertEqual(report["perfect_verifier_fraction"], "8/8")
            self.assertTrue(report["development_ready"])

    def test_registered_development_panel_is_fixed_but_reusable(self) -> None:
        problem_id = "MoveCube-P-contact"
        state = self.state_with_problem(problem_id)
        panel_value = panel(problem_id, "development", (1, 2, 3, 4))
        first = V5.register_panel(state, panel_value, config(), Path("panel.json"))
        second = V5.register_panel(state, panel_value, config(), Path("panel.json"))
        self.assertEqual(first["panel_sha256"], second["panel_sha256"])
        changed = json.loads(json.dumps(panel_value))
        changed["samples"][0]["episode_id"] = 99
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            V5.register_panel(state, changed, config(), Path("changed.json"))

    def test_later_revision_may_register_fresh_generalization_panel(self) -> None:
        problem_id = "MoveCube-P-contact"
        state = self.state_with_problem(problem_id)
        development = panel(problem_id, "development", (1, 2, 3, 4))
        first = panel(problem_id, "generalization", (11, 12))
        V5.register_panel(state, development, config(), Path("dev.json"))
        first_record = V5.register_panel(
            state, first, config(), Path("first-gen.json")
        )
        state["round_history"] = [
            {
                "problem_id": problem_id,
                "generalization_panel_sha256": first_record["panel_sha256"],
                "verifier_reports": {"generalization": {"generalization_pass": False}},
            }
        ]
        state["current_candidate"] = {
            "candidate_id": "movecube_contact_gate_v2",
            "problem_id": problem_id,
            "status": "development_decision_pending",
        }
        overlapping = panel(problem_id, "generalization", (11, 23))
        with self.assertRaisesRegex(ValueError, "all prior"):
            V5.register_panel(state, overlapping, config(), Path("overlap.json"))
        refreshed = panel(problem_id, "generalization", (21, 22))
        refreshed_record = V5.register_panel(
            state, refreshed, config(), Path("second-gen.json")
        )
        self.assertEqual(
            state["panel_registry"][problem_id]["generalization"]["panel_sha256"],
            refreshed_record["panel_sha256"],
        )
        self.assertEqual(len(state["panel_history"]), 1)
        self.assertEqual(
            state["panel_history"][0]["record"]["panel_sha256"],
            first_record["panel_sha256"],
        )

    def test_retrospective_is_mandatory_and_revision_keeps_same_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem_id = "MoveCube-P-contact"
            state = self.state_with_problem(problem_id)
            development = panel(problem_id, "development", (1, 2, 3, 4))
            V5.register_panel(state, development, config(), root / "dev.json")
            candidate_path = self.materialized_candidate(root, problem_id)
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            current = V5.start_candidate(state, candidate_path, lineage(candidate))
            report = evaluate(
                candidate,
                self.rows(
                    candidate,
                    development,
                    current["base_bundle_sha256"],
                    current["candidate_bundle_sha256"],
                ),
                development,
                config(),
                candidate_dir=candidate_path.parent,
            )
            V5.record_verifier(state, report)
            with self.assertRaisesRegex(ValueError, "mandatory retrospective"):
                V5.decide_candidate(state, "revise_same_problem", "needs refinement")
            V5.record_retrospective(
                state,
                retrospective(candidate, development, "revise_same_problem"),
            )
            V5.decide_candidate(state, "revise_same_problem", "needs refinement")
            self.assertEqual(state["problems"][problem_id]["status"], "active")
            self.assertEqual(
                state["panel_registry"][problem_id]["development"]["panel"]["panel_id"],
                development["panel_id"],
            )
            second_path = materialize_candidate(EXAMPLE, root / "candidate-v2")
            second = json.loads(second_path.read_text(encoding="utf-8"))
            second["candidate_id"] = "movecube_contact_gate_v2"
            second["problem_id"] = problem_id
            second["revision"] = 2
            second["authoring"]["analysis_inputs"] = [
                {"dataset_split": "train", "episode_id": 1}
            ]
            second["components"][1]["config"]["max_blocked_transitions"] = 3
            second_path.write_text(json.dumps(second), encoding="utf-8")
            second_current = V5.start_candidate(
                state,
                second_path,
                lineage(second, parent=candidate["candidate_id"]),
            )
            self.assertEqual(
                second_current["development_panel_id"], development["panel_id"]
            )

    def test_generalization_is_disjoint_and_required_before_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            problem_id = "MoveCube-P-contact"
            state = self.state_with_problem(problem_id)
            development = panel(problem_id, "development", (1, 2, 3, 4))
            V5.register_panel(state, development, config(), root / "dev.json")
            candidate_path = self.materialized_candidate(root, problem_id)
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            current = V5.start_candidate(state, candidate_path, lineage(candidate))
            dev_report = evaluate(
                candidate,
                self.rows(
                    candidate,
                    development,
                    current["base_bundle_sha256"],
                    current["candidate_bundle_sha256"],
                ),
                development,
                config(),
                candidate_dir=candidate_path.parent,
            )
            V5.record_verifier(state, dev_report)
            V5.record_retrospective(
                state,
                retrospective(candidate, development, "promote_to_generalization"),
            )
            with self.assertRaisesRegex(ValueError, "Register a fresh"):
                V5.decide_candidate(state, "promote_to_generalization", "ready")
            overlapping = panel(problem_id, "generalization", (3, 4))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                V5.register_panel(state, overlapping, config(), root / "bad.json")
            generalization = panel(problem_id, "generalization", (11, 12))
            V5.register_panel(state, generalization, config(), root / "gen.json")
            V5.decide_candidate(state, "promote_to_generalization", "ready")
            gen_report = evaluate(
                candidate,
                self.rows(
                    candidate,
                    generalization,
                    current["base_bundle_sha256"],
                    current["candidate_bundle_sha256"],
                ),
                generalization,
                config(),
                candidate_dir=candidate_path.parent,
            )
            V5.record_verifier(state, gen_report)
            V5.record_retrospective(
                state, retrospective(candidate, generalization, "accept")
            )
            V5.decide_candidate(state, "accept", "fresh panel passed")
            self.assertEqual(state["task_evolution_status"], "early_exit_eligible")
            self.assertEqual(
                state["working_bundle"][0]["candidate"]["status"], "approved"
            )

    def test_problem_switch_requires_structured_evidence(self) -> None:
        candidate = {"candidate_id": "c1", "problem_id": "p1"}
        panel_value = {
            "panel_id": "dev",
            "panel_role": "development",
        }
        value = retrospective(candidate, panel_value, "switch_problem")
        state = {
            "current_candidate": {
                "candidate_id": "c1",
                "problem_id": "p1",
                "status": "development_retrospective_pending",
                "verifier_reports": {"development": {"panel_id": "dev"}},
            }
        }
        with self.assertRaisesRegex(ValueError, "supported reason"):
            V5.validate_retrospective(value, state)
        value["switch_justification"] = {
            "reason_code": "hypothesis_disproved",
            "evidence": ["success counterexample contradicts the hypothesis"],
        }
        V5.validate_retrospective(value, state)

    def test_advance_evolution_switches_to_next_problem_with_budget(self) -> None:
        state = V5.create_state(config(), "MoveCube", "abc123")
        state["problems"] = {
            "MoveCube-P-primary": {
                "problem_id": "MoveCube-P-primary",
                "status": "deferred",
                "support": 10,
                "priority_score": 10.0,
            },
            "MoveCube-P-secondary": {
                "problem_id": "MoveCube-P-secondary",
                "status": "active",
                "support": 4,
                "priority_score": 4.0,
            },
        }
        state["problem_attempts"] = {
            "MoveCube-P-primary": 1,
            "MoveCube-P-secondary": 0,
        }
        state["active_problem_id"] = None
        selected = V5.advance_evolution(
            state, config(), exclude_problem_id="MoveCube-P-primary"
        )
        self.assertEqual(selected, "MoveCube-P-secondary")
        self.assertEqual(state["active_problem_id"], "MoveCube-P-secondary")
        self.assertEqual(state["stage"], "problem_selected")
        self.assertEqual(state["task_evolution_status"], "active")
        self.assertEqual(state["remaining_problem_budget"], 5)

    def test_advance_evolution_marks_terminal_only_without_selectable_problem(self) -> None:
        state = V5.create_state(config(), "MoveCube", "abc123")
        state["problems"] = {
            "MoveCube-P-primary": {
                "problem_id": "MoveCube-P-primary",
                "status": "deferred",
                "support": 10,
                "priority_score": 10.0,
            }
        }
        state["problem_attempts"] = {"MoveCube-P-primary": 1}
        state["active_problem_id"] = None
        selected = V5.advance_evolution(
            state, config(), exclude_problem_id="MoveCube-P-primary"
        )
        self.assertIsNone(selected)
        self.assertEqual(state["stage"], "evolution_terminal")
        self.assertEqual(state["task_evolution_status"], "no_improvement")
        self.assertEqual(
            state["task_terminal_reason"], "no_selectable_problem_remaining"
        )

    def test_closing_last_round_records_no_improvement_without_candidate(self) -> None:
        problem_id = "MoveCube-P-primary"
        state = self.state_with_problem(problem_id)
        state["remaining_problem_budget"] = 0
        state["current_candidate"] = {
            "round": 5,
            "candidate_id": "candidate-final",
            "problem_id": problem_id,
            "candidate": {"candidate_id": "candidate-final"},
        }
        V5._close_round(
            state,
            "switch_problem",
            "final repair did not change paired outcomes",
            "deferred",
        )
        self.assertEqual(state["stage"], "evolution_budget_exhausted")
        self.assertEqual(state["task_evolution_status"], "no_improvement")
        self.assertEqual(
            state["task_terminal_reason"],
            "round_budget_exhausted_no_candidate",
        )
        self.assertIsNone(state["active_problem_id"])

    def test_closing_last_round_keeps_fallback_selectable(self) -> None:
        problem_id = "MoveCube-P-primary"
        state = self.state_with_problem(problem_id)
        state["remaining_problem_budget"] = 0
        state["fallback_candidates"] = [
            {"candidate": {"candidate_id": "fallback-one"}}
        ]
        state["current_candidate"] = {
            "round": 5,
            "candidate_id": "candidate-final",
            "problem_id": problem_id,
            "candidate": {"candidate_id": "candidate-final"},
        }
        V5._close_round(
            state,
            "retain_fallback",
            "retain the final +1 candidate",
            "deferred",
        )
        self.assertEqual(
            state["task_evolution_status"], "fallback_selection_pending"
        )
        self.assertEqual(
            state["task_terminal_reason"],
            "round_budget_exhausted_fallback_available",
        )
        selected = V4.select_final_fallback(state, "fallback-one")
        self.assertEqual(selected["candidate"]["candidate_id"], "fallback-one")

    def test_switch_decision_auto_advances_when_config_is_supplied(self) -> None:
        state = V5.create_state(config(), "MoveCube", "abc123")
        state["problems"] = {
            "MoveCube-P-primary": {
                "problem_id": "MoveCube-P-primary",
                "status": "active",
                "support": 10,
                "priority_score": 10.0,
            },
            "MoveCube-P-secondary": {
                "problem_id": "MoveCube-P-secondary",
                "status": "active",
                "support": 4,
                "priority_score": 4.0,
            },
        }
        state["problem_attempts"] = {
            "MoveCube-P-primary": 1,
            "MoveCube-P-secondary": 0,
        }
        state["active_problem_id"] = "MoveCube-P-primary"
        state["current_candidate"] = {
            "round": 1,
            "problem_id": "MoveCube-P-primary",
            "candidate_id": "candidate-primary-r1",
            "status": "development_decision_pending",
            "verifier_reports": {
                "development": {"panel_id": "development-panel"}
            },
            "retrospectives": {
                "development": {"recommended_action": "switch_problem"}
            },
        }
        record = V5.decide_candidate(
            state,
            "switch_problem",
            "candidate had no effect and another cluster remains",
            config(),
        )
        self.assertEqual(record["next_problem_id"], "MoveCube-P-secondary")
        self.assertEqual(state["active_problem_id"], "MoveCube-P-secondary")
        self.assertEqual(state["stage"], "problem_selected")
        self.assertEqual(state["problems"]["MoveCube-P-primary"]["status"], "deferred")

    def test_switch_at_problem_attempt_limit_marks_problem_exhausted(self) -> None:
        state = V5.create_state(config(), "MoveCube", "abc123")
        state["problems"] = {
            "MoveCube-P-primary": {
                "problem_id": "MoveCube-P-primary",
                "status": "active",
                "support": 10,
                "priority_score": 10.0,
            },
            "MoveCube-P-secondary": {
                "problem_id": "MoveCube-P-secondary",
                "status": "unactionable",
                "support": 4,
                "priority_score": 4.0,
            },
        }
        state["problem_attempts"] = {
            "MoveCube-P-primary": state["max_attempts_per_problem"],
            "MoveCube-P-secondary": 0,
        }
        state["active_problem_id"] = "MoveCube-P-primary"
        state["current_candidate"] = {
            "round": state["max_attempts_per_problem"],
            "problem_id": "MoveCube-P-primary",
            "candidate_id": "candidate-primary-final",
            "status": "development_decision_pending",
            "verifier_reports": {
                "development": {"panel_id": "development-panel"}
            },
            "retrospectives": {
                "development": {"recommended_action": "switch_problem"}
            },
        }
        record = V5.decide_candidate(
            state,
            "switch_problem",
            "problem attempt budget exhausted without a corrected failure",
            config(),
        )
        self.assertIsNone(record["next_problem_id"])
        self.assertEqual(
            state["problems"]["MoveCube-P-primary"]["status"], "exhausted"
        )
        self.assertEqual(state["stage"], "evolution_terminal")
        self.assertEqual(state["task_evolution_status"], "no_improvement")

    def test_fallback_is_retained_while_revising_same_problem(self) -> None:
        state = V5.create_state(config(), "MoveCube", "abc123")
        state["current_candidate"] = {
            "round": 1,
            "problem_id": "MoveCube-P-primary",
            "candidate_id": "candidate-fallback",
            "candidate": {
                "candidate_id": "candidate-fallback",
                "status": "candidate",
            },
            "candidate_manifest_path": "/tmp/candidate-fallback.json",
            "candidate_manifest_sha256": "manifest-hash",
        }
        report = {
            "acceptance_tier": "fallback",
            "fallback_eligible": True,
            "candidate_successes": 1,
            "paired_verifier": {
                "net_improvements": 1,
                "regressed": 0,
            },
        }
        first = V5._retain_current_fallback(state, report)
        second = V5._retain_current_fallback(state, report)
        self.assertEqual(first, second)
        self.assertEqual(len(state["fallback_candidates"]), 1)
        self.assertEqual(
            state["fallback_candidates"][0]["candidate"]["candidate_id"],
            "candidate-fallback",
        )
        self.assertEqual(state["fallback_candidates"][0]["net_improvements"], 1)

    def test_historical_fallback_reconciliation_restores_budget_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "candidate.json"
            manifest.write_text(
                json.dumps(
                    {
                        "candidate_id": "candidate-historical",
                        "status": "candidate",
                    }
                )
            )
            state = V5.create_state(config(), "MoveCube", "abc123")
            state["remaining_problem_budget"] = 0
            state["task_evolution_status"] = "no_improvement"
            state["task_terminal_reason"] = "round_budget_exhausted_no_candidate"
            state["round_history"] = [
                {
                    "round": 1,
                    "problem_id": "MoveCube-P-primary",
                    "candidate_id": "candidate-historical",
                    "candidate_manifest_path": str(manifest),
                    "candidate_manifest_sha256": "manifest-hash",
                    "verifier_reports": {
                        "development": {
                            "acceptance_tier": "fallback",
                            "fallback_eligible": True,
                            "candidate_successes": 1,
                            "paired_verifier": {
                                "net_improvements": 1,
                                "regressed": 0,
                            },
                        }
                    },
                }
            ]
            added = V5.reconcile_historical_fallbacks(state)
            self.assertEqual(added, ["candidate-historical"])
            self.assertEqual(
                state["task_evolution_status"], "fallback_selection_pending"
            )
            self.assertEqual(
                state["task_terminal_reason"],
                "round_budget_exhausted_fallback_available",
            )


if __name__ == "__main__":
    unittest.main()
