from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import self_evolve_v4 as V4
from build_self_evolve_v4_candidate import (
    compose_candidates,
    materialize_candidate,
)
from interventions import load_runtime_intervention_bundle
from interventions.bundle import sha256_file
from prepare_self_evolve_v4_repair_request import build_request
from self_evolve_v4 import reassess_fullstack_actionability
from text_progress_planner import (
    PATTERNLOCK_CANONICAL_DIRECTIONS,
    PICKHIGHLIGHT_POINT_PICKUP_TEMPLATES,
    CodexTextProgressPlanner,
    ensure_pickhighlight_pickup_candidates,
    ensure_patternlock_direction_candidates,
    evaluate_button_swap_press_grounding_recovery,
    evaluate_pickhighlight_pickup_grounding_recovery,
    evaluate_unmask_pickup_grounding_recovery,
    reconcile_button_swap_press_state_after_runtime_overrides,
)
from verify_self_evolve_v4_candidate import evaluate


EXAMPLE = (
    SCRIPT_DIR
    / "examples"
    / "self_evolve_v4"
    / "movecube_v1_compatible"
    / "candidate.json"
)


def config() -> dict:
    return {
        "experiment_id": "test-v4",
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
        "verification": {
            "required_train_attempts": 3,
            "required_val_attempts": 3,
            "minimum_total_attempts": 6,
            "maximum_total_attempts": 12,
            "require_verifier_episode_ids": True,
            "reject_analysis_consumed_samples": True,
            "evolution_acceptance_ladder": [
                {"tier": "strict", "min_net_improvements": 2, "max_regressions": 0},
                {
                    "tier": "fallback",
                    "min_net_improvements": 1,
                    "max_regressions": 0,
                    "only_if_no_strict_candidate": True,
                },
            ],
            "early_exit_tiers": ["strict", "perfect"],
            "fallback_release_policy": "only_after_total_round_budget_exhausted",
        },
    }


def authoring() -> dict:
    return {
        "skill_name": "self-evolve-intervention-author",
        "source": "unit-test train/val diagnosis",
        "analysis_inputs": [],
        "runtime_uses_episode_identity": False,
        "offline_acceptance_verifier_modified": False,
    }


def component(
    component_id: str,
    component_type: str,
    task: str,
    *,
    priority: int,
    config_value: dict,
    claims: list[str],
    namespace: str | None = None,
    artifacts: dict | None = None,
) -> dict:
    return {
        "component_id": component_id,
        "type": component_type,
        "task_scope": [task],
        "priority": priority,
        "state_namespace": namespace,
        "claims": claims,
        "config": config_value,
        "artifacts": artifacts or {},
    }


def candidate_manifest(task: str, candidate_id: str, components: list[dict]) -> dict:
    return {
        "schema_version": "self-evolve-v4-candidate-intervention-v1",
        "candidate_id": candidate_id,
        "revision": 1,
        "status": "candidate",
        "problem_id": f"{task}-P-test",
        "task_scope": [task],
        "base_profile": "retrieval_v8_5_refine_then_lock",
        "diagnosis": {
            "first_divergence": "A runtime-observable proposal violates the task contract.",
            "responsible_module": "planner_executor_interface",
            "runtime_observable": True,
            "repair_surfaces": [row["type"] for row in components],
        },
        "components": components,
        "authoring": authoring(),
    }


class SelfEvolveV4FullStackTests(unittest.TestCase):
    def test_pickhighlight_all_pickup_templates_retrieval_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            candidate = candidate_manifest(
                "PickHighlight",
                "pickhighlight-retrieval-test",
                [
                    component(
                        "pickhighlight-all-pickups",
                        "retrieval_overlay",
                        "PickHighlight",
                        priority=20,
                        config_value={
                            "policy": "pickhighlight_all_pickup_templates"
                        },
                        claims=["retrieval:PickHighlight:all-point-pickups"],
                    )
                ],
            )
            source_path = source / "candidate.json"
            source_path.write_text(json.dumps(candidate), encoding="utf-8")
            materialized = materialize_candidate(source_path, root / "candidate")
            bundle = compose_candidates(
                [materialized], root / "bundle", bundle_id="pickhighlight", revision=1
            )
            plan = load_runtime_intervention_bundle(bundle)
            self.assertTrue(plan.pickhighlight_all_pickup_templates)
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle,
                verify_login=False,
            )
            self.assertTrue(planner.enable_pickhighlight_all_pickup_templates)
            self.assertIsNotNone(planner.skill_retriever)
            candidates = ensure_pickhighlight_pickup_candidates(
                [], retriever=planner.skill_retriever
            )
            self.assertEqual(len(candidates), 12)
            self.assertEqual(
                {row["canonical_template"] for row in candidates},
                set(PICKHIGHLIGHT_POINT_PICKUP_TEMPLATES),
            )
            self.assertTrue(
                all(
                    row["injected_by_experience"]
                    == "pickhighlight_all_pickup_templates"
                    for row in candidates
                )
            )

    def test_pickhighlight_grounding_recovery_bundle_and_runtime_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            gate_config = {
                "trigger_after_failed_pickups": 2,
                "minimum_point_delta_norm": 10,
                "max_forced_regrounds_per_target": 2,
            }
            candidate = candidate_manifest(
                "PickHighlight",
                "pickhighlight-grounding-test",
                [
                    component(
                        "pickhighlight-grounding-gate",
                        "pickhighlight_pickup_grounding_recovery_gate",
                        "PickHighlight",
                        priority=20,
                        config_value=gate_config,
                        claims=["grounding:PickHighlight:pickup-recovery"],
                        namespace="pickhighlight-grounding-recovery",
                    )
                ],
            )
            source_path = source / "candidate.json"
            source_path.write_text(json.dumps(candidate), encoding="utf-8")
            materialized = materialize_candidate(source_path, root / "candidate")
            bundle = compose_candidates(
                [materialized], root / "bundle", bundle_id="pickhighlight", revision=1
            )
            plan = load_runtime_intervention_bundle(bundle)
            self.assertEqual(plan.pickhighlight_pickup_grounding_recovery, gate_config)
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle,
                verify_login=False,
            )
            self.assertEqual(
                planner.pickhighlight_pickup_grounding_recovery_config, gate_config
            )

    def test_pickhighlight_grounding_recovery_rejects_stale_point(self) -> None:
        config_value = {
            "trigger_after_failed_pickups": 2,
            "minimum_point_delta_norm": 10,
            "max_forced_regrounds_per_target": 2,
        }
        state = {
            "target_signature": "second::red",
            "consecutive_failed_pickups": 1,
            "last_dispatched_point_norm": [112, 155],
            "recent_failed_points_norm": [[110, 154]],
            "forced_regrounds": 0,
        }
        failure = {
            "recent_status": "stagnant",
            "execution_state": {
                "held_object": "none",
                "object_support": "table",
                "gripper_state": "open",
            },
        }
        with self.assertRaisesRegex(ValueError, "repeats a recent failed location"):
            evaluate_pickhighlight_pickup_grounding_recovery(
                state=state,
                previous_subtask=(
                    "pick up the second highlighted cube at <112, 155>, which is red"
                ),
                proposed_subtask=(
                    "pick up the second highlighted cube at <114, 151>, which is red"
                ),
                proposed_points=[{"x_norm": 114, "y_norm": 151}],
                response=failure,
                config=config_value,
            )
        updated, audit = evaluate_pickhighlight_pickup_grounding_recovery(
            state=state,
            previous_subtask=(
                "pick up the second highlighted cube at <112, 155>, which is red"
            ),
            proposed_subtask=(
                "pick up the second highlighted cube at <130, 170>, which is red"
            ),
            proposed_points=[{"x_norm": 130, "y_norm": 170}],
            response=failure,
            config=config_value,
        )
        self.assertEqual(audit["decision"], "accepted_reground")
        self.assertEqual(updated["forced_regrounds"], 1)
        self.assertIn([112, 155], updated["recent_failed_points_norm"])

    def test_unmask_grounding_recovery_bundle_and_runtime_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            gate_config = {
                "trigger_after_failed_pickups": 2,
                "minimum_point_delta_norm": 80,
                "max_forced_regrounds_per_target": 2,
            }
            candidate = candidate_manifest(
                "ButtonUnmask",
                "button-unmask-grounding-test",
                [
                    component(
                        "button-unmask-grounding-gate",
                        "unmask_pickup_grounding_recovery_gate",
                        "ButtonUnmask",
                        priority=20,
                        config_value=gate_config,
                        claims=["grounding:ButtonUnmask:pickup-recovery"],
                        namespace="button-unmask-grounding-recovery",
                    )
                ],
            )
            source_path = source / "candidate.json"
            source_path.write_text(json.dumps(candidate), encoding="utf-8")
            materialized = materialize_candidate(source_path, root / "candidate")
            bundle = compose_candidates(
                [materialized], root / "bundle", bundle_id="button-unmask", revision=1
            )
            plan = load_runtime_intervention_bundle(bundle)
            self.assertEqual(plan.unmask_pickup_grounding_recovery, gate_config)
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle,
                verify_login=False,
            )
            self.assertEqual(
                planner.unmask_pickup_grounding_recovery_config, gate_config
            )

    def test_unmask_grounding_recovery_rejects_recent_failed_points(self) -> None:
        config_value = {
            "trigger_after_failed_pickups": 2,
            "minimum_point_delta_norm": 80,
            "max_forced_regrounds_per_target": 2,
        }
        state = {
            "target_color": "red",
            "consecutive_failed_pickups": 1,
            "last_dispatched_point_norm": [120, 100],
            "recent_failed_points_norm": [[160, 120]],
            "forced_regrounds": 0,
        }
        failure = {
            "execution_state": {
                "held_object": "none",
                "object_support": "table",
                "gripper_state": "open",
            },
            "refine_check": {"decision": "not_satisfied"},
        }
        criterion = {"criterion_id": "unmask_pickup_identity_and_separation"}
        with self.assertRaisesRegex(ValueError, "repeats a recent failed location"):
            evaluate_unmask_pickup_grounding_recovery(
                state=state,
                previous_subtask=(
                    "pick up the container at <120, 100> that hides the red cube"
                ),
                proposed_subtask=(
                    "pick up the container at <155, 118> that hides the red cube"
                ),
                proposed_points=[{"x_norm": 155, "y_norm": 118}],
                response=failure,
                refine_criterion=criterion,
                config=config_value,
            )
        updated, audit = evaluate_unmask_pickup_grounding_recovery(
            state=state,
            previous_subtask=(
                "pick up the container at <120, 100> that hides the red cube"
            ),
            proposed_subtask=(
                "pick up the container at <280, 220> that hides the red cube"
            ),
            proposed_points=[{"x_norm": 280, "y_norm": 220}],
            response=failure,
            refine_criterion=criterion,
            config=config_value,
        )
        self.assertEqual(audit["decision"], "accepted_reground")
        self.assertEqual(updated["forced_regrounds"], 1)
        self.assertIn([120, 100], updated["recent_failed_points_norm"])

    def test_button_swap_press_recovery_bundle_and_runtime_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            gate_config = {
                "trigger_after_failed_presses": 2,
                "minimum_point_delta_norm": 40,
                "minimum_completed_button_delta_norm": 100,
                "max_forced_regrounds_per_button": 3,
            }
            candidate = candidate_manifest(
                "ButtonUnmaskSwap",
                "button-swap-press-grounding-test",
                [
                    component(
                        "button-swap-press-grounding-gate",
                        "button_swap_press_grounding_recovery_gate",
                        "ButtonUnmaskSwap",
                        priority=20,
                        config_value=gate_config,
                        claims=["grounding:ButtonUnmaskSwap:distinct-press-recovery"],
                        namespace="button-swap-press-grounding-recovery",
                    )
                ],
            )
            source_path = source / "candidate.json"
            source_path.write_text(json.dumps(candidate), encoding="utf-8")
            materialized = materialize_candidate(source_path, root / "candidate")
            bundle = compose_candidates(
                [materialized], root / "bundle", bundle_id="button-swap", revision=1
            )
            plan = load_runtime_intervention_bundle(bundle)
            self.assertEqual(
                plan.button_swap_press_grounding_recovery, gate_config
            )
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle,
                verify_login=False,
            )
            self.assertEqual(
                planner.button_swap_press_grounding_recovery_config,
                gate_config,
            )
            planner.start_episode(
                run_dir=root / "planner-run",
                detailed_task_description=(
                    "Press two distinct buttons, then unmask the requested cubes."
                ),
                task="ButtonUnmaskSwap",
                planner_guidance={},
            )
            self.assertEqual(
                planner.button_swap_press_grounding_recovery_state[
                    "completed_points_norm"
                ],
                [],
            )

    def test_button_swap_press_recovery_can_start_under_runtime_gate_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_config = {
                "trigger_after_failed_presses": 2,
                "minimum_point_delta_norm": 40,
                "minimum_completed_button_delta_norm": 100,
                "max_forced_regrounds_per_button": 3,
            }
            candidate = candidate_manifest(
                "ButtonUnmaskSwap",
                "button-swap-press-policy-test",
                [
                    component(
                        "button-swap-press-policy-gate",
                        "button_swap_press_grounding_recovery_gate",
                        "ButtonUnmaskSwap",
                        priority=20,
                        config_value=gate_config,
                        claims=["grounding:ButtonUnmaskSwap:policy-test"],
                        namespace="button-swap-press-policy-test",
                    )
                ],
            )
            candidate["authoring"]["analysis_inputs"] = [
                {"dataset_split": "train", "episode_id": 1}
            ]
            source = root / "candidate.json"
            source.write_text(json.dumps(candidate), encoding="utf-8")
            materialized = materialize_candidate(source, root / "materialized")
            policy_config = config()
            policy_config["task_order"] = ["ButtonUnmaskSwap"]
            policy_config["intervention_policy"] = {"allow_runtime_gate": True}
            state = V4.create_state(
                policy_config, "ButtonUnmaskSwap", "policy-test-commit"
            )
            problem_id = "ButtonUnmaskSwap-P-test"
            state["problems"] = {
                problem_id: {"problem_id": problem_id, "status": "active"}
            }
            state["problem_attempts"] = {problem_id: 0}
            state["active_problem_id"] = problem_id
            current = V4.start_candidate(state, materialized)
            self.assertEqual(
                current["candidate_id"], "button-swap-press-policy-test"
            )

    def test_button_swap_press_recovery_rejects_stale_failed_point(self) -> None:
        config_value = {
            "trigger_after_failed_presses": 2,
            "minimum_point_delta_norm": 40,
            "minimum_completed_button_delta_norm": 100,
            "max_forced_regrounds_per_button": 3,
        }
        state = {
            "completed_points_norm": [[260, 400]],
            "counted_action_instances": ["first-press"],
            "active_ordinal": 2,
            "last_dispatched_point_norm": [265, 590],
            "recent_failed_points_norm": [],
            "consecutive_failed_presses": 1,
            "forced_regrounds": 0,
        }
        failure = {
            "recent_status": "stagnant",
            "execution_state": {
                "held_object": "none",
                "object_support": "not_applicable",
                "gripper_state": "open",
            },
            "refine_check": {"decision": "not_satisfied"},
        }
        criterion = {"criterion_id": "button_unmask_press_local_event"}
        with self.assertRaisesRegex(ValueError, "near-identical button point"):
            evaluate_button_swap_press_grounding_recovery(
                state=state,
                previous_subtask="press the second button at <68, 151>",
                proposed_subtask="press the second button at <68, 151>",
                proposed_points=[{"x_norm": 267, "y_norm": 592}],
                response=failure,
                refine_criterion=criterion,
                action_instance_id="second-press",
                config=config_value,
            )
        updated, audit = evaluate_button_swap_press_grounding_recovery(
            state=state,
            previous_subtask="press the second button at <68, 151>",
            proposed_subtask="press the second button at <82, 165>",
            proposed_points=[{"x_norm": 320, "y_norm": 650}],
            response=failure,
            refine_criterion=criterion,
            action_instance_id="second-press",
            config=config_value,
        )
        self.assertEqual(audit["decision"], "accepted_reground")
        self.assertEqual(updated["forced_regrounds"], 1)
        self.assertIn([265, 590], updated["recent_failed_points_norm"])

    def test_button_swap_press_ledger_defers_proposal_replayed_by_refinement_lock(
        self,
    ) -> None:
        config_value = {
            "trigger_after_failed_presses": 2,
            "minimum_point_delta_norm": 40,
            "minimum_completed_button_delta_norm": 100,
            "max_forced_regrounds_per_button": 3,
        }
        state = {
            "completed_points_norm": [],
            "counted_action_instances": [],
            "active_ordinal": 1,
            "last_dispatched_point_norm": [410, 249],
            "recent_failed_points_norm": [],
            "consecutive_failed_presses": 0,
            "forced_regrounds": 0,
        }
        proposed, guard = evaluate_button_swap_press_grounding_recovery(
            state=state,
            previous_subtask="press the first button at <64, 105>",
            proposed_subtask="press the second button at <68, 150>",
            proposed_points=[{"x_norm": 586, "y_norm": 266}],
            response={
                "recent_status": "completed",
                "refine_check": {"decision": "satisfied"},
            },
            refine_criterion={"criterion_id": "button_unmask_press_local_event"},
            action_instance_id="first-button-action",
            config=config_value,
        )
        self.assertEqual(proposed["active_ordinal"], 2)
        committed, reconciled = (
            reconcile_button_swap_press_state_after_runtime_overrides(
                proposed_state=proposed,
                guard=guard,
                planner_proposed_subtask="press the second button at <68, 150>",
                dispatched_subtask="press the first button at <64, 105>",
                override_source="refinement_lock",
            )
        )
        self.assertEqual(committed["completed_points_norm"], [[410, 249]])
        self.assertIsNone(committed["active_ordinal"])
        self.assertIsNone(committed["last_dispatched_point_norm"])
        self.assertEqual(reconciled["decision"], "deferred_by_runtime_override")
        self.assertEqual(
            reconciled["runtime_override"]["source"], "refinement_lock"
        )

    def test_button_swap_press_recovery_rejects_completed_button_location(self) -> None:
        config_value = {
            "trigger_after_failed_presses": 2,
            "minimum_point_delta_norm": 40,
            "minimum_completed_button_delta_norm": 100,
            "max_forced_regrounds_per_button": 3,
        }
        state = {
            "completed_points_norm": [[260, 400]],
            "counted_action_instances": ["first-press"],
            "active_ordinal": None,
            "last_dispatched_point_norm": None,
            "recent_failed_points_norm": [],
            "consecutive_failed_presses": 0,
            "forced_regrounds": 0,
        }
        with self.assertRaisesRegex(ValueError, "already completed button"):
            evaluate_button_swap_press_grounding_recovery(
                state=state,
                previous_subtask="press the first button at <66, 102>",
                proposed_subtask="press the second button at <67, 104>",
                proposed_points=[{"x_norm": 266, "y_norm": 410}],
                response={
                    "recent_status": "completed",
                    "refine_check": {"decision": "satisfied"},
                },
                refine_criterion={
                    "criterion_id": "button_unmask_press_local_event"
                },
                action_instance_id="first-press",
                config=config_value,
            )
        updated, audit = evaluate_button_swap_press_grounding_recovery(
            state=state,
            previous_subtask="press the first button at <66, 102>",
            proposed_subtask="press the second button at <67, 104>",
            proposed_points=[{"x_norm": 266, "y_norm": 410}],
            response={
                "recent_status": "completed",
                "refine_check": {"decision": "satisfied"},
            },
            refine_criterion={"criterion_id": "button_unmask_press_local_event"},
            action_instance_id="first-press",
            config=config_value,
            allow_retry_budget_bypass=True,
        )
        self.assertEqual(
            audit["decision"],
            "untrusted_completed_button_point_after_validation_retry_budget",
        )
        self.assertTrue(audit["untrusted_dispatch"])
        self.assertIsNone(updated["active_ordinal"])
        self.assertIsNone(updated["last_dispatched_point_norm"])

    def test_button_swap_press_recovery_bounds_stale_ordinal_and_early_pickup(
        self,
    ) -> None:
        config_value = {
            "trigger_after_failed_presses": 2,
            "minimum_point_delta_norm": 40,
            "minimum_completed_button_delta_norm": 100,
            "max_forced_regrounds_per_button": 3,
        }
        state = {
            "completed_points_norm": [[260, 400]],
            "counted_action_instances": ["first-press"],
            "active_ordinal": None,
            "last_dispatched_point_norm": None,
            "recent_failed_points_norm": [],
            "consecutive_failed_presses": 0,
            "forced_regrounds": 0,
        }
        for proposed_subtask, proposed_points, expected_decision in (
            (
                "press the first button at <66, 102>",
                [{"x_norm": 260, "y_norm": 400}],
                "untrusted_stale_ordinal_after_validation_retry_budget",
            ),
            (
                "pick up the container at <95, 93> that hides the blue cube",
                [{"x_norm": 363, "y_norm": 371}],
                "untrusted_early_pickup_after_validation_retry_budget",
            ),
        ):
            updated, audit = evaluate_button_swap_press_grounding_recovery(
                state=state,
                previous_subtask="press the first button at <66, 102>",
                proposed_subtask=proposed_subtask,
                proposed_points=proposed_points,
                response={
                    "recent_status": "stagnant",
                    "refine_check": {"decision": "not_satisfied"},
                },
                refine_criterion={
                    "criterion_id": "button_unmask_press_local_event"
                },
                action_instance_id="first-press",
                config=config_value,
                allow_retry_budget_bypass=True,
            )
            self.assertEqual(audit["decision"], expected_decision)
            self.assertTrue(audit["untrusted_dispatch"])
            self.assertEqual(updated, state)

    def test_button_swap_press_recovery_blocks_early_pickup(self) -> None:
        config_value = {
            "trigger_after_failed_presses": 2,
            "minimum_point_delta_norm": 40,
            "minimum_completed_button_delta_norm": 100,
            "max_forced_regrounds_per_button": 3,
        }
        state = {
            "completed_points_norm": [[260, 400]],
            "counted_action_instances": ["first-press"],
            "active_ordinal": None,
            "last_dispatched_point_norm": None,
            "recent_failed_points_norm": [],
            "consecutive_failed_presses": 0,
            "forced_regrounds": 0,
        }
        with self.assertRaisesRegex(ValueError, "pickup before two distinct"):
            evaluate_button_swap_press_grounding_recovery(
                state=state,
                previous_subtask="press the first button at <66, 102>",
                proposed_subtask=(
                    "pick up the container at <120, 170> that hides the red cube"
                ),
                proposed_points=[{"x_norm": 470, "y_norm": 660}],
                response={
                    "recent_status": "completed",
                    "refine_check": {"decision": "satisfied"},
                },
                refine_criterion={
                    "criterion_id": "button_unmask_press_local_event"
                },
                action_instance_id="first-press",
                config=config_value,
            )

    def materialized_v1(self, root: Path) -> Path:
        return materialize_candidate(EXAMPLE, root / "candidate")

    def test_empty_bundle_has_reusable_v85_behavior_id(self) -> None:
        planner = CodexTextProgressPlanner(
            prompt_profile="self_evolve_v4_fullstack_trainval",
            verify_login=False,
        )
        self.assertEqual(
            planner.rollout_behavior_id,
            "v8.5-refine-then-lock-empty-intervention-v1",
        )

    def test_patternlock_retrieval_overlay_only_adds_frozen_directions(self) -> None:
        class Retriever:
            entries = [
                {
                    "template_id": f"direction-{index}",
                    "canonical_template": canonical,
                    "tasks": ["PatternLock"],
                    "point_arity": 0,
                }
                for index, canonical in enumerate(PATTERNLOCK_CANONICAL_DIRECTIONS)
            ]

        initial = [{"rank": 1, **Retriever.entries[0]}]
        result = ensure_patternlock_direction_candidates(
            initial, retriever=Retriever()
        )
        self.assertEqual(len(result), 8)
        self.assertEqual(
            {row["canonical_template"] for row in result},
            set(PATTERNLOCK_CANONICAL_DIRECTIONS),
        )
        self.assertNotIn("injected_by_experience", result[0])
        self.assertTrue(
            all(
                row.get("injected_by_experience") == "patternlock_all_directions"
                for row in result[1:]
            )
        )

    def test_patternlock_overlay_is_applied_before_prompt_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.json"
            source.write_text(
                json.dumps(
                    candidate_manifest(
                        "PatternLock",
                        "patternlock-retrieval-test",
                        [
                            component(
                                "patternlock-directions",
                                "retrieval_overlay",
                                "PatternLock",
                                priority=10,
                                config_value={"policy": "patternlock_all_directions"},
                                claims=["retrieval:PatternLock:all-directions"],
                            )
                        ],
                    )
                ),
                encoding="utf-8",
            )
            materialized = materialize_candidate(source, root / "materialized")
            bundle = compose_candidates(
                [materialized], root / "bundle", bundle_id="patternlock", revision=1
            )
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle,
                verify_login=False,
            )
            planner.start_episode(
                run_dir=root / "episode",
                detailed_task_description="Trace the ordered PatternLock grid.",
                task="PatternLock",
                planner_guidance={},
            )
            prompt = planner._render_prompt("r_f0000: frame", mode="normal")
            for canonical in PATTERNLOCK_CANONICAL_DIRECTIONS:
                self.assertIn(f'"canonical_template": "{canonical}"', prompt)
            self.assertEqual(len(planner.current_retrieval_candidates), 8)

    def test_fullstack_reopens_supported_builtin_repair(self) -> None:
        discovery = {
            "task": "MoveCube",
            "clusters": [{
                "problem_id": "MoveCube-P-completion",
                "status": "unactionable",
                "support": 3,
                "runtime_observable_ratio": 1.0,
                "signature": {
                    "task_variant": "peg_push",
                    "failure_stage": "completion_judgment",
                    "primary_problem": "completion_error",
                    "responsible_module": "completion_judge",
                },
                "first_divergence_examples": [
                    "The planner advances from peg pickup before held motion."
                ],
                "secondary_symptoms": [],
            }],
        }
        config = {
            "intervention_policy": {
                "allow_runtime_criterion": True,
                "allow_runtime_gate": True,
            }
        }
        revised = reassess_fullstack_actionability(discovery, config)
        cluster = revised["clusters"][0]
        self.assertEqual(cluster["status"], "active")
        self.assertIn(
            "bounded_transition_gate",
            cluster["v4_actionability_reassessment"]["suggested_component_types"],
        )
        self.assertEqual(discovery["clusters"][0]["status"], "unactionable")

    def test_v1_candidate_materializes_and_enables_existing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = self.materialized_v1(root)
            bundle_path = compose_candidates(
                [candidate_path], root / "bundle", bundle_id="v1-equivalent", revision=1
            )
            plan = load_runtime_intervention_bundle(bundle_path)
            self.assertIn("move_cube_peg_pickup_held_motion", plan.transition_gates)
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle_path,
                verify_login=False,
            )
            self.assertIsNone(planner.rollout_behavior_id)
            self.assertIn(
                "move_cube_peg_pickup_held_motion",
                planner.experience_transition_gate_config,
            )
            self.assertIn(
                "move_cube_peg_pickup_held_motion",
                {
                    row["criterion_id"]
                    for row in planner.refine_check_catalog["criteria"]
                },
            )
            run_dir = root / "episode"
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description="Use the peg to move the cube.",
                task="MoveCube",
                planner_guidance={},
            )
            episode_config = json.loads(
                (run_dir / "episode_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                episode_config["runtime_intervention_bundle"]["bundle_sha256"],
                load_runtime_intervention_bundle(bundle_path).bundle["bundle_sha256"],
            )
            self.assertTrue(
                (
                    run_dir
                    / "runtime_intervention_bundle"
                    / "runtime_intervention_bundle.json"
                ).is_file()
            )

    def test_component_claim_conflict_is_rejected_before_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = self.materialized_v1(root / "first")
            second_dir = root / "second"
            second_path = self.materialized_v1(second_dir)
            second = json.loads(second_path.read_text(encoding="utf-8"))
            second["candidate_id"] = "movecube_duplicate"
            second["components"][0]["component_id"] = "duplicate_criterion"
            second["components"][1]["component_id"] = "duplicate_gate"
            second_path.write_text(json.dumps(second), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicts between"):
                compose_candidates(
                    [first_path, second_path],
                    root / "bundle",
                    bundle_id="conflict",
                    revision=2,
                )

    def test_runtime_component_rejects_episode_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = self.materialized_v1(root)
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["components"][1]["config"]["situation_class"] = "episode 17"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "episode-specific"):
                load_runtime_intervention_bundle(
                    compose_candidates(
                        [candidate_path], root / "bundle", bundle_id="leak", revision=1
                    )
                )

    def test_state_controller_accepts_full_bundle_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = self.materialized_v1(root)
            candidate_value = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate_value["authoring"]["analysis_inputs"] = [
                {"dataset_split": "train", "episode_id": 1}
            ]
            candidate_path.write_text(json.dumps(candidate_value), encoding="utf-8")
            state = V4.create_state(config(), "MoveCube", "abc123")
            problem_id = "MoveCube-P-peg-pickup-unverified"
            state["problems"] = {
                problem_id: {
                    "problem_id": problem_id,
                    "status": "active",
                    "priority_score": 1.0,
                    "support": 3,
                }
            }
            state["problem_attempts"] = {problem_id: 0}
            state["active_problem_id"] = problem_id
            current = V4.start_candidate(state, candidate_path)
            report = {
                "schema_version": "self-evolve-v4-paired-verifier-report-v1",
                "candidate_id": current["candidate_id"],
                "problem_id": problem_id,
                "task": "MoveCube",
                "bundle_comparison": {
                    "base_bundle_sha256": current["base_bundle_sha256"],
                    "candidate_bundle_sha256": current["candidate_bundle_sha256"],
                },
                "sample_provenance": [
                    {
                        "record_id": f"{split}-{episode}",
                        "dataset_split": split,
                        "episode_id": episode,
                    }
                    for split, episode in (
                        ("train", 31),
                        ("train", 32),
                        ("train", 33),
                        ("val", 41),
                        ("val", 42),
                        ("val", 43),
                    )
                ],
                "pilot_verifier": {"regressed": 0},
                "strict_pass": True,
                "perfect_pass": False,
                "early_task_exit_eligible": True,
                "test_release_eligible": True,
                "acceptance_tier": "strict",
            }
            V4.decide_candidate(state, report, "accept", "paired improvement")
            self.assertEqual(state["bundle_revision"], 1)
            self.assertEqual(
                state["working_bundle"][0]["candidate"]["status"], "approved"
            )

    def test_v4_verifier_keeps_offline_thresholds_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = self.materialized_v1(root)
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            base_hash = V4.candidate_bundle_hash([])
            candidate_hash = V4.candidate_bundle_hash([candidate])
            rows = []
            for split, episodes in (("train", (1, 2, 3)), ("val", (4, 5, 6))):
                for episode in episodes:
                    improved = episode in {1, 4}
                    rows.append(
                        {
                            "record_id": f"{split}-{episode}",
                            "dataset_split": split,
                            "episode_id": episode,
                            "task": "MoveCube",
                            "baseline_success": not improved,
                            "experience_success": True,
                            "target_failure_corrected": improved,
                            "consumed_for_analysis": False,
                            "base_bundle_sha256": base_hash,
                            "candidate_bundle_sha256": candidate_hash,
                        }
                    )
            report = evaluate(
                candidate, rows, config(), candidate_dir=candidate_path.parent
            )
            self.assertEqual(report["acceptance_tier"], "strict")
            self.assertFalse(report["offline_acceptance_verifier_mutable_by_candidate"])

    def test_fallback_is_retained_without_terminating_or_mutating_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = self.materialized_v1(root)
            value = json.loads(candidate_path.read_text(encoding="utf-8"))
            value["authoring"]["analysis_inputs"] = [
                {"dataset_split": "train", "episode_id": 1}
            ]
            candidate_path.write_text(json.dumps(value), encoding="utf-8")
            state = V4.create_state(config(), "MoveCube", "abc123")
            problem_id = "MoveCube-P-peg-pickup-unverified"
            state["problems"] = {
                problem_id: {"problem_id": problem_id, "status": "active"}
            }
            state["problem_attempts"] = {problem_id: 0}
            state["active_problem_id"] = problem_id
            current = V4.start_candidate(state, candidate_path)
            report = {
                "schema_version": "self-evolve-v4-paired-verifier-report-v1",
                "candidate_id": current["candidate_id"],
                "problem_id": problem_id,
                "task": "MoveCube",
                "bundle_comparison": {
                    "base_bundle_sha256": current["base_bundle_sha256"],
                    "candidate_bundle_sha256": current["candidate_bundle_sha256"],
                },
                "sample_provenance": [
                    {"dataset_split": split, "episode_id": episode}
                    for split, episode in (
                        ("train", 31),
                        ("train", 32),
                        ("train", 33),
                        ("val", 41),
                        ("val", 42),
                        ("val", 43),
                    )
                ],
                "pilot_verifier": {"net_improvements": 1, "regressed": 0},
                "candidate_successes": 3,
                "strict_pass": False,
                "perfect_pass": False,
                "fallback_pool_eligible": True,
                "early_task_exit_eligible": False,
                "test_release_eligible": False,
                "acceptance_tier": "fallback",
            }
            V4.decide_candidate(
                state, report, "retain_fallback", "keep +1 and continue"
            )
            self.assertEqual(len(state["fallback_candidates"]), 1)
            self.assertEqual(state["working_bundle"], [])
            self.assertEqual(state["task_evolution_status"], "active")
            self.assertEqual(state["remaining_problem_budget"], 4)
            self.assertEqual(len(state["verifier_consumed_samples"]), 6)
            self.assertIn(
                {"dataset_split": "val", "episode_id": 43},
                state["analysis_consumed_samples"],
            )
            with self.assertRaisesRegex(ValueError, "Test snapshot requires"):
                V4.assert_test_snapshot_eligible(state)

    def test_perfect_six_of_six_can_terminate_without_net_gain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = self.materialized_v1(root)
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            base_hash = V4.candidate_bundle_hash([])
            candidate_hash = V4.candidate_bundle_hash([candidate])
            rows = []
            for split, episodes in (("train", (1, 2, 3)), ("val", (4, 5, 6))):
                for episode in episodes:
                    rows.append(
                        {
                            "record_id": f"{split}-{episode}",
                            "dataset_split": split,
                            "episode_id": episode,
                            "task": "MoveCube",
                            "baseline_success": True,
                            "experience_success": True,
                            "target_failure_corrected": False,
                            "consumed_for_analysis": False,
                            "base_bundle_sha256": base_hash,
                            "candidate_bundle_sha256": candidate_hash,
                        }
                    )
            report = evaluate(
                candidate, rows, config(), candidate_dir=candidate_path.parent
            )
            self.assertTrue(report["perfect_pass"])
            self.assertTrue(report["early_task_exit_eligible"])
            self.assertEqual(report["acceptance_tier"], "perfect")

    def test_fallback_can_only_be_selected_after_five_rounds(self) -> None:
        state = V4.create_state(config(), "MoveCube", "abc123")
        retained = {
            "candidate": {"candidate_id": "fallback-one", "status": "candidate"},
            "candidate_manifest_path": "/tmp/fallback.json",
        }
        state["fallback_candidates"] = [retained]
        state["remaining_problem_budget"] = 1
        with self.assertRaisesRegex(ValueError, "round exhaustion"):
            V4.select_final_fallback(state, "fallback-one")
        state["remaining_problem_budget"] = 0
        selected = V4.select_final_fallback(state, "fallback-one")
        self.assertEqual(selected["candidate"]["status"], "approved")
        self.assertEqual(
            state["task_evolution_status"], "budget_fallback_selected"
        )
        V4.assert_test_snapshot_eligible(state)

    def test_swing_gate_is_enabled_by_manifest_not_profile_fork(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            shutil.copy2(
                SCRIPT_DIR
                / "knowledge"
                / "action_success_criteria_v13_self_evolve_swing_binding.json",
                source / "criteria.json",
            )
            candidate = candidate_manifest(
                "SwingXtimes",
                "swing-binding-test",
                [
                    component(
                        "swing-side-criterion",
                        "criterion_overlay",
                        "SwingXtimes",
                        priority=10,
                        config_value={},
                        claims=["criterion:swing_target_side_visual_binding"],
                        artifacts={
                            "catalog": {
                                "path": "criteria.json",
                                "sha256": "AUTO",
                            }
                        },
                    ),
                    component(
                        "swing-side-binding",
                        "swing_visual_binding_gate",
                        "SwingXtimes",
                        priority=20,
                        config_value={"experience_id": "swing-exp", "revision": 1},
                        claims=["grounding:SwingXtimes:target-side"],
                    ),
                ],
            )
            source_path = source / "candidate.json"
            source_path.write_text(json.dumps(candidate), encoding="utf-8")
            materialized = materialize_candidate(source_path, root / "candidate")
            bundle = compose_candidates(
                [materialized], root / "bundle", bundle_id="swing", revision=1
            )
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle,
                verify_login=False,
            )
            self.assertTrue(planner.enable_swing_side_grounding_guard)
            self.assertIn(
                "swing_target_side_visual_binding",
                planner.experience_transition_gate_config,
            )

    def test_pick_bundle_enables_ledger_retrieval_ordinal_and_reinforcement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            criteria_source = (
                SCRIPT_DIR
                / "knowledge"
                / "action_success_criteria_v14_self_evolve_counting_ledger.json"
            )
            shutil.copy2(criteria_source, source / "criteria.json")
            components = [
                component(
                    "pick-counting-criteria",
                    "criterion_overlay",
                    "PickXtimes",
                    priority=10,
                    config_value={},
                    claims=["criteria:PickXtimes:counting"],
                    artifacts={"catalog": {"path": "criteria.json", "sha256": "AUTO"}},
                ),
                component(
                    "pick-counting-ledger",
                    "counting_event_ledger",
                    "PickXtimes",
                    priority=20,
                    config_value={
                        "explicit_first_pickup": True,
                        "exact_pickup_ordinal": True,
                    },
                    claims=["state:PickXtimes:counting-ledger"],
                    namespace="pick-counting-ledger",
                ),
                component(
                    "pick-first-retrieval",
                    "retrieval_overlay",
                    "PickXtimes",
                    priority=30,
                    config_value={"policy": "pickxtimes_explicit_first_pickup"},
                    claims=["retrieval:PickXtimes:first-pickup"],
                ),
                component(
                    "pick-next-retrieval",
                    "retrieval_overlay",
                    "PickXtimes",
                    priority=31,
                    config_value={"policy": "counting_live_ordinal"},
                    claims=["retrieval:PickXtimes:live-ordinal"],
                ),
                component(
                    "pick-ordinal-canonicalizer",
                    "action_canonicalizer",
                    "PickXtimes",
                    priority=40,
                    config_value={"policy": "counting_live_ordinal"},
                    claims=["canonicalizer:PickXtimes:live-ordinal"],
                ),
                component(
                    "pick-pickup-reinforcement",
                    "reinforcement_overlay",
                    "PickXtimes",
                    priority=50,
                    config_value={
                        "dispatch_counts": {"counting_requested_cube_pickup": 1}
                    },
                    claims=["reinforcement:PickXtimes:pickup"],
                ),
            ]
            candidate = candidate_manifest(
                "PickXtimes", "pick-fullstack-test", components
            )
            source_path = source / "candidate.json"
            source_path.write_text(json.dumps(candidate), encoding="utf-8")
            materialized = materialize_candidate(source_path, root / "candidate")
            bundle = compose_candidates(
                [materialized], root / "bundle", bundle_id="pick", revision=1
            )
            planner = CodexTextProgressPlanner(
                prompt_profile="self_evolve_v4_fullstack_trainval",
                runtime_intervention_bundle_path=bundle,
                verify_login=False,
            )
            self.assertTrue(planner.enable_counting_event_ledger_guard)
            self.assertTrue(planner.enable_counting_explicit_first_pickup)
            self.assertTrue(planner.enable_counting_exact_pickup_ordinal)
            self.assertEqual(
                planner.reinforcement_lock_config["counting_requested_cube_pickup"], 1
            )
            criterion_ids = {
                row["criterion_id"] for row in planner.refine_check_catalog["criteria"]
            }
            self.assertIn("counting_requested_cube_pickup", criterion_ids)

    def test_problem_router_preserves_train_val_evidence_for_authoring_skill(
        self,
    ) -> None:
        cluster = {
            "problem_id": "PickXtimes-P-count",
            "signature": {
                "task_variant": "PickXtimes",
                "failure_stage": "repetition counting",
                "primary_problem": "stale ordinal after release",
                "responsible_module": "planner_executor_interface",
            },
            "first_divergence_examples": ["second pickup reuses first ordinal"],
            "secondary_symptoms": ["premature stop"],
            "episode_refs": [
                {"dataset_split": "train", "episode_id": 3},
                {"dataset_split": "val", "episode_id": 7},
            ],
        }
        discovery = {"task": "PickXtimes", "clusters": [cluster]}
        records = [
            {
                "dataset_split": split,
                "episode_id": episode,
                "first_divergence": "stale ordinal",
                "responsible_module": "planner_executor_interface",
                "runtime_observable": True,
            }
            for split, episode in (("train", 3), ("val", 7))
        ]
        request = build_request(discovery, records, "PickXtimes-P-count")
        self.assertEqual(request["skill_name"], "self-evolve-intervention-author")
        self.assertIn("counting_event_ledger", request["suggested_component_types"])
        self.assertIn("action_canonicalizer", request["suggested_component_types"])

    def test_problem_router_suggests_button_swap_press_recovery(self) -> None:
        cluster = {
            "problem_id": "ButtonUnmaskSwap-P-recovery",
            "signature": {
                "task_variant": "unknown",
                "failure_stage": "failure_recovery",
                "primary_problem": "recovery_failure",
                "responsible_module": "recovery_policy",
            },
            "first_divergence_examples": [
                "the pending button repeats the same failed contact point"
            ],
            "secondary_symptoms": ["timeout"],
            "episode_refs": [
                {"dataset_split": "train", "episode_id": 1},
                {"dataset_split": "val", "episode_id": 2},
            ],
        }
        discovery = {"task": "ButtonUnmaskSwap", "clusters": [cluster]}
        records = [
            {
                "dataset_split": split,
                "episode_id": episode,
                "first_divergence": "repeated failed button contact",
                "responsible_module": "recovery_policy",
                "runtime_observable": True,
            }
            for split, episode in (("train", 1), ("val", 2))
        ]
        request = build_request(
            discovery, records, "ButtonUnmaskSwap-P-recovery"
        )
        self.assertIn(
            "button_swap_press_grounding_recovery_gate",
            request["suggested_component_types"],
        )


if __name__ == "__main__":
    unittest.main()
