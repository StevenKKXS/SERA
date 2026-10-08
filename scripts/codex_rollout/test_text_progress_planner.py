from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "text_progress_planner.py"
SPEC = importlib.util.spec_from_file_location("codex_text_progress_planner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
import planner_server as PLANNER_SERVER


class TextProgressPlannerTests(unittest.TestCase):
    def test_video_repick_event_ledger_requires_verified_ordered_cycles(self) -> None:
        requested = MODULE.video_repick_required_repetitions(
            "Pick up and put down the same cube exactly two times.",
            {"fixed_facts": ["Complete exactly two repetitions."]},
        )
        self.assertEqual(requested, 2)
        state = {
            "requested": 2,
            "completed": 0,
            "active_ordinal": None,
            "pickup_verified": False,
            "counted_release_instances": [],
        }
        state, audit = MODULE.evaluate_video_repick_event_ledger(
            state=state,
            previous_subtask=(
                "pick up the correct cube at <90, 120> for the first time"
            ),
            proposed_subtask="put it down",
            response={
                "recent_status": "completed",
                "execution_state": {
                    "held_object": "cube",
                    "object_support": "airborne",
                    "gripper_state": "closed",
                },
                "refine_check": {"decision": "satisfied"},
            },
            refine_criterion={"criterion_id": "videorepick_verified_target_pickup"},
            action_instance_id="pickup-1",
        )
        self.assertEqual(audit["event"], "pickup_verified")
        state, audit = MODULE.evaluate_video_repick_event_ledger(
            state=state,
            previous_subtask="put it down",
            proposed_subtask=(
                "pick up the correct cube at <90, 120> for the second time"
            ),
            response={
                "recent_status": "completed",
                "execution_state": {
                    "held_object": "none",
                    "object_support": "table",
                    "gripper_state": "open",
                },
                "refine_check": {"decision": "satisfied"},
            },
            refine_criterion={"criterion_id": "videorepick_verified_table_release"},
            action_instance_id="release-1",
        )
        self.assertEqual(audit["event"], "cycle_verified")
        self.assertEqual(state["completed"], 1)
        with self.assertRaisesRegex(ValueError, "early finish-button"):
            MODULE.evaluate_video_repick_event_ledger(
                state=state,
                previous_subtask=(
                    "pick up the correct cube at <90, 120> for the second time"
                ),
                proposed_subtask="press the button to finish",
                response={
                    "recent_status": "stagnant",
                    "execution_state": {
                        "held_object": "none",
                        "object_support": "table",
                        "gripper_state": "open",
                    },
                    "refine_check": {"decision": "not_satisfied"},
                },
                refine_criterion={
                    "criterion_id": "videorepick_verified_target_pickup"
                },
                action_instance_id="pickup-2",
            )

    def test_video_repick_grounding_recovery_requires_material_reground(self) -> None:
        config = {
            "trigger_after_failed_pickups": 2,
            "minimum_point_delta_norm": 32,
            "max_forced_regrounds_per_ordinal": 2,
        }
        state = {
            "ordinal": 1,
            "consecutive_failed_pickups": 0,
            "last_dispatched_point_norm": [300, 400],
            "forced_regrounds": 0,
        }
        failure = {
            "recent_status": "stagnant",
            "execution_state": {
                "held_object": "none",
                "object_support": "table",
                "gripper_state": "open",
            },
            "refine_check": {"decision": "not_satisfied"},
        }
        criterion = {"criterion_id": "videorepick_verified_target_pickup"}
        state, audit = MODULE.evaluate_video_repick_grounding_recovery(
            state=state,
            previous_subtask=(
                "pick up the correct cube at <77, 102> for the first time"
            ),
            proposed_subtask=(
                "pick up the correct cube at <78, 102> for the first time"
            ),
            proposed_points=[{"x_norm": 304, "y_norm": 400}],
            response=failure,
            refine_criterion=criterion,
            config=config,
        )
        self.assertEqual(state["consecutive_failed_pickups"], 1)
        self.assertEqual(audit["decision"], "accepted")
        with self.assertRaisesRegex(ValueError, "nearly identical pickup point"):
            MODULE.evaluate_video_repick_grounding_recovery(
                state=state,
                previous_subtask=(
                    "pick up the correct cube at <78, 102> for the first time"
                ),
                proposed_subtask=(
                    "pick up the correct cube at <79, 102> for the first time"
                ),
                proposed_points=[{"x_norm": 308, "y_norm": 400}],
                response=failure,
                refine_criterion=criterion,
                config=config,
            )
        bypassed, audit = MODULE.evaluate_video_repick_grounding_recovery(
            state=state,
            previous_subtask=(
                "pick up the correct cube at <78, 102> for the first time"
            ),
            proposed_subtask=(
                "pick up the correct cube at <79, 102> for the first time"
            ),
            proposed_points=[{"x_norm": 308, "y_norm": 400}],
            response=failure,
            refine_criterion=criterion,
            config=config,
            allow_retry_budget_bypass=True,
        )
        self.assertEqual(
            audit["decision"], "accepted_after_validation_retry_budget"
        )
        self.assertEqual(bypassed["forced_regrounds"], 1)
        state, audit = MODULE.evaluate_video_repick_grounding_recovery(
            state=state,
            previous_subtask=(
                "pick up the correct cube at <78, 102> for the first time"
            ),
            proposed_subtask=(
                "pick up the correct cube at <92, 112> for the first time"
            ),
            proposed_points=[{"x_norm": 360, "y_norm": 438}],
            response=failure,
            refine_criterion=criterion,
            config=config,
        )
        self.assertEqual(audit["decision"], "accepted_reground")
        self.assertEqual(state["forced_regrounds"], 1)

    def test_self_evolve_v4_counting_contracts_and_event_ledger(self) -> None:
        pick_requirements = MODULE.counting_requirements_from_guidance(
            "PickXtimes",
            {
                "fixed_facts": [
                    "The required object is the red cube.",
                    "Complete exactly five valid repetitions.",
                ]
            },
        )
        self.assertEqual(pick_requirements, {"red": 5})
        bin_requirements = MODULE.counting_requirements_from_guidance(
            "BinFill",
            {
                "fixed_facts": [
                    "Required bin totals: 2 red, 1 blue, and 2 green cubes.",
                    "Place exactly 5 cubes in total.",
                ]
            },
        )
        self.assertEqual(bin_requirements, {"red": 2, "blue": 1, "green": 2})

        state = {
            "requested": {"red": 2},
            "completed": {"red": 0},
            "active_color": None,
            "pickup_verified": False,
            "integrated_first_pending": False,
            "integrated_first_consumed": False,
            "counted_action_instances": [],
        }
        pickup_response = {
            "recent_status": "completed",
            "execution_state": {
                "held_object": "cube",
                "object_support": "airborne",
                "gripper_state": "closed",
            },
            "refine_check": {"decision": "satisfied"},
        }
        state, audit = MODULE.evaluate_counting_event_ledger(
            task="PickXtimes",
            state=state,
            previous_subtask="pick up the red cube at <80, 90> for the first time",
            proposed_subtask="place the red cube onto the target at <100, 120>",
            response=pickup_response,
            refine_criterion={"criterion_id": "counting_requested_cube_pickup"},
            action_instance_id="pickup-1",
        )
        self.assertEqual(audit["event"], "pickup_verified")
        self.assertTrue(state["pickup_verified"])

        release_response = {
            "recent_status": "completed",
            "execution_state": {
                "held_object": "none",
                "object_support": "target",
                "gripper_state": "open",
            },
            "refine_check": {"decision": "satisfied"},
        }
        state, audit = MODULE.evaluate_counting_event_ledger(
            task="PickXtimes",
            state=state,
            previous_subtask="place the red cube onto the target at <100, 120>",
            proposed_subtask="pick up the red cube at <100, 120> for the second time",
            response=release_response,
            refine_criterion={"criterion_id": "pickxtimes_verified_target_release"},
            action_instance_id="release-1",
        )
        self.assertEqual(audit["event"], "placement_verified")
        self.assertEqual(state["completed"], {"red": 1})
        with self.assertRaisesRegex(ValueError, "early stop-button"):
            MODULE.evaluate_counting_event_ledger(
                task="PickXtimes",
                state=state,
                previous_subtask="pick up the red cube at <100, 120> for the second time",
                proposed_subtask="press the button at <60, 90> to stop",
                response={
                    "recent_status": "progressing",
                    "execution_state": {
                        "held_object": "none",
                        "object_support": "table",
                        "gripper_state": "open",
                    },
                    "refine_check": {"decision": "uncertain"},
                },
                refine_criterion={"criterion_id": "counting_requested_cube_pickup"},
                action_instance_id="pickup-2",
            )

    def test_self_evolve_v4_profile_is_identity_isolated_and_task_scoped(self) -> None:
        profile = MODULE.PROMPT_PROFILES["retrieval_v8_5_self_evolve_v4"]
        catalog = MODULE.load_action_success_criteria(
            profile["refine_check_criteria_path"]
        )
        memory, gates = MODULE.load_runtime_experience_memory(
            profile["runtime_experience_memory_path"], refine_catalog=catalog
        )
        self.assertTrue(profile["enable_counting_event_ledger_guard"])
        self.assertIn(
            "retrieval_v8_5_self_evolve_v4", MODULE.IDENTITY_ISOLATED_PROMPT_PROFILES
        )
        self.assertEqual(memory["revision"], 4)
        self.assertEqual(
            gates["pickxtimes_verified_target_release"]["task_scope"],
            ["PickXtimes"],
        )
        self.assertEqual(
            gates["binfill_verified_bin_release"]["task_scope"], ["BinFill"]
        )

    def test_self_evolve_counting_recovers_same_cube_after_failed_release(self) -> None:
        state = {
            "requested": {"red": 3, "green": 1},
            "completed": {"red": 2, "green": 0},
            "active_color": "red",
            "pickup_verified": True,
            "integrated_first_pending": False,
            "integrated_first_consumed": False,
            "exact_pickup_ordinal_required": True,
            "counted_action_instances": [],
        }
        dropped_response = {
            "recent_status": "regressed",
            "execution_state": {
                "held_object": "none",
                "object_support": "table",
                "gripper_state": "open",
            },
            "refine_check": {"decision": "not_satisfied"},
        }
        recovered, audit = MODULE.evaluate_counting_event_ledger(
            task="BinFill",
            state=state,
            previous_subtask="put it into the bin at <130, 126>",
            proposed_subtask="pick up the third red cube at <155, 128>",
            response=dropped_response,
            refine_criterion={"criterion_id": "binfill_verified_bin_release"},
            action_instance_id="release-red-3-failed",
        )
        self.assertEqual(audit["event"], "placement_recovery_required")
        self.assertEqual(recovered["completed"], {"red": 2, "green": 0})
        self.assertEqual(recovered["active_color"], "red")
        self.assertFalse(recovered["pickup_verified"])

        with self.assertRaisesRegex(ValueError, "same dropped red cube"):
            MODULE.evaluate_counting_event_ledger(
                task="BinFill",
                state=state,
                previous_subtask="put it into the bin at <130, 126>",
                proposed_subtask="pick up the first green cube at <80, 90>",
                response=dropped_response,
                refine_criterion={"criterion_id": "binfill_verified_bin_release"},
                action_instance_id="release-red-3-failed",
            )

    def test_self_evolve_v4_allows_only_the_integrated_first_pickxtimes_place(self) -> None:
        state = {
            "requested": {"green": 2},
            "completed": {"green": 0},
            "active_color": None,
            "pickup_verified": False,
            "integrated_first_pending": False,
            "integrated_first_consumed": False,
            "counted_action_instances": [],
        }
        state, audit = MODULE.evaluate_counting_event_ledger(
            task="PickXtimes",
            state=state,
            previous_subtask="None. These are the initial rollout observations.",
            proposed_subtask="place the green cube onto the target at <60, 80>",
            response={
                "recent_status": "initial",
                "execution_state": {
                    "held_object": "none",
                    "object_support": "table",
                    "gripper_state": "open",
                },
                "refine_check": {"decision": "not_applicable"},
            },
            refine_criterion=None,
            action_instance_id=None,
        )
        self.assertEqual(audit["event"], "integrated_first_started")
        self.assertTrue(state["integrated_first_pending"])
        state, audit = MODULE.evaluate_counting_event_ledger(
            task="PickXtimes",
            state=state,
            previous_subtask="place the green cube onto the target at <60, 80>",
            proposed_subtask="pick up the green cube at <60, 80> for the second time",
            response={
                "recent_status": "completed",
                "execution_state": {
                    "held_object": "none",
                    "object_support": "target",
                    "gripper_state": "open",
                },
                "refine_check": {"decision": "satisfied"},
            },
            refine_criterion={"criterion_id": "pickxtimes_verified_target_release"},
            action_instance_id="release-integrated-first",
        )
        self.assertEqual(audit["event"], "placement_verified")
        self.assertEqual(state["completed"], {"green": 1})
        self.assertFalse(state["integrated_first_pending"])
        with self.assertRaisesRegex(ValueError, "without a verified current pickup"):
            MODULE.evaluate_counting_event_ledger(
                task="PickXtimes",
                state=state,
                previous_subtask="pick up the green cube at <60, 80> for the second time",
                proposed_subtask="place the green cube onto the target at <60, 80>",
                response={
                    "recent_status": "progressing",
                    "execution_state": {
                        "held_object": "none",
                        "object_support": "table",
                        "gripper_state": "open",
                    },
                    "refine_check": {"decision": "uncertain"},
                },
                refine_criterion={"criterion_id": "counting_requested_cube_pickup"},
                action_instance_id="pickup-second",
            )

    def test_self_evolve_v5_requires_and_injects_explicit_first_pickup(self) -> None:
        profile = MODULE.PROMPT_PROFILES["retrieval_v8_5_self_evolve_v5"]
        self.assertTrue(profile["enable_counting_explicit_first_pickup"])
        self.assertIn(
            "retrieval_v8_5_self_evolve_v5", MODULE.IDENTITY_ISOLATED_PROMPT_PROFILES
        )
        catalog = MODULE.load_action_success_criteria(
            profile["refine_check_criteria_path"]
        )
        memory, gates = MODULE.load_runtime_experience_memory(
            profile["runtime_experience_memory_path"], refine_catalog=catalog
        )
        self.assertEqual(memory["revision"], 5)
        self.assertEqual(
            gates["counting_requested_cube_pickup"]["revision"], 2
        )

        state = {
            "requested": {"green": 3},
            "completed": {"green": 0},
            "active_color": None,
            "pickup_verified": False,
            "integrated_first_pending": False,
            "integrated_first_consumed": False,
            "explicit_first_pickup_required": True,
            "counted_action_instances": [],
        }
        retriever = MODULE.ExecutorSkillRetriever(MODULE.GLOBAL_VOCABULARY_PATH)
        candidates = retriever.retrieve(
            query="PickXtimes green three times",
            task="PickXtimes",
            previous_template_id=None,
            recent_status="initial",
            top_k=5,
            scheme="R0",
        )
        injected = MODULE.ensure_pickxtimes_first_pickup_candidate(
            candidates, retriever=retriever, state=state
        )
        matching = [
            candidate
            for candidate in injected
            if candidate["canonical_template"]
            == "pick up the green cube at <point_1> for the first time"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(
            matching[0]["injected_by_experience"],
            "pickxtimes_explicit_first_pickup",
        )
        with self.assertRaisesRegex(ValueError, "without a verified current pickup"):
            MODULE.evaluate_counting_event_ledger(
                task="PickXtimes",
                state=state,
                previous_subtask="None. These are the initial rollout observations.",
                proposed_subtask="place the green cube onto the target at <60, 80>",
                response={
                    "recent_status": "initial",
                    "execution_state": {
                        "held_object": "none",
                        "object_support": "table",
                        "gripper_state": "open",
                    },
                    "refine_check": {"decision": "not_applicable"},
                },
                refine_criterion=None,
                action_instance_id=None,
            )

    def test_self_evolve_v6_injects_and_enforces_live_ledger_ordinals(self) -> None:
        profile = MODULE.PROMPT_PROFILES["retrieval_v8_5_self_evolve_v6"]
        self.assertTrue(profile["enable_counting_exact_pickup_ordinal"])
        self.assertIn(
            "retrieval_v8_5_self_evolve_v6", MODULE.IDENTITY_ISOLATED_PROMPT_PROFILES
        )
        catalog = MODULE.load_action_success_criteria(
            profile["refine_check_criteria_path"]
        )
        memory, gates = MODULE.load_runtime_experience_memory(
            profile["runtime_experience_memory_path"], refine_catalog=catalog
        )
        self.assertEqual(memory["revision"], 6)
        self.assertEqual(gates["counting_requested_cube_pickup"]["revision"], 3)
        retriever = MODULE.ExecutorSkillRetriever(MODULE.GLOBAL_VOCABULARY_PATH)

        pick_state = {
            "requested": {"red": 5},
            "completed": {"red": 4},
            "active_color": None,
            "pickup_verified": False,
            "integrated_first_pending": False,
            "integrated_first_consumed": False,
            "explicit_first_pickup_required": True,
            "exact_pickup_ordinal_required": True,
            "counted_action_instances": [],
        }
        injected = MODULE.ensure_counting_next_pickup_candidates(
            [], retriever=retriever, task="PickXtimes", state=pick_state
        )
        self.assertEqual(
            [candidate["canonical_template"] for candidate in injected],
            ["pick up the red cube at <point_1> for the fifth time"],
        )
        stale_pick = retriever.by_id["skill_0b2137af06ef"]
        allowed = {
            stale_pick["template_id"]: stale_pick,
            injected[0]["template_id"]: injected[0],
        }
        replacement, audit = MODULE.canonicalize_counting_pickup_skill(
            stale_pick["template_id"],
            allowed_skills=allowed,
            task="PickXtimes",
            state=pick_state,
        )
        self.assertEqual(replacement, "skill_f23f4598f21f")
        self.assertTrue(audit["canonicalized"])
        self.assertEqual(audit["proposed_ordinal"], 3)
        self.assertEqual(audit["expected_ordinal"], 5)
        with self.assertRaisesRegex(ValueError, "stale pickup ordinal"):
            MODULE.evaluate_counting_event_ledger(
                task="PickXtimes",
                state=pick_state,
                previous_subtask="place the red cube onto the target at <100, 120>",
                proposed_subtask="pick up the red cube at <90, 100> for the third time",
                response={
                    "recent_status": "progressing",
                    "execution_state": {
                        "held_object": "none",
                        "object_support": "target",
                        "gripper_state": "open",
                    },
                    "refine_check": {"decision": "not_satisfied"},
                },
                refine_criterion={"criterion_id": "pickxtimes_verified_target_release"},
                action_instance_id="release-4",
            )

        transition_state = {
            "requested": {"green": 4},
            "completed": {"green": 0},
            "active_color": "green",
            "pickup_verified": True,
            "integrated_first_pending": False,
            "integrated_first_consumed": False,
            "explicit_first_pickup_required": False,
            "exact_pickup_ordinal_required": True,
            "counted_action_instances": [],
        }
        release_response = {
            "recent_status": "completed",
            "execution_state": {
                "held_object": "none",
                "object_support": "bin",
                "gripper_state": "open",
            },
            "refine_check": {"decision": "satisfied"},
        }
        state_after_release, event = MODULE.advance_counting_previous_event(
            task="BinFill",
            state=transition_state,
            previous_subtask="put it into the bin at <88, 88>",
            response=release_response,
            refine_criterion={"criterion_id": "binfill_verified_bin_release"},
            action_instance_id="release-green-1",
        )
        self.assertEqual(event, "placement_verified")
        self.assertEqual(state_after_release["completed"], {"green": 1})
        next_green = MODULE.ensure_counting_next_pickup_candidates(
            [], retriever=retriever, task="BinFill", state=state_after_release
        )
        first_green = retriever.by_id["skill_5dec3992237b"]
        transition_allowed = {first_green["template_id"]: first_green}
        with self.assertRaisesRegex(ValueError, "could not canonicalize"):
            MODULE.canonicalize_counting_pickup_skill(
                first_green["template_id"],
                allowed_skills=transition_allowed,
                task="BinFill",
                state=state_after_release,
            )
        transition_allowed.update(
            {candidate["template_id"]: candidate for candidate in next_green}
        )
        replacement, audit = MODULE.canonicalize_counting_pickup_skill(
            first_green["template_id"],
            allowed_skills=transition_allowed,
            task="BinFill",
            state=state_after_release,
        )
        self.assertEqual(
            transition_allowed[replacement]["canonical_template"],
            "pick up the second green cube at <point_1>",
        )
        self.assertTrue(audit["canonicalized"])
        validated_state, validated_audit = MODULE.evaluate_counting_event_ledger(
            task="BinFill",
            state=transition_state,
            previous_subtask="put it into the bin at <88, 88>",
            proposed_subtask="pick up the second green cube at <57, 142>",
            response=release_response,
            refine_criterion={"criterion_id": "binfill_verified_bin_release"},
            action_instance_id="release-green-1",
        )
        self.assertEqual(validated_audit["event"], "placement_verified")
        self.assertEqual(validated_state["completed"], {"green": 1})

        bin_state = {
            "requested": {"red": 3, "green": 1},
            "completed": {"red": 2, "green": 0},
            "active_color": None,
            "pickup_verified": False,
            "integrated_first_pending": False,
            "integrated_first_consumed": False,
            "explicit_first_pickup_required": False,
            "exact_pickup_ordinal_required": True,
            "counted_action_instances": [],
        }
        injected = MODULE.ensure_counting_next_pickup_candidates(
            [], retriever=retriever, task="BinFill", state=bin_state
        )
        self.assertEqual(
            {candidate["canonical_template"] for candidate in injected},
            {
                "pick up the third red cube at <point_1>",
                "pick up the first green cube at <point_1>",
            },
        )
        self.assertEqual(
            MODULE.counting_pickup_ordinal(
                "BinFill", "pick up the third red cube at <90, 100>"
            ),
            3,
        )

    def test_self_evolve_v3_swing_side_binding_rejects_swapped_points(self) -> None:
        right_response = {
            "current_subtask_template": (
                "move to the top of the right-side target at <point_1> "
                "for the second time"
            )
        }
        with self.assertRaisesRegex(ValueError, "target swap"):
            MODULE.validate_swing_side_grounding(
                right_response,
                [{"x_norm": 680}],
                experience_id="experience_test",
                experience_revision=1,
            )

        accepted = MODULE.validate_swing_side_grounding(
            right_response,
            [{"x_norm": 280}],
            experience_id="experience_test",
            experience_revision=1,
        )
        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(accepted["proposed_side"], "right")

        left_response = {
            "current_subtask_template": (
                "move to the top of the left-side target at <point_1> "
                "for the third time"
            )
        }
        with self.assertRaisesRegex(ValueError, "image-right"):
            MODULE.validate_swing_side_grounding(
                left_response,
                [{"x_norm": 310}],
                experience_id="experience_test",
                experience_revision=1,
            )
        accepted = MODULE.validate_swing_side_grounding(
            left_response,
            [{"x_norm": 720}],
            experience_id="experience_test",
            experience_revision=1,
        )
        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(accepted["proposed_side"], "left")

    def test_self_evolve_v3_memory_is_task_scoped_and_identity_isolated(self) -> None:
        profile = MODULE.PROMPT_PROFILES["retrieval_v8_5_self_evolve_v3"]
        catalog = MODULE.load_action_success_criteria(
            profile["refine_check_criteria_path"]
        )
        memory, gates = MODULE.load_runtime_experience_memory(
            profile["runtime_experience_memory_path"],
            refine_catalog=catalog,
        )
        self.assertTrue(profile["enable_swing_side_grounding_guard"])
        self.assertIn(
            "retrieval_v8_5_self_evolve_v3",
            MODULE.IDENTITY_ISOLATED_PROMPT_PROFILES,
        )
        self.assertEqual(memory["revision"], 3)
        self.assertEqual(
            gates["swing_target_side_visual_binding"]["task_scope"],
            ["SwingXtimes"],
        )
        self.assertEqual(
            gates["move_cube_peg_pickup_held_motion"]["task_scope"],
            ["MoveCube"],
        )

    def test_v8_5_prompt_reserves_initial_status_for_round_zero(self) -> None:
        profile = MODULE.PROMPT_PROFILES["retrieval_v8_5_refine_then_lock"]
        prompt = profile["system_prompt"].read_text(encoding="utf-8")
        self.assertIn(
            "Use initial if and only if this is planner round 0",
            prompt,
        )
        self.assertIn(
            "After any skill has been executed, never use initial",
            prompt,
        )
        self.assertIn(
            "recent_status is a transition label",
            prompt,
        )
        self.assertIn(
            "Phase: initial` after one or more failed attempts",
            prompt,
        )

    def test_v8_5_refine_is_advisory_and_lock_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt=sys.stdin.read()\n"
                "counter=pathlib.Path(__file__).with_name('counter.txt')\n"
                "index=int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index+1))\n"
                "frame='r_f%04d' % (index*16)\n"
                "if index == 0:\n"
                " status='initial'; skill='skill_6bb26132e502'; template='press the button at <point_1>'\n"
                " points=[{'id':'point_1','object_description':'button','source_frame_id':frame,'x_norm':500,'y_norm':500}]\n"
                " refine={'criterion_id':'none','decision':'not_applicable','checks':[],'reason':'No previous configured action.'}\n"
                "else:\n"
                " assert 'advisory' in prompt.lower()\n"
                " status='completed'; skill='skill_e4ffdfcaa9f6'; template='pick up the container at <point_1> that hides the blue cube'\n"
                " points=[{'id':'point_1','object_description':'blue target container','source_frame_id':frame,'x_norm':520,'y_norm':520}]\n"
                " second_verdict='uncertain' if index == 1 else 'no'\n"
                " decision='uncertain' if index == 1 else 'not_satisfied'\n"
                " checks=[\n"
                "  {'check_id':'button_local_aligned_contact','verdict':'yes','evidence':'The gripper reached the button.'},\n"
                "  {'check_id':'aligned_gripper_downstroke','verdict':second_verdict,'evidence':'The narrow final downstroke test is inconclusive.'}]\n"
                " refine={'criterion_id':'button_unmask_press_local_event','decision':decision,'checks':checks,'reason':'The narrow check does not control the integrated final decision.'}\n"
                "response={'recent_status':status,'recent_outcome':'Refined final stage judgment.',\n"
                " 'refine_check':refine,'updated_progress':'Phase: refined planning.',\n"
                " 'skill_key':skill,'execution_state':{'held_object':'none','object_support':'table','gripper_state':'open'},\n"
                " 'instruction_strategy':'advance_skill' if index else 'normal',\n"
                " 'current_subtask_template':template,'points':points}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1]); out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="retrieval_v8_5_refine_then_lock",
                retrieval_scheme="R0",
                retrieval_top_k=5,
                verify_login=False,
                timeout_seconds=10,
            )
            planner.start_episode(
                run_dir=temporary / "run",
                detailed_task_description=(
                    "First press the button, then pick up the container hiding the blue cube."
                ),
                planner_guidance={},
                task="ButtonUnmask",
                episode=0,
                init_case_id="ButtonUnmask_episode_00",
            )
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            outputs = []
            records = []
            for index in range(3):
                output, record = planner.plan(
                    front_frames=[frame], wrist_frames=[frame], frame_ids=[index * 16]
                )
                outputs.append(output)
                records.append(record)

            self.assertIn("press the button", outputs[0])
            self.assertEqual(outputs[1], outputs[0])
            self.assertIn("pick up the container", outputs[2])
            self.assertEqual(
                records[1]["response"]["refine_check"]["decision"], "uncertain"
            )
            self.assertFalse(records[1]["completion_gate"]["applicable"])
            self.assertTrue(records[1]["refinement_lock"]["triggered"])
            self.assertTrue(records[2]["refinement_lock"]["released"])
            self.assertEqual(
                records[2]["response"]["refine_check"]["decision"],
                "not_satisfied",
            )
            self.assertFalse(
                records[2]["refinement_lock"]["refine_check_has_veto_authority"]
            )
            state = records[2]["refinement_lock"]["state_after"]
            self.assertEqual(state["status"], "exhausted")
            self.assertEqual(state["extra_commands_consumed"], 1)

    def test_self_evolve_peg_pickup_gate_is_scoped_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt=sys.stdin.read()\n"
                "counter=pathlib.Path(__file__).with_name('counter.txt')\n"
                "index=int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index+1))\n"
                "frame='r_f%04d' % (index*16)\n"
                "if index == 0:\n"
                " status='initial'; skill='skill_ac7483a073f8'; template='pick up the peg at <point_1>'\n"
                " points=[{'id':'point_1','object_description':'peg','source_frame_id':frame,'x_norm':450,'y_norm':650}]\n"
                " refine={'criterion_id':'none','decision':'not_applicable','checks':[],'reason':'No previous pickup exists.'}\n"
                " state={'held_object':'none','object_support':'table','gripper_state':'open'}\n"
                "else:\n"
                " assert 'move_cube_peg_pickup_held_motion' in prompt\n"
                " status='completed'; skill='skill_5ace02f82c6b'; template='hook the cube at <point_1> to the target at <point_2> with the peg'\n"
                " points=[{'id':'point_1','object_description':'cube','source_frame_id':frame,'x_norm':500,'y_norm':500},{'id':'point_2','object_description':'target','source_frame_id':frame,'x_norm':350,'y_norm':400}]\n"
                " verdicts=['yes','yes','yes'] if index == 3 else (['yes','no','no'] if index == 2 else ['yes','uncertain','uncertain'])\n"
                " decision='satisfied' if index == 3 else ('not_satisfied' if index == 2 else 'uncertain')\n"
                " ids=['peg_retained_between_fingers','peg_moves_with_gripper_across_frames','peg_not_left_table_supported']\n"
                " checks=[{'check_id':key,'verdict':value,'evidence':'Chronological visual check.'} for key,value in zip(ids,verdicts)]\n"
                " refine={'criterion_id':'move_cube_peg_pickup_held_motion','decision':decision,'checks':checks,'reason':'Require held-together motion before hook.'}\n"
                " state={'held_object':'peg','object_support':'held_unknown','gripper_state':'closed'}\n"
                "response={'recent_status':status,'recent_outcome':'Observed the peg pickup attempt.',\n"
                " 'refine_check':refine,'updated_progress':'Phase: peg pickup review.',\n"
                " 'skill_key':skill,'execution_state':state,'instruction_strategy':'advance_skill' if index else 'normal',\n"
                " 'current_subtask_template':template,'points':points}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1]); out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="retrieval_v8_5_self_evolve_v1",
                retrieval_scheme="R0",
                retrieval_top_k=5,
                verify_login=False,
                timeout_seconds=10,
            )
            planner.start_episode(
                run_dir=temporary / "run",
                detailed_task_description="Use the peg-pushing method to move the cube.",
                planner_guidance={"fixed_facts": ["Required method: peg pushing."]},
                task="MoveCube",
            )
            episode_config = json.loads(
                (temporary / "run" / "episode_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(episode_config["decision_identity_isolation"])
            self.assertNotIn("episode", episode_config)
            self.assertNotIn("init_case_id", episode_config)
            self.assertEqual(
                episode_config["runtime_experience_memory"]["memory_id"],
                "retrieval_v8_5_self_evolve_v1",
            )
            self.assertTrue(
                (temporary / "run" / "runtime_experience_memory.json").is_file()
            )
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            outputs = []
            records = []
            for index in range(4):
                output, record = planner.plan(
                    front_frames=[frame], wrist_frames=[frame], frame_ids=[index * 16]
                )
                outputs.append(output)
                records.append(record)

            self.assertIn("pick up the peg", outputs[0])
            self.assertEqual(outputs[1], outputs[0])
            self.assertEqual(outputs[2], outputs[0])
            self.assertIn("hook the cube", outputs[3])
            self.assertTrue(records[1]["experience_gate"]["transition_blocked"])
            self.assertTrue(records[2]["experience_gate"]["transition_blocked"])
            self.assertTrue(
                records[3]["experience_gate"][
                    "released_due_to_satisfied_evidence"
                ]
            )
            state = records[3]["experience_gate"]["state_after"]
            self.assertEqual(state["experience_id"], "experience_121bab65ef1b")
            self.assertEqual(state["blocked_transition_count"], 2)
            prompts = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted((temporary / "run" / "rounds").glob("*/prompt.txt"))
            )
            self.assertNotIn("MoveCube_episode_01", prompts)
            self.assertNotIn("failures 01", prompts)

    def test_self_evolve_rejects_episode_identity_at_decision_boundary(self) -> None:
        planner = MODULE.CodexTextProgressPlanner(
            prompt_profile="retrieval_v8_5_self_evolve_v1",
            verify_login=False,
        )
        with tempfile.TemporaryDirectory() as temporary_value:
            with self.assertRaisesRegex(ValueError, "must not receive episode"):
                planner.start_episode(
                    run_dir=Path(temporary_value) / "run",
                    detailed_task_description="Use the peg to move the cube.",
                    planner_guidance={},
                    task="MoveCube",
                    episode=3,
                    init_case_id="MoveCube_episode_03",
                )

    def test_self_evolve_v2_loads_task_scoped_swing_experience(self) -> None:
        catalog = MODULE.load_action_success_criteria(
            MODULE.SELF_EVOLVE_V2_CRITERIA_PATH
        )
        memory, gates = MODULE.load_runtime_experience_memory(
            MODULE.SELF_EVOLVE_V2_RUNTIME_MEMORY_PATH,
            refine_catalog=catalog,
        )
        self.assertEqual(memory["memory_id"], "retrieval_v8_5_self_evolve_v2")
        self.assertEqual(memory["revision"], 2)
        self.assertEqual(
            set(gates),
            {
                "move_cube_peg_pickup_held_motion",
                "swing_expected_target_activation_ledger",
            },
        )
        swing_gate = gates["swing_expected_target_activation_ledger"]
        self.assertEqual(swing_gate["task_scope"], ["SwingXtimes"])
        self.assertEqual(swing_gate["max_blocked_transitions"], 1)
        criterion = MODULE.match_action_success_criterion(
            "move to the top of the left-side target at <79, 85> for the third time",
            catalog,
            task="SwingXtimes",
        )
        self.assertIsNotNone(criterion)
        self.assertEqual(
            criterion["criterion_id"],
            "swing_expected_target_activation_ledger",
        )
        self.assertIsNone(
            MODULE.match_action_success_criterion(
                "move to the top of the left-side target at <79, 85> for the third time",
                catalog,
                task="MoveCube",
            )
        )

    def test_self_evolve_v2_rejects_episode_identity_at_decision_boundary(self) -> None:
        planner = MODULE.CodexTextProgressPlanner(
            prompt_profile="retrieval_v8_5_self_evolve_v2",
            verify_login=False,
        )
        with tempfile.TemporaryDirectory() as temporary_value:
            with self.assertRaisesRegex(ValueError, "must not receive episode"):
                planner.start_episode(
                    run_dir=Path(temporary_value) / "run",
                    detailed_task_description="Swing the cube twice.",
                    planner_guidance={},
                    task="SwingXtimes",
                    episode=3,
                    init_case_id="SwingXtimes_episode_03",
                )

    def test_runtime_experience_rejects_source_episode_metadata(self) -> None:
        catalog = MODULE.load_action_success_criteria(
            MODULE.SELF_EVOLVE_V1_CRITERIA_PATH
        )
        with tempfile.TemporaryDirectory() as temporary_value:
            memory_path = Path(temporary_value) / "memory.json"
            memory = json.loads(
                MODULE.SELF_EVOLVE_V1_RUNTIME_MEMORY_PATH.read_text(encoding="utf-8")
            )
            memory["experiences"][0]["source_episode_refs"] = ["01"]
            memory_path.write_text(json.dumps(memory), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden key"):
                MODULE.load_runtime_experience_memory(
                    memory_path, refine_catalog=catalog
                )

    def test_v2_runtime_lesson_loads_task_scoped_prompt_experience(self) -> None:
        memory = {
            "schema_version": "self-evolve-runtime-lessons-v2",
            "memory_id": "train-memory",
            "revision": 1,
            "base_profile": "retrieval_v8_5_refine_then_lock",
            "experiences": [
                {
                    "experience_id": "experience_train",
                    "revision": 1,
                    "status": "approved",
                    "task_scope": ["MoveCube"],
                    "condition": "The target identity is visually uncertain.",
                    "lesson": "Re-observe before changing the target binding.",
                    "recommended_skill_or_policy": "Use another visible observation.",
                    "success_evidence": ["The same target remains bound."],
                    "forbidden_behavior": ["Do not guess a hidden target."],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_value:
            path = Path(temporary_value) / "memory.json"
            path.write_text(json.dumps(memory), encoding="utf-8")
            loaded = MODULE.load_runtime_lesson_memory(path)
            self.assertEqual(loaded["experiences"][0]["experience_id"], "experience_train")
            memory["experiences"][0]["episode_id"] = 7
            path.write_text(json.dumps(memory), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden key"):
                MODULE.load_runtime_lesson_memory(path)

    def test_server_keeps_episode_identity_out_of_self_evolve_planner(self) -> None:
        class FakePlanner:
            instances = []

            def __init__(self, *, prompt_profile: str) -> None:
                self.prompt_profile = prompt_profile
                self.codex_version = "fake"
                self.model = "fake-model"
                self.reasoning_effort = "fake-effort"
                self.start_kwargs = None
                self.__class__.instances.append(self)

            def start_episode(self, **kwargs: object) -> None:
                self.start_kwargs = kwargs

        for profile in (
            "retrieval_v8_5_self_evolve_v1",
            "retrieval_v8_5_self_evolve_v2",
            "retrieval_v8_5_self_evolve_v3",
            "self_evolve_v2_train_active",
            "self_evolve_v3_train30_val20",
            "self_evolve_v3_multiproblem_auto_hard20",
        ):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary_value:
                service = PLANNER_SERVER.PlannerService(
                    FakePlanner,
                    "text-progress",
                    {"prompt_profile": profile},
                )
                service.start_episode(
                    {
                        "session_id": f"test-session-{profile}",
                        "task": "MoveCube",
                        "episode": 17,
                        "run_dir": str(Path(temporary_value) / "planner"),
                        "init_case_id": "MoveCube_episode_17",
                        "detailed_task_description": "Use the peg to move the cube.",
                        "planner_guidance": {},
                    }
                )
                start_kwargs = FakePlanner.instances[-1].start_kwargs
                self.assertIsNotNone(start_kwargs)
                self.assertNotIn("episode", start_kwargs)
                self.assertNotIn("init_case_id", start_kwargs)

    def test_multiproblem_v3_runtime_starts_from_mature_v8_5(self) -> None:
        mature = MODULE.PROMPT_PROFILES["retrieval_v8_5_refine_then_lock"]
        evolved = MODULE.PROMPT_PROFILES[
            "self_evolve_v3_multiproblem_auto_hard20"
        ]
        for field in (
            "system_prompt",
            "user_prompt",
            "output_schema",
            "refine_check_criteria_path",
            "reinforcement_lock",
        ):
            self.assertEqual(evolved[field], mature[field])
        self.assertIn(
            "self_evolve_v3_multiproblem_auto_hard20",
            MODULE.IDENTITY_ISOLATED_PROMPT_PROFILES,
        )

    def test_bounded_reinforcement_yields_without_saved_successor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "counter = pathlib.Path(__file__).with_name('counter.txt')\n"
                "index = int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index + 1))\n"
                "frame = 'r_f%04d' % (index * 16)\n"
                "identity = {'requested_color':'none','binding_status':'not_applicable',\n"
                " 'binding_basis':'not_applicable','target_description':'button',\n"
                " 'evidence_frame_ids':[],'revealed_color':'none',\n"
                " 'same_as_bound_target':'uncertain','reason':'No cover target is active.'}\n"
                "if index == 0:\n"
                " status='initial'\n"
                " check={'criterion_id':'none','decision':'not_applicable','checks':[],\n"
                "        'reason':'No prior action.'}\n"
                "else:\n"
                " status='completed'\n"
                " checks=[\n"
                "  {'check_id':'button_local_aligned_contact','verdict':'yes','evidence':'Aligned contact.'},\n"
                "  {'check_id':'aligned_gripper_downstroke','verdict':'yes','evidence':'Downstroke visible.'}]\n"
                " check={'criterion_id':'button_unmask_press_local_event',\n"
                "        'decision':'satisfied','checks':checks,'reason':'Press is complete.'}\n"
                "propose_transition = index in (1, 4)\n"
                "skill = 'skill_f2376d9bd67c' if propose_transition else 'skill_6bb26132e502'\n"
                "template = 'put down the container' if propose_transition else 'press the button at <point_1>'\n"
                "points = [] if propose_transition else [{'id':'point_1','object_description':'button',\n"
                " 'source_frame_id':frame,'x_norm':500,'y_norm':500}]\n"
                "response={'recent_status':status,'recent_outcome':'The button was pressed.',\n"
                " 'target_identity':identity,'binding_updates':[],\n"
                " 'completion_check':check,'updated_progress':'Phase: button press.',\n"
                " 'skill_key':skill,\n"
                " 'execution_state':{'held_object':'none','object_support':'table','gripper_state':'open'},\n"
                " 'instruction_strategy':'normal',\n"
                " 'current_subtask_template':template,'points':points}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1])\n"
                "out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            profile_name = "_test_bounded_reinforcement_k2"
            profile = dict(
                MODULE.PROMPT_PROFILES[
                    "retrieval_v8_4_6_pickhighlight_instances"
                ]
            )
            profile["one_shot_completion_reinforcement"] = {
                "button_unmask_press_local_event": 2,
            }
            MODULE.PROMPT_PROFILES[profile_name] = profile
            try:
                planner = MODULE.CodexTextProgressPlanner(
                    codex_bin=str(fake_codex),
                    prompt_profile=profile_name,
                    retrieval_scheme="R0",
                    retrieval_top_k=5,
                    verify_login=False,
                    timeout_seconds=10,
                )
                planner.start_episode(
                    run_dir=temporary / "run",
                    detailed_task_description=(
                        "First press the button, then pick up a container."
                    ),
                    planner_guidance={},
                    task="ButtonUnmask",
                    episode=0,
                    init_case_id="ButtonUnmask_episode_00",
                )
                frame = np.zeros((100, 200, 3), dtype=np.uint8)
                outputs = []
                records = []
                for index in range(5):
                    output, record = planner.plan(
                        front_frames=[frame],
                        wrist_frames=[frame],
                        frame_ids=[index * 16],
                    )
                    outputs.append(output)
                    records.append(record)
            finally:
                MODULE.PROMPT_PROFILES.pop(profile_name, None)

            self.assertEqual(
                ["press the button" in value for value in outputs],
                [True, True, True, True, False],
            )
            self.assertEqual(outputs[4], "put down the container")
            self.assertTrue(
                records[1]["completion_gate"]["one_shot_reinforcement_triggered"]
            )
            self.assertTrue(
                records[1]["completion_gate"]["action_transition_lock_active_after"]
            )
            self.assertEqual(
                records[1]["completion_gate"]["action_transition_lock_blocked_proposal"][
                    "grounded_subtask"
                ],
                "put down the container",
            )
            self.assertTrue(records[2]["completion_gate"]["semantic_switch_blocked"])
            self.assertEqual(
                records[2]["completion_gate"]["one_shot_reinforcement_state_after"][
                    "extra_execution_consumed"
                ],
                1,
            )
            self.assertTrue(
                records[3]["completion_gate"]["one_shot_reinforcement_completed"]
            )
            self.assertFalse(records[3]["completion_gate"]["semantic_switch_blocked"])
            final_state = records[3]["completion_gate"][
                "one_shot_reinforcement_state_after"
            ]
            self.assertEqual(final_state["status"], "exhausted")
            self.assertEqual(final_state["extra_execution_consumed"], 2)
            self.assertIsNone(final_state["deferred_successor"])
            self.assertEqual(
                final_state["blocked_transition_proposal"]["grounded_subtask"],
                "put down the container",
            )
            self.assertIn(
                "press the button",
                final_state["latest_planner_proposal"]["grounded_subtask"],
            )
            self.assertEqual(
                [
                    "press the button" in proposal["grounded_subtask"]
                    for proposal in final_state["planner_proposal_history"]
                ],
                [False, True, True],
            )
            self.assertTrue(
                records[3]["completion_gate"]["action_transition_lock_released"]
            )
            self.assertFalse(
                records[4]["completion_gate"]["one_shot_reinforcement_triggered"]
            )

    def test_v8_4_6_pickhighlight_instances_and_release_margin(self) -> None:
        profile = MODULE.PROMPT_PROFILES[
            "retrieval_v8_4_6_pickhighlight_instances"
        ]
        self.assertTrue(profile["enable_highlight_target_ledger"])
        self.assertEqual(
            profile["one_shot_completion_reinforcement"][
                "pickhighlight_put_down_stable_release"
            ],
            1,
        )
        catalog = MODULE.load_action_success_criteria(
            profile["action_success_criteria_path"]
        )
        release = MODULE.match_action_success_criterion(
            "place the cube onto the table", catalog, task="PickHighlight"
        )
        self.assertIsNotNone(release)
        self.assertEqual(
            release["criterion_id"], "pickhighlight_put_down_stable_release"
        )
        self.assertEqual(
            [item["check_id"] for item in release["required_checks"]],
            [
                "cube_table_supported",
                "gripper_visibly_open",
                "cube_decoupled_with_gap",
                "released_cube_stable",
            ],
        )
        unrelated = MODULE.match_action_success_criterion(
            "place the cube onto the table", catalog, task="MoveCube"
        )
        self.assertIsNone(unrelated)

    def test_pickhighlight_instance_ids_are_action_slot_not_color(self) -> None:
        first = MODULE.highlight_target_instance_id(
            "skill_first_red",
            "pick up the first highlighted cube at <point_1>, which is red",
            "pick up the first highlighted cube at <70, 120>, which is red",
        )
        first_moved = MODULE.highlight_target_instance_id(
            "skill_first_red",
            "pick up the first highlighted cube at <point_1>, which is red",
            "pick up the first highlighted cube at <93, 155>, which is red",
        )
        second = MODULE.highlight_target_instance_id(
            "skill_second_red",
            "pick up the second highlighted cube at <point_1>, which is red",
            "pick up the second highlighted cube at <102, 92>, which is red",
        )
        self.assertEqual(first, first_moved)
        self.assertNotEqual(first, second)
        self.assertEqual(
            MODULE.parse_highlighted_cube_pickup(
                "pick up the highlighted cube at <81, 135>, which is green"
            ),
            ("single", "green"),
        )

    def test_v8_4_5_pickhighlight_button_uses_one_shot_completion_gate(self) -> None:
        profile = MODULE.PROMPT_PROFILES[
            "retrieval_v8_4_5_pickhighlight_button"
        ]
        catalog = MODULE.load_action_success_criteria(
            profile["action_success_criteria_path"]
        )
        pickhighlight = MODULE.match_action_success_criterion(
            "press the button at <74, 95>", catalog, task="PickHighlight"
        )
        self.assertIsNotNone(pickhighlight)
        self.assertEqual(
            pickhighlight["criterion_id"], "button_unmask_press_local_event"
        )
        self.assertEqual(
            [item["check_id"] for item in pickhighlight["required_checks"]],
            ["button_local_aligned_contact", "aligned_gripper_downstroke"],
        )
        self.assertEqual(
            profile["one_shot_completion_reinforcement"][
                "button_unmask_press_local_event"
            ],
            1,
        )
        unrelated = MODULE.match_action_success_criterion(
            "press the button at <74, 95>", catalog, task="PickHighLightUnrelated"
        )
        self.assertIsNone(unrelated)
        prompt = profile["system_prompt"].read_text(encoding="utf-8")
        self.assertIn("temporary white highlight cues", prompt)
        self.assertIn("can never count as button-press evidence", prompt)

    def test_duplicate_completion_check_with_same_verdict_is_sanitized(self) -> None:
        criterion = {
            "criterion_id": "button_press",
            "required_checks": [
                {"check_id": "contact"},
                {"check_id": "downstroke"},
            ],
        }
        completion_check = {
            "criterion_id": "button_press",
            "decision": "satisfied",
            "checks": [
                {"check_id": "contact", "verdict": "yes", "evidence": "Contact."},
                {
                    "check_id": "downstroke",
                    "verdict": "yes",
                    "evidence": "Downstroke visible.",
                },
                {
                    "check_id": "downstroke",
                    "verdict": "yes",
                    "evidence": "The aligned gripper moved down.",
                },
            ],
            "reason": "Both required signals are visible.",
        }
        repaired, metadata = MODULE.sanitize_duplicate_completion_checks(
            completion_check, criterion
        )
        self.assertTrue(metadata["changed"])
        self.assertEqual(
            [item["check_id"] for item in repaired["checks"]],
            ["contact", "downstroke"],
        )
        MODULE.validate_completion_check(repaired, criterion)

    def test_conflicting_duplicate_completion_check_is_not_sanitized(self) -> None:
        criterion = {
            "criterion_id": "button_press",
            "required_checks": [{"check_id": "contact"}],
        }
        completion_check = {
            "criterion_id": "button_press",
            "decision": "uncertain",
            "checks": [
                {"check_id": "contact", "verdict": "yes", "evidence": "Contact."},
                {"check_id": "contact", "verdict": "no", "evidence": "No contact."},
            ],
            "reason": "Conflicting observations.",
        }
        repaired, metadata = MODULE.sanitize_duplicate_completion_checks(
            completion_check, criterion
        )
        self.assertFalse(metadata["changed"])
        self.assertIs(repaired, completion_check)
        with self.assertRaisesRegex(ValueError, "criterion order exactly"):
            MODULE.validate_completion_check(repaired, criterion)

    def test_v8_4_4_reinforces_each_unmask_action_instance_once(self) -> None:
        profile = MODULE.PROMPT_PROFILES[
            "retrieval_v8_4_4_one_shot_reinforcement"
        ]
        self.assertEqual(
            profile["one_shot_completion_reinforcement"],
            {
                "button_unmask_press_local_event": 1,
                "unmask_pickup_identity_and_separation": 1,
                "unmask_put_down_stable_release": 1,
            },
        )
        red = MODULE.action_instance_signature(
            "unmask_pickup_identity_and_separation",
            "skill_red",
            "pick up the container at <point_1> that hides the red cube",
            "pick up the container at <96, 86> that hides the red cube",
        )
        red_moved = MODULE.action_instance_signature(
            "unmask_pickup_identity_and_separation",
            "skill_red",
            "pick up the container at <point_1> that hides the red cube",
            "pick up the container at <101, 91> that hides the red cube",
        )
        green = MODULE.action_instance_signature(
            "unmask_pickup_identity_and_separation",
            "skill_green",
            "pick up the container at <point_1> that hides the green cube",
            "pick up the container at <130, 173> that hides the green cube",
        )
        self.assertEqual(red, red_moved)
        self.assertNotEqual(red, green)
        prompt = profile["system_prompt"].read_text(encoding="utf-8")
        self.assertIn("exactly one additional chunk", prompt)
        self.assertIn("separate instances", prompt)

    def test_v8_4_3_accepts_visual_downstroke_without_mandatory_cap_travel(self) -> None:
        profile = MODULE.PROMPT_PROFILES[
            "retrieval_v8_4_3_visual_downstroke"
        ]
        catalog = MODULE.load_action_success_criteria(
            profile["action_success_criteria_path"]
        )
        button = MODULE.match_action_success_criterion(
            "press the button at <66, 121>", catalog, task="ButtonUnmask"
        )
        self.assertEqual(button["completion_latch_key"], "button_prerequisite")
        self.assertEqual(
            [item["check_id"] for item in button["required_checks"]],
            ["button_local_aligned_contact", "aligned_gripper_downstroke"],
        )
        text = json.dumps(button).lower()
        self.assertIn("not required when the gripper occludes", text)
        self.assertNotIn("visible_button_cap_travel", text)
        self.assertEqual(button["transition_policy"]["confirmation_retries"], 0)

    def test_v8_4_2_uses_strict_button_evidence_without_forced_retry(self) -> None:
        profile = MODULE.PROMPT_PROFILES[
            "retrieval_v8_4_2_strict_button_bindings"
        ]
        self.assertTrue(profile["require_binding_updates"])
        self.assertTrue(profile["enable_action_completion_latches"])
        self.assertNotIn("completion_confirmation_retries", profile)
        catalog = MODULE.load_action_success_criteria(
            profile["action_success_criteria_path"]
        )
        button = MODULE.match_action_success_criterion(
            "press the button at <66, 121>", catalog, task="ButtonUnmask"
        )
        self.assertEqual(button["completion_latch_key"], "button_prerequisite")
        self.assertEqual(
            [item["check_id"] for item in button["required_checks"]],
            ["button_local_aligned_contact", "visible_button_cap_travel"],
        )
        self.assertEqual(button["transition_policy"]["confirmation_retries"], 0)
        self.assertNotIn("withdrawal without evidence", json.dumps(button).lower())

    def test_v8_4_2_binding_updates_persist_and_cannot_be_downgraded(self) -> None:
        update = {
            "color": "red",
            "binding_status": "locked",
            "binding_basis": "observed_cover_event",
            "target_description": "left container with a dark rim",
            "evidence_frame_ids": ["r_f0032"],
            "grounding_point": {
                "object_description": "left container",
                "source_frame_id": "r_f0032",
                "x_norm": 300,
                "y_norm": 500,
            },
            "revealed_color": "unknown",
            "reason": "The red cube became covered by this container.",
        }
        accepted, metadata = MODULE.sanitize_binding_updates(
            [update], allowed_front_ids={"r_f0032"}
        )
        self.assertEqual(metadata["accepted"], 1)
        bindings, decisions = MODULE.merge_target_binding_updates({}, accepted)
        self.assertEqual(decisions[0]["decision"], "accepted")
        weaker = dict(update)
        weaker["target_description"] = "a newly guessed container"
        bindings, decisions = MODULE.merge_target_binding_updates(bindings, [weaker])
        self.assertEqual(decisions[0]["decision"], "preserved_existing")
        self.assertEqual(
            bindings["red"]["target_description"], "left container with a dark rim"
        )

    def test_v8_4_2_pickup_can_reuse_direct_stored_binding(self) -> None:
        stored = {
            "requested_color": "green",
            "binding_status": "locked",
            "binding_basis": "observed_cover_event",
            "target_description": "rear container",
            "evidence_frame_ids": ["r_f0032"],
            "revealed_color": "unknown",
            "same_as_bound_target": "yes",
            "reason": "Direct cover event.",
        }
        current = {
            **stored,
            "binding_status": "candidate",
            "binding_basis": "inference_only",
            "evidence_frame_ids": [],
            "same_as_bound_target": "yes",
        }
        safe, reason, source = MODULE.resolve_pickup_binding(
            current, "green", stored
        )
        self.assertTrue(safe)
        self.assertEqual(reason, "direct_binding_verified")
        self.assertEqual(source, "stored_binding")
        current["same_as_bound_target"] = "uncertain"
        safe, _, source = MODULE.resolve_pickup_binding(current, "green", stored)
        self.assertFalse(safe)
        self.assertEqual(source, "none")

    def test_v8_4_1_patches_are_task_scoped_and_do_not_double_press(self) -> None:
        profile = MODULE.PROMPT_PROFILES[
            "retrieval_v8_4_1_task_scoped_patches"
        ]
        self.assertNotIn("completion_confirmation_retries", profile)
        catalog = MODULE.load_action_success_criteria(
            profile["action_success_criteria_path"]
        )
        button = MODULE.match_action_success_criterion(
            "press the button at <66, 121>", catalog, task="ButtonUnmask"
        )
        self.assertEqual(button["criterion_id"], "button_unmask_press_local_event")
        self.assertEqual(button["transition_policy"]["confirmation_retries"], 0)
        self.assertEqual(
            button["transition_policy"]["on_satisfied"], "advance_immediately"
        )
        self.assertIsNone(
            MODULE.match_action_success_criterion(
                "press the button at <66, 121>", catalog, task="VideoUnmask"
            )
        )
        self.assertIsNone(
            MODULE.match_action_success_criterion(
                "put down the container", catalog, task="PickHighLight"
            )
        )
        pickup = MODULE.match_action_success_criterion(
            "pick up the container at <120, 160> that hides the red cube",
            catalog,
            task="ButtonUnmask",
        )
        self.assertEqual(
            pickup["criterion_id"], "unmask_pickup_identity_and_separation"
        )

    def test_target_identity_frame_sanitizer_accepts_history_and_drops_fabrication(
        self,
    ) -> None:
        identity = {
            "requested_color": "red",
            "binding_status": "locked",
            "binding_basis": "observed_cover_event",
            "target_description": "upper-left container",
            "evidence_frame_ids": ["r_f0032", "r_f02800444"],
            "revealed_color": "unknown",
            "same_as_bound_target": "yes",
            "reason": "Bound from the observed cover event.",
        }
        sanitized, metadata = MODULE.sanitize_target_identity_frame_ids(
            identity, allowed_front_ids={"r_f0032", "r_f0048"}
        )
        self.assertEqual(sanitized["evidence_frame_ids"], ["r_f0032"])
        self.assertEqual(metadata["dropped_frame_ids"], ["r_f02800444"])
        self.assertFalse(metadata["direct_binding_downgraded"])

        downgraded, metadata = MODULE.sanitize_target_identity_frame_ids(
            identity, allowed_front_ids={"r_f0048"}
        )
        self.assertEqual(downgraded["evidence_frame_ids"], [])
        self.assertEqual(downgraded["binding_status"], "candidate")
        self.assertEqual(downgraded["binding_basis"], "inference_only")
        self.assertTrue(metadata["direct_binding_downgraded"])

    def test_v8_4_identity_binding_requires_direct_evidence(self) -> None:
        base = {
            "requested_color": "red",
            "binding_status": "locked",
            "binding_basis": "inference_only",
            "target_description": "upper-left container",
            "evidence_frame_ids": [],
            "revealed_color": "unknown",
            "same_as_bound_target": "yes",
            "reason": "Spatial guess only.",
        }
        safe, reason = MODULE.target_binding_is_safe_for_pickup(base, "red")
        self.assertFalse(safe)
        self.assertEqual(reason, "binding_basis_not_direct")
        observed = dict(base)
        observed["binding_basis"] = "observed_cover_event"
        observed["evidence_frame_ids"] = ["r_f0016"]
        safe, reason = MODULE.target_binding_is_safe_for_pickup(observed, "red")
        self.assertTrue(safe)
        self.assertEqual(reason, "direct_binding_verified")
        wrong = dict(observed)
        wrong["revealed_color"] = "green"
        safe, reason = MODULE.target_binding_is_safe_for_pickup(wrong, "red")
        self.assertFalse(safe)
        self.assertEqual(reason, "revealed_color_mismatch")

    def test_v8_4_has_strict_pickup_and_put_down_criteria(self) -> None:
        profile = MODULE.PROMPT_PROFILES[
            "retrieval_v8_4_identity_pickup_release"
        ]
        self.assertTrue(profile["enable_target_identity_gate"])
        self.assertFalse(profile.get("enable_completion_evidence_latch", False))
        catalog = MODULE.load_action_success_criteria(
            profile["action_success_criteria_path"]
        )
        pickup = MODULE.match_action_success_criterion(
            "pick up the container at <120, 160> that hides the red cube", catalog
        )
        release = MODULE.match_action_success_criterion(
            "put down the container", catalog
        )
        self.assertEqual(
            pickup["criterion_id"],
            "pickup_container_identity_and_separation",
        )
        self.assertEqual(
            [item["check_id"] for item in pickup["required_checks"]],
            [
                "same_bound_container",
                "requested_cube_color_matches",
                "full_persistent_cube_container_separation",
                "secure_sustained_airborne_hold",
            ],
        )
        self.assertEqual(
            release["criterion_id"], "put_down_container_stable_release"
        )
        self.assertEqual(len(release["required_checks"]), 4)

    def test_v8_3_pickup_evidence_latch_accumulates_and_resets(self) -> None:
        profile = MODULE.PROMPT_PROFILES["retrieval_v8_3_pickup_latch"]
        self.assertTrue(profile["enable_completion_evidence_latch"])
        catalog = MODULE.load_action_success_criteria(
            profile["action_success_criteria_path"]
        )
        criterion = MODULE.match_action_success_criterion(
            "pick up the container at <120, 160> that hides the red cube", catalog
        )
        self.assertEqual(criterion["criterion_id"], "pickup_container_calibrated_lift")

        def check(coupled: str, cube: str, airborne: str, sustained: str) -> dict:
            verdicts = [coupled, cube, airborne, sustained]
            decision = (
                "satisfied"
                if all(value == "yes" for value in verdicts)
                else "not_satisfied"
                if any(value == "no" for value in verdicts)
                else "uncertain"
            )
            return {
                "criterion_id": "pickup_container_calibrated_lift",
                "decision": decision,
                "checks": [
                    {"check_id": key, "verdict": value, "evidence": "test"}
                    for key, value in zip(
                        [
                            "target_coupled_to_closed_gripper",
                            "hidden_cube_fully_exposed",
                            "clear_airborne_displacement",
                            "sustained_or_accumulated_hold",
                        ],
                        verdicts,
                        strict=True,
                    )
                ],
                "reason": "test",
            }

        candidate = check("uncertain", "yes", "yes", "uncertain")
        first = MODULE.update_completion_evidence_state(None, candidate, criterion)
        self.assertEqual(first["candidate_round_count"], 1)
        self.assertFalse(first["completion_latched"])
        second = MODULE.update_completion_evidence_state(first, candidate, criterion)
        self.assertEqual(second["candidate_round_count"], 2)
        self.assertTrue(second["completion_latched"])
        missed_grasp = check("no", "no", "no", "no")
        reset = MODULE.update_completion_evidence_state(
            second, missed_grasp, criterion
        )
        self.assertEqual(reset["candidate_round_count"], 0)
        self.assertFalse(reset["completion_latched"])
        self.assertEqual(reset["last_update"], "hard_reset")

    def test_v8_3_latched_pickup_allows_successor_with_uncertain_raw_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt=sys.stdin.read()\n"
                "counter=pathlib.Path(__file__).with_name('counter.txt')\n"
                "index=int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index+1))\n"
                "if index == 0:\n"
                " check={'criterion_id':'none','decision':'not_applicable','checks':[],'reason':'Initial.'}\n"
                " status='initial'; skill='skill_b8064b602776'; points=[{'id':'point_1','object_description':'green container','source_frame_id':'r_f0000','x_norm':500,'y_norm':500}]\n"
                "else:\n"
                " assert 'pickup_container_calibrated_lift' in prompt\n"
                " if index == 3: assert '\"completion_latched\": true' in prompt\n"
                " checks=[\n"
                "  {'check_id':'target_coupled_to_closed_gripper','verdict':'uncertain','evidence':'Target moves with closed gripper under occlusion.'},\n"
                "  {'check_id':'hidden_cube_fully_exposed','verdict':'yes','evidence':'Hidden cube is fully exposed.'},\n"
                "  {'check_id':'clear_airborne_displacement','verdict':'yes','evidence':'Container left table support.'},\n"
                "  {'check_id':'sustained_or_accumulated_hold','verdict':'uncertain','evidence':'Current and accumulated evidence are consistent.'}]\n"
                " check={'criterion_id':'pickup_container_calibrated_lift','decision':'uncertain','checks':checks,'reason':'Occluded hold but consistent lift.'}\n"
                " status='progressing' if index < 3 else 'completed'\n"
                " skill='skill_b8064b602776' if index < 3 else 'skill_f2376d9bd67c'\n"
                " frame='r_f%04d' % (index*16)\n"
                " points=[{'id':'point_1','object_description':'green container','source_frame_id':frame,'x_norm':500,'y_norm':500}] if index < 3 else []\n"
                "response={'recent_status':status,'recent_outcome':'Visible lift evidence.',\n"
                " 'completion_check':check,'updated_progress':'Phase: evaluate pickup.',\n"
                " 'skill_key':skill,'execution_state':{'held_object':'container','object_support':'airborne','gripper_state':'closed'},\n"
                " 'instruction_strategy':'advance_skill' if index == 3 else 'normal',\n"
                " 'current_subtask_template':'placeholder','points':points}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1]); out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="retrieval_v8_3_pickup_latch",
                retrieval_scheme="R0",
                retrieval_top_k=5,
                verify_login=False,
                timeout_seconds=10,
            )
            planner.start_episode(
                run_dir=temporary / "run",
                detailed_task_description="Pick up and then put down the green container.",
                planner_guidance={},
                task="VideoUnmask",
                episode=1,
                init_case_id="VideoUnmask_episode_01",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            outputs = []
            records = []
            for index in range(4):
                output, record = planner.plan(
                    front_frames=[front],
                    wrist_frames=[wrist],
                    frame_ids=[index * 16],
                )
                outputs.append(output)
                records.append(record)
            self.assertIn("pick up", outputs[0])
            self.assertFalse(
                records[1]["completion_gate"]["completion_evidence_state_after"][
                    "completion_latched"
                ]
            )
            self.assertTrue(
                records[2]["completion_gate"]["completion_evidence_state_after"][
                    "completion_latched"
                ]
            )
            self.assertEqual(
                records[3]["completion_gate"]["raw_decision"], "uncertain"
            )
            self.assertEqual(
                records[3]["completion_gate"]["effective_decision"], "satisfied"
            )
            self.assertFalse(records[3]["completion_gate"]["semantic_switch_blocked"])
            self.assertEqual(outputs[3], "put down the container")

    def test_v8_action_success_profile_and_catalog_are_generic(self) -> None:
        v6 = MODULE.PROMPT_PROFILES["retrieval_v6"]
        v8 = MODULE.PROMPT_PROFILES["retrieval_v8_action_success"]
        self.assertTrue(v8["enable_action_success_criteria"])
        self.assertTrue(v8["enable_global_retrieval"])
        self.assertTrue(v8["catalog_retrieval_only"])
        self.assertNotEqual(v8["output_schema"], v6["output_schema"])
        catalog = MODULE.load_action_success_criteria()
        criterion = MODULE.match_action_success_criterion(
            "pick up the container at <83, 126> that hides the green cube",
            catalog,
        )
        self.assertEqual(
            criterion["criterion_id"], "pickup_container_strict_clearance"
        )
        self.assertIsNone(
            MODULE.match_action_success_criterion("put down the container", catalog)
        )
        prompt = v8["system_prompt"].read_text(encoding="utf-8")
        self.assertIn("stricter than ordinary human visual judgment", prompt)
        self.assertIn("clear and generous separation", prompt)
        self.assertNotIn("VideoUnmask", prompt)

    def test_v8_1_extends_harness_with_button_criterion_without_changing_v8_0(self) -> None:
        v8 = MODULE.PROMPT_PROFILES["retrieval_v8_action_success"]
        v8_1 = MODULE.PROMPT_PROFILES["retrieval_v8_1_action_success"]
        self.assertNotEqual(
            v8["action_success_criteria_path"],
            v8_1["action_success_criteria_path"],
        )
        v8_catalog = MODULE.load_action_success_criteria(
            v8["action_success_criteria_path"]
        )
        v8_1_catalog = MODULE.load_action_success_criteria(
            v8_1["action_success_criteria_path"]
        )
        self.assertIsNone(
            MODULE.match_action_success_criterion(
                "press the button at <66, 121>", v8_catalog
            )
        )
        criterion = MODULE.match_action_success_criterion(
            "press the button at <66, 121>", v8_1_catalog
        )
        self.assertEqual(
            criterion["criterion_id"], "press_button_target_local_actuation"
        )
        self.assertEqual(
            [item["check_id"] for item in criterion["required_checks"]],
            [
                "confirmed_button_press_contact_phase",
                "button_press_event_closed",
            ],
        )
        self.assertEqual(
            MODULE.match_action_success_criterion(
                "pick up the container at <83, 126> that hides the green cube",
                v8_1_catalog,
            )["criterion_id"],
            "pickup_container_strict_clearance",
        )
        self.assertIsNone(
            MODULE.match_action_success_criterion(
                "press the button to stop the cube", v8_1_catalog
            )
        )
        prompt = v8_1["system_prompt"].read_text(encoding="utf-8")
        self.assertIn("A press is an event", prompt)
        self.assertIn("Container appearance", prompt)
        self.assertNotIn("ButtonUnmask", prompt)
        legacy = MODULE.PROMPT_PROFILES["retrieval_v9_action_success"]
        self.assertEqual(
            legacy["action_success_criteria_path"],
            v8_1["action_success_criteria_path"],
        )

    def test_v8_completion_gate_blocks_unsupported_skill_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt=sys.stdin.read()\n"
                "counter=pathlib.Path(__file__).with_name('counter.txt')\n"
                "index=int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index+1))\n"
                "if index == 0:\n"
                " assert '\"criterion_id\": \"none\"' in prompt\n"
                " response={\n"
                "  'recent_status':'initial','recent_outcome':'Initial scene.',\n"
                "  'completion_check':{'criterion_id':'none','decision':'not_applicable','checks':[],'reason':'No criterion applies.'},\n"
                "  'updated_progress':'Phase: green pickup. Completed: none.',\n"
                "  'skill_key':'skill_b8064b602776',\n"
                "  'execution_state':{'held_object':'none','object_support':'table','gripper_state':'open'},\n"
                "  'instruction_strategy':'normal',\n"
                "  'current_subtask_template':'wrong wording',\n"
                "  'points':[{'id':'point_1','object_description':'green container','source_frame_id':'r_f0000','x_norm':500,'y_norm':500}]}\n"
                "else:\n"
                " assert 'pickup_container_strict_clearance' in prompt\n"
                " checks=[\n"
                "  {'check_id':'correct_container_held','verdict':'yes','evidence':'Container appears held.'},\n"
                "  {'check_id':'hidden_cube_fully_visible','verdict':'yes','evidence':'Cube is visible.'},\n"
                "  {'check_id':'clear_generous_vertical_separation','verdict':'no','evidence':'The gap remains small.'},\n"
                "  {'check_id':'sustained_across_recent_frames','verdict':'yes','evidence':'Lift persists across frames.'}]\n"
                " response={\n"
                "  'recent_status':'completed','recent_outcome':'The container appears lifted.',\n"
                "  'completion_check':{'criterion_id':'pickup_container_strict_clearance','decision':'not_satisfied','checks':checks,'reason':'Clear generous separation is absent.'},\n"
                "  'updated_progress':'Phase: incorrectly advanced. Completed: green pickup.',\n"
                "  'skill_key':'skill_f2376d9bd67c',\n"
                "  'execution_state':{'held_object':'container','object_support':'airborne','gripper_state':'closed'},\n"
                "  'instruction_strategy':'advance_skill',\n"
                "  'current_subtask_template':'wrong successor wording','points':[]}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1])\n"
                "out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="retrieval_v8_action_success",
                retrieval_scheme="R0",
                retrieval_top_k=5,
                verify_login=False,
                timeout_seconds=10,
            )
            run_dir = temporary / "run"
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description=(
                    "Pick up the upper-left container hiding the green cube."
                ),
                planner_guidance={},
                task="VideoUnmask",
                episode=1,
                init_case_id="VideoUnmask_episode_01",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            first, _ = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
            )
            second, second_record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[16]
            )
            self.assertEqual(first, "pick up the container at <50, 100> that hides the green cube")
            self.assertEqual(second, first)
            self.assertTrue(second_record["completion_gate"]["semantic_switch_blocked"])
            self.assertTrue(second_record["completion_gate"]["completed_status_blocked"])
            self.assertEqual(second_record["response"]["recent_status"], "uncertain")
            self.assertEqual(second_record["skill_key"], "skill_b8064b602776")
            self.assertIn("Completion gate", second_record["response"]["updated_progress"])

    def test_v8_2_button_confirmation_requires_one_extra_press_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt=sys.stdin.read()\n"
                "counter=pathlib.Path(__file__).with_name('counter.txt')\n"
                "index=int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index+1))\n"
                "if index == 0:\n"
                " status='initial'; skill='skill_6bb26132e502'; template='press';\n"
                " check={'criterion_id':'none','decision':'not_applicable','checks':[],'reason':'No criterion applies.'}\n"
                " points=[{'id':'point_1','object_description':'button','source_frame_id':'r_f0000','x_norm':500,'y_norm':500}]\n"
                "else:\n"
                " assert 'press_button_v6_visual_event' in prompt\n"
                " status='completed'; skill='skill_e4ffdfcaa9f6'; template='pickup'\n"
                " checks=[\n"
                "  {'check_id':'button_local_press_like_interaction','verdict':'yes','evidence':'Aligned press-like interaction is visible.'},\n"
                "  {'check_id':'physically_consistent_press_completion','verdict':'yes','evidence':'Downstroke and withdrawal are physically consistent.'}]\n"
                " check={'criterion_id':'press_button_v6_visual_event','decision':'satisfied','checks':checks,'reason':'Button-local press event is visible.'}\n"
                " frame='r_f0016' if index == 1 else 'r_f0032'\n"
                " points=[{'id':'point_1','object_description':'blue container','source_frame_id':frame,'x_norm':400,'y_norm':600}]\n"
                "response={'recent_status':status,'recent_outcome':'Visible event.',\n"
                " 'completion_check':check,'updated_progress':'Phase: advance.',\n"
                " 'skill_key':skill,'execution_state':{'held_object':'none','object_support':'table','gripper_state':'open'},\n"
                " 'instruction_strategy':'advance_skill' if index else 'normal',\n"
                " 'current_subtask_template':template,'points':points}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1])\n"
                "out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="retrieval_v8_2_v6_button_confirm",
                retrieval_scheme="R0",
                retrieval_top_k=5,
                verify_login=False,
                timeout_seconds=10,
            )
            planner.start_episode(
                run_dir=temporary / "run",
                detailed_task_description=(
                    "Press the button, then pick up the container hiding the blue cube."
                ),
                planner_guidance={},
                task="ButtonUnmask",
                episode=1,
                init_case_id="ButtonUnmask_episode_01",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            first, _ = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
            )
            second, second_record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[16]
            )
            third, third_record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[32]
            )
            self.assertIn("press the button", first)
            self.assertEqual(second, first)
            self.assertTrue(second_record["completion_gate"]["confirmation_retry_required"])
            self.assertEqual(second_record["completion_gate"]["confirmation_count_after"], 1)
            self.assertIn("pick up the container", third)
            self.assertFalse(third_record["completion_gate"]["confirmation_retry_required"])
            self.assertEqual(third_record["completion_gate"]["confirmation_count_before"], 1)

    def test_v7_target_local_profile_is_generic_v6_plus_completion_gate(self) -> None:
        v6 = MODULE.PROMPT_PROFILES["retrieval_v6"]
        v7 = MODULE.PROMPT_PROFILES["retrieval_v7_target_local"]
        self.assertEqual(v7["user_prompt"], v6["user_prompt"])
        self.assertEqual(v7["output_schema"], v6["output_schema"])
        self.assertTrue(v7["enable_global_retrieval"])
        self.assertTrue(v7["catalog_retrieval_only"])
        prompt = v7["system_prompt"].read_text(encoding="utf-8")
        self.assertIn("Completion-evidence discipline", prompt)
        self.assertIn("local to that same target", prompt)
        self.assertIn("A change to another object", prompt)
        self.assertNotIn("ButtonUnmask", prompt)

    def test_v6_global_retrieval_selects_known_predecessor_and_successor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt = sys.stdin.read()\n"
                "counter = pathlib.Path(__file__).with_name('counter.txt')\n"
                "index = int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index + 1))\n"
                "assert 'automatically extracted released pi0.5 training labels' in prompt\n"
                "assert 'skill_ac7483a073f8' in prompt\n"
                "assert 'skill_5ace02f82c6b' in prompt\n"
                "if index == 0:\n"
                " status = 'initial'; skill = 'skill_ac7483a073f8'\n"
                " template = 'close only the gripper'\n"
                " points = [{'id':'point_1','object_description':'peg',\n"
                "  'source_frame_id':'r_f0000','x_norm':500,'y_norm':500}]\n"
                " state = {'held_object':'none','object_support':'table','gripper_state':'open'}\n"
                "else:\n"
                " status = 'completed'; skill = 'skill_5ace02f82c6b'\n"
                " template = 'move the tool a little'\n"
                " points = [\n"
                "  {'id':'point_1','object_description':'cube','source_frame_id':'r_f0016','x_norm':400,'y_norm':500},\n"
                "  {'id':'point_2','object_description':'target','source_frame_id':'r_f0016','x_norm':600,'y_norm':500}]\n"
                " state = {'held_object':'peg','object_support':'table','gripper_state':'closed'}\n"
                "response={'recent_status':status,'recent_outcome':'Visible '+status+'.',\n"
                " 'updated_progress':'Phase: peg pushing. Execution state: '+status+'.',\n"
                " 'skill_key':skill,'execution_state':state,'instruction_strategy':'normal',\n"
                " 'current_subtask_template':template,'points':points}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1])\n"
                "out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="retrieval_v6",
                retrieval_scheme="R0",
                retrieval_top_k=5,
                verify_login=False,
                timeout_seconds=10,
            )
            run_dir = temporary / "run"
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description=(
                    "Use the demonstrated peg-pushing method: grasp the peg, then "
                    "move the cube to the target with the held peg."
                ),
                planner_guidance={"fixed_facts": ["Required method: peg pushing."]},
                task="MoveCube",
                episode=3,
                init_case_id="MoveCube_episode_03",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            first, first_record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
            )
            second, second_record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[16]
            )
            self.assertEqual(first, "pick up the peg at <50, 100>")
            self.assertEqual(
                second,
                "hook the cube at <50, 80> to the target at <50, 120> with the peg",
            )
            self.assertTrue(first_record["template_canonicalized_by_harness"])
            self.assertTrue(second_record["template_canonicalized_by_harness"])
            self.assertEqual(first_record["retrieval_scheme"], "R0")
            self.assertEqual(len(first_record["retrieval_candidates"]), 5)
            config = json.loads((run_dir / "episode_config.json").read_text())
            self.assertEqual(config["vocabulary_entry_count"], 122)
            self.assertFalse(config["novel_instruction_exploration_enabled"])

    def test_v5_move_cube_direct_push_enforces_oracle_skill_granularity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt = sys.stdin.read()\n"
                "assert 'direct_gripper_push_cube_to_target' in prompt\n"
                "assert 'pick_up_cube' not in prompt\n"
                "response = {\n"
                " 'recent_status': 'initial',\n"
                " 'recent_outcome': 'The cube and target are separated on the table.',\n"
                " 'updated_progress': 'Phase: direct push. Completed: none.',\n"
                " 'skill_key': 'direct_gripper_push_cube_to_target',\n"
                " 'execution_state': {'held_object': 'none', 'object_support': 'table',\n"
                "                     'gripper_state': 'open'},\n"
                " 'instruction_strategy': 'normal',\n"
                " 'current_subtask_template': 'set the empty gripper aperture to zero',\n"
                " 'points': [\n"
                "  {'id': 'point_1', 'object_description': 'red cube',\n"
                "   'source_frame_id': 'r_f0000', 'x_norm': 600, 'y_norm': 500},\n"
                "  {'id': 'point_2', 'object_description': 'target',\n"
                "   'source_frame_id': 'r_f0000', 'x_norm': 400, 'y_norm': 500}]}\n"
                "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
                "output.write_text(json.dumps(response))\n"
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'fresh'}))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="oracle_skills_v5",
                verify_login=False,
                timeout_seconds=10,
            )
            run_dir = temporary / "run"
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description=(
                    "Use the demonstrated direct-gripper-pushing method. Do not grasp "
                    "or lift the cube."
                ),
                planner_guidance={"fixed_facts": ["Use direct gripper pushing."]},
                task="MoveCube",
                episode=23,
                init_case_id="MoveCube_episode_23",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            subtask, record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
            )

            self.assertEqual(
                subtask,
                "Close the gripper and push the cube at <50, 120> to the target at <50, 80>",
            )
            self.assertTrue(record["template_canonicalized_by_harness"])
            self.assertEqual(
                record["planner_proposed_subtask_template"],
                "set the empty gripper aperture to zero",
            )
            self.assertEqual(
                record["allowed_executor_skill_ids"],
                ["direct_gripper_push_cube_to_target"],
            )
            config = json.loads((run_dir / "episode_config.json").read_text())
            self.assertEqual(config["oracle_skill_method"], "direct_gripper_push")

    def test_v4_lessons_trigger_only_after_confirmed_stall_and_reset_per_episode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt = sys.stdin.read()\n"
                "counter = pathlib.Path(__file__).with_name('counter.txt')\n"
                "index = int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index + 1))\n"
                "statuses = ['initial'] + ['stagnant'] * 7 + ['completed']\n"
                "status = statuses[index]\n"
                "if index <= 5:\n"
                " assert '\"mode\": \"explore_instruction\"' not in prompt\n"
                "if index == 6:\n"
                " assert '\"mode\": \"explore_instruction\"' in prompt\n"
                " assert 'place the cube on the table' in prompt\n"
                " assert '\"do_not_retry\"' in prompt\n"
                " assert '\"exploration_candidates_tried\": 0' in prompt\n"
                "if index == 7:\n"
                " assert '\"exploration_candidates_tried\": 1' in prompt\n"
                "if index in (6, 7):\n"
                " skill = 'put_down_cube'; instruction = 'open the gripper'\n"
                " strategy = 'atomic_breakdown'\n"
                " state = {'held_object': 'cube', 'object_support': 'airborne',\n"
                "          'gripper_state': 'closed'}\n"
                "elif index == 8:\n"
                " skill = 'reach_next_cube'; instruction = 'reach for the next cube'\n"
                " strategy = 'normal'\n"
                " state = {'held_object': 'none', 'object_support': 'table',\n"
                "          'gripper_state': 'open'}\n"
                "else:\n"
                " skill = 'put_down_cube'; instruction = 'place the cube on the table'\n"
                " strategy = 'normal'\n"
                " state = {'held_object': 'cube', 'object_support': 'airborne',\n"
                "          'gripper_state': 'closed'}\n"
                "response = {\n"
                " 'recent_status': status,\n"
                " 'recent_outcome': 'Visible evidence for ' + status + '.',\n"
                " 'updated_progress': 'Phase: placing. Execution state: ' + status + '.',\n"
                " 'skill_key': skill, 'execution_state': state,\n"
                " 'instruction_strategy': strategy,\n"
                " 'current_subtask_template': instruction, 'points': []}\n"
                "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
                "output.write_text(json.dumps(response))\n"
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'fresh'}))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="lesson_v4",
                verify_login=False,
                timeout_seconds=10,
            )
            run_dir = temporary / "run"
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description="Put the held cube down, then get the next cube.",
                task="MoveCube",
                episode=0,
                init_case_id="MoveCube_episode_00",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            records = []
            for _ in range(9):
                _, record = planner.plan(
                    front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
                )
                records.append(record)

            self.assertEqual(
                [record["planner_mode"] for record in records],
                [
                    "normal",
                    "normal",
                    "normal",
                    "normal",
                    "rethink_verify",
                    "rethink_verify",
                    "explore_instruction",
                    "explore_instruction",
                    "explore_instruction",
                ],
            )
            self.assertEqual(records[5]["episode_local_lesson_count"], 0)
            memory = json.loads((run_dir / "lesson_memory.json").read_text())
            lessons = memory["lessons"]
            self.assertEqual(memory["scope"], "episode_local")
            self.assertFalse(memory["cross_episode_loading"])
            by_instruction = {
                item["candidate_instruction"]: item["lesson_status"] for item in lessons
            }
            self.assertEqual(
                by_instruction["place the cube on the table"], "ineffective"
            )
            self.assertEqual(by_instruction["open the gripper"], "successful")
            events = [
                json.loads(line)
                for line in (run_dir / "exploration_events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(any(row["event"] == "exploration_started" for row in events))
            self.assertTrue(any(row["event"] == "exploration_ended" for row in events))

            next_run_dir = temporary / "next_run"
            planner.start_episode(
                run_dir=next_run_dir,
                detailed_task_description="Put the held cube down.",
                task="MoveCube",
                episode=1,
                init_case_id="MoveCube_episode_01",
            )
            reset_memory = json.loads(
                (next_run_dir / "lesson_memory.json").read_text()
            )
            self.assertEqual(reset_memory["lessons"], [])

    def test_v3_rethinking_status_drives_harness_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt = sys.stdin.read()\n"
                "counter = pathlib.Path(__file__).with_name('counter.txt')\n"
                "index = int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index + 1))\n"
                "statuses = ['initial', 'stagnant', 'uncertain', 'stagnant', 'completed']\n"
                "status = statuses[index]\n"
                "if index == 0:\n"
                " assert 'method=peg_hook' in prompt\n"
                "if index == 4:\n"
                " assert '\"rethinking_required_this_turn\": true' in prompt\n"
                "if index < 4:\n"
                " template = 'pick up the peg at <point_1>'\n"
                " points = [{'id': 'point_1', 'object_description': 'peg',\n"
                "   'source_frame_id': 'r_f0000', 'x_norm': 500, 'y_norm': 500}]\n"
                "else:\n"
                " template = 'hook the cube at <point_1> to the target at <point_2> with the peg'\n"
                " points = [\n"
                "  {'id': 'point_1', 'object_description': 'cube',\n"
                "   'source_frame_id': 'r_f0000', 'x_norm': 400, 'y_norm': 500},\n"
                "  {'id': 'point_2', 'object_description': 'target',\n"
                "   'source_frame_id': 'r_f0000', 'x_norm': 600, 'y_norm': 500}]\n"
                "response = {\n"
                " 'recent_status': status,\n"
                " 'recent_outcome': 'Visible evidence for ' + status + '.',\n"
                " 'updated_progress': 'Phase: tool use. Execution state: status=' + status + '.',\n"
                " 'current_subtask_template': template, 'points': points}\n"
                "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
                "output.write_text(json.dumps(response))\n"
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'fresh'}))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="rethinking_v3",
                verify_login=False,
                timeout_seconds=10,
            )
            run_dir = temporary / "run"
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description="Move the cube with the demonstrated peg.",
                planner_guidance={
                    "fixed_facts": ["method=peg_hook"],
                    "state_to_track": ["peg_held"],
                    "progress_rules": ["Advance after the peg is held."],
                    "execution_constraints": ["Do not grasp the cube directly."],
                },
                task="MoveCube",
                episode=1,
                init_case_id="MoveCube_episode_01",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            records = []
            for _ in range(5):
                _, record = planner.plan(
                    front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
                )
                records.append(record)

            self.assertEqual(
                [record["response"]["recent_status"] for record in records],
                ["initial", "stagnant", "uncertain", "stagnant", "completed"],
            )
            self.assertEqual(records[3]["consecutive_no_progress_rounds_after"], 3)
            self.assertTrue(records[4]["rethinking_required_this_turn"])
            self.assertEqual(records[4]["consecutive_no_progress_rounds_after"], 0)
            config = json.loads((run_dir / "episode_config.json").read_text())
            self.assertEqual(config["prompt_profile"], "rethinking_v3")
            self.assertEqual(config["planner_guidance"]["fixed_facts"], ["method=peg_hook"])

    def test_v3_validation_rejects_unknown_recent_status(self) -> None:
        response = {
            "recent_status": "stuck",
            "recent_outcome": "No visible change.",
            "updated_progress": "Phase: acquiring the peg.",
            "current_subtask_template": "pick up the peg",
            "points": [],
        }
        with self.assertRaisesRegex(ValueError, "recent_status"):
            MODULE.validate_response(
                response,
                allowed_recent_front_ids=set(),
                require_recent_status=True,
            )

    def test_two_rounds_pass_only_canonical_text_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt = sys.stdin.read()\n"
                "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
                "if 'No rollout progress has been established yet.' in prompt:\n"
                " response = {\n"
                "  'recent_outcome': 'No action has run yet.',\n"
                "  'updated_progress': 'The cube remains ungrasped beside the target.',\n"
                "  'current_subtask_template': 'pick up the cube at <point_1>',\n"
                "  'points': [{'id': 'point_1', 'object_description': 'cube',\n"
                "    'source_frame_id': 'r_f0000', 'x_norm': 500, 'y_norm': 500}]}\n"
                "else:\n"
                " assert 'The cube remains ungrasped beside the target.' in prompt\n"
                " assert 'pick up the cube at <50, 100>' in prompt\n"
                " response = {\n"
                "  'recent_outcome': 'The cube is now securely grasped.',\n"
                "  'updated_progress': 'The cube is grasped; placement is not complete.',\n"
                "  'current_subtask_template': 'place the cube on <point_1>',\n"
                "  'points': [{'id': 'point_1', 'object_description': 'target',\n"
                "    'source_frame_id': 'r_f0016', 'x_norm': 250, 'y_norm': 750}]}\n"
                "output.write_text(json.dumps(response))\n"
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'fresh'}))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            run_dir = temporary / "run"
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex), verify_login=False, timeout_seconds=10
            )
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description="Move the cube onto the target.",
                task="MoveCube",
                episode=1,
                init_case_id="MoveCube_episode_01",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            first_subtask, first_record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
            )
            second_subtask, second_record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[16]
            )

            self.assertEqual(first_subtask, "pick up the cube at <50, 100>")
            self.assertEqual(second_subtask, "place the cube on <74, 50>")
            self.assertEqual(
                second_record["previous_progress"],
                first_record["response"]["updated_progress"],
            )
            self.assertEqual(second_record["historical_image_count"], 0)
            self.assertFalse((run_dir / "keyframes").exists())
            history = json.loads((run_dir / "progress_history.json").read_text())
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]["updated_progress"], planner.progress)
            for round_index in range(2):
                manifest = json.loads(
                    (
                        run_dir
                        / "rounds"
                        / f"round_{round_index:04d}"
                        / "input_manifest.json"
                    ).read_text()
                )
                self.assertEqual(manifest["historical_image_count"], 0)
                self.assertEqual(manifest["attachment_count"], 2)

    def test_stale_grounding_and_noninitial_status_retry_same_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "prompt=sys.stdin.read()\n"
                "counter=pathlib.Path(__file__).with_name('counter.txt')\n"
                "index=int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(index+1))\n"
                "if index == 0:\n"
                " status='initial'; frame='r_f0000'\n"
                "elif index == 1:\n"
                " status='initial'; frame='r_f0000'\n"
                "else:\n"
                " assert 'rejected by the harness' in prompt\n"
                " assert 'Historical frame IDs in text memory are provenance only' in prompt\n"
                " status='progressing'; frame='r_f0016'\n"
                "response={'recent_status':status,'recent_outcome':'The cube is visible.',\n"
                " 'updated_progress':'Phase: pickup. Completed: none.',\n"
                " 'current_subtask_template':'pick up the cube at <point_1>',\n"
                " 'points':[{'id':'point_1','object_description':'cube',\n"
                " 'source_frame_id':frame,'x_norm':500,'y_norm':500}]}\n"
                "out=pathlib.Path(args[args.index('--output-last-message')+1])\n"
                "out.write_text(json.dumps(response))\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fresh'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            run_dir = temporary / "run"
            planner = MODULE.CodexTextProgressPlanner(
                codex_bin=str(fake_codex),
                prompt_profile="rethinking_v3",
                verify_login=False,
                timeout_seconds=10,
            )
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description="Pick up the cube.",
                task="MoveCube",
                episode=1,
                init_case_id="MoveCube_episode_01",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            planner.plan(front_frames=[front], wrist_frames=[wrist], frame_ids=[0])
            subtask, record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[16]
            )
            self.assertEqual(subtask, "pick up the cube at <50, 100>")
            self.assertEqual(record["response"]["recent_status"], "progressing")
            self.assertEqual(record["codex_output_validation_retry_count"], 1)
            self.assertEqual(record["codex_usage"]["input_tokens"], 20)
            self.assertEqual(planner.round_index, 2)
            retry = json.loads(
                (run_dir / "rounds/round_0001/validation_retries.json").read_text()
            )
            self.assertFalse(retry["environment_advanced_between_attempts"])
            self.assertTrue(
                (run_dir / "rounds/round_0001/invalid_response.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
