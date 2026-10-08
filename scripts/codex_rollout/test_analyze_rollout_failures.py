from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "analyze_rollout_failures.py"
SPEC = importlib.util.spec_from_file_location("analyze_rollout_failures", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def summary(description: str, result: str = "fail") -> dict:
    return {
        "result": result,
        "detailed_task_description_for_planner": description,
        "planner_guidance_for_planner": {"fixed_facts": [description]},
        "final_progress": "incomplete",
    }


def swing_summary(description: str, result: str = "fail") -> dict:
    value = summary(description, result)
    value["task"] = "SwingXtimes"
    return value


def diagnostic(round_index: int, stage: str) -> dict:
    return {
        "round_index": round_index,
        "after": {"current_task_name": stage},
    }


def history_row(round_index: int, status: str, subtask: str) -> dict:
    return {
        "round_index": round_index,
        "recent_rollout_frame_ids": [round_index * 16],
        "recent_status": status,
        "recent_outcome": status,
        "next_grounded_subtask": subtask,
        "execution_state": {},
    }


class FailureAnalyzerTests(unittest.TestCase):
    def test_test_split_is_rejected_by_analysis_guard(self) -> None:
        with self.assertRaisesRegex(ValueError, "only on train or val"):
            MODULE.assert_analysis_split({"dataset_split": "test"}, "test")
        with self.assertRaisesRegex(ValueError, "split audit failed"):
            MODULE.assert_analysis_split({"dataset_split": "test"}, "train")
        MODULE.assert_analysis_split({"dataset_split": "train"}, "train")

    def test_stagnation_early_stop_is_analyzable_as_recovery_failure(self) -> None:
        rows = [history_row(0, "initial", "close the gripper and push the cube")]
        rows.extend(
            history_row(index, "stagnant", "close the gripper and push the cube")
            for index in range(1, 5)
        )
        value = MODULE.heuristic_analysis(
            summary("Use direct gripper pushing.", result="stagnation_early_stop"),
            rows,
            [diagnostic(0, "close the gripper and push the cube")],
        )
        self.assertEqual(value["failure_type"], "recovery_failure")
        self.assertEqual(value["responsible_module"], "recovery_policy")

    def test_detects_init_method_mismatch(self) -> None:
        value = MODULE.heuristic_analysis(
            summary("Use direct gripper pushing."),
            [history_row(0, "initial", "close the gripper and push the cube")],
            [diagnostic(0, "Pick up the cube")],
        )
        self.assertEqual(value["failure_type"], "task_understanding_error")
        self.assertEqual(value["responsible_module"], "init_processor")
        self.assertEqual(value["first_divergence_round"], 0)

    def test_detects_premature_peg_transition(self) -> None:
        rows = [
            history_row(0, "initial", "pick up the peg"),
            history_row(1, "progressing", "hook the cube with the peg"),
        ]
        value = MODULE.heuristic_analysis(
            summary("Required method: peg pushing."),
            rows,
            [diagnostic(0, "Pick up the peg"), diagnostic(1, "Pick up the peg")],
        )
        self.assertEqual(value["failure_type"], "completion_error")
        self.assertEqual(value["first_divergence_round"], 1)

    def test_does_not_blame_recovered_premature_transition(self) -> None:
        rows = [
            history_row(0, "initial", "pick up the peg"),
            history_row(1, "stagnant", "hook the cube with the peg"),
            history_row(2, "progressing", "pick up the peg"),
        ]
        rows.extend(
            history_row(i, "stagnant", "hook the cube with the peg")
            for i in range(3, 7)
        )
        value = MODULE.heuristic_analysis(
            summary("Required method: peg pushing.", result="timeout"),
            rows,
            [
                diagnostic(0, "Pick up the peg"),
                diagnostic(1, "Pick up the peg"),
                diagnostic(2, "Hook the cube with the peg"),
                diagnostic(3, "Hook the cube with the peg"),
                diagnostic(4, "Hook the cube with the peg"),
                diagnostic(5, "Hook the cube with the peg"),
                diagnostic(6, "Hook the cube with the peg"),
            ],
        )
        self.assertEqual(value["failure_type"], "recovery_failure")
        self.assertEqual(value["first_divergence_round"], 3)

    def test_visible_peg_operation_can_override_stale_evaluator_stage(self) -> None:
        rows = [
            history_row(0, "initial", "pick up the peg"),
            history_row(1, "completed", "hook the cube with the peg"),
            history_row(2, "progressing", "hook the cube with the peg"),
        ]
        rows[2]["execution_state"] = {"held_object": "peg"}
        rows[2]["recent_outcome"] = "The held peg moved the cube toward the target."
        rows.extend(
            history_row(i, "stagnant", "hook the cube with the peg")
            for i in range(3, 7)
        )
        value = MODULE.heuristic_analysis(
            summary("Required method: peg pushing.", result="timeout"),
            rows,
            [diagnostic(i, "Pick up the peg") for i in range(7)],
        )
        self.assertEqual(value["failure_type"], "recovery_failure")

    def test_disagreement_is_not_allowed_to_create_experience(self) -> None:
        heuristic = {
            "failure_type": "completion_error",
            "responsible_module": "completion_judge",
        }
        model = {
            "failure_type": "recovery_failure",
            "responsible_module": "recovery_policy",
        }
        selected, adjudication = MODULE.adjudicate_analysis(heuristic, model)
        self.assertIs(selected, heuristic)
        self.assertEqual(adjudication["status"], "disputed")
        self.assertFalse(adjudication["candidate_allowed"])

    def test_detects_recovery_failure(self) -> None:
        rows = [history_row(0, "initial", "close the gripper and push the cube")]
        rows.extend(
            history_row(i, "stagnant", "close the gripper and push the cube")
            for i in range(1, 5)
        )
        value = MODULE.heuristic_analysis(
            summary("Use direct gripper pushing.", result="timeout"),
            rows,
            [diagnostic(i, "Close the gripper and push the cube to the target") for i in range(5)],
        )
        self.assertEqual(value["failure_type"], "recovery_failure")
        self.assertEqual(value["responsible_module"], "recovery_policy")
        self.assertEqual(value["first_divergence_round"], 1)

    def test_detects_unresolved_swing_sequence_advance(self) -> None:
        rows = [
            history_row(0, "initial", "pick up the blue cube"),
            history_row(
                1,
                "completed",
                "move to the top of the right-side target for the first time",
            ),
            history_row(
                2,
                "completed",
                "move to the top of the left-side target for the first time",
            ),
            history_row(3, "completed", "put the blue cube on the table"),
            history_row(4, "completed", "press the button"),
        ]
        value = MODULE.heuristic_analysis(
            swing_summary("Complete exactly two right-to-left cycles."),
            rows,
            [
                diagnostic(0, "pick up the blue cube"),
                diagnostic(
                    1, "move to the top of the right-side target for the first time"
                ),
                diagnostic(
                    2, "move to the top of the left-side target for the first time"
                ),
                diagnostic(
                    3, "move to the top of the right-side target for the second time"
                ),
                diagnostic(
                    4, "move to the top of the right-side target for the second time"
                ),
            ],
        )
        self.assertEqual(value["failure_type"], "completion_error")
        self.assertEqual(value["first_divergence_round"], 3)
        self.assertEqual(
            value["derived_context"]["experience_scope"],
            "swing_ordered_cycle_ledger",
        )

    def test_swing_advance_is_not_causal_if_later_stage_catches_up(self) -> None:
        rows = [
            history_row(0, "initial", "pick up the red cube"),
            history_row(
                1,
                "completed",
                "move to the top of the right-side target for the first time",
            ),
            history_row(
                2,
                "completed",
                "move to the top of the left-side target for the first time",
            ),
            history_row(
                3,
                "progressing",
                "move to the top of the left-side target for the first time",
            ),
        ]
        value = MODULE.heuristic_analysis(
            swing_summary("Complete one right-to-left cycle.", result="timeout"),
            rows,
            [
                diagnostic(0, "pick up the red cube"),
                diagnostic(
                    1, "move to the top of the right-side target for the first time"
                ),
                diagnostic(
                    2, "move to the top of the right-side target for the first time"
                ),
                diagnostic(
                    3, "move to the top of the left-side target for the first time"
                ),
            ],
        )
        self.assertNotEqual(value["failure_type"], "completion_error")

    def test_aggregate_writes_candidate_report(self) -> None:
        analysis = {
            "episode": 1,
            "source_result": "fail",
            "model_analysis": None,
            "heuristic_analysis": {
                "failure_type": "completion_error",
                "responsible_module": "completion_judge",
                "first_divergence_round": 2,
                "experience_eligibility": "eligible_candidate",
                "confidence": 0.9,
                "analysis_summary": "premature transition",
                "derived_context": {
                    "evaluator_method": "peg_push",
                    "guidance_method": "peg_push",
                    "selected_method": "peg_push",
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            MODULE.write_aggregate(output, [analysis])
            payload = json.loads((output / "aggregate.json").read_text())
            self.assertEqual(payload["failure_type_counts"], {"completion_error": 1})
            self.assertIn("candidate_only", (output / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
