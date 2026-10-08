from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("planner.py")
SPEC = importlib.util.spec_from_file_location("codex_rollout_planner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PlannerTests(unittest.TestCase):
    def test_normalized_coordinates_render_in_robomme_y_x_order(self) -> None:
        response = {
            "stage_summary": "The cube is on the table.",
            "current_subtask_template": "pick up the cube at <point_1>",
            "points": [
                {
                    "id": "point_1",
                    "object_description": "cube center",
                    "source_frame_id": "r_f0000",
                    "x_norm": 552,
                    "y_norm": 396,
                }
            ],
            "new_keyframes": [],
        }
        MODULE.validate_response(
            response,
            allowed_recent_ids={"r_f0000", "r_w0000"},
            allowed_recent_front_ids={"r_f0000"},
        )
        rendered, points = MODULE.render_grounded_subtask(
            response, {"r_f0000": (256, 256)}
        )
        self.assertEqual(rendered, "pick up the cube at <101, 141>")
        self.assertEqual((points[0]["y_px"], points[0]["x_px"]), (101, 141))

    def test_validation_rejects_wrist_grounding_and_unmatched_placeholder(self) -> None:
        response = {
            "stage_summary": "Visible state.",
            "current_subtask_template": "pick at <point_1>",
            "points": [
                {
                    "id": "point_1",
                    "object_description": "object",
                    "source_frame_id": "r_w0000",
                    "x_norm": 1,
                    "y_norm": 2,
                }
            ],
            "new_keyframes": [],
        }
        with self.assertRaisesRegex(ValueError, "recent front frame"):
            MODULE.validate_response(
                response,
                allowed_recent_ids={"r_f0000", "r_w0000"},
                allowed_recent_front_ids={"r_f0000"},
            )
        response["points"] = []
        with self.assertRaisesRegex(ValueError, "placeholders must match"):
            MODULE.validate_response(
                response,
                allowed_recent_ids={"r_f0000", "r_w0000"},
                allowed_recent_front_ids={"r_f0000"},
            )

    def test_normalization_drops_only_unreferenced_points(self) -> None:
        response = {
            "stage_summary": "The cube is already held.",
            "current_subtask_template": "lift the cube",
            "points": [
                {
                    "id": "point_1",
                    "object_description": "cube center",
                    "source_frame_id": "r_f0000",
                    "x_norm": 500,
                    "y_norm": 500,
                }
            ],
            "new_keyframes": [],
        }
        normalized, notes = MODULE.normalize_response(response)
        self.assertEqual(normalized["points"], [])
        self.assertEqual(notes[0]["point_ids"], ["point_1"])
        self.assertEqual(len(response["points"]), 1, "raw response must remain unchanged")
        MODULE.validate_response(
            normalized,
            allowed_recent_ids={"r_f0000", "r_w0000"},
            allowed_recent_front_ids={"r_f0000"},
        )

        missing = dict(response)
        missing["current_subtask_template"] = "lift at <point_2>"
        normalized_missing, _ = MODULE.normalize_response(missing)
        with self.assertRaisesRegex(ValueError, "placeholders must match"):
            MODULE.validate_response(
                normalized_missing,
                allowed_recent_ids={"r_f0000", "r_w0000"},
                allowed_recent_front_ids={"r_f0000"},
            )
    def test_fake_codex_round_saves_reason_and_keyframe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['--version']:\n"
                " print('codex-cli fake'); raise SystemExit(0)\n"
                "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
                "response = {\n"
                " 'stage_summary': 'The cube is visible and not yet grasped.',\n"
                " 'current_subtask_template': 'pick up the cube at <point_1>',\n"
                " 'points': [{'id': 'point_1', 'object_description': 'cube center',\n"
                "   'source_frame_id': 'r_f0000', 'x_norm': 999, 'y_norm': 0}],\n"
                " 'new_keyframes': [{'frame_id': 'r_w0000',\n"
                "   'description': 'Open gripper above the cube',\n"
                "   'reason': 'Records the pre-grasp state'}]}\n"
                "output.write_text(json.dumps(response))\n"
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'fake-thread'}))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {\n"
                " 'input_tokens': 100, 'cached_input_tokens': 20, 'output_tokens': 30}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            run_dir = temporary / "run"
            planner = MODULE.CodexRolloutPlanner(
                codex_bin=str(fake_codex), verify_login=False, timeout_seconds=10
            )
            planner.start_episode(
                run_dir=run_dir,
                detailed_task_description="Move the cube onto the marked target.",
                task="MoveCube",
                episode=0,
                init_case_id="MoveCube_episode_00",
            )
            front = np.zeros((100, 200, 3), dtype=np.uint8)
            wrist = np.zeros((80, 120, 3), dtype=np.uint8)
            subtask, record = planner.plan(
                front_frames=[front], wrist_frames=[wrist], frame_ids=[0]
            )
            self.assertEqual(subtask, "pick up the cube at <0, 199>")
            self.assertEqual(record["response"]["stage_summary"], "The cube is visible and not yet grasped.")
            self.assertEqual(record["codex_usage"]["input_tokens"], 100)
            self.assertTrue((run_dir / "keyframes" / "k_w0000.png").is_file())
            self.assertFalse((run_dir / "rounds" / "round_0000" / "recent_inputs").exists())
            index = json.loads((run_dir / "keyframes" / "index.json").read_text())
            self.assertEqual(index[0]["reason"], "Records the pre-grasp state")


if __name__ == "__main__":
    unittest.main()
