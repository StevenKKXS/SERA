from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_init_video.py")
SPEC = importlib.util.spec_from_file_location("run_init_video", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HarnessTests(unittest.TestCase):
    def test_task_filter_precedes_deterministic_sharding(self) -> None:
        cases = [
            MODULE.CaseSpec(f"A_episode_{index:02d}", "A", index, "a", None)
            for index in range(3)
        ] + [
            MODULE.CaseSpec(f"B_episode_{index:02d}", "B", index, "b", None)
            for index in range(3)
        ]
        selected = MODULE.select_cases(cases, [], ["B"], 1, 2, None)
        self.assertEqual([case.case_id for case in selected], ["B_episode_01"])

    def test_prompt_is_stable_and_contains_ordered_frame_manifest(self) -> None:
        prompt = MODULE.render_prompt(
            "system",
            (
                "Task: {task_name}\nSpecification: {task_specification}\n"
                "Description: {task_description}\n{video_context}"
            ),
            "MoveCube",
            "exactly three methods",
            "move the cube",
            [
                {"frame_id": "i_f0000", "timestamp_seconds": 0.0},
                {"frame_id": "i_f0003", "timestamp_seconds": 0.1},
            ],
        )
        self.assertEqual(
            prompt,
            "system\n\nTask: MoveCube\nSpecification: exactly three methods\n"
            "Description: move the cube\nThe attached frames are the complete initial "
            "demonstration video downsampled from 30 FPS to 10 FPS. They are "
            "provided in chronological order and labeled with their original frame "
            "identifiers.\n\n"
            "Frame manifest in attachment order:\n"
            "i_f0000: 0.00 s\n"
            "i_f0003: 0.10 s\n",
        )

    def test_v2_response_validation_has_no_keyframes(self) -> None:
        response = {
            "video_description": "visible event",
            "detailed_task_description": "precise task",
        }
        MODULE.validate_response(response)
        response["keyframes"] = []
        MODULE.validate_response(response, set())  # Legacy v1 remains readable.

    def test_planner_guidance_v3_response_validation(self) -> None:
        response = {
            "video_description": "The gripper pushes the cube into the target.",
            "detailed_task_description": "Move the cube by closed-gripper pushing.",
            "planner_guidance": {
                "fixed_facts": ["method=closed-gripper pushing"],
                "state_to_track": ["cube_target_relation"],
                "progress_rules": ["Complete only when the cube is inside the target."],
                "execution_constraints": ["Do not grasp or hook the cube."],
            },
        }
        MODULE.validate_response(response)

        response["planner_guidance"]["fixed_facts"] = [""]
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            MODULE.validate_response(response)

    def test_prompt_profile_selects_v3_without_changing_v2_default(self) -> None:
        common = [
            "run_init_video.py",
            "run",
            "--input-jsonl",
            "/unused/input.jsonl",
            "--output-dir",
            "/unused/output",
            "--cache-dir",
            "/unused/cache",
        ]
        with mock.patch.object(sys, "argv", common):
            default_args = MODULE.parse_args()
        self.assertEqual(default_args.prompt_profile, "appendix_v2")
        self.assertEqual(default_args.system_prompt, MODULE.DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(default_args.output_schema, MODULE.DEFAULT_OUTPUT_SCHEMA)

        with mock.patch.object(
            sys,
            "argv",
            common + ["--prompt-profile", "planner_guidance_v3"],
        ):
            v3_args = MODULE.parse_args()
        self.assertEqual(v3_args.system_prompt, MODULE.PLANNER_GUIDANCE_SYSTEM_PROMPT)
        self.assertEqual(v3_args.output_schema, MODULE.PLANNER_GUIDANCE_OUTPUT_SCHEMA)

    def test_inference_cache_key_changes_with_prompt_and_model(self) -> None:
        prepared = MODULE.PreparedFrames(
            cache_key="frame-key",
            cache_dir=Path("/unused"),
            source_sha256="source-hash",
            source_fps=30.0,
            source_frame_count=3,
            target_fps=10.0,
            frames=[],
        )
        common = {
            "case": MODULE.CaseSpec(
                case_id="Task_episode_00",
                task="Task",
                episode=0,
                task_description="description",
                video_path=Path("/unused/video.mp4"),
            ),
            "prepared": prepared,
            "system_prompt": "system",
            "user_prompt_template": "user {task_description}",
            "output_schema_text": "{}",
            "reasoning_effort": "high",
            "codex_version": "codex-cli test",
        }
        first = MODULE.inference_cache_key(prompt="prompt-a", model="model-a", **common)
        same = MODULE.inference_cache_key(prompt="prompt-a", model="model-a", **common)
        new_prompt = MODULE.inference_cache_key(prompt="prompt-b", model="model-a", **common)
        new_model = MODULE.inference_cache_key(prompt="prompt-a", model="model-b", **common)
        self.assertEqual(first, same)
        self.assertNotEqual(first, new_prompt)
        self.assertNotEqual(first, new_model)

    def test_local_chatgpt_login_and_jsonl_usage_parsing_with_fake_codex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--version\" ]; then echo 'codex-cli fake'; exit 0; fi\n"
                "if [ \"$1\" = \"login\" ]; then echo 'Logged in using ChatGPT'; exit 0; fi\n"
                "exit 2\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            environment = os.environ.copy()
            self.assertEqual(
                MODULE.get_codex_version(str(fake_codex), environment),
                "codex-cli fake",
            )
            self.assertIn(
                "ChatGPT",
                MODULE.require_chatgpt_login(str(fake_codex), environment),
            )

            events = temporary / "events.jsonl"
            events.write_text(
                json.dumps({"type": "thread.started", "thread_id": "thread-1"})
                + "\n"
                + json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 40,
                            "output_tokens": 10,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            thread_id, usage = MODULE.parse_codex_events(events)
            self.assertEqual(thread_id, "thread-1")
            self.assertEqual(usage["cached_input_tokens"], 40)

    def test_noninteractive_codex_command_records_structured_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
                "response = {\n"
                "  'video_description': 'visible event',\n"
                "  'detailed_task_description': 'precise task',\n"
                "}\n"
                "output.write_text(json.dumps(response))\n"
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'thread-2'}))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {\n"
                "  'input_tokens': 12, 'cached_input_tokens': 4, 'output_tokens': 3,\n"
                "  'reasoning_output_tokens': 1}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            image_path = temporary / "i_f0000.jpg"
            image_path.touch()
            schema_path = temporary / "schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            destination = temporary / "result"
            result = MODULE.run_codex_inference(
                codex_bin=str(fake_codex),
                model="model",
                reasoning_effort="high",
                prompt="prompt",
                output_schema=schema_path,
                image_paths=[image_path],
                destination=destination,
                environment=os.environ.copy(),
                timeout_seconds=10.0,
            )
            MODULE.validate_response(result["response"])
            self.assertEqual(result["thread_id"], "thread-2")
            self.assertEqual(result["usage"]["cached_input_tokens"], 4)
            self.assertTrue((destination / "codex_events.jsonl").is_file())

    def test_no_video_prompt_includes_specification_without_frame_manifest(self) -> None:
        prompt = MODULE.render_prompt(
            "system",
            (
                "Task: {task_name}\nSpecification: {task_specification}\n"
                "Description: {task_description}\n{video_context}"
            ),
            "StopCube",
            "count visits and account for button delay",
            "stop on the third visit",
            [],
        )
        self.assertIn("Task: StopCube", prompt)
        self.assertIn("account for button delay", prompt)
        self.assertIn("No initial demonstration video is provided", prompt)
        self.assertNotIn("Frame manifest", prompt)

    def test_default_appendix_specifications_cover_all_tasks(self) -> None:
        specifications = MODULE.load_task_specifications(MODULE.DEFAULT_TASK_SPECIFICATIONS)
        self.assertEqual(set(specifications), MODULE.EXPECTED_ROBOMME_TASKS)
        self.assertIn("one of three manipulation methods", specifications["MoveCube"])
        self.assertIn("not by itself evidence", specifications["MoveCube"])
        self.assertIn("must not be grasped or lifted", specifications["MoveCube"])
        self.assertEqual(
            MODULE.DEFAULT_TASK_SPECIFICATIONS.name,
            "robomme_appendix_e_v4.json",
        )
        self.assertIn("physically press the button", specifications["ButtonUnmask"])
        self.assertNotIn("button-press phase", specifications["ButtonUnmask"])

        previous_path = (
            MODULE.SCRIPT_DIR
            / "task_specifications"
            / "robomme_appendix_e_v1.json"
        )
        previous = MODULE.load_task_specifications(previous_path)
        self.assertIn("button-press phase", previous["ButtonUnmask"])

        v2_path = (
            MODULE.SCRIPT_DIR
            / "task_specifications"
            / "robomme_appendix_e_v2.json"
        )
        v2 = MODULE.load_task_specifications(v2_path)
        self.assertEqual(specifications["ButtonUnmask"], v2["ButtonUnmask"])
        self.assertNotEqual(specifications["MoveCube"], v2["MoveCube"])

        v3_path = (
            MODULE.SCRIPT_DIR
            / "task_specifications"
            / "robomme_appendix_e_v3.json"
        )
        v3 = MODULE.load_task_specifications(v3_path)
        self.assertEqual(specifications["ButtonUnmask"], v3["ButtonUnmask"])
        self.assertNotEqual(specifications["MoveCube"], v3["MoveCube"])

    def test_no_video_case_runs_text_only_codex_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            temporary = Path(temporary_value)
            fake_codex = temporary / "codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "assert '--image' not in args\n"
                "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
                "response = {'video_description': 'No demonstration video is provided.', "
                "'detailed_task_description': 'Count visits and account for button delay.'}\n"
                "output.write_text(json.dumps(response))\n"
                "print(json.dumps({'type': 'thread.started', 'thread_id': 'thread-text'}))\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {"
                "'input_tokens': 20, 'cached_input_tokens': 0, 'output_tokens': 4}}))\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            schema = temporary / "schema.json"
            schema.write_text("{}", encoding="utf-8")
            output_dir = temporary / "output"
            case = MODULE.CaseSpec(
                case_id="StopCube_episode_00",
                task="StopCube",
                episode=0,
                task_description="press the button to stop the cube on the target",
                video_path=None,
            )
            summary = MODULE.process_case(
                case=case,
                output_dir=output_dir,
                cache_root=temporary / "cache",
                system_prompt="system",
                user_prompt_template=(
                    "Task: {task_name}\nSpecification: {task_specification}\n"
                    "Description: {task_description}\n{video_context}"
                ),
                task_specifications={"StopCube": "count visits and button delay"},
                output_schema=schema,
                output_schema_text="{}",
                codex_bin=str(fake_codex),
                codex_version="codex-cli test",
                model="model",
                reasoning_effort="high",
                target_fps=10.0,
                jpeg_quality=95,
                environment=os.environ.copy(),
                timeout_seconds=10.0,
            )
            response = json.loads(
                (output_dir / "cases" / case.case_id / "response.json").read_text()
            )
            self.assertIn("button delay", response["detailed_task_description"])
            self.assertNotIn("keyframes", response)
            self.assertIsNotNone(summary["usage"])
            self.assertIsNone(summary["frames_link"])


if __name__ == "__main__":
    unittest.main()
