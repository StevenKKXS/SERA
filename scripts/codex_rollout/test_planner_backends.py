from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from text_progress_planner import CodexTextProgressPlanner  # noqa: E402


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, value: dict) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._send(
            {
                "object": "list",
                "data": [{"id": "qwen3-vl-32b-thinking", "object": "model"}],
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.requests.append(request)
        response = {
            "recent_status": "initial",
            "recent_outcome": "The red cube is visible on the table.",
            "refine_check": {
                "criterion_id": "none",
                "decision": "not_applicable",
                "checks": [],
                "reason": "There is no previously executed action.",
            },
            "updated_progress": (
                "Phase: pickup. Completed: none. Task state: pick up the red cube. "
                "Physical state: cube on table. Execution state: nothing held. "
                "Uncertainty: none material."
            ),
            "skill_key": "skill_cc4eb96c796f",
            "execution_state": {
                "held_object": "none",
                "object_support": "table",
                "gripper_state": "unknown",
            },
            "instruction_strategy": "normal",
            "current_subtask_template": "pick up the cube at <point_1>",
            "points": [
                {
                    "id": "point_1",
                    "object_description": "red cube",
                    "source_frame_id": "r_f0000",
                    "x_norm": 500,
                    "y_norm": 300,
                }
            ],
        }
        self._send(
            {
                "id": "chatcmpl-local-test",
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps(response)}}
                ],
                "usage": {
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                    "total_tokens": 168,
                },
            }
        )


class TwoStageOpenAIHandler(FakeOpenAIHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.requests.append(request)
        if request["chat_template_kwargs"]["enable_thinking"]:
            self._send(
                {
                    "id": "chatcmpl-thinking-draft",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The visual relation is ambiguous, so I will keep analyzing",
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 128,
                        "total_tokens": 228,
                    },
                }
            )
            return
        response = {
            "recent_status": "initial",
            "recent_outcome": "The red cube is visible on the table.",
            "refine_check": {
                "criterion_id": "none",
                "decision": "not_applicable",
                "checks": [],
                "reason": "There is no previously executed action.",
            },
            "updated_progress": (
                "Phase: pickup. Completed: none. Task state: pick up the red cube. "
                "Physical state: cube on table. Execution state: nothing held. "
                "Uncertainty: none material."
            ),
            "skill_key": "skill_cc4eb96c796f",
            "execution_state": {
                "held_object": "none",
                "object_support": "table",
                "gripper_state": "unknown",
            },
            "instruction_strategy": "normal",
            "current_subtask_template": "pick up the cube at <point_1>",
            "points": [
                {
                    "id": "point_1",
                    "object_description": "red cube",
                    "source_frame_id": "r_f0000",
                    "x_norm": 500,
                    "y_norm": 300,
                }
            ],
        }
        self._send(
            {
                "id": "chatcmpl-json-finalizer",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(response)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 45,
                    "total_tokens": 245,
                },
            }
        )


class LocalPlannerBackendTests(unittest.TestCase):
    def test_local_backend_reuses_v8_5_planner_and_harness(self) -> None:
        FakeOpenAIHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary_value:
                planner = CodexTextProgressPlanner(
                    inference_provider="local_openai",
                    openai_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="qwen3-vl-32b-thinking",
                    prompt_profile="retrieval_v8_5_refine_then_lock",
                    retrieval_scheme="R0",
                    retrieval_top_k=5,
                    timeout_seconds=10,
                    keep_recent_inputs=True,
                )
                run_dir = Path(temporary_value) / "run"
                planner.start_episode(
                    run_dir=run_dir,
                    detailed_task_description=(
                        "Pick up the red cube and place it on the demonstrated target."
                    ),
                    planner_guidance={},
                    task="VideoPlaceButton",
                    episode=0,
                    init_case_id="VideoPlaceButton_episode_00",
                )
                frame = np.zeros((100, 200, 3), dtype=np.uint8)
                subtask, record = planner.plan(
                    front_frames=[frame], wrist_frames=[frame], frame_ids=[0]
                )
                self.assertEqual(subtask, "pick up the cube at <30, 100>")
                self.assertEqual(record["planner_provider"], "local_openai")
                self.assertEqual(record["planner_usage"]["input_tokens"], 123)
                self.assertEqual(record["codex_usage"]["output_tokens"], 45)
                episode_config = json.loads(
                    (run_dir / "episode_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(episode_config["planner_provider"], "local_openai")
                request = FakeOpenAIHandler.requests[0]
                self.assertEqual(request["response_format"]["type"], "json_schema")
                content = request["messages"][0]["content"]
                self.assertEqual(
                    [row["type"] for row in content],
                    ["text", "image_url", "image_url"],
                )
                self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png"))
                request_record = json.loads(
                    (
                        run_dir
                        / "rounds"
                        / "round_0000"
                        / "provider_request.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(request_record["image_count"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_bounded_thinking_uses_nonthinking_json_finalizer(self) -> None:
        TwoStageOpenAIHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), TwoStageOpenAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary_value:
                planner = CodexTextProgressPlanner(
                    inference_provider="local_openai",
                    openai_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="qwen3-vl-32b-thinking",
                    prompt_profile="retrieval_v8_5_refine_then_lock",
                    retrieval_scheme="R0",
                    retrieval_top_k=5,
                    timeout_seconds=10,
                    keep_recent_inputs=True,
                    max_output_tokens=24576,
                    enable_thinking=True,
                    thinking_budget_tokens=128,
                    thinking_finalizer_max_output_tokens=1024,
                )
                run_dir = Path(temporary_value) / "run"
                planner.start_episode(
                    run_dir=run_dir,
                    detailed_task_description=(
                        "Pick up the red cube and place it on the demonstrated target."
                    ),
                    planner_guidance={},
                    task="VideoPlaceButton",
                    episode=0,
                    init_case_id="VideoPlaceButton_episode_00",
                )
                frame = np.zeros((100, 200, 3), dtype=np.uint8)
                subtask, record = planner.plan(
                    front_frames=[frame], wrist_frames=[frame], frame_ids=[0]
                )
                self.assertEqual(subtask, "pick up the cube at <30, 100>")
                self.assertTrue(record["thinking_finalizer_used"])
                self.assertEqual(record["thinking_stage_finish_reason"], "length")
                self.assertEqual(record["planner_usage"]["output_tokens"], 173)
                self.assertEqual(len(TwoStageOpenAIHandler.requests), 2)
                thinking_request, finalizer_request = TwoStageOpenAIHandler.requests
                self.assertTrue(
                    thinking_request["chat_template_kwargs"]["enable_thinking"]
                )
                self.assertEqual(thinking_request["max_tokens"], 128)
                self.assertFalse(
                    finalizer_request["chat_template_kwargs"]["enable_thinking"]
                )
                self.assertEqual(finalizer_request["max_tokens"], 1024)
                round_dir = run_dir / "rounds" / "round_0000"
                self.assertTrue((round_dir / "thinking_stage" / "response.txt").is_file())
                self.assertTrue(
                    (round_dir / "finalization_stage" / "response.txt").is_file()
                )
                self.assertTrue((round_dir / "reasoning.txt").is_file())
                root_response = json.loads(
                    (round_dir / "response.txt").read_text(encoding="utf-8")
                )
                self.assertEqual(root_response["skill_key"], "skill_cc4eb96c796f")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
