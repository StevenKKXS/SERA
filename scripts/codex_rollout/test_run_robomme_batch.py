from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_robomme_batch.py")
SPEC = importlib.util.spec_from_file_location("codex_rollout_batch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakePlanner:
    def __init__(self) -> None:
        self.starts = []
        self.calls = []
        self.ended = False

    def start_episode(self, **kwargs):
        self.starts.append(kwargs)

    def plan(self, *, front_frames, wrist_frames, frame_ids):
        self.calls.append((front_frames, wrist_frames, frame_ids))
        return "pick up the cube at <10, 20>", {
            "round_index": len(self.calls) - 1,
            "recent_rollout_frame_ids": frame_ids,
            "response": {"stage_summary": "The cube is visible."},
            "added_keyframes": [],
            "codex_usage": {"input_tokens": 100, "output_tokens": 10},
            "codex_wall_seconds": 0.25,
        }

    def end_episode(self):
        self.ended = True


class FakePlannerThenJsonFailures(FakePlanner):
    def plan(self, *, front_frames, wrist_frames, frame_ids):
        if not self.calls:
            return super().plan(
                front_frames=front_frames,
                wrist_frames=wrist_frames,
                frame_ids=frame_ids,
            )
        self.calls.append((front_frames, wrist_frames, frame_ids))
        raise RuntimeError(
            'Codex planner HTTP 500: {"error":"JSONDecodeError: Expecting value"}'
        )


class FakePolicy:
    def __init__(self) -> None:
        self.buffers = []
        self.elements = []

    def reset(self):
        return {"reset_finished": True}

    def add_buffer(self, value):
        self.buffers.append(value)
        return {"add_buffer_finished": True}

    def infer(self, element):
        self.elements.append(element)
        return {"actions": np.zeros((2, 8), dtype=np.float32)}


class FakeEnv:
    env_id = "MoveCube"

    def __init__(self) -> None:
        self.steps = 0
        self.closed = False

    def make_env(self, episode_id):
        self.episode_id = episode_id

    def get_init_obs(self):
        front = np.zeros((16, 16, 3), dtype=np.uint8)
        wrist = np.ones((16, 16, 3), dtype=np.uint8)
        state = np.zeros(8, dtype=np.float32)
        return {
            "images": [front.copy(), front.copy(), front.copy()],
            "wrist_images": [wrist.copy(), wrist.copy(), wrist.copy()],
            "states": [state.copy(), state.copy(), state.copy()],
            "task_goal": "official task goal",
        }

    def step(self, action):
        self.steps += 1
        front = np.full((16, 16, 3), self.steps, dtype=np.uint8)
        wrist = np.full((16, 16, 3), self.steps + 10, dtype=np.uint8)
        state = np.full(8, self.steps, dtype=np.float32)
        result = "success" if self.steps == 2 else "unknown"
        return (front, wrist, state), result == "success", result

    def close_env(self):
        self.closed = True


class FakeNeverTerminalEnv(FakeEnv):
    def step(self, action):
        self.steps += 1
        front = np.full((16, 16, 3), self.steps, dtype=np.uint8)
        wrist = np.full((16, 16, 3), self.steps + 10, dtype=np.uint8)
        state = np.full(8, self.steps, dtype=np.float32)
        return (front, wrist, state), False, "unknown"


class BatchTests(unittest.TestCase):
    def test_semantic_stagnation_stop_is_bounded_and_opt_in(self) -> None:
        record = {"consecutive_no_progress_rounds_after": 10}
        self.assertFalse(
            MODULE.should_stop_for_semantic_stagnation(record, 0, "unknown")
        )
        self.assertFalse(
            MODULE.should_stop_for_semantic_stagnation(record, 11, "unknown")
        )
        self.assertTrue(
            MODULE.should_stop_for_semantic_stagnation(record, 10, "unknown")
        )
        self.assertFalse(
            MODULE.should_stop_for_semantic_stagnation(record, 10, "success")
        )

    def test_planner_round_limit_is_bounded_and_opt_in(self) -> None:
        self.assertFalse(MODULE.should_stop_for_planner_round_limit(40, 0, "unknown"))
        self.assertFalse(MODULE.should_stop_for_planner_round_limit(39, 40, "unknown"))
        self.assertTrue(MODULE.should_stop_for_planner_round_limit(40, 40, "unknown"))
        self.assertFalse(MODULE.should_stop_for_planner_round_limit(40, 40, "success"))
    def test_video_episode_parser_supports_compact_all_selector(self) -> None:
        self.assertEqual(MODULE.parse_video_episodes("all", 4), {0, 1, 2, 3})
        self.assertEqual(MODULE.parse_video_episodes("0, 2", 4), {0, 2})
        with self.assertRaisesRegex(ValueError, "comma-separated integers"):
            MODULE.parse_video_episodes("0,two", 4)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MODULE.parse_video_episodes("-1", 4)

    def test_json_failures_repeat_previous_chunk_and_stop_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            episode_dir = Path(temporary_value) / "episode"
            episode_dir.mkdir()
            planner = FakePlannerThenJsonFailures()
            policy = FakePolicy()
            env = FakeNeverTerminalEnv()
            args = argparse.Namespace(
                session_prefix="test-json-fallback",
                max_steps=64,
                obs_horizon=16,
                planner_frame_stride=2,
                max_consecutive_planner_json_failures=10,
            )

            summary = MODULE.run_episode(
                args=args,
                planner=planner,
                policy_client=policy,
                env_runner=env,
                pack_buffer=lambda images, states, exec_start_idx: {},
                episode_id=0,
                episode_dir=episode_dir,
                init_case={
                    "case_id": "MoveCube_episode_00",
                    "detailed_task_description": "Move the cube.",
                },
                save_video=False,
            )

            self.assertEqual(summary["result"], "planner_json_failure")
            self.assertFalse(summary["success"])
            self.assertEqual(summary["planner_json_failure_count"], 10)
            self.assertEqual(summary["planner_json_fallback_chunk_count"], 9)
            self.assertEqual(summary["maximum_consecutive_planner_json_failures"], 10)
            self.assertEqual(summary["steps_executed"], 20)
            self.assertEqual(len(policy.elements), 10)
            self.assertTrue(
                all(
                    element["grounded_subgoal"] == "pick up the cube at <10, 20>"
                    for element in policy.elements
                )
            )
            failures = [
                json.loads(line)
                for line in (episode_dir / "planner_json_failures.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(failures), 10)
            self.assertEqual(failures[-1]["consecutive_failures"], 10)

    def test_recent_observation_subsampling_keeps_newest_synchronized_pair(self) -> None:
        front = [np.full((1, 1, 3), value, dtype=np.uint8) for value in range(16)]
        wrist = [
            np.full((1, 1, 3), value + 100, dtype=np.uint8) for value in range(16)
        ]
        selected_front, selected_wrist, selected_ids = (
            MODULE.subsample_recent_observations(
                front, wrist, list(range(1, 17)), stride=2
            )
        )
        self.assertEqual(selected_ids, [2, 4, 6, 8, 10, 12, 14, 16])
        self.assertEqual(
            [int(value[0, 0, 0]) for value in selected_front],
            list(range(1, 16, 2)),
        )
        self.assertEqual(
            [int(value[0, 0, 0]) for value in selected_wrist],
            list(range(101, 116, 2)),
        )

        _, _, odd_ids = MODULE.subsample_recent_observations(
            front[:5], wrist[:5], list(range(1, 6)), stride=2
        )
        self.assertEqual(odd_ids, [1, 3, 5])

    def test_init_loader_preserves_planner_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            case_dir = root / "cases" / "MoveCube_episode_01"
            case_dir.mkdir(parents=True)
            guidance = {
                "fixed_facts": ["method=peg_hook"],
                "state_to_track": ["peg_held"],
                "progress_rules": ["Advance after the peg is held."],
                "execution_constraints": ["Do not grasp the cube directly."],
            }
            (case_dir / "response.json").write_text(
                json.dumps(
                    {
                        "video_description": "The robot uses a peg.",
                        "detailed_task_description": "Move the cube with the peg.",
                        "planner_guidance": guidance,
                    }
                ),
                encoding="utf-8",
            )
            loaded = MODULE.load_detailed_task_description(root, "MoveCube", 1)
            self.assertEqual(loaded["planner_guidance"], guidance)

    def test_train_init_loader_requires_and_preserves_split_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            case_dir = root / "cases" / "train_MoveCube_episode_01"
            case_dir.mkdir(parents=True)
            (case_dir / "response.json").write_text(
                json.dumps(
                    {
                        "detailed_task_description": "Move the cube.",
                        "planner_guidance": {},
                    }
                ),
                encoding="utf-8",
            )
            (case_dir / "case_summary.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "case_id": "train_MoveCube_episode_01",
                        "dataset_split": "train",
                        "task": "MoveCube",
                        "episode": 1,
                    }
                ),
                encoding="utf-8",
            )
            loaded = MODULE.load_detailed_task_description(
                root, "MoveCube", 1, "train"
            )
            self.assertEqual(loaded["dataset_split"], "train")
            self.assertEqual(loaded["case_id"], "train_MoveCube_episode_01")

    def test_grounded_pi05_contract_and_no_init_images_for_planner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            episode_dir = Path(temporary_value) / "episode"
            episode_dir.mkdir()
            planner = FakePlanner()
            policy = FakePolicy()
            env = FakeEnv()
            args = argparse.Namespace(
                session_prefix="test",
                max_steps=32,
                obs_horizon=16,
                planner_frame_stride=2,
            )

            def pack_buffer(images, states, exec_start_idx):
                return {
                    "image_count": len(images),
                    "state_count": len(states),
                    "exec_start_idx": exec_start_idx,
                }

            summary = MODULE.run_episode(
                args=args,
                planner=planner,
                policy_client=policy,
                env_runner=env,
                pack_buffer=pack_buffer,
                episode_id=0,
                episode_dir=episode_dir,
                init_case={
                    "case_id": "MoveCube_episode_00",
                    "detailed_task_description": "Detailed move-cube task.",
                },
                save_video=False,
            )
            self.assertTrue(summary["success"])
            self.assertEqual(planner.calls[0][2], [0])
            self.assertEqual(len(planner.calls[0][0]), 1)
            self.assertEqual(policy.buffers[0], {"image_count": 3, "state_count": 3, "exec_start_idx": 2})
            self.assertEqual(policy.elements[0]["prompt"], "official task goal")
            self.assertEqual(
                policy.elements[0]["grounded_subgoal"], "pick up the cube at <10, 20>"
            )
            self.assertNotIn("simple_subgoal", policy.elements[0])
            self.assertTrue(planner.ended)
            self.assertTrue(env.closed)
            self.assertTrue(summary["initial_demo_history_preserved_for_pi05"])
            self.assertFalse(summary["initial_demo_images_sent_to_planner"])
            self.assertEqual(summary["planner_frame_stride"], 2)
            self.assertEqual(planner.starts[0]["planner_guidance"], {})
            timing = json.loads((episode_dir / "timings.jsonl").read_text())
            self.assertEqual(timing["source_recent_frame_ids"], [0])
            self.assertEqual(timing["source_recent_frame_count"], 1)
            self.assertEqual(timing["planner_input_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
