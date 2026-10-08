from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = load_module("capture_robomme_init_demos", "capture_robomme_init_demos.py")
PLANNER = load_module("plan_train_exploration", "plan_train_exploration.py")


class TrainActiveReferenceTests(unittest.TestCase):
    def test_expert_demo_excludes_execution_initial_observation(self) -> None:
        frames, available = CAPTURE.select_reference_frames(["demo-a", "demo-b", "exec"])
        self.assertTrue(available)
        self.assertEqual(frames, ["demo-a", "demo-b"])

    def test_no_demo_uses_initial_observation_for_selection_only(self) -> None:
        frames, available = CAPTURE.select_reference_frames(["exec"])
        self.assertFalse(available)
        self.assertEqual(frames, ["exec"])
        card = {"video_path": "/selection/reference.mp4", "expert_demo_available": False}
        self.assertIsNone(PLANNER.init_analyzer_video_path(card))

    def test_true_demo_is_forwarded_to_init_analyzer(self) -> None:
        card = {"video_path": "/ground/truth.mp4", "expert_demo_available": True}
        self.assertEqual(
            PLANNER.init_analyzer_video_path(card), "/ground/truth.mp4"
        )

    def test_empty_observation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no image observations"):
            CAPTURE.select_reference_frames([])

    def test_legacy_expert_demo_cache_is_read_without_rewrite(self) -> None:
        legacy = {
            "video_path": "/cache/initial_demo.mp4",
            "video_sha256": "abc",
        }
        normalized = CAPTURE.normalize_cached_summary(legacy)
        self.assertTrue(normalized["expert_demo_available"])
        self.assertEqual(normalized["reference_kind"], "expert_demonstration")
        self.assertEqual(normalized["reference_video_path"], legacy["video_path"])
        self.assertNotIn("expert_demo_available", legacy)


if __name__ == "__main__":
    unittest.main()
