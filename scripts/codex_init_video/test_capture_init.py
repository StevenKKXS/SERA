from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from scripts.codex_init_video.capture_init import capture_cases
from scripts.codex_init_video.run_init_video import load_cases
import json


class CaptureTests(unittest.TestCase):
    def test_manifest_loads_and_execution_frame_is_excluded(self):
        runner = Mock(num_episodes=2)
        runner.get_init_obs.side_effect = [
            {"images": ["demo", "execution"], "task_goal": "Move the cube"},
            {"images": ["execution"], "task_goal": "Count objects"},
        ]
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            video_calls = []

            def save(path, frames, **kwargs):
                video_calls.append((frames, kwargs))
                path.write_bytes(b"video")

            rows = capture_cases(["MoveCube"], 2, output, lambda _: runner, save)
            self.assertEqual(video_calls, [(["demo"], {"fps": 30})])
            self.assertIsNone(rows[1]["video_path"])
            manifest = output / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cases = load_cases(manifest)
            self.assertEqual(cases[0].episode, 0)
            self.assertTrue(cases[0].video_path.is_file())
            self.assertIsNone(cases[1].video_path)
            self.assertEqual(runner.make_env.call_count, 2)

    def test_cleanup_after_reset_failure(self):
        runner = Mock(num_episodes=1)
        runner.get_init_obs.side_effect = RuntimeError("reset failed")
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                capture_cases(["MoveCube"], 1, Path(temporary), lambda _: runner, Mock())
        runner.close_env.assert_called()

    def test_episode_bounds_checked_before_reset(self):
        runner = Mock(num_episodes=1)
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "only 1"):
                capture_cases(["MoveCube"], 2, Path(temporary), lambda _: runner, Mock())
        runner.make_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
