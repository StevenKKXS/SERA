from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "self_evolve_v2.py"
SPEC = importlib.util.spec_from_file_location("self_evolve_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelfEvolveV2Tests(unittest.TestCase):
    def test_round_limit_override_and_historical_task_defaults(self) -> None:
        config = {
            "evolution": {
                "max_rounds": 3,
                "task_max_rounds": {"MoveCube": 1, "SwingXtimes": 2},
            }
        }
        self.assertEqual(MODULE.resolve_round_limit(config, "MoveCube", None), 1)
        self.assertEqual(MODULE.resolve_round_limit(config, "SwingXtimes", None), 2)
        self.assertEqual(MODULE.resolve_round_limit(config, "PatternLock", None), 3)
        self.assertEqual(MODULE.resolve_round_limit(config, "MoveCube", 4), 4)

    def test_candidate_pool_is_stratified_and_never_returns_seed(self) -> None:
        records = []
        for episode in range(100):
            difficulty = ("easy", "easy", "medium", "hard")[episode % 4]
            records.append(
                {
                    "task": "MoveCube",
                    "episode": episode,
                    "difficulty": difficulty,
                    "seed": 123 + episode,
                }
            )
        pool = MODULE.stratified_pool(records, 16)
        self.assertEqual(len(pool), 16)
        self.assertTrue(all("seed" not in row for row in pool))
        counts = {
            difficulty: sum(row["difficulty"] == difficulty for row in pool)
            for difficulty in ("easy", "medium", "hard")
        }
        self.assertEqual(counts, {"easy": 8, "medium": 4, "hard": 4})

    def test_split_specific_pool_sizes_fall_back_to_v2_size(self) -> None:
        v3 = {
            "evolution": {
                "candidate_pool_size": 16,
                "train_pool_size": 30,
                "val_pool_size": 20,
            }
        }
        self.assertEqual(MODULE.resolve_pool_limit(v3, "train"), 30)
        self.assertEqual(MODULE.resolve_pool_limit(v3, "val"), 20)
        v2 = {"evolution": {"candidate_pool_size": 16}}
        self.assertEqual(MODULE.resolve_pool_limit(v2, "train"), 16)

    def test_experience_runtime_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden runtime key"):
            MODULE.assert_no_privileged_identity(
                {"task_scope": ["MoveCube"], "episode_id": 3}
            )

    def test_rollout_cache_key_changes_with_experience_bundle(self) -> None:
        kwargs = {
            "task": "MoveCube",
            "split": "train",
            "episode": 3,
            "planner_commit": "abc",
            "planner_config_sha256": "cfg",
            "executor_version": "pi05",
            "experience_bundle_sha256": "empty",
            "metadata_sha256": "meta",
        }
        first = MODULE.behavior_cache_key(**kwargs)
        kwargs["experience_bundle_sha256"] = "lesson-v1"
        second = MODULE.behavior_cache_key(**kwargs)
        self.assertNotEqual(first, second)

    def test_init_is_resumable_and_round_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "test",
                        "base_profile": "retrieval_v8_5_refine_then_lock",
                        "task_order": ["MoveCube"],
                        "evolution": {
                            "max_rounds": 3,
                            "task_max_rounds": {"MoveCube": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                config=config_path,
                task="MoveCube",
                run_root=root / "run",
                max_evolution_rounds=None,
                repo_root=root,
            )
            with mock.patch.object(MODULE, "git_commit", return_value="abc"):
                MODULE.init_command(args)
                MODULE.init_command(args)
            MODULE.advance_command(argparse.Namespace(run_root=root / "run"))
            with self.assertRaisesRegex(ValueError, "max_evolution_rounds"):
                MODULE.advance_command(argparse.Namespace(run_root=root / "run"))


if __name__ == "__main__":
    unittest.main()
