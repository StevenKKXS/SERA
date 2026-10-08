from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from prepare_evolution_dataset import prepare
except ModuleNotFoundError:
    from scripts.codex_rollout.prepare_evolution_dataset import prepare


class EvolutionDatasetGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps({
                "episode_sampling": {"dataset_id": "example"},
                "evolution_dataset_gate": {
                    "dataset_name": "example_evolution50_dataset",
                    "required_train_episodes": 2,
                    "required_val_episodes": 1,
                    "required_baseline_profile": "profile",
                },
            }), encoding="utf-8",
        )
        for split, ids in (("train", [1, 3]), ("val", [2])):
            (self.root / f"{split}_pool.json").write_text(
                json.dumps({"task": "Task", "dataset_split": split,
                            "candidates": [{"episode": value} for value in ids]}),
                encoding="utf-8",
            )
            for episode in ids:
                path = self.root / f"{split}_run" / f"episode_{episode}" / "summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "complete": True, "task": "Task", "dataset_split": split,
                    "episode": episode, "planner_prompt_profile": "profile",
                    "experience_ids_used": [], "success": episode != 3,
                }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke(self):
        return prepare(
            config_path=self.config, task="Task",
            train_pool=self.root / "train_pool.json", val_pool=self.root / "val_pool.json",
            train_run_root=self.root / "train_run", val_run_root=self.root / "val_run",
            output_dir=self.root / "evolution_dataset",
        )

    def test_unlocks_only_complete_evolution_dataset(self) -> None:
        result = self.invoke()
        self.assertTrue(result["analysis_unlocked"])
        self.assertEqual(
            result["evolution_dataset_name"], "example_evolution50_dataset"
        )
        self.assertEqual(result["evolution_episode_count"], 3)
        self.assertEqual(result["failures"], 1)

    def test_missing_val_episode_blocks_analysis(self) -> None:
        (self.root / "val_run" / "episode_2" / "summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete val baseline"):
            self.invoke()

    def test_loaded_experience_blocks_baseline(self) -> None:
        path = self.root / "train_run" / "episode_1" / "summary.json"
        value = json.loads(path.read_text())
        value["experience_ids_used"] = ["leak"]
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "loaded experience"):
            self.invoke()

    def test_behavior_equivalent_legacy_profile_is_reused(self) -> None:
        config = json.loads(self.config.read_text())
        gate = config["evolution_dataset_gate"]
        gate["required_baseline_behavior_id"] = "behavior-v1"
        gate["compatible_baseline_profiles"] = ["legacy-profile"]
        self.config.write_text(json.dumps(config), encoding="utf-8")
        for path in self.root.rglob("summary.json"):
            value = json.loads(path.read_text())
            value["planner_prompt_profile"] = "legacy-profile"
            path.write_text(json.dumps(value), encoding="utf-8")

        result = self.invoke()

        self.assertTrue(result["cross_campaign_cache_reused"])
        self.assertEqual(result["source_baseline_profiles"], ["legacy-profile"])
        self.assertEqual(result["baseline_behavior_id"], "behavior-v1")
        import_record = json.loads(
            (self.root / "evolution_dataset" / "baseline_cache_import.json").read_text()
        )
        self.assertTrue(import_record["read_only"])
        self.assertEqual(len(import_record["records"]), 3)

    def test_recorded_behavior_id_mismatch_blocks_reuse(self) -> None:
        config = json.loads(self.config.read_text())
        config["evolution_dataset_gate"]["required_baseline_behavior_id"] = (
            "behavior-v1"
        )
        self.config.write_text(json.dumps(config), encoding="utf-8")
        path = self.root / "train_run" / "episode_1" / "summary.json"
        value = json.loads(path.read_text())
        value["planner_behavior_id"] = "different-behavior"
        path.write_text(json.dumps(value), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "behavior id mismatch"):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
