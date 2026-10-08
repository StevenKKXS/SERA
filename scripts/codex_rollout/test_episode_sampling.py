from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from episode_sampling import allocate_strata, select_dataset_episodes
except ModuleNotFoundError:
    from scripts.codex_rollout.episode_sampling import allocate_strata, select_dataset_episodes


class EpisodeSamplingTests(unittest.TestCase):
    def test_historical_allocation_is_preserved(self) -> None:
        weights = {"easy": 0.5, "medium": 0.25, "hard": 0.25}
        self.assertEqual(
            allocate_strata(30, ["easy", "medium", "hard"], weights),
            {"easy": 15, "medium": 7, "hard": 8},
        )
        self.assertEqual(
            allocate_strata(20, ["easy", "medium", "hard"], weights),
            {"easy": 10, "medium": 5, "hard": 5},
        )

    def test_dataset_can_choose_uniform_sampler_without_difficulty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            (root / "train").mkdir()
            (root / "train" / "Task.json").write_text(
                json.dumps({"items": [{"id": value, "seed": value + 99} for value in range(10)]}),
                encoding="utf-8",
            )
            config = {
                "episode_sampling": {
                    "dataset_id": "plain_dataset",
                    "datasets": {
                        "plain_dataset": {
                            "loader": {
                                "name": "json_records",
                                "path_template": "{split}/{task}.json",
                                "records_path": ["items"],
                                "episode_id_field": "id",
                                "safe_sampling_fields": [],
                            },
                            "algorithm": {"name": "uniform_even"},
                        }
                    },
                }
            }
            selected, provenance = select_dataset_episodes(
                config=config, source_root=root, split="train", task="Task", limit=4
            )
            self.assertEqual([row["episode_id"] for row in selected], [0, 3, 6, 9])
            self.assertTrue(all("seed" not in row["attributes"] for row in selected))
            self.assertEqual(provenance["algorithm"], "uniform_even")

    def test_explicit_sampler_materializes_requested_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            root.mkdir(exist_ok=True)
            (root / "records.json").write_text(
                json.dumps([{"episode": value} for value in range(6)]), encoding="utf-8"
            )
            config = {
                "episode_sampling": {
                    "dataset_id": "fixed",
                    "datasets": {
                        "fixed": {
                            "loader": {
                                "name": "json_records",
                                "path_template": "records.json",
                                "episode_id_field": "episode",
                            },
                            "algorithm": {"name": "explicit_ids", "episode_ids": [4, 1]},
                        }
                    },
                }
            }
            selected, _ = select_dataset_episodes(
                config=config, source_root=root, split="train", task="Task", limit=2
            )
            self.assertEqual([row["episode_id"] for row in selected], [4, 1])


if __name__ == "__main__":
    unittest.main()
