from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARD = load_module("prepare_hard_extension", "prepare_hard_extension.py")
RESULTS = load_module(
    "prepare_hard_extension_results", "prepare_hard_extension_results.py"
)
DISCOVERY = load_module(
    "build_failure_discovery_pool", "build_failure_discovery_pool.py"
)


def config() -> dict:
    return {
        "episode_sampling": {
            "dataset_id": "robomme",
            "datasets": {
                "robomme": {
                    "loader": {
                        "name": "json_records",
                        "path_template": "{split}/record_dataset_{task}_metadata.json",
                        "records_path": ["records"],
                        "episode_id_field": "episode",
                        "safe_sampling_fields": ["difficulty"],
                    },
                    "algorithm": {
                        "name": "stratified_even",
                        "stratum_field": "difficulty",
                        "stratum_order": ["easy", "medium", "hard"],
                        "stratum_weights": {"easy": 0.5, "medium": 0.25, "hard": 0.25},
                    },
                }
            },
        },
        "hard_exploration": {
            "mode": "auto",
            "trigger_min_success_rate": 0.5,
            "trigger_max_success_rate": 0.95,
            "train_count": 12,
            "val_count": 8,
            "selection_algorithm": "stratified_even_excluding_consumed_v1",
            "baseline_profile": "self_evolve_v3_multiproblem_auto_hard20",
            "run_at_most_once": True,
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def metadata_rows() -> list[dict]:
    rows = []
    for episode in range(120):
        difficulty = "easy" if episode < 60 else "medium" if episode < 90 else "hard"
        rows.append(
            {"episode": episode, "difficulty": difficulty, "seed": episode + 1000}
        )
    return rows


class PrepareHardExtensionTests(unittest.TestCase):
    def test_auto_trigger_selects_fresh_train12_val8_outcome_blind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            config_path = root / "experiment.json"
            gate_path = root / "gate.json"
            evolution_path = root / "evolution.jsonl"
            write_json(config_path, config())
            write_json(
                gate_path,
                {
                    "complete": True,
                    "task": "MoveCube",
                    "evolution_episode_count": 50,
                    "successes": 40,
                    "failures": 10,
                },
            )
            evolution = [
                {
                    "task": "MoveCube",
                    "dataset_split": "train" if index < 30 else "val",
                    "episode": index if index < 30 else index - 30,
                    "success": index < 40,
                }
                for index in range(50)
            ]
            write_jsonl(evolution_path, evolution)
            for split in ("train", "val"):
                write_json(
                    root / "metadata" / split / "record_dataset_MoveCube_metadata.json",
                    {"records": metadata_rows()},
                )

            trigger, selected = HARD.prepare(
                config_path=config_path,
                gate_report_path=gate_path,
                evolution_manifest_path=evolution_path,
                metadata_root=root / "metadata",
                output_dir=root / "hard",
            )
            self.assertEqual(trigger["decision"], "trigger")
            self.assertEqual(trigger["success_rate"], 0.8)
            self.assertEqual(
                {
                    split: sum(row["dataset_split"] == split for row in selected)
                    for split in ("train", "val")
                },
                {"train": 12, "val": 8},
            )
            self.assertTrue(
                all(row["outcome_used_for_selection"] is False for row in selected)
            )
            self.assertTrue(
                all(
                    row["episode_id"] >= (30 if row["dataset_split"] == "train" else 20)
                    for row in selected
                )
            )
            difficulty_counts = {
                (split, difficulty): sum(
                    row["dataset_split"] == split
                    and row["attributes"]["difficulty"] == difficulty
                    for row in selected
                )
                for split in ("train", "val")
                for difficulty in ("easy", "medium", "hard")
            }
            self.assertEqual(
                difficulty_counts,
                {
                    ("train", "easy"): 6,
                    ("train", "medium"): 3,
                    ("train", "hard"): 3,
                    ("val", "easy"): 4,
                    ("val", "medium"): 2,
                    ("val", "hard"): 2,
                },
            )

    def test_auto_trigger_skips_success_rate_above_range(self) -> None:
        report = HARD.decide_trigger(
            {
                "complete": True,
                "task": "MoveCube",
                "evolution_episode_count": 50,
                "successes": 49,
                "failures": 1,
            },
            config()["hard_exploration"],
        )
        self.assertEqual(report["decision"], "skip")
        self.assertEqual(report["train_count"], 0)

    def test_hard_results_are_separate_and_failures_join_discovery_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_value:
            root = Path(temporary_value)
            config_path = root / "experiment.json"
            selection_path = root / "selection.jsonl"
            write_json(config_path, config())
            selected = []
            for split, count, offset in (("train", 12, 30), ("val", 8, 20)):
                for index in range(count):
                    selected.append(
                        {
                            "task": "MoveCube",
                            "dataset_split": split,
                            "episode_id": offset + index,
                        }
                    )
                    summary_path = (
                        root
                        / f"{split}_runs"
                        / "MoveCube"
                        / f"episode_{offset + index:02d}"
                        / "summary.json"
                    )
                    write_json(
                        summary_path,
                        {
                            "task": "MoveCube",
                            "episode": offset + index,
                            "complete": True,
                            "dataset_split": split,
                            "planner_prompt_profile": "self_evolve_v3_multiproblem_auto_hard20",
                            "experience_ids_used": [],
                            "success": index >= 2,
                        },
                    )
            write_jsonl(selection_path, selected)
            report, hard_rows = RESULTS.prepare(
                config_path=config_path,
                selection_manifest_path=selection_path,
                train_run_root=root / "train_runs",
                val_run_root=root / "val_runs",
                output_dir=root / "hard_results",
            )
            self.assertEqual(report["attempts"], 20)
            self.assertEqual(report["failures"], 4)
            self.assertFalse(report["representative_benchmark"])

            evolution_path = root / "evolution.jsonl"
            evolution = [
                {
                    "task": "MoveCube",
                    "dataset_split": "train" if index < 30 else "val",
                    "episode": index if index < 30 else index - 30,
                    "success": index not in {0, 1, 30, 31},
                }
                for index in range(50)
            ]
            write_jsonl(evolution_path, evolution)
            hard_manifest_path = (
                root / "hard_results" / "hard_extension20_manifest.jsonl"
            )
            pool_report, failures, successes = DISCOVERY.build(
                evolution_manifest_path=evolution_path,
                hard_manifest_path=hard_manifest_path,
                output_dir=root / "discovery",
            )
            self.assertEqual(
                pool_report["failure_counts"], {"evolution50": 4, "hard_extension20": 4}
            )
            self.assertEqual(len(failures), 8)
            self.assertEqual(len(successes), 62)
            self.assertFalse(pool_report["success_rates_merged_for_reporting"])
            self.assertEqual(len(hard_rows), 20)


if __name__ == "__main__":
    unittest.main()
