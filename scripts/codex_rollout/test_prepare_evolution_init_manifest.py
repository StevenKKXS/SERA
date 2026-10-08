import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_evolution_init_manifest.py")
SPEC = importlib.util.spec_from_file_location("prepare_evolution_init_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareEvolutionInitManifestTest(unittest.TestCase):
    def test_combines_split_qualified_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {}
            for split, count in (("train", 2), ("val", 1)):
                manifest = root / f"{split}.jsonl"
                rows = []
                for episode in range(count):
                    video = root / f"{split}_{episode}.mp4"
                    video.touch()
                    rows.append(
                        {
                            "case_id": f"{split}_MoveCube_episode_{episode:02d}",
                            "dataset_split": split,
                            "task": "MoveCube",
                            "episode": episode,
                            "video_path": str(video),
                        }
                    )
                manifest.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                manifests[split] = manifest
            output = root / "combined.jsonl"
            rows = MODULE.prepare(
                task="MoveCube",
                train_manifest=manifests["train"],
                val_manifest=manifests["val"],
                output=output,
                expected_train=2,
                expected_val=1,
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)

    def test_rejects_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            row = {
                "case_id": "train_MoveCube_episode_00",
                "dataset_split": "train",
                "task": "MoveCube",
                "episode": 0,
                "video_path": str(video),
            }
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            val.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "task/split mismatch"):
                MODULE.prepare(
                    task="MoveCube",
                    train_manifest=train,
                    val_manifest=val,
                    output=root / "out.jsonl",
                    expected_train=1,
                    expected_val=1,
                )


if __name__ == "__main__":
    unittest.main()
