from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "prepare_v3_problem_records", SCRIPT_DIR / "prepare_v3_problem_records.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareV3ProblemRecordsTests(unittest.TestCase):
    def test_converts_adjudicated_analysis_without_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "analysis" / "episode_07"
            episode.mkdir(parents=True)
            source = {
                "task": "MoveCube",
                "dataset_split": "train",
                "episode": 7,
                "source_result": "stagnation_early_stop",
                "adjudication": {"status": "consensus", "candidate_allowed": True},
                "heuristic_analysis": {
                    "derived_context": {"selected_method": "peg_push"}
                },
                "final_analysis": {
                    "failure_type": "recovery_failure",
                    "responsible_module": "recovery_policy",
                    "analysis_summary": "Repeated no-contact commands.",
                    "causal_evidence": [
                        {"source": "rollout_video", "observation": "No contact."}
                    ],
                    "alternative_causes": ["Initial executor lapse."],
                    "experience_eligibility": "eligible_candidate",
                    "candidate_experience": {"lesson": "Change contact geometry."},
                    "confidence": 0.9,
                },
            }
            (episode / "analysis.json").write_text(json.dumps(source), encoding="utf-8")
            output = root / "records.jsonl"
            rows = MODULE.prepare([root / "analysis"], output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task_variant"], "peg_push")
            self.assertEqual(rows[0]["failure_stage"], "failure_recovery")
            self.assertEqual(rows[0]["planner_actionability"], "high")
            self.assertTrue(rows[0]["runtime_observable"])
            self.assertNotIn("seed", rows[0])

    def test_rejects_test_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            path.write_text(
                json.dumps(
                    {
                        "task": "MoveCube",
                        "dataset_split": "test",
                        "episode": 1,
                        "final_analysis": {"failure_type": "unknown"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "reject split"):
                MODULE.convert(path)

    def test_success_is_preserved_as_variant_counterexample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "analysis" / "episode_03"
            episode.mkdir(parents=True)
            source = {
                "task": "SwingXtimes",
                "dataset_split": "val",
                "episode": 3,
                "source_result": "success",
                "adjudication": {"status": "consensus", "candidate_allowed": False},
                "heuristic_analysis": {
                    "derived_context": {"selected_method": "ordered_target_cycles"}
                },
                "final_analysis": {
                    "failure_type": "unknown",
                    "responsible_module": "unknown",
                    "analysis_summary": "The ordered cycle and stop were visibly completed.",
                    "causal_evidence": [
                        {"source": "rollout_video", "observation": "Success."}
                    ],
                    "alternative_causes": [],
                    "experience_eligibility": "not_experience_problem",
                    "candidate_experience": None,
                    "confidence": 0.95,
                },
            }
            (episode / "analysis.json").write_text(json.dumps(source), encoding="utf-8")
            rows = MODULE.prepare([root / "analysis"], root / "records.jsonl")
            self.assertEqual(rows[0]["outcome"], "success")
            self.assertEqual(rows[0]["task_variant"], "ordered_target_cycles")
            self.assertEqual(rows[0]["counterexample_for"], ["*"])
            self.assertEqual(rows[0]["planner_actionability"], "none")


if __name__ == "__main__":
    unittest.main()
