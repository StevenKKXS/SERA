from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "verify_self_evolve_v2_experience",
    SCRIPT_DIR / "verify_self_evolve_v2_experience.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyExperienceTests(unittest.TestCase):
    def test_paired_summary_counts_improvements_and_regressions(self) -> None:
        rows = [
            {
                "dataset_split": "train",
                "record_id": "a",
                "baseline_success": False,
                "experience_success": True,
            },
            {
                "dataset_split": "train",
                "record_id": "b",
                "baseline_success": True,
                "experience_success": False,
            },
        ]
        summary = MODULE.summarize(rows, "train")
        self.assertEqual(summary["improved"], 1)
        self.assertEqual(summary["regressed"], 1)
        self.assertEqual(summary["net_improvements"], 0)


if __name__ == "__main__":
    unittest.main()
