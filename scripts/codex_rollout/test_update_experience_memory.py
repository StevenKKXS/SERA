from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
MODULE_PATH = SCRIPT_DIR / "update_experience_memory.py"
SPEC = importlib.util.spec_from_file_location("update_experience_memory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def memory() -> dict:
    return {
        "revision": 1,
        "status": "candidate_only",
        "experiences": [
            {
                "experience_id": "experience_test",
                "revision": 1,
                "status": "candidate",
                "validation": {
                    "attempts": 0,
                    "improved": 0,
                    "unchanged_success": 0,
                    "unchanged_failure": 0,
                    "regressed": 0,
                    "records": [],
                },
            }
        ],
    }


class ExperienceMemoryUpdateTests(unittest.TestCase):
    def test_promotes_only_after_positive_held_out_evidence(self) -> None:
        rows = [
            {
                "experience_id": "experience_test",
                "task": "MoveCube",
                "episode": episode,
                "baseline_success": False,
                "experience_success": episode < 2,
            }
            for episode in range(5)
        ]
        result = MODULE.update_memory(
            memory(), rows, min_attempts=5, min_net_improvements=2
        )
        experience = result["experiences"][0]
        self.assertEqual(experience["status"], "validated")
        self.assertEqual(experience["validation"]["improved"], 2)

    def test_regression_forces_revision(self) -> None:
        rows = [
            {
                "experience_id": "experience_test",
                "task": "MoveCube",
                "episode": 7,
                "baseline_success": True,
                "experience_success": False,
            }
        ]
        result = MODULE.update_memory(
            memory(), rows, min_attempts=1, min_net_improvements=0
        )
        self.assertEqual(result["experiences"][0]["status"], "needs_revision")

    def test_duplicate_validation_is_idempotent(self) -> None:
        row = {
            "experience_id": "experience_test",
            "task": "MoveCube",
            "episode": 1,
            "baseline_success": False,
            "experience_success": True,
        }
        result = MODULE.update_memory(
            memory(), [row, row], min_attempts=5, min_net_improvements=2
        )
        self.assertEqual(result["experiences"][0]["validation"]["attempts"], 1)
        revision = result["revision"]
        repeated = MODULE.update_memory(
            result, [row], min_attempts=5, min_net_improvements=2
        )
        self.assertEqual(repeated["revision"], revision)


if __name__ == "__main__":
    unittest.main()
