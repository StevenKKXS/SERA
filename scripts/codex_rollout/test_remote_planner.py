from __future__ import annotations

import json
import unittest

from remote_planner import planner_http_error


class RemotePlannerTests(unittest.TestCase):
    def test_ordinary_server_error_remains_retryable_runtime_error(self) -> None:
        error = planner_http_error(
            500,
            json.dumps(
                {
                    "ok": False,
                    "error": "TimeoutError: planner timed out",
                    "error_code": "planner_server_error",
                    "retryable": True,
                }
            ),
        )
        self.assertIs(type(error), RuntimeError)


if __name__ == "__main__":
    unittest.main()
