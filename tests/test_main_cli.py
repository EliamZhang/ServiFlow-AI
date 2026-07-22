from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from main import main


class MainCliTests(unittest.TestCase):
    def test_main_prints_one_compact_timing_line(self) -> None:
        timings = {
            "read": 0.01,
            "classify": 2.34,
            "output": 0.45,
            "total": 2.80,
        }
        mock_result = MagicMock()
        mock_result.executions = []
        stdout = io.StringIO()

        with patch.object(sys, "argv", ["main"]), patch(
            "main._execute_classification",
            return_value=(mock_result, timings),
        ), redirect_stdout(stdout):
            main()

        self.assertEqual(
            stdout.getvalue(),
            "Timing | read 0.01s | classify 2.34s | output 0.45s | total 2.80s\n",
        )


if __name__ == "__main__":
    unittest.main()
