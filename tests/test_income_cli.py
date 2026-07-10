from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from income_classification_engine.cli import main, parse_args


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "sample.csv"


class IncomeCliTests(unittest.TestCase):
    def test_default_output_is_compact(self) -> None:
        with patch.object(sys, "argv", ["income"]):
            args = parse_args()
        self.assertFalse(args.full)

    def test_full_output_requires_explicit_flag(self) -> None:
        with patch.object(sys, "argv", ["income", "--full"]):
            args = parse_args()
        self.assertTrue(args.full)

    def test_income_cli_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "income_report.xlsx"
            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "income",
                    "--input",
                    str(SAMPLE_FILE),
                    "--output",
                    str(output_file),
                ],
            ), redirect_stdout(stdout):
                main()

            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(output_file.exists())


if __name__ == "__main__":
    unittest.main()
