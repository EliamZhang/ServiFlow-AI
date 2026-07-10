from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from income_classification_engine.domain.summary import build_summary
from income_classification_engine.pipeline import run_pipeline
from income_classification_engine.presentation.reporting import write_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "sample.csv"


class IncomePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        transactions = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig")
        cls.result = run_pipeline(transactions)

    def test_centrelink_payment_type_is_not_output(self) -> None:
        self.assertNotIn("centrelink_payment_type", self.result.transactions.columns)

        summary = build_summary(self.result.transactions)
        self.assertNotIn("centrelink_payment_type", summary.columns)

    def test_full_report_does_not_output_centrelink_payment_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "income_full.xlsx"
            detail_df, summary_df = write_report(
                self.result,
                output_file,
                full=True,
            )

        self.assertNotIn("centrelink_payment_type", detail_df.columns)
        self.assertNotIn("centrelink_payment_type", summary_df.columns)


if __name__ == "__main__":
    unittest.main()
