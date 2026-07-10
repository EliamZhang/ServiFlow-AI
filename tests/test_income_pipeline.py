from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from income_classification_engine.domain.summary import build_summary
from income_classification_engine.pipeline import run_pipeline


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


if __name__ == "__main__":
    unittest.main()
