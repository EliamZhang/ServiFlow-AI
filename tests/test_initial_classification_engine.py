from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from initial_classification_engine.pipeline import run_pipeline


class InitialClassificationEngineTests(unittest.TestCase):
    def test_financial_institutions_category_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kb_path = Path(temporary_directory) / "merchant_kb.csv"
            kb_path.write_text(
                "merchant_name,keywords,category\n"
                "CashFaster,CashFaster,Financial Institutions\n",
                encoding="utf-8",
            )

            result = run_pipeline(pd.DataFrame({"text": ["CashFaster"]}), kb_path)

        transaction = result.transactions.loc[0]
        self.assertFalse(transaction["matched"])
        self.assertEqual(transaction["counterparty"], "CashFaster")
        self.assertEqual(transaction["finv_category"], "")


if __name__ == "__main__":
    unittest.main()
