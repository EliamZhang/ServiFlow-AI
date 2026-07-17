from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import liability_classification_engine.domain.summary as summary_module
from liability_classification_engine.domain.special_rules import apply_special_rules
from liability_classification_engine.pipeline import (
    DEFAULT_RESOURCES_DIR,
    run_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "sample.csv"


class LiabilityPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        transactions = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig")
        cls.result = run_pipeline(transactions)

    def test_pipeline_does_not_calculate_unused_diagnostics(self) -> None:
        self.assertEqual(self.result.diagnostics, {})

    def test_summary_prepares_transactions_once(self) -> None:
        prepare_summary_input = summary_module.prepare_summary_input
        with patch.object(
            summary_module,
            "prepare_summary_input",
            wraps=prepare_summary_input,
        ) as wrapped:
            summary_module.build_summary(
                self.result.transactions,
                limits_file=DEFAULT_RESOURCES_DIR / "bnpl_maximum_limits.csv",
            )

        self.assertEqual(wrapped.call_count, 1)

    def test_cash_converters_uses_its_existing_product_classification(self) -> None:
        transactions = pd.DataFrame(
            {
                "counterparty": ["Cash Converters"],
                "text": ["cash converters advance"],
                "amount": [100],
                "is_dishonours": ["No"],
                "dr_cr": ["credit"],
                "product_type": ["personal_loan"],
                "untouched": ["keep"],
            }
        )

        result = apply_special_rules(transactions)

        self.assertEqual(result.at[0, "product_type"], "personal_loan")
        self.assertEqual(result.at[0, "untouched"], "keep")
        self.assertEqual(transactions.at[0, "product_type"], "personal_loan")


if __name__ == "__main__":
    unittest.main()
