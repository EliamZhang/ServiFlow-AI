from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from classification_core.config import load_category_owners, load_pipeline_config
from classification_core.orchestrator import ClassificationOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = PROJECT_ROOT / "sample.csv"


class CurrentSampleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        transactions = pd.read_csv(SAMPLE_FILE, encoding="utf-8-sig")
        cls.result = ClassificationOrchestrator(
            load_pipeline_config(),
            load_category_owners(),
        ).run(transactions)

    def test_row_count_and_transaction_keys_are_preserved(self) -> None:
        self.assertEqual(len(self.result.transactions), 825)
        self.assertEqual(
            int(
                self.result.transactions.duplicated(
                    ["application_id", "transaction_id"]
                ).sum()
            ),
            0,
        )

    def test_current_engine_claim_counts(self) -> None:
        counts = self.result.transactions["classification_engine"].value_counts()
        self.assertEqual(int(counts["income"]), 13)
        self.assertEqual(int(counts["liability"]), 457)
        self.assertEqual(
            int(
                self.result.transactions["classification_status"]
                .eq("unclassified")
                .sum()
            ),
            355,
        )

    def test_summaries_are_generated(self) -> None:
        summaries = {
            artifact.name: artifact.data for artifact in self.result.summaries
        }
        self.assertEqual(len(summaries["income_summary"]), 1)
        self.assertEqual(len(summaries["liability_summary"]), 15)
        self.assertEqual(len(summaries["run_summary"]), 2)
        self.assertIn("finv_category", summaries["liability_summary"].columns)

    def test_classified_rows_have_both_core_fields(self) -> None:
        classified = self.result.transactions[
            self.result.transactions["classification_status"].eq("classified")
        ]
        self.assertFalse(classified["counterparty"].isna().any())
        self.assertFalse(classified["finv_category"].isna().any())


if __name__ == "__main__":
    unittest.main()
