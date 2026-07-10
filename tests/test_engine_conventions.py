from __future__ import annotations

import unittest

import income_classification_engine as income
import liability_classification_engine as liability
from income_classification_engine import reporting as income_reporting
from liability_classification_engine import reporting as liability_reporting
from serviflow.models import PipelineResult


class EngineConventionTests(unittest.TestCase):
    def test_both_packages_expose_the_same_public_api(self) -> None:
        common_api = {
            "PipelineResult",
            "build_summary",
            "run_pipeline",
            "write_report",
        }
        self.assertTrue(common_api.issubset(set(income.__all__)))
        self.assertTrue(common_api.issubset(set(liability.__all__)))
        self.assertIs(income.PipelineResult, PipelineResult)
        self.assertIs(liability.PipelineResult, PipelineResult)

    def test_report_sheet_names_follow_the_same_convention(self) -> None:
        self.assertEqual(income_reporting.TRANSACTIONS_SHEET_NAME, "transactions")
        self.assertEqual(liability_reporting.TRANSACTIONS_SHEET_NAME, "transactions")
        self.assertEqual(income_reporting.SUMMARY_SHEET_NAME, "income_summary")
        self.assertEqual(
            liability_reporting.SUMMARY_SHEET_NAME,
            "liability_summary",
        )

    def test_engine_ids_match_package_domains(self) -> None:
        self.assertEqual(income.IncomeEngine.engine_id, "income")
        self.assertEqual(liability.LiabilityEngine.engine_id, "liability")


if __name__ == "__main__":
    unittest.main()
