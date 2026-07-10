from __future__ import annotations

import unittest
from pathlib import Path

import income_classification_engine as income
import liability_classification_engine as liability
from income_classification_engine.cli import DEFAULT_INPUT as INCOME_DEFAULT_INPUT
from income_classification_engine.presentation import reporting as income_reporting
from liability_classification_engine.cli import (
    DEFAULT_INPUT as LIABILITY_DEFAULT_INPUT,
)
from liability_classification_engine.presentation import reporting as liability_reporting
from classification_core.models import PipelineResult


class EngineConventionTests(unittest.TestCase):
    def test_both_engines_use_the_root_sample(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        expected = project_root / "sample.csv"
        self.assertEqual(INCOME_DEFAULT_INPUT, expected)
        self.assertEqual(LIABILITY_DEFAULT_INPUT, expected)
        samples = list(project_root.rglob("sample.csv"))
        self.assertEqual(samples, [expected])

    def test_top_level_python_files_are_identical(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        expected = {
            "__init__.py",
            "__main__.py",
            "cli.py",
            "engine.py",
            "pipeline.py",
        }
        for package_name in (
            "income_classification_engine",
            "liability_classification_engine",
        ):
            package_root = project_root / package_name
            actual = {path.name for path in package_root.glob("*.py")}
            self.assertEqual(actual, expected)

    def test_both_packages_have_domain_and_presentation_layers(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        for package_name in (
            "income_classification_engine",
            "liability_classification_engine",
        ):
            package_root = project_root / package_name
            self.assertTrue((package_root / "domain" / "__init__.py").is_file())
            self.assertTrue(
                (package_root / "presentation" / "__init__.py").is_file()
            )

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
