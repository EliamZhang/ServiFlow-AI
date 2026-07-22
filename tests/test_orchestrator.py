from __future__ import annotations

import unittest

import pandas as pd

from classification_core.config import EngineSpec, PipelineConfig
from classification_core.models import EngineContext, EngineResult, SummaryArtifact
from classification_core.orchestrator import ClassificationOrchestrator


class FakeEngine:
    engine_version = "test"

    def __init__(
        self,
        engine_id: str,
        category: str,
        matching_transaction_ids: set[int],
    ) -> None:
        self.engine_id = engine_id
        self.category = category
        self.matching_transaction_ids = matching_transaction_ids

    def classify(self, context: EngineContext) -> EngineResult:
        matched = context.candidates[
            context.candidates["transaction_id"].isin(
                self.matching_transaction_ids
            )
        ].copy()
        predictions = matched[["application_id", "transaction_id"]].copy()
        predictions["matched"] = True
        predictions["counterparty"] = self.engine_id.upper()
        predictions["finv_category"] = self.category
        predictions["classification_rule_id"] = f"{self.engine_id}_rule"
        predictions["classification_reason"] = "test"
        return EngineResult(predictions, matched)

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        summary = pd.DataFrame(
            [{"engine_id": self.engine_id, "count": len(accepted_predictions)}]
        )
        return [SummaryArtifact(f"{self.engine_id}_summary", "test", summary)]


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transactions = pd.DataFrame(
            {
                "application_id": [10, 10],
                "transaction_id": [1, 2],
                "amount": [100.0, 50.0],
            }
        )
        self.config = PipelineConfig(
            engines=(
                EngineSpec("income", 100),
                EngineSpec("liability", 200),
            )
        )

    def test_later_engine_overwrites_earlier_at_row_level(self) -> None:
        """Later engines see all transactions and overwrite earlier claims entirely."""
        engines = {
            "income": FakeEngine("income", "salary_payg", {1}),
            "liability": FakeEngine("liability", "bnpl", {1, 2}),
        }
        result = ClassificationOrchestrator(
            config=self.config,
            category_owners={"salary_payg": "income", "bnpl": "liability"},
            engine_factory=engines.__getitem__,
        ).run(self.transactions)

        row_one = result.transactions.loc[
            result.transactions["transaction_id"].eq(1)
        ].iloc[0]
        row_two = result.transactions.loc[
            result.transactions["transaction_id"].eq(2)
        ].iloc[0]
        # Liability runs later (priority 200 > 100), so it overwrites income's
        # claim on row 1, and also claims row 2.
        self.assertEqual(row_one["classification_engine"], "liability")
        self.assertEqual(row_one["finv_category"], "bnpl")
        self.assertEqual(row_two["classification_engine"], "liability")
        self.assertEqual(row_two["finv_category"], "bnpl")
        # Both engines see all transactions.
        self.assertEqual(result.executions[0].candidate_count, 2)
        self.assertEqual(result.executions[1].candidate_count, 2)

    def test_duplicate_transaction_keys_are_rejected(self) -> None:
        duplicated = pd.concat(
            [self.transactions.iloc[[0]], self.transactions.iloc[[0]]],
            ignore_index=True,
        )
        orchestrator = ClassificationOrchestrator(
            config=self.config,
            category_owners={"salary_payg": "income", "bnpl": "liability"},
            engine_factory=lambda engine_id: FakeEngine(
                engine_id,
                "salary_payg" if engine_id == "income" else "bnpl",
                set(),
            ),
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            orchestrator.run(duplicated)

    def test_engine_cannot_return_category_owned_by_another_engine(self) -> None:
        engines = {
            "income": FakeEngine("income", "bnpl", {1}),
            "liability": FakeEngine("liability", "bnpl", set()),
        }
        orchestrator = ClassificationOrchestrator(
            config=self.config,
            category_owners={"salary_payg": "income", "bnpl": "liability"},
            engine_factory=engines.__getitem__,
        )
        with self.assertRaisesRegex(ValueError, "unowned category"):
            orchestrator.run(self.transactions)


if __name__ == "__main__":
    unittest.main()
