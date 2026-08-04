"""Initial-classification engine — merchant keyword matching via Aho-Corasick."""

from __future__ import annotations

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)

from .pipeline import DEFAULT_KB_PATH, run_pipeline


class InitialClassificationEngine:
    """Classify transactions by matching text against a merchant knowledge base.

    The engine compares each transaction's ``text`` field against ~30k keyword
    variants sourced from ~9k categorised merchant rows.  Matching uses a
    pure-Python Aho-Corasick automaton — construction is a one-off cost and
    per-text search is linear in text length.
    """

    engine_id = "initial"
    engine_version = "1.0"

    # ------------------------------------------------------------------
    # ClassificationEngine protocol
    # ------------------------------------------------------------------

    def classify(self, context: EngineContext) -> EngineResult:
        result = run_pipeline(context.candidates, kb_path=DEFAULT_KB_PATH)
        details = result.transactions

        matched = details[details["matched"].eq(True)].copy()
        predictions = pd.DataFrame(
            {
                **{col: matched[col].values for col in TRANSACTION_KEY_COLUMNS},
                "matched": True,
                "counterparty": matched["counterparty"].values,
                "finv_category": matched["finv_category"].values,
                "stream_id": "",
                "classification_rule_id": matched[
                    "classification_rule_id"
                ].values,
                "classification_reason": matched[
                    "classification_reason"
                ].values,
            }
        )

        return EngineResult(
            predictions=predictions,
            transactions=details,
            diagnostics=result.diagnostics,
        )

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        return []
