from __future__ import annotations

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)

from .pipeline import run_pipeline


class FeeEngine:
    """Classify fee transactions by matching text against regex patterns.

    The engine examines each transaction's ``text`` field and matches against
    ~70 ordered regex rules covering international transaction fees, ATM operator
    fees, bank account fees, overdrawn/dishonour/late-payment fees, cash advance
    fees, and third-party maintenance/membership fees.

    Execution order is defined by the engine's ``priority`` in
    ``configs/pipeline.json`` (fee = 500), not by this file.
    """

    engine_id = "fee"
    engine_version = "1.0"

    # ------------------------------------------------------------------
    # ClassificationEngine protocol
    # ------------------------------------------------------------------

    def classify(self, context: EngineContext) -> EngineResult:
        result = run_pipeline(context.candidates)
        details = result.transactions

        matched = details[details["is_fee_pred"].eq(1)].copy()
        predictions = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
        predictions["matched"] = True
        predictions["counterparty"] = matched["counterparty"].values
        predictions["finv_category"] = matched["finv_category"].values
        predictions["stream_id"] = matched["stream_id"].values
        predictions["classification_rule_id"] = matched[
            "fee_rule_name"
        ].values
        predictions["classification_reason"] = matched[
            "fee_pred_reason"
        ].values

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
        # Fee engine is simple — no detailed summary needed beyond counts.
        return []
