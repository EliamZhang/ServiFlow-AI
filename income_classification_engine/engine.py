from __future__ import annotations

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)
from classification_core.transaction_keys import filter_to_transaction_keys

from .domain.summary import build_summary
from .pipeline import run_pipeline


class IncomeEngine:
    engine_id = "income"
    engine_version = "1.0"

    def classify(self, context: EngineContext) -> EngineResult:
        result = run_pipeline(context.candidates)
        details = result.transactions
        matched = details[details["is_income_pred"].eq(1)].copy()
        predictions = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
        predictions["matched"] = True
        predictions["proposed_counterparty"] = matched["counterparty"].values
        predictions["proposed_finv_category"] = matched["finv_category"].values
        predictions["stream_id"] = matched["stream_id"].values
        predictions["rule_id"] = matched["income_type_rule_name"].values
        predictions["reason"] = matched["income_type_pred_reason"].values
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
        accepted_details = filter_to_transaction_keys(
            result.transactions,
            accepted_predictions,
        )
        return [
            SummaryArtifact(
                "income_summary",
                "1.0",
                build_summary(accepted_details),
            )
        ]
