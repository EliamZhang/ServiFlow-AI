from __future__ import annotations

import pandas as pd

from serviflow.models import (
    EngineContext,
    PredictionBatch,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)
from serviflow.transaction_keys import filter_to_transaction_keys

from .wages_detector import build_income_summary, classify_income_transactions


class IncomeEngine:
    engine_id = "income"
    engine_version = "1.0"

    def classify(self, context: EngineContext) -> PredictionBatch:
        result = classify_income_transactions(
            context.candidates,
            include_centrelink_payment_type=True,
        )
        details = result.transactions
        matched = details[details["is_income_pred"].eq(1)].copy()
        predictions = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
        predictions["matched"] = True
        predictions["proposed_counterparty"] = matched["counterparty"].values
        predictions["proposed_finv_category"] = matched["finv_category"].values
        predictions["stream_id"] = matched["stream_id"].values
        predictions["rule_id"] = matched["income_type_rule_name"].values
        predictions["reason"] = matched["income_type_pred_reason"].values
        return PredictionBatch(
            predictions=predictions,
            details=details,
            diagnostics={
                "predicted_income_rows": int(details["is_income_pred"].sum()),
                "predicted_wages_rows": int(details["is_wages_pred"].sum()),
            },
        )

    def summarize(
        self,
        context: EngineContext,
        batch: PredictionBatch,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        accepted_details = filter_to_transaction_keys(
            batch.details,
            accepted_predictions,
        )
        return [
            SummaryArtifact(
                "income_summary",
                "1.0",
                build_income_summary(accepted_details),
            )
        ]
