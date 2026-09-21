from __future__ import annotations

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)

from .pipeline import run_pipeline


class TransferEngine:
    engine_id = "transfer"
    # 1.1: transfer rows no longer emit a stream_id (the removed value echoed the
    # transfer category).  Row-level output change, so the claims archive shows it.
    engine_version = "1.1"

    def classify(self, context: EngineContext) -> EngineResult:
        result = run_pipeline(
            context.candidates,
            all_rows=context.all_transactions,
        )
        details = result.transactions
        matched = details[details["is_transfer_pred"].eq(1)].copy()
        predictions = pd.DataFrame(
            {
                **{col: matched[col].values for col in TRANSACTION_KEY_COLUMNS},
                "matched": True,
                "counterparty": matched["counterparty"].values,
                "bscat": matched["bscat"].values,
                "classification_rule_id": matched[
                    "prediction_rule"
                ].values,
                "classification_reason": matched[
                    "transfer_pred_reason"
                ].values,
            }
        )
        # Transfer rows own no stream: pinned to NA (rather than omitted) so the
        # claim archive keeps its fixed column set.
        predictions["stream_id"] = pd.NA
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
