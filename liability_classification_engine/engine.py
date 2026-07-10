from __future__ import annotations

from pathlib import Path

import pandas as pd

from serviflow.models import (
    EngineContext,
    PredictionBatch,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)
from serviflow.transaction_keys import filter_to_transaction_keys

from .loan_summary import build_loan_summary
from .pipeline import DEFAULT_RESOURCES_DIR, classify_liability_transactions


class LiabilityEngine:
    engine_id = "liability"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path = DEFAULT_RESOURCES_DIR) -> None:
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> PredictionBatch:
        result = classify_liability_transactions(
            context.candidates,
            resources_dir=self.resources_dir,
        )
        details = result.transactions
        category = details["finv_category"].astype("string").str.strip()
        matched = details[category.notna() & category.ne("")].copy()
        predictions = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
        predictions["matched"] = True
        predictions["proposed_counterparty"] = matched["counterparty"].values
        predictions["proposed_finv_category"] = matched["finv_category"].values
        predictions["stream_id"] = matched["stream_id"].values
        predictions["rule_id"] = matched["product_type"].map(
            lambda value: f"liability_product:{value}"
        ).values
        predictions["reason"] = matched["product_type"].map(
            lambda value: f"Matched liability product type {value}"
        ).values
        return PredictionBatch(
            predictions=predictions,
            details=details,
            diagnostics={
                "stream_counts": result.diagnostics,
                "predicted_liability_rows": int(len(matched)),
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
        summary = build_loan_summary(
            accepted_details,
            limits_file=self.resources_dir / "bnpl_maximum_limits.csv",
        )
        return [SummaryArtifact("liability_summary", "1.0", summary)]
