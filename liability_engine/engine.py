from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)
from classification_core.reasons import format_classification_reason
from classification_core.transaction_keys import filter_to_transaction_keys

from .domain.summary import build_summary
from .pipeline import DEFAULT_RESOURCES_DIR, run_pipeline


class LiabilityEngine:
    engine_id = "liability"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path = DEFAULT_RESOURCES_DIR) -> None:
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> EngineResult:
        result = run_pipeline(
            context.candidates,
            resources_dir=self.resources_dir,
        )
        details = result.transactions
        category = details["finv_category"].astype("string").str.strip()
        matched = details[
            category.notna()
            & category.ne("")
            & details["counterparty"].notna()
            & details["counterparty"].astype("string").str.strip().ne("")
        ].copy()
        predictions = pd.DataFrame(
            {
                **{col: matched[col].values for col in TRANSACTION_KEY_COLUMNS},
                "matched": True,
                "counterparty": matched["counterparty"].values,
                "finv_category": matched["finv_category"].values,
                "stream_id": matched["stream_id"].values,
                "classification_rule_id": matched["product_type"].map(
                    lambda value: f"liability_product:{value}"
                ).values,
                "classification_reason": matched.apply(
                    lambda row: format_classification_reason(
                        category=row["finv_category"],
                        rule=f"liability_product:{row['product_type']}",
                        evidence=[
                            f"product_type={row['product_type']}",
                            f"counterparty={row['counterparty']}",
                        ],
                    ),
                    axis=1,
                ).values,
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
        accepted_details = filter_to_transaction_keys(
            result.transactions,
            accepted_predictions,
        )
        summary = build_summary(
            accepted_details,
            limits_file=self.resources_dir / "bnpl_maximum_limits.csv",
        )
        return [SummaryArtifact("liability_summary", summary)]
