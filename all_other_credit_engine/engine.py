from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from classification_core.claims import exclude_prior_claimed
from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)
from classification_core.rules import load_dishonour_style_rules


class AllOtherCreditEngine:
    engine_id = "all_other_credit"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path | None = None) -> None:
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parent / "resources"
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> EngineResult:
        candidates = context.candidates.copy()

        is_credit = candidates["dr_cr"].astype(str).str.lower().eq("credit")
        candidates = candidates[is_credit].copy()

        if candidates.empty:
            return EngineResult(
                predictions=pd.DataFrame(columns=[*TRANSACTION_KEY_COLUMNS, "matched", "counterparty", "finv_category"]),
                transactions=pd.DataFrame(),
            )

        # Exclude rows already classified by prior engines,
        # except External Transfers which may be re-matched.
        candidates = exclude_prior_claimed(
            candidates,
            context.prior_claims,
            keep_categories={"External Transfers"},
        )

        if candidates.empty:
            return EngineResult(
                predictions=pd.DataFrame(columns=[*TRANSACTION_KEY_COLUMNS, "matched", "counterparty", "finv_category"]),
                transactions=pd.DataFrame(),
            )

        rules = load_dishonour_style_rules(
            self.resources_dir / "all_other_credit_rules.csv"
        )
        text_col = candidates["text"].fillna("").astype(str)

        mask = pd.Series(False, index=candidates.index)
        for rule_type, pattern, _required_terms in rules:
            if rule_type == "keyword":
                mask |= text_col.str.contains(
                    re.escape(pattern), case=False, na=False
                )

        matched = candidates[mask].copy()
        predictions = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
        predictions["matched"] = True
        predictions["counterparty"] = "-"
        predictions["finv_category"] = "All Other Credits"

        return EngineResult(
            predictions=predictions,
            transactions=pd.DataFrame(),
        )

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        return []
