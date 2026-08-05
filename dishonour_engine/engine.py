from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)
from classification_core.rules import load_dishonour_style_rules


class DishonourEngine:
    engine_id = "dishonour"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path | None = None) -> None:
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parent / "resources"
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> EngineResult:
        rules = load_dishonour_style_rules(
            self.resources_dir / "dishonour_rules.csv"
        )
        candidates = context.candidates.copy()
        text_col = candidates["text"].fillna("").astype(str)

        mask = pd.Series(False, index=candidates.index)
        for rule_type, pattern, required_terms in rules:
            if rule_type == "keyword":
                mask |= text_col.str.contains(re.escape(pattern), case=False, na=False)
            else:
                lower_text = text_col.str.lower()
                term_mask = pd.Series(True, index=candidates.index)
                for term in required_terms:
                    term_mask &= lower_text.str.contains(term, na=False)
                mask |= term_mask & text_col.str.contains(pattern, case=False, na=False, regex=True)

        matched = candidates[mask].copy()
        predictions = pd.DataFrame(
            {
                **{col: matched[col].values for col in TRANSACTION_KEY_COLUMNS},
                "matched": True,
                "counterparty": "-",
                "finv_category": "Dishonours",
                "stream_id": pd.NA,
                "classification_rule_id": "dishonour:generic",
                "classification_reason": "Dishonour transaction detected",
            }
        )

        return EngineResult(predictions=predictions, transactions=pd.DataFrame())

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        return []
