from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)


class DishonourEngine:
    engine_id = "dishonour"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path | None = None) -> None:
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parent / "resources"
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> EngineResult:
        rules = _load_rules(self.resources_dir / "dishonour_rules.csv")
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


def _load_rules(rules_file):
    rules = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_type = (row.get("rule_type") or "").strip().lower()
            pattern = row.get("pattern") or ""
            required_terms = [x.strip().lower() for x in (row.get("required_terms") or "").split(";") if x.strip()]
            if rule_type and pattern:
                rules.append((rule_type, pattern, required_terms))
    return rules
