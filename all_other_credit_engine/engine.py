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
        if not context.prior_claims.empty:
            prior_map: dict[tuple[str, str], str] = {}
            for _, prow in context.prior_claims.iterrows():
                key = (
                    str(prow["application_id"]),
                    str(prow["transaction_id"]),
                )
                prior_map[key] = str(prow["finv_category"])
            keep = pd.Series(True, index=candidates.index)
            for idx, row in candidates.iterrows():
                key = (str(row["application_id"]), str(row["transaction_id"]))
                prior_cat = prior_map.get(key)
                if prior_cat is not None and prior_cat != "External Transfers":
                    keep.at[idx] = False
            candidates = candidates[keep].copy()

        if candidates.empty:
            return EngineResult(
                predictions=pd.DataFrame(columns=[*TRANSACTION_KEY_COLUMNS, "matched", "counterparty", "finv_category"]),
                transactions=pd.DataFrame(),
            )

        rules = _load_rules(self.resources_dir / "all_other_credit_rules.csv")
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


def _load_rules(rules_file: Path) -> list[tuple[str, str, list[str]]]:
    rules = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_type = (row.get("rule_type") or "").strip().lower()
            pattern = (row.get("pattern") or "").strip()
            required_terms = [
                x.strip().lower()
                for x in (row.get("required_terms") or "").split(";")
                if x.strip()
            ]
            if rule_type and pattern:
                rules.append((rule_type, pattern, required_terms))
    return rules
