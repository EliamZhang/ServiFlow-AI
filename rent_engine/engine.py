"""Rent classification engine — keyword/regex → Rent category.

Runs after income and liability (priority 800) and catches transactions that
those engines haven't claimed, using rent-specific keyword and regex rules.
Does NOT use exclude_prior_claimed — income/liability protection is handled
at the orchestrator level (candidates are pre-filtered).

Text is normalised before matching (uppercase, alphanumerics + spaces only)
so whole-word matching is reliable.

Each rule carries a confidence score (0.0–1.0). When multiple rules match
the same transaction the one with the highest confidence wins.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    TRANSACTION_KEY_COLUMNS,
)
from classification_core.reasons import format_classification_reason
from classification_core.text import clean_text


class RentEngine:
    """Classify transactions as Rent via keyword/regex matching."""

    engine_id = "rent"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path | None = None) -> None:
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parent / "resources"
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> EngineResult:
        rules = _load_rules(self.resources_dir / "rent_rules.csv")
        candidates = context.candidates.copy()

        empty_pred = EngineResult(
            predictions=pd.DataFrame(
                columns=[
                    *TRANSACTION_KEY_COLUMNS,
                    "matched",
                    "counterparty",
                    "finv_category",
                ]
            ),
            transactions=pd.DataFrame(),
        )

        if candidates.empty:
            return empty_pred

        text_clean = candidates["text"].apply(clean_text)

        matched_mask = pd.Series(False, index=candidates.index)
        best_rule_names: list[str] = []
        best_confidences: list[float] = []

        for idx in candidates.index:
            text = text_clean.at[idx]
            if not text:
                matched_mask.at[idx] = False
                best_rule_names.append("")
                best_confidences.append(0.0)
                continue

            best_conf = 0.0
            best_rule = ""
            text_len = len(text)

            for rule_name, _category, keyword, match_type, confidence in rules:
                if confidence <= best_conf:
                    continue

                if match_type == "keyword":
                    pos = text.find(keyword)
                    if pos == -1:
                        continue
                    if pos > 0 and text[pos - 1] != " ":
                        continue
                    end = pos + len(keyword)
                    if end < text_len and text[end] != " ":
                        continue
                else:  # regex
                    if not re.search(keyword, text):
                        continue

                best_conf = confidence
                best_rule = rule_name

            if best_conf > 0:
                matched_mask.at[idx] = True
                best_rule_names.append(best_rule)
                best_confidences.append(best_conf)
            else:
                best_rule_names.append("")
                best_confidences.append(0.0)

        matched = candidates[matched_mask].copy()
        if matched.empty:
            return empty_pred

        predictions = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
        predictions["matched"] = True
        predictions["counterparty"] = "-"
        predictions["finv_category"] = "Rent"

        m_confidences = [
            c for i, c in enumerate(best_confidences) if matched_mask.iloc[i]
        ]
        m_rule_names = [
            r for i, r in enumerate(best_rule_names) if matched_mask.iloc[i]
        ]

        reasons: list[tuple[str, str]] = []
        for rule, conf in zip(m_rule_names, m_confidences):
            reason = format_classification_reason(
                category="Rent",
                rule=rule,
                evidence=[f"confidence={conf:.2f}"],
            )
            reasons.append((rule, reason))

        predictions["classification_rule_id"] = [r[0] for r in reasons]
        predictions["classification_reason"] = [r[1] for r in reasons]
        predictions["stream_id"] = pd.NA

        return EngineResult(
            predictions=predictions,
            transactions=pd.DataFrame(),
            diagnostics={
                "rules_loaded": len(rules),
                "candidates_considered": len(candidates),
                "matched_count": len(predictions),
            },
        )

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list:
        return []


def _load_rules(
    rules_file: Path,
) -> list[tuple[str, str, str, str, float]]:
    """Load rent rules from CSV, sorted by confidence descending.

    Returns list of (rule_name, category, pattern, match_type, confidence).
    """
    rules: list[tuple[str, str, str, str, float]] = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_name = (row.get("rule_name") or "").strip()
            category = (row.get("category") or "").strip()
            pattern = (row.get("pattern") or "").strip()
            match_type = (row.get("match_type") or "keyword").strip().lower()
            confidence_str = (row.get("confidence") or "0.5").strip()

            if not rule_name or not category or not pattern:
                continue

            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.5

            rules.append((rule_name, category, pattern, match_type, confidence))

    rules.sort(key=lambda r: r[4], reverse=True)
    return rules
