"""Catch-all classification engine — descriptive term → category inference.

Runs as the LAST engine (priority 999) and catches transactions that no other
engine could classify.  Instead of matching specific merchant names (which is
the initial engine's job), this engine looks for **generic category-indicating
terms** in the transaction text — e.g. "RESTAURANT" → Dining Out, "PHARMACY" →
Health, "HOTEL" → Travel.

Text is normalised before matching (uppercase, alphanumerics + spaces only —
same as the initial engine's ``clean_text``) so whole-word matching is reliable.

Each rule in the CSV carries a *confidence* score (0.0–1.0).  When multiple
rules match the same transaction the one with the highest confidence wins.
This engine only emits a prediction when at least one rule matches — it never
blindly tags everything as a catch-all bucket.
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

# ── text normalisation (same semantics as initial engine's clean_text) ──────
_CLEAN_RE = re.compile(r"[^A-Z0-9]+")


def _normalise(value: object) -> str:
    """Uppercase + strip non-alphanumeric — identical to initial engine."""
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = _CLEAN_RE.sub(" ", text)
    return " ".join(text.split())


class CatchAllEngine:
    """Classify unclassified transactions via descriptive term matching.

    Inspects the ``text`` field of each candidate for category-indicating words
    (e.g. "BAKERY", "PHARMACY", "SALON") and maps them to the corresponding
    finv_category.  Rules are loaded from a CSV file with per-rule confidence
    scores — the highest-confidence match wins.
    """

    engine_id = "catch_all"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path | None = None) -> None:
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parent / "resources"
        self.resources_dir = Path(resources_dir)

    # ------------------------------------------------------------------
    # ClassificationEngine protocol
    # ------------------------------------------------------------------

    def classify(self, context: EngineContext) -> EngineResult:
        rules = _load_rules(self.resources_dir / "catch_all_rules.csv")
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

        # Only consider rows NOT already classified by prior engines.
        if not context.prior_claims.empty:
            prior_keys = {
                (str(row["application_id"]), str(row["transaction_id"]))
                for _, row in context.prior_claims.iterrows()
            }
            keep = pd.Series(True, index=candidates.index)
            for idx, row in candidates.iterrows():
                key = (str(row["application_id"]), str(row["transaction_id"]))
                if key in prior_keys:
                    keep.at[idx] = False
            candidates = candidates[keep].copy()

        if candidates.empty:
            return empty_pred

        # Normalise text column
        text_clean = candidates["text"].apply(_normalise)

        # Per-row best-match
        matched_mask = pd.Series(False, index=candidates.index)
        best_rule_names: list[str] = []
        best_categories: list[str] = []
        best_confidences: list[float] = []

        for idx in candidates.index:
            text = text_clean.at[idx]
            if not text:
                matched_mask.at[idx] = False
                best_rule_names.append("")
                best_categories.append("")
                best_confidences.append(0.0)
                continue

            best_conf = 0.0
            best_cat = ""
            best_rule = ""
            text_len = len(text)

            for rule_name, category, keyword, match_type, confidence in rules:
                if confidence <= best_conf:
                    continue  # already matched a higher-confidence rule

                if match_type == "keyword":
                    pos = text.find(keyword)
                    if pos == -1:
                        continue
                    # whole-word check
                    if pos > 0 and text[pos - 1] != " ":
                        continue
                    end = pos + len(keyword)
                    if end < text_len and text[end] != " ":
                        continue
                else:  # regex
                    if not re.search(keyword, text):
                        continue

                best_conf = confidence
                best_cat = category
                best_rule = rule_name

            if best_conf > 0:
                matched_mask.at[idx] = True
                best_rule_names.append(best_rule)
                best_categories.append(best_cat)
                best_confidences.append(best_conf)
            else:
                best_rule_names.append("")
                best_categories.append("")
                best_confidences.append(0.0)

        matched = candidates[matched_mask].copy()
        if matched.empty:
            return empty_pred

        predictions = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
        predictions["matched"] = True
        predictions["counterparty"] = "-"
        predictions["finv_category"] = [
            c for i, c in enumerate(best_categories)
            if matched_mask.iloc[i]
        ]

        # Build per-row reason strings
        m_confidences = [
            c for i, c in enumerate(best_confidences)
            if matched_mask.iloc[i]
        ]
        m_rule_names = [
            r for i, r in enumerate(best_rule_names)
            if matched_mask.iloc[i]
        ]
        m_cats = predictions["finv_category"].tolist()

        reasons: list[tuple[str, str]] = []
        for cat, rule, conf in zip(m_cats, m_rule_names, m_confidences):
            reason = format_classification_reason(
                category=cat,
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


# ------------------------------------------------------------------
# Rule loading
# ------------------------------------------------------------------

def _load_rules(
    rules_file: Path,
) -> list[tuple[str, str, str, str, float]]:
    """Load catch-all rules from CSV, sorted by confidence descending.

    Returns list of (rule_name, category, pattern, match_type, confidence).

    For *keyword* rules the pattern is a plain uppercase keyword (e.g. ``CAFE``)
    and matching uses ``str.find`` + whole-word boundary check on normalised
    text.  For *regex* rules the pattern is a compiled-on-the-fly regex applied
    to the normalised text.
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
