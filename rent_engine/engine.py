"""Rent classification engine — merchant-KB institutions + keyword/regex → Rent.

Runs after income and liability (priority 800) and catches transactions that
those engines haven't claimed, using two layers of rules loaded from
``rent_rules.csv``:

* **institution rows** (``source=institution``) — merchants that were moved out
  of ``initial_engine/merchant_kb.csv`` (category Rent) so rent recognition
  lives entirely in this engine.  Matched with an Aho-Corasick automaton
  (longest keyword wins, whole-word boundaries) and the same channel-prefix
  text cleaning the initial engine used, so matches behave identically to the
  old KB claims.  Counterparty is the merchant name.  To preserve the
  pre-move outcome matrix, an institution match is *dropped* when:

  * the fee or dishonour engine already claimed the row (those engines ran
    earlier and would have beaten the old initial-engine claim too), or
  * the initial engine already claimed the row with a keyword **at least as
    long** as the rent keyword — the pre-move automaton ranked keywords by
    length across *all* categories, so a longer non-Rent match would have won
    the row then as well (equal lengths resolve in the initial engine's
    favour).

  Keyword rows may still claim such rows, exactly as before.
* **keyword / regex rows** (``source=rule``) — the original rent rules.  Each
  rule carries a confidence score (0.0–1.0); when multiple rule rows match the
  same transaction the highest confidence wins.  Counterparty is "-".

Income/liability protection is handled at the orchestrator level (candidates
are pre-filtered).  Text is normalised before matching (uppercase, alphanumerics
+ spaces only) so whole-word matching is reliable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from classification_core.merchant_institution import (
    build_institution_automaton,
    initial_claim_keyword_lengths,
    load_rules,
    match_institutions,
    prior_claim_keys,
)
from classification_core.models import (
    EngineContext,
    EngineResult,
    TRANSACTION_KEY_COLUMNS,
)
from classification_core.reasons import format_classification_reason
from classification_core.text import clean_text, clean_text_with_channel_prefix

# Engines whose earlier claims must not be overridden by the institution layer.
# Mirrors the pre-move semantics: when the rent merchants lived in the initial
# engine (priority 10), fee (500) / dishonour (150) claims beat them; only the
# rent keyword layer (priority 800) was able to re-claim afterwards.
_PRIOR_CLAIM_ENGINES_INSTITUTION_EXCLUDES = ("fee", "dishonour")

_INSTITUTION_RULE_ID = "rent_merchant_kb"


class RentEngine:
    """Classify transactions as Rent via institution + keyword/regex matching."""

    engine_id = "rent"
    engine_version = "2.0"

    def __init__(self, resources_dir: str | Path | None = None) -> None:
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parent / "resources"
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> EngineResult:
        rules, institutions = load_rules(
            self.resources_dir / "rent_rules.csv"
        )
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

        # ------------------------------------------------------------------
        # Keyword / regex layer (unchanged semantics: confidence wins, claims
        # regardless of prior claims, counterparty "-")
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Institution layer (Aho-Corasick, longest keyword wins, counterparty
        # = merchant name).  Skips rows already claimed by fee/dishonour, and
        # rows the initial engine claimed with a keyword at least as long as
        # the rent keyword (mirroring the pre-move across-category ranking).
        # ------------------------------------------------------------------
        excluded_keys = prior_claim_keys(
            context.prior_claims, _PRIOR_CLAIM_ENGINES_INSTITUTION_EXCLUDES
        )
        initial_keyword_lens = initial_claim_keyword_lengths(
            context.prior_claims
        )
        inst_merchants: list[str] = []
        inst_keywords: list[str] = []

        if institutions:
            automaton = build_institution_automaton(institutions)
            text_inst_clean = candidates["text"].apply(
                clean_text_with_channel_prefix
            )
            for idx in candidates.index:
                text = text_inst_clean.at[idx]
                if not text:
                    inst_merchants.append("")
                    inst_keywords.append("")
                    continue
                hit = match_institutions(text, automaton)
                if hit is None:
                    inst_merchants.append("")
                    inst_keywords.append("")
                    continue
                key = (
                    str(candidates.at[idx, "application_id"]),
                    str(candidates.at[idx, "transaction_id"]),
                )
                if key in excluded_keys:
                    inst_merchants.append("")
                    inst_keywords.append("")
                    continue
                if (
                    initial_keyword_lens.get(key, 0)
                    >= len(hit[0])
                ):
                    inst_merchants.append("")
                    inst_keywords.append("")
                    continue
                inst_merchants.append(hit[1])
                inst_keywords.append(hit[0])
        else:
            inst_merchants = [""] * len(candidates)
            inst_keywords = [""] * len(candidates)

        # ------------------------------------------------------------------
        # Combine: institution layer beats keyword layer when both match.
        # ------------------------------------------------------------------
        inst_win = pd.Series(
            [bool(m) for m in inst_merchants], index=candidates.index
        )
        kw_win = matched_mask & ~inst_win
        matched_mask = inst_win | kw_win

        predictions = candidates.loc[
            matched_mask, list(TRANSACTION_KEY_COLUMNS)
        ].copy()
        predictions["matched"] = True
        predictions["finv_category"] = "Rent"

        counterparts = [
            (
                inst_merchants[i]
                if inst_win.iloc[i]
                else "-"
            )
            for i in range(len(candidates))
        ]
        predictions["counterparty"] = [
            counterparts[i]
            for i in range(len(candidates))
            if matched_mask.iloc[i]
        ]

        reasons: list[tuple[str, str]] = []
        for i in range(len(candidates)):
            if not matched_mask.iloc[i]:
                continue
            if inst_win.iloc[i]:
                reason = format_classification_reason(
                    category="Rent",
                    rule=_INSTITUTION_RULE_ID,
                    evidence=[
                        f"keyword={inst_keywords[i]}",
                        f"merchant={inst_merchants[i]}",
                    ],
                )
                reasons.append((_INSTITUTION_RULE_ID, reason))
            else:
                rule = best_rule_names[i]
                conf = best_confidences[i]
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
                "institutions_loaded": len(institutions),
                "candidates_considered": len(candidates),
                "matched_count": len(predictions),
                "matched_by_institution": int(inst_win.sum()),
                "matched_by_keyword": int(kw_win.sum()),
            },
        )

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list:
        return []
