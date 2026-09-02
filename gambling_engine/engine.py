"""Gambling classification engine — merchant-KB institutions + keyword/regex.

Runs after fee (priority 700) and before rent (priority 800), using two layers
of rules loaded from ``gambling_rules.csv``:

* **institution rows** (``source=institution``) — merchants that were moved out
  of ``initial_engine/merchant_kb.csv`` (category Gambling) so gambling
  recognition lives entirely in this engine.  Matched with the shared
  Aho-Corasick machinery (longest keyword wins, whole-word boundaries) and the
  same channel-prefix text cleaning the initial engine used, so matches behave
  identically to the old KB claims.  Counterparty is the merchant name.  To
  preserve the pre-move outcome matrix, an institution match is *dropped* when:

  * the fee or dishonour engine already claimed the row (those engines ran
    earlier and would have beaten the old initial-engine claim too), or
  * the initial engine already claimed the row with a keyword **at least as
    long** as the gambling keyword — the pre-move automaton ranked keywords by
    length across *all* categories, so a longer non-Gambling match would have
    won the row then as well (equal lengths resolve in the initial engine's
    favour).

  Rows claimed by all_other_credit are NOT dropped: before the move the
  initial engine claimed gambling credit rows, which made all_other_credit
  skip them; now all_other_credit (priority 400) claims them first and the
  institution layer must re-claim them to keep the outcome.

* **keyword / regex rows** (``source=rule``) — the original catch_all rules.
  Before the move these lived in the catch_all engine (priority 999), which
  only claims rows no earlier engine claimed — so this layer defers to ALL
  prior claims via ``exclude_prior_claimed`` (unlike the rent engine's keyword
  layer, which may re-claim anything).  Each rule carries a confidence score;
  when multiple rule rows match the same transaction the highest confidence
  wins.  Counterparty is "-".

Income/liability protection is handled at the orchestrator level (candidates
are pre-filtered).  When both layers match, the institution wins (confidence
0.95 > rules <= 0.90) and its merchant name is used as counterparty.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from classification_core.claims import exclude_prior_claimed
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
# Mirrors the pre-move semantics: when the gambling merchants lived in the
# initial engine (priority 10), fee (500) / dishonour (150) claims beat them.
_PRIOR_CLAIM_ENGINES_INSTITUTION_EXCLUDES = ("fee", "dishonour")

_INSTITUTION_RULE_ID = "gambling_merchant_kb"


class GamblingEngine:
    """Classify transactions as Gambling via institution + keyword/regex."""

    engine_id = "gambling"
    engine_version = "1.0"

    def __init__(self, resources_dir: str | Path | None = None) -> None:
        if resources_dir is None:
            resources_dir = Path(__file__).resolve().parent / "resources"
        self.resources_dir = Path(resources_dir)

    def classify(self, context: EngineContext) -> EngineResult:
        rules, institutions = load_rules(
            self.resources_dir / "gambling_rules.csv"
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
        # Keyword / regex layer.  Mirrors the old catch_all semantics: only
        # rows no prior engine claimed are considered.
        # ------------------------------------------------------------------
        kw_candidates = exclude_prior_claimed(
            candidates, context.prior_claims
        )
        text_clean = kw_candidates["text"].apply(clean_text)

        kw_hits = set()
        best_rules: dict[object, str] = {}
        best_confs: dict[object, float] = {}

        for idx in kw_candidates.index:
            text = text_clean.at[idx]
            if not text:
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
                kw_hits.add(idx)
                best_rules[idx] = best_rule
                best_confs[idx] = best_conf

        # ------------------------------------------------------------------
        # Institution layer (shared Aho-Corasick machinery, counterparty =
        # merchant name).  Skips rows claimed by fee/dishonour, and rows the
        # initial engine claimed with a keyword at least as long as the
        # gambling keyword.
        # ------------------------------------------------------------------
        excluded_keys = prior_claim_keys(
            context.prior_claims, _PRIOR_CLAIM_ENGINES_INSTITUTION_EXCLUDES
        )
        initial_keyword_lens = initial_claim_keyword_lengths(
            context.prior_claims
        )
        inst_hits: dict[object, tuple[str, str]] = {}

        if institutions:
            automaton = build_institution_automaton(institutions)
            text_inst_clean = candidates["text"].apply(
                clean_text_with_channel_prefix
            )
            for idx in candidates.index:
                text = text_inst_clean.at[idx]
                if not text:
                    continue
                hit = match_institutions(text, automaton)
                if hit is None:
                    continue
                key = (
                    str(candidates.at[idx, "application_id"]),
                    str(candidates.at[idx, "transaction_id"]),
                )
                if key in excluded_keys:
                    continue
                if initial_keyword_lens.get(key, 0) >= len(hit[0]):
                    continue
                inst_hits[idx] = hit

        # ------------------------------------------------------------------
        # Combine: institution layer beats keyword layer when both match.
        # ------------------------------------------------------------------
        matched_idx = [i for i in candidates.index
                       if i in inst_hits or i in kw_hits]
        matched_mask = pd.Series(False, index=candidates.index)
        matched_mask.loc[matched_idx] = True

        predictions = candidates.loc[
            matched_mask, list(TRANSACTION_KEY_COLUMNS)
        ].copy()
        predictions["matched"] = True
        predictions["finv_category"] = "Gambling"

        counterparts: list[str] = []
        reasons: list[tuple[str, str]] = []
        for i in matched_idx:
            if i in inst_hits:
                keyword, merchant = inst_hits[i]
                counterparts.append(merchant)
                reason = format_classification_reason(
                    category="Gambling",
                    rule=_INSTITUTION_RULE_ID,
                    evidence=[
                        f"keyword={keyword}",
                        f"merchant={merchant}",
                    ],
                )
                reasons.append((_INSTITUTION_RULE_ID, reason))
            else:
                rule = best_rules[i]
                conf = best_confs[i]
                counterparts.append("-")
                reason = format_classification_reason(
                    category="Gambling",
                    rule=rule,
                    evidence=[f"confidence={conf:.2f}"],
                )
                reasons.append((rule, reason))

        predictions["counterparty"] = counterparts
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
                "keyword_candidates_considered": len(kw_candidates),
                "matched_count": len(predictions),
                "matched_by_institution": len(inst_hits),
                "matched_by_keyword": len(kw_hits - set(inst_hits)),
            },
        )

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list:
        return []
