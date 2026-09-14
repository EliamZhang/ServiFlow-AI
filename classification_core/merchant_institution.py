"""Shared institution layer for merchant-KB engines (rent, gambling, ...).

Engines that own a category whose merchants were moved out of
``initial_engine/merchant_kb.csv`` share the same matching machinery:

* a rules file with two kinds of rows — ``source=rule`` (keyword/regex rows
  with a confidence score) and ``source=institution`` (merchant rows with
  pipe-separated keyword variants and a counterparty/merchant name);
* an Aho-Corasick automaton over the institution keyword variants — longest
  keyword wins, whole-word boundaries, later rows win for duplicated keywords
  (mirroring the old merchant-KB automaton);
* two claim guards that preserve the pre-move outcome matrix:

  * rows already claimed by engines that used to beat the initial-engine claim
    (fee / dishonour) are skipped;
  * rows the initial engine (priority 10) claimed with a keyword *at least as
    long* as the institution keyword are skipped — the pre-move automaton
    ranked keywords by length across all categories, so the non-Rent/non-
    Gambling category would have won then as well.

The engine-specific parts stay in the engine module: the double-layer
combination (institution beats keyword rules), reason construction, and the
keyword-layer prior-claim semantics (which differ per engine — rent's keyword
layer ignores prior claims, gambling's defers to them).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import ahocorasick
import pandas as pd

from classification_core.models import TRANSACTION_KEY_COLUMNS

# Confidence for institution rows.  Above every keyword/regex rule (<= 0.90) so
# that when both layers match, the institution wins and its merchant name is
# used as counterparty.
INSTITUTION_CONFIDENCE = 0.95

# Cap on pipe-separated keyword variants per institution, matching the initial
# engine's guard against KB entries with hundreds of generic keywords.
MAX_VARIANTS_PER_INSTITUTION = 50

# The initial engine (priority 10) runs before rent/gambling and claims every
# row that matches a non-Rent/non-Gambling keyword, with the *longest* keyword
# winning (its Aho-Corasick ranking).  Before the category move, the single KB
# automaton ranked all keywords together by length, so a row whose initial-
# engine keyword is at least as long as the institution keyword was claimed by
# the other category back then — the institution layer must defer to it.  The
# keyword is recovered from the initial engine's claim reason
# (``evidence=keyword=K, ...``; keywords are ``[A-Z0-9 ]``, no commas, so this
# parses reliably).
INITIAL_KEYWORD_RE = re.compile(r"keyword=([A-Z0-9 ]+?)(?:,|$)")


def load_rules(
    rules_file: Path,
) -> tuple[
    list[tuple[str, str, str, str, float]],
    list[tuple[str, str, str]],
]:
    """Load an engine rules CSV.

    Returns (rule_rows, institution_rows):

    * rule_rows — ``(rule_name, category, pattern, match_type, confidence)``
      for rows with ``source != institution`` (keyword/regex rules), sorted by
      confidence descending.
    * institution_rows — ``(rule_name, pipe_joined_keywords, counterparty)`` in
      file order (later rows win for shared keywords, mirroring the old KB
      automaton).
    """
    rules: list[tuple[str, str, str, str, float]] = []
    institutions: list[tuple[str, str, str]] = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_name = (row.get("rule_name") or "").strip()
            category = (row.get("category") or "").strip()
            pattern = (row.get("pattern") or "").strip()
            match_type = (row.get("match_type") or "keyword").strip().lower()
            confidence_str = (row.get("confidence") or "0.5").strip()
            source = (row.get("source") or "").strip().lower()
            counterparty = (row.get("counterparty") or "").strip()

            if not rule_name or not category or not pattern:
                continue

            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.5

            if source == "institution":
                institutions.append((rule_name, pattern, counterparty))
            else:
                rules.append(
                    (rule_name, category, pattern, match_type, confidence)
                )

    rules.sort(key=lambda r: r[4], reverse=True)
    return rules, institutions


def build_institution_automaton(
    institutions: list[tuple[str, str, str]],
    max_variants: int = MAX_VARIANTS_PER_INSTITUTION,
) -> ahocorasick.Automaton:
    """Build an automaton from institution rows.

    Each pipe-separated keyword variant is inserted as an independent keyword
    (capped at *max_variants* per institution).  Later rows win for duplicated
    keywords, mirroring the old merchant-KB automaton.  The automaton value
    carries the row's counterparty (falling back to the rule/merchant name),
    which becomes the prediction's counterparty.
    """
    automaton = ahocorasick.Automaton()
    for merchant, keywords, counterparty in institutions:
        display = counterparty or merchant
        for variant in keywords.split("|")[:max_variants]:
            kw = variant.strip()
            if not kw:
                continue
            automaton.add_word(kw, (kw, len(kw), display))
    automaton.make_automaton()
    return automaton


def match_institutions(
    text_clean: str, automaton: ahocorasick.Automaton
) -> tuple[str, str] | None:
    """Return (keyword, merchant) of the longest whole-word institution match.

    Whole-word boundaries are spaces or string edges (cleaned text is
    ``[A-Z0-9 ]``).  Longest keyword wins — the most specific merchant — which
    mirrors the initial engine's ranking.  ``None`` when nothing matches.
    """
    text_len = len(text_clean)
    best: tuple[int, str, str] | None = None

    for end_pos, (kw, kw_len, merchant) in automaton.iter(text_clean):
        # pyahocorasick 2.x returns index of the last matching character
        # (closed interval), so start = end_pos - kw_len + 1.
        # pyahocorasick 1.x returns one-past-the-end (Python slice end),
        # so start = end_pos - kw_len.  We try 2.x first (current version),
        # then 1.x, then fall back to str.find().
        pos = end_pos - kw_len + 1
        if pos < 0 or text_clean[pos:pos + kw_len] != kw:
            pos = end_pos - kw_len
            if pos < 0 or text_clean[pos:pos + kw_len] != kw:
                pos = text_clean.find(kw)
                if pos == -1:
                    continue

        # Whole-word check
        if pos > 0 and text_clean[pos - 1] != " ":
            continue
        end = pos + kw_len
        if end < text_len and text_clean[end] != " ":
            continue

        # Longest keyword wins
        if best is None or kw_len > best[0]:
            best = (kw_len, kw, merchant)

    if best is None:
        return None
    return (best[1], best[2])


def prior_claim_keys(
    prior_claims: pd.DataFrame,
    excludes: tuple[str, ...],
) -> set[tuple[str, str]]:
    """Keys of rows already claimed by *excludes* engines the institution layer
    must not override (fee / dishonour)."""
    if prior_claims is None or prior_claims.empty:
        return set()
    mask = prior_claims["classification_engine"].isin(excludes)
    excluded = prior_claims.loc[
        mask, list(TRANSACTION_KEY_COLUMNS)
    ].astype("string").fillna("")
    return {
        (row.application_id, row.transaction_id)
        for row in excluded.itertuples(index=False)
    }


def initial_claim_keyword_lengths(
    prior_claims: pd.DataFrame,
) -> dict[tuple[str, str], int]:
    """Map (application_id, transaction_id) -> initial-engine keyword length.

    For every row the initial engine claimed, recover the length of the
    keyword it matched (its reason carries ``evidence=keyword=K``).  The
    institution layer uses this to defer to the initial engine when its
    keyword is at least as long as the institution keyword — see module
    docstring.
    """
    if prior_claims is None or prior_claims.empty:
        return {}
    mask = prior_claims["classification_engine"].eq("initial")
    rows = prior_claims.loc[mask].copy()
    if rows.empty:
        return {}
    reasons = (
        rows["classification_reason"]
        .astype("string")
        .fillna("")
        .str.extract(INITIAL_KEYWORD_RE, expand=False)
        .fillna("")
        .str.strip()
    )
    lengths = reasons.str.len()
    keys = rows[list(TRANSACTION_KEY_COLUMNS)].astype("string").fillna("")
    return {
        (row.application_id, row.transaction_id): length
        for row, length in zip(keys.itertuples(index=False), lengths)
        if length > 0
    }
