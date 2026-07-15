# -*- coding: utf-8 -*-
"""
Merchant keyword matching via pure-Python Aho-Corasick automaton.

Loads merchant_kb.csv in chunks, builds a trie from all keyword variants of
categorised rows, then scans transaction text in a single pass per row.

Performance characteristics
---------------------------
- Build: O(total keyword characters) — one-off cost.
- Search: O(text length + number of matches) per transaction.
- Memory: ~10 MB for the automaton (~30k keywords from ~9k rows).
"""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from classification_core.reasons import format_classification_reason

# ---------------------------------------------------------------------------
# Aho-Corasick automaton (pure Python)
# ---------------------------------------------------------------------------

type _Value = tuple[str, str]  # (merchant_name, category)


class _TrieNode:
    __slots__ = ("children", "fail", "outputs")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.fail: _TrieNode | None = None
        self.outputs: list[tuple[str, _Value]] = []  # (keyword, (merchant, cat))


class KeywordAutomaton:
    """Case-insensitive Aho-Corasick automaton.

    Usage
    -----
    >>> automaton = KeywordAutomaton()
    >>> automaton.add_word("PIZZAHUT", "Pizza Hut", "Dining Out")
    >>> automaton.add_word("PIZZA HUT", "Pizza Hut", "Dining Out")
    >>> automaton.build()
    >>> automaton.search("PAID AT PIZZAHUT SYDNEY")
    [('PIZZAHUT', 'Pizza Hut', 'Dining Out')]
    """

    def __init__(self) -> None:
        self.root = _TrieNode()
        self.root.fail = self.root  # root fails to itself
        self._built = False
        self._keyword_count = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_word(self, keyword: str, merchant_name: str, category: str) -> None:
        """Insert one keyword variant into the trie."""
        if self._built:
            raise RuntimeError("Cannot add words after build().")
        if not keyword:
            return
        node = self.root
        for char in keyword:
            if char not in node.children:
                node.children[char] = _TrieNode()
            node = node.children[char]
        node.outputs.append((keyword, (merchant_name, category)))
        self._keyword_count += 1

    def build(self) -> None:
        """BFS to compute failure links and propagate output sets."""
        if self._built:
            return
        queue: deque[_TrieNode] = deque()

        # Depth-1 children fail to root.
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        # BFS for deeper levels.
        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                queue.append(child)

                # Walk failure links to find the deepest node whose child
                # matches `char`.
                fail = current.fail
                while fail is not self.root and char not in fail.children:
                    fail = fail.fail
                if char in fail.children and fail.children[char] is not child:
                    child.fail = fail.children[char]
                else:
                    child.fail = self.root

                # Inherit outputs from the failure node so we don't need to
                # walk the failure chain at search time.
                child.outputs.extend(child.fail.outputs)

        self._built = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, text: str) -> list[tuple[str, str, str]]:
        """Return all matches as ``(keyword, merchant_name, category)`` tuples.

        An empty list means no keyword was found in *text*.
        When multiple keywords match, the caller should pick the longest.
        """
        if not self._built or not text:
            return []
        matches: list[tuple[str, str, str]] = []
        node = self.root
        for char in text:
            while node is not self.root and char not in node.children:
                node = node.fail
            if char in node.children:
                node = node.children[char]
            for kw, (merchant, category) in node.outputs:
                matches.append((kw, merchant, category))
        return matches

    @property
    def keyword_count(self) -> int:
        return self._keyword_count


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Keep only A-Z, 0-9 and spaces — matches how keywords are normalised.
_CLEAN_RE = re.compile(r"[^A-Z0-9]+")


def clean_text(value: object) -> str:
    """Normalise a text field for keyword matching (uppercase, alphanum only)."""
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = _CLEAN_RE.sub(" ", text)
    return " ".join(text.split())  # collapse whitespace


# ---------------------------------------------------------------------------
# KB loader
# ---------------------------------------------------------------------------

def load_merchant_kb(kb_path: str | Path) -> KeywordAutomaton:
    """Chunk-read *kb_path* and return a ready-to-use automaton.

    Only rows with a non-empty *category* are indexed.  Each pipe-separated
    variant in the *keywords* column is inserted as an independent keyword.
    """
    kb_path = Path(kb_path)
    automaton = KeywordAutomaton()
    seen: set[tuple[str, str, str]] = set()  # (keyword_upper, merchant, cat)

    chunks = pd.read_csv(
        kb_path,
        usecols=["merchant_name", "keywords", "category"],
        chunksize=100_000,
        encoding="utf-8-sig",
        dtype="string",
    )

    for chunk in chunks:
        valid = chunk[
            chunk["category"].notna() & (chunk["category"].str.strip() != "")
        ]
        for _, row in valid.iterrows():
            merchant = str(row["merchant_name"]).strip()
            category = str(row["category"]).strip()
            raw_keywords = str(row["keywords"])
            variant_count = 0
            for variant in raw_keywords.split("|"):
                if variant_count >= _MAX_VARIANTS_PER_MERCHANT:
                    break
                kw = clean_text(variant)
                if len(kw) < _MIN_KEYWORD_LEN:  # skip very short / meaningless tokens
                    continue
                if kw in _STOPWORDS:  # skip generic banking-artifact tokens
                    continue
                key = (kw, merchant, category)
                if key not in seen:
                    seen.add(key)
                    automaton.add_word(kw, merchant, category)
                variant_count += 1

    automaton.build()
    return automaton


# Module-level cache so that downstream engines (e.g. income) can reuse the
# same automaton without reloading the 395 MB CSV.
_cached_automaton: KeywordAutomaton | None = None
_cached_kb_path: str | None = None
_DEFAULT_KB_PATH: str | None = None


def get_cached_automaton(kb_path: str | Path | None = None) -> KeywordAutomaton:
    """Return a cached automaton, building it on first call.

    The first call triggers a full load + build (~30-60 s for 395 MB CSV).
    Subsequent calls return the cached instance instantly.
    """
    global _cached_automaton, _cached_kb_path, _DEFAULT_KB_PATH
    if _DEFAULT_KB_PATH is None:
        _DEFAULT_KB_PATH = str(
            Path(__file__).resolve().parent.parent / "merchant_kb.csv"
        )
    resolved = str(kb_path or _DEFAULT_KB_PATH)
    if _cached_automaton is None or _cached_kb_path != resolved:
        _cached_automaton = load_merchant_kb(resolved)
        _cached_kb_path = resolved
    return _cached_automaton


# ---------------------------------------------------------------------------
# Batch matcher
# ---------------------------------------------------------------------------

_MIN_KEYWORD_LEN = 4
_MAX_VARIANTS_PER_MERCHANT = 50  # cap to guard against KB entries with hundreds of
                                # unrelated generic keywords (data-quality issue)

# Generic banking-artifact tokens that appear in the *text* of almost every
# transaction (e.g. "... AUS Card xx9327 Value Date: 10/01/2026"). If any of
# these is admitted into the automaton as a standalone keyword, it matches
# nearly every row and drowns out the real merchant signal — that is exactly
# how the junk KB row `Card | CARD PTY LTD -> Retail` ended up tagging ~12% of
# a sample as counterparty "Card". We never insert a single-token keyword that
# is one of these. Multi-word phrases that merely *contain* one of these tokens
# (e.g. "AUSSIE CARD SERVICES") are still kept — only the bare token is dropped.
# All entries are stored post-`clean_text` (uppercase, alphanumerics only).
_STOPWORDS: frozenset[str] = frozenset(
    {
        # --- card / payment-instrument artifacts ---
        "CARD",
        "CARDS",
        "VISA",
        "MASTERCARD",
        "AMEX",
        "EFTPOS",
        "ATM",
        "PAYWAVE",
        "PAYPASS",
        "CONTACTLESS",
        "CHIP",
        # --- payment rails / scheme names ---
        "BPAY",
        "OSKO",
        "PAYID",
        "PAYTO",
        "NPP",
        "DIRECT",
        "DEBIT",
        "CREDIT",
        "AUTOPAY",
        "RECURRING",
        # --- transaction-type / ledger words ---
        "PAYMENT",
        "PAYMENTS",
        "PYMT",
        "PYMNT",
        "PURCHASE",
        "PURCHASES",
        "TRANSFER",
        "TRANSFERS",
        "XFER",
        "WITHDRAWAL",
        "WITHDRAW",
        "DEPOSIT",
        "DEPOSITS",
        "REFUND",
        "REVERSAL",
        "REVERSED",
        "REBATE",
        "ADJUSTMENT",
        "CHARGE",
        "CHARGES",
        "TRANSACTION",
        "TRANS",
        "SETTLEMENT",
        "PENDING",
        "CLEARED",
        "AUTHORISATION",
        "AUTHORIZATION",
        # --- fees / interest ---
        "FEE",
        "FEES",
        "INTEREST",
        "OVERDRAWN",
        "OVERDRAFT",
        "SURCHARGE",
        # --- channel words ---
        "ONLINE",
        "INTERNET",
        "MOBILE",
        "BANKING",
        "BRANCH",
        "COUNTER",
        "TELLER",
        "PHONE",
        # --- currency / geography generic ---
        "AUS",
        "AUD",
        "AUSTRALIA",
        "AUSTRALIAN",
        "INTERNATIONAL",
        "OVERSEAS",
        "FOREIGN",
        "CONVERSION",
        # --- date / reference scaffolding ---
        "VALUE",
        "DATE",
        "REFERENCE",
        "RECEIPT",
        "INVOICE",
        "ORDER",
        "NUMBER",
        # --- generic filler nouns ---
        "MISCELLANEOUS",
        "SUNDRY",
        "GENERAL",
        "OTHER",
        "PAYEE",
        "MERCHANT",
        "STORE",
        "RETAIL",
        "ACCOUNT",
        "FUNDS",
        "CASH",
        "MONEY",
        # --- generic company suffixes (>= _MIN_KEYWORD_LEN chars) ---
        "LIMITED",
        "GROUP",
        "HOLDINGS",
        "COMPANY",
        "CORPORATION",
        "ENTERPRISES",
        "SERVICES",
    }
)


def _is_whole_word(keyword: str, text: str) -> bool:
    """Check that *keyword* appears as a whole word in *text*.

    After ``clean_text`` the text contains only ``[A-Z0-9 ]`` — word
    boundaries are simply spaces or string edges.
    """
    pos = text.find(keyword)
    if pos == -1:
        return False
    if pos > 0 and text[pos - 1] != " ":
        return False
    end = pos + len(keyword)
    if end < len(text) and text[end] != " ":
        return False
    return True


def match_transactions(
    transactions: pd.DataFrame,
    automaton: KeywordAutomaton,
) -> pd.DataFrame:
    """Add *counterparty*, *finv_category* and match metadata columns.

    Returns a DataFrame with the same row order as *transactions*, containing
    the original columns plus the classification columns defined by the engine
    protocol.
    """
    out = transactions.copy()
    out["_text_clean"] = out["text"].apply(clean_text)

    counterparties: list[str] = []
    categories: list[str] = []
    matched_flags: list[bool] = []
    matched_keywords: list[str] = []
    rule_ids: list[str] = []
    reasons: list[str] = []

    for _, row in out.iterrows():
        text_clean = str(row["_text_clean"])
        hits = automaton.search(text_clean)
        # Keep only whole-word matches.
        whole_word_hits = [
            h for h in hits if _is_whole_word(h[0], text_clean)
        ]
        if whole_word_hits:
            # Longest keyword = most specific match.
            best_kw, best_merchant, best_cat = max(
                whole_word_hits, key=lambda h: len(h[0])
            )
            # Use the KB merchant_name directly — the source CSV has been
            # pre-cleaned (see clean_merchant_kb.py).
            matched_flags.append(True)
            counterparties.append(best_merchant)
            categories.append(best_cat)
            matched_keywords.append(best_kw)
            rule_ids.append("merchant_kb_match")
            reasons.append(
                format_classification_reason(
                    category=best_cat,
                    rule="merchant_kb_match",
                    evidence=[f"keyword={best_kw}", f"merchant={best_merchant}"],
                )
            )
        else:
            matched_flags.append(False)
            counterparties.append("")
            categories.append("")
            matched_keywords.append("")
            rule_ids.append("")
            reasons.append("")

    out["matched"] = matched_flags
    out["counterparty"] = counterparties
    out["finv_category"] = categories
    out["_matched_keyword"] = matched_keywords
    out["classification_rule_id"] = rule_ids
    out["classification_reason"] = reasons

    return out.drop(columns=["_text_clean"])
