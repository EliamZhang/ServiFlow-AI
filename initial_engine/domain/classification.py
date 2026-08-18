# -*- coding: utf-8 -*-
"""
Merchant keyword matching via pyahocorasick (C-extension) automaton.

Loads merchant_kb.csv in chunks, builds a trie from all keyword variants,
then scans transaction text in a single pass per row.

Performance characteristics
---------------------------
- Build: O(total keyword characters) — one-off cost (C level).
- Search: O(text length + number of matches) per transaction (C level).
"""

from __future__ import annotations

import re
from pathlib import Path

import ahocorasick
import pandas as pd

from classification_core.reasons import format_classification_reason

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Payment-channel prefixes that should not be treated as merchant names.
# Stripped from the beginning of transaction text before keyword matching so the
# actual counterparty name can be matched at position 0.  Applied only to
# transaction text, never to keywords — a KB entry named "Bill Pay Services"
# would still have its keywords inserted as-is.
_CHANNEL_PREFIX_RE = re.compile(
    r"^(?:"
    r"BILL\s*PAY(?:MENT)?\s+|"
    r"VISA\s+(?:WDL|PURCHASE|DEBIT)\s+(?:PURCHASE\s+)?(?:\d+\w+\s+)?\s*|"
    r"VISA\s+CREDIT\s+\d*\s*|"
    r"EFTPOS\s+(?:DEBIT|WDL)(?:\s+\d+)?\s+|"
    r"MISCELLANEOUS\s+DEBIT\s+V\d+\s+\d+\s+\d+\s+|"
    r"DEBIT\s+CARD\s+PURCHASE\s+|"
    r"EFT\s+Dep\s+"
    r")",
    re.IGNORECASE,
)

# EFTPOS receipt timestamps ("EFTPOS DEBIT EFTPOS 21/05 10:14") are glued to the
# merchant name in the raw text ("...10:14Subway").  They must be stripped
# *before* non-alphanumerics become spaces — after that step the minute field is
# glued to the merchant ("14SUBWAY") and the whole-word boundary check rejects the
# merchant keyword.  The optional second "EFTPOS" covers the common
# "EFTPOS DEBIT EFTPOS DD/MM HH:MM<merchant>" shape.
_EFTPOS_TS_RE = re.compile(
    r"^EFTPOS\s+DEBIT\s+(?:EFTPOS\s+)?\d{1,2}/\d{1,2}\s+\d{1,2}:\d{1,2}",
    re.IGNORECASE,
)


def _clean_transaction_text(series: pd.Series) -> pd.Series:
    """Vectorized: normalise transaction text and strip payment-channel prefixes.

    Equivalent to the scalar clean_text() + _CHANNEL_PREFIX_RE chain, but
    operates on the entire Series at C level — no per-row Python calls.
    """
    # Step 0 — strip EFTPOS receipt timestamps before alnum cleaning
    s = series.fillna("").str.upper()
    s = s.str.replace(_EFTPOS_TS_RE, "", regex=True)
    # Step 1 — clean_text equivalent (alphanumerics → collapse spaces)
    s = s.str.replace(r"[^A-Z0-9]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # Step 2 — strip payment-channel prefixes
    s = s.str.replace(_CHANNEL_PREFIX_RE, "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


# ---------------------------------------------------------------------------
# Automaton wrapper
# ---------------------------------------------------------------------------

class _Automaton:
    """Thin wrapper around ``ahocorasick.Automaton`` with keyword metadata.

    ``pyahocorasick`` C-extension objects do not support ad-hoc attribute
    assignment, so we wrap the automaton to carry *keyword_count* (and any
    future metadata) alongside it.  ``.iter()`` delegates directly to the
    underlying C automaton for zero-overhead search.
    """

    __slots__ = ("_a", "keyword_count")

    def __init__(self, automaton: ahocorasick.Automaton, keyword_count: int) -> None:
        self._a = automaton
        self.keyword_count = keyword_count

    def iter(self, text: str):
        """Yield ``(end_pos, value)`` tuples from the underlying automaton."""
        return self._a.iter(text)

    def search(self, text: str) -> list[tuple[str, str, str]]:
        """Return whole-word ``(keyword, merchant, category)`` hits — compatible
        with the legacy pure-Python automaton API used by downstream engines
        (e.g. income)."""
        hits: list[tuple[str, str, str]] = []
        text_len = len(text)
        for end_pos, (kw, kw_len, merchant, cat) in self._a.iter(text):
            # O(1) position from automaton end_pos.
            # pyahocorasick 2.x: end_pos = last matching char index → start = end_pos - kw_len + 1.
            # pyahocorasick 1.x: end_pos = one-past-the-end → start = end_pos - kw_len.
            pos = end_pos - kw_len + 1  # try 2.x convention first
            if pos < 0 or text[pos:pos + kw_len] != kw:
                pos = end_pos - kw_len  # try 1.x convention
                if pos < 0 or text[pos:pos + kw_len] != kw:
                    pos = text.find(kw)  # safety fallback
                    if pos == -1:
                        continue
            # Whole-word check
            if pos > 0 and text[pos - 1] != " ":
                continue
            end = pos + kw_len
            if end < text_len and text[end] != " ":
                continue
            hits.append((kw, merchant, cat))
        return hits


# ---------------------------------------------------------------------------
# KB loader
# ---------------------------------------------------------------------------

def load_merchant_kb(kb_path: str | Path) -> _Automaton:
    """Chunk-read *kb_path* and return a ready-to-use automaton.

    All rows are indexed regardless of whether *category* is populated.
    Each pipe-separated variant in the *keywords* column is inserted as an
    independent keyword.
    """
    kb_path = Path(kb_path)
    automaton = ahocorasick.Automaton()
    # (keyword_upper, merchant, cat) keyed in file order.  A dict (not a set)
    # preserves insertion order, so rebuilds are deterministic even when the
    # KB lists the same keyword under several merchants — the last row wins,
    # exactly mirroring add_word's overwrite semantics.
    seen: dict[tuple[str, str, str], None] = {}

    chunks = pd.read_csv(
        kb_path,
        usecols=["merchant_name", "keywords", "category"],
        chunksize=100_000,
        encoding="utf-8-sig",
        dtype="string",
    )

    for chunk in chunks:
        # Drop rows with empty keywords.
        chunk = chunk[
            chunk["keywords"].notna() & (chunk["keywords"].str.strip() != "")
        ]
        # Exclude "Financial Institutions" — handled by liability/dishonour engines.
        chunk = chunk[chunk["category"] != "Financial Institutions"]
        if chunk.empty:
            continue

        # Split pipe-separated keywords and cap variants per merchant
        # (vectorized — avoids Python-level per-row split loop).
        kw_lists = chunk["keywords"].str.split("|").str[:_MAX_VARIANTS_PER_MERCHANT]

        # Explode: one row per keyword variant.  Handled at C level by pandas
        # instead of Python-level iterrows.
        exploded = chunk[["merchant_name", "category"]].copy()
        exploded["_kw_raw"] = kw_lists
        exploded = exploded.explode("_kw_raw").dropna(subset=["_kw_raw"])

        # Strip whitespace around "|" separators that survives the split.
        # Without this, "KEYWORD_A | KEYWORD_B" produces "KEYWORD_A " /
        # " KEYWORD_B", and trailing spaces break the whole-word boundary
        # check in _classify_one.
        exploded["_kw_raw"] = exploded["_kw_raw"].str.strip()

        # Keywords are already pre-cleaned by Merchant-Extraction's
        # dedup_keywords.py --preclean (upper, alphanumerics, collapsed
        # spaces).  No apply(clean_text) needed — the 650k+ Python-level
        # clean_text calls that were the dominant startup cost are gone.

        # Filter: stopwords.
        exploded = exploded[~exploded["_kw_raw"].isin(_STOPWORDS)]

        if exploded.empty:
            continue

        # Dedup within chunk (most duplicates eliminated here at C level).
        exploded = exploded.drop_duplicates(
            subset=["_kw_raw", "merchant_name", "category"]
        )

        # Pre-clean at C level — avoids per-row str()/strip()/pd.notna()
        # inside the zip loop body (~650k items).
        exploded["merchant_name"] = exploded["merchant_name"].str.strip()
        exploded["category"] = exploded["category"].fillna("").str.strip()

        # Cross-chunk dedup via fast zip() over raw numpy arrays — avoids
        # pandas Series iterator overhead.  iterrows() is ~13× slower.
        for kw, merchant, category in zip(
            exploded["_kw_raw"].values,
            exploded["merchant_name"].values,
            exploded["category"].values,
        ):
            key = (kw, merchant, category)
            if key not in seen:
                seen[key] = None

    for kw, merchant, cat in seen:
        kw_len = len(kw)
        automaton.add_word(kw, (kw, kw_len, merchant, cat))

    automaton.make_automaton()
    return _Automaton(automaton, len(seen))


# Module-level cache so that downstream engines (e.g. income) can reuse the
# same automaton without reloading the 395 MB CSV.
_cached_automaton: _Automaton | None = None
_cached_kb_path: str | None = None
_DEFAULT_KB_PATH: str | None = None


def get_cached_automaton(kb_path: str | Path | None = None) -> _Automaton:
    """Return a cached automaton, building it on first call.

    The automaton is cached in memory within the same process so that downstream
    engines (e.g. income) can reuse it without reloading the 395 MB CSV.  Each
    invocation rebuilds from CSV — there is no persistent disk cache.
    """
    global _cached_automaton, _cached_kb_path, _DEFAULT_KB_PATH
    if _DEFAULT_KB_PATH is None:
        _DEFAULT_KB_PATH = str(
            Path(__file__).resolve().parent.parent / "merchant_kb.csv"
        )
    resolved = str(kb_path or _DEFAULT_KB_PATH)

    # In-memory cache hit.
    if _cached_automaton is not None and _cached_kb_path == resolved:
        return _cached_automaton

    # Build from scratch.
    _cached_automaton = load_merchant_kb(resolved)
    _cached_kb_path = resolved
    return _cached_automaton


# ---------------------------------------------------------------------------
# Batch matcher
# ---------------------------------------------------------------------------

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
        "BILL",
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
        # --- generic company suffixes ---
        "LIMITED",
        "GROUP",
        "HOLDINGS",
        "COMPANY",
        "CORPORATION",
        "ENTERPRISES",
        "SERVICES",
    }
)


def _classify_one(
    text_clean: str,
    automaton: _Automaton,
) -> tuple[bool, str, str, str, str, str]:
    """Classify a single cleaned text by whole-word keyword match.

    Returns (matched, counterparty, category, keyword, rule_id, reason).

    The automaton yields every matching keyword; a keyword only counts if
    it appears as a whole word (boundary check — the cleaned text is
    ``[A-Z0-9 ]``, so boundaries are spaces or string edges).  Matches
    are ranked by longest keyword first — the longest (most specific) one wins.
    """
    if not text_clean:
        return (False, "", "", "", "", "")

    text_len = len(text_clean)

    best: tuple[int, str, str, str] | None = None

    for end_pos, (kw, kw_len, merchant, cat) in automaton.iter(text_clean):
        # O(1) position from automaton end_pos.
        # pyahocorasick 2.x returns index of the last matching character
        # (closed interval), so start = end_pos - kw_len + 1.
        # pyahocorasick 1.x returns one-past-the-end (Python slice end),
        # so start = end_pos - kw_len.  We try 2.x first (current version),
        # then 1.x, then fall back to str.find().
        pos = end_pos - kw_len + 1
        if pos < 0 or text_clean[pos:pos + kw_len] != kw:
            pos = end_pos - kw_len  # try 1.x convention
            if pos < 0 or text_clean[pos:pos + kw_len] != kw:
                pos = text_clean.find(kw)  # safety fallback
                if pos == -1:
                    continue

        # ── Whole-word check (inlined) ──
        if pos > 0 and text_clean[pos - 1] != " ":
            continue
        end = pos + kw_len
        if end < text_len and text_clean[end] != " ":
            continue

        # ── Longest keyword wins ──
        if best is None or kw_len > best[0]:
            best = (kw_len, kw, merchant, cat)

    if best is None:
        return (False, "", "", "", "", "")

    _, best_kw, best_merchant, best_cat = best

    reason = format_classification_reason(
        category=best_cat,
        rule="merchant_kb_match",
        evidence=[
            f"keyword={best_kw}",
            f"merchant={best_merchant}",
        ],
    )
    return (
        True,
        best_merchant,
        best_cat,
        best_kw,
        "merchant_kb_match",
        reason,
    )


# Sentinel for empty-text rows — avoids function-call overhead for the
# common case where _text_clean is "" after stripping channel prefixes.
_EMPTY_RESULT = (False, "", "", "", "", "")


def match_transactions(
    transactions: pd.DataFrame,
    automaton: _Automaton,
) -> pd.DataFrame:
    """Add *counterparty*, *finv_category* and match metadata columns.

    Returns a DataFrame with the same row order as *transactions*, containing
    the original columns plus the classification columns defined by the engine
    protocol.
    """
    out = transactions.copy()
    out["_text_clean"] = _clean_transaction_text(out["text"])

    # List comprehension over raw numpy array — avoids pandas Series
    # indexing overhead per element (measurably faster than .apply()).
    texts = out["_text_clean"].values
    results = [
        _classify_one(t, automaton) if t else _EMPTY_RESULT
        for t in texts
    ]
    (
        out["matched"],
        out["counterparty"],
        out["finv_category"],
        out["_matched_keyword"],
        out["classification_rule_id"],
        out["classification_reason"],
    ) = zip(*results)

    return out.drop(columns=["_text_clean"])




