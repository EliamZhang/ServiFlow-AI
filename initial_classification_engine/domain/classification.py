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

import math
import pickle
import re
from collections import defaultdict
from pathlib import Path

import ahocorasick
import numpy as np
import pandas as pd

from classification_core.reasons import format_classification_reason

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Keep only A-Z, 0-9 and spaces — matches how keywords are normalised.
_CLEAN_RE = re.compile(r"[^A-Z0-9]+")

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


def clean_text(value: object) -> str:
    """Normalise a text field for keyword matching (uppercase, alphanum only)."""
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = _CLEAN_RE.sub(" ", text)
    return " ".join(text.split())  # collapse whitespace


def _clean_transaction_text(value: object) -> str:
    """Normalise transaction text and strip payment-channel prefixes."""
    text = clean_text(value)
    text = _CHANNEL_PREFIX_RE.sub("", text)
    return text


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
        """Return ``(keyword, merchant, category)`` tuples — compatible with the
        legacy pure-Python automaton API used by downstream engines (e.g. income)."""
        return [
            (kw, merchant, cat)
            for _, (kw, merchant, cat, _p) in self._a.iter(text)
        ]


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
    seen: set[tuple[str, str, str]] = set()  # (keyword_upper, merchant, cat)

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

        # Clean keywords.
        exploded["_kw_clean"] = exploded["_kw_raw"].apply(clean_text)

        # Filter: minimum length.
        exploded = exploded[
            exploded["_kw_clean"].str.len() >= _MIN_KEYWORD_LEN
        ]

        # Filter: stopwords.
        exploded = exploded[~exploded["_kw_clean"].isin(_STOPWORDS)]

        if exploded.empty:
            continue

        # Dedup within chunk (most duplicates eliminated here at C level).
        exploded = exploded.drop_duplicates(
            subset=["_kw_clean", "merchant_name", "category"]
        )

        # Cross-chunk dedup (collect, automaton built after purity calc).
        for _, row in exploded.iterrows():
            kw = row["_kw_clean"]
            merchant = str(row["merchant_name"]).strip()
            category = (
                str(row["category"]).strip() if pd.notna(row["category"]) else ""
            )
            key = (kw, merchant, category)
            if key not in seen:
                seen.add(key)

    # Compute per-keyword merchant count for uniqueness scoring.
    # More distinct merchants sharing a keyword → more generic → lower purity.
    _kw_merchants: dict[str, set[str]] = defaultdict(set)
    for kw, merchant, _cat in seen:
        _kw_merchants[kw].add(merchant)

    total = len(seen)
    for kw, merchant, cat in seen:
        uniqueness = math.log(total / len(_kw_merchants[kw]))
        completeness = 2.0 if cat else 1.0
        purity = uniqueness * completeness
        automaton.add_word(kw, (kw, merchant, cat, purity))

    automaton.make_automaton()
    return _Automaton(automaton, total)


# Module-level cache so that downstream engines (e.g. income) can reuse the
# same automaton without reloading the 395 MB CSV.
_cached_automaton: _Automaton | None = None
_cached_kb_path: str | None = None
_DEFAULT_KB_PATH: str | None = None


def get_cached_automaton(kb_path: str | Path | None = None) -> _Automaton:
    """Return a cached automaton, building it on first call.

    Caching is two-tier:
    1. **Disk** — a pickle file (``merchant_kb.csv.pickle``) is written after
       the first build.  On subsequent runs the automaton is loaded from disk
       in seconds, provided the CSV has not been modified since.
    2. **Memory** — within the same process the automaton is reused across
       engine invocations.
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

    # Disk cache: load from pickle if newer than the source CSV.
    cache_path = resolved + ".pickle"
    try:
        kb_mtime = Path(resolved).stat().st_mtime
        if Path(cache_path).stat().st_mtime > kb_mtime:
            with open(cache_path, "rb") as fh:
                _cached_automaton = pickle.load(fh)
            _cached_kb_path = resolved
            return _cached_automaton
    except (FileNotFoundError, pickle.UnpicklingError, EOFError, OSError):
        pass  # No cache, corrupted, or inaccessible — build from scratch.

    # Build from scratch (one-off per CSV version).
    _cached_automaton = load_merchant_kb(resolved)
    _cached_kb_path = resolved

    # Persist to disk for next run.
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(_cached_automaton, fh)
    except OSError:
        pass  # Non-critical — next run will just rebuild.

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
    automaton: _Automaton,
) -> pd.DataFrame:
    """Add *counterparty*, *finv_category* and match metadata columns.

    Returns a DataFrame with the same row order as *transactions*, containing
    the original columns plus the classification columns defined by the engine
    protocol.
    """
    out = transactions.copy()
    out["_text_clean"] = out["text"].apply(_clean_transaction_text)

    def _classify_one(text_clean: str) -> tuple[bool, str, str, str, str, str]:
        """Classify a single cleaned text via purity × position scoring.

        Returns (matched, counterparty, category, keyword, rule_id, reason).
        """
        text_clean = str(text_clean)
        if not text_clean:
            return (False, "", "", "", "", "")

        text_len = max(len(text_clean), 1)
        scored: list[tuple[float, str, str, str]] = []  # (score, kw, merchant, cat)

        # pyahocorasick.iter() yields (end_pos, (kw, merchant, cat, purity)).
        for _, (kw, merchant, cat, purity) in automaton.iter(text_clean):
            if not _is_whole_word(kw, text_clean):
                continue
            pos = text_clean.find(kw)
            position_weight = 1.0 - pos / text_len
            score = purity * position_weight
            scored.append((score, kw, merchant, cat))

        if not scored:
            return (False, "", "", "", "", "")

        # Prefer categorised matches; fall back to uncategorised if none exist.
        cat_hits = [h for h in scored if h[3]]
        pick_from = cat_hits if cat_hits else scored
        best_score, best_kw, best_merchant, best_cat = max(
            pick_from, key=lambda h: h[0]
        )

        reason = format_classification_reason(
            category=best_cat,
            rule="merchant_kb_match",
            evidence=[
                f"keyword={best_kw}",
                f"merchant={best_merchant}",
                f"purity={best_score:.2f}",
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

    # apply() uses C-level iteration — much faster than iterrows().
    results = out["_text_clean"].apply(_classify_one)
    (
        out["matched"],
        out["counterparty"],
        out["finv_category"],
        out["_matched_keyword"],
        out["classification_rule_id"],
        out["classification_reason"],
    ) = zip(*results)

    # ── Post-processing: correct known KB misclassifications ────────────────
    out = _apply_kb_corrections(out)

    # ── Fallback: classify KB-matched rows that lack a category ─────────────
    out = _apply_fallback_classify(out)

    return out.drop(columns=["_text_clean"])


# ── KB correction rules ───────────────────────────────────────────────────
# High-precision fixes for systematic merchant KB misclassifications.
# Each entry: (text_pattern, wrong_category, correct_category, counterparty)

_KB_CORRECTIONS: list[tuple[str, str, str, str]] = [
    # Uber Eats contains "UBER *EATS" (food delivery) but the KB keyword
    # "UBER" matches Transport.  Correct these to Dining Out.
    (r"UBER\s*\*?\s*EATS", "Transport", "Dining Out", "Uber Eats"),
]


def _apply_kb_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Apply high-precision overrides for known KB misclassifications."""
    out = df.copy()
    raw_text = out.get("text", pd.Series("", index=out.index))

    for pattern, wrong_cat, correct_cat, correct_cp in _KB_CORRECTIONS:
        mask = (
            (out["finv_category"] == wrong_cat)
            & raw_text.str.upper().str.contains(pattern, na=False, regex=True)
        )
        if mask.any():
            out.loc[mask, "finv_category"] = correct_cat
            out.loc[mask, "counterparty"] = correct_cp
            out.loc[mask, "classification_rule_id"] = (
                "merchant_kb_match_corrected"
            )
            out.loc[mask, "classification_reason"] = out.loc[
                mask, "classification_reason"
            ].apply(
                lambda r: r.replace(
                    "rule=merchant_kb_match",
                    f"rule=merchant_kb_match_corrected({wrong_cat}→{correct_cat})",
                )
            )

    return out


# ── Fallback classification for uncategorised KB matches ──────────────────
# When the KB has a merchant keyword but the ``category`` column is empty,
# the automaton returns matched=True with category="".  The pipeline treats
# empty finv_category as unclassified, which is the #1 cause of our
# coverage gap (84% of missed records).
#
# These rules assign a category based on the matched merchant name or the
# full transaction text.  Only applied when matched=True AND category="".

# Priority-ordered regex -> category mapping.
_FALLBACK_RULES: list[tuple[str, str]] = [
    # -- Gambling (high-precision operator names) --
    (r"\bSPORTSBET\b", "Gambling"),
    (r"\bBET365\b", "Gambling"),
    (r"\bLADBROKES\b", "Gambling"),
    (r"\bBETFAIR\b", "Gambling"),
    (r"\bPOINTSBET\b", "Gambling"),
    (r"\bUNIBET\b", "Gambling"),
    (r"\bDRAFTKINGS\b", "Gambling"),
    (r"\bTAB\b", "Gambling"),
    (r"\bTATTSBET\b", "Gambling"),
    (r"\bNEDS\b", "Gambling"),
    (r"\bBLUEBET\b", "Gambling"),
    (r"\bPLAYUP\b", "Gambling"),
    (r"\bPALMERBET\b", "Gambling"),
    (r"\bBETR\b", "Gambling"),
    (r"\bBOSSBET\b", "Gambling"),
    (r"\bCLASSICBET\b", "Gambling"),
    (r"\bBETCHAIN\b", "Gambling"),
    (r"\bMADBOOKIE\b", "Gambling"),
    (r"\bTOPBETTA\b", "Gambling"),
    # -- Groceries --
    (r"\bWOOLWORTHS\b", "Groceries"),
    (r"\bCOLES\b", "Groceries"),
    (r"\bBWS\b", "Groceries"),             # Beer Wine Spirits - bottle shop
    (r"\bALDI\b", "Groceries"),
    (r"\bIGA\b", "Groceries"),
    (r"\bFOODWORKS\b", "Groceries"),
    (r"\bFOODLAND\b", "Groceries"),
    (r"\bDRAKES\b", "Groceries"),
    (r"\bRITCHIES\b", "Groceries"),
    (r"\bSUPA\s*IGA\b", "Groceries"),
    (r"\bFREE CHOICE\b", "Groceries"),     # tobacconist / convenience
    # -- Dining Out (restaurants, pubs, cafes, fast food) --
    (r"\bBAKEHOUSE\b", "Dining Out"),
    (r"\bBAKERY\b", "Dining Out"),
    (r"\bSURF CLUB\b", "Dining Out"),
    (r"\bBOWLS CLUB\b", "Dining Out"),
    (r"\bBOWLING CLUB\b", "Dining Out"),
    (r"\bLEAGUES CLUB\b", "Dining Out"),
    (r"\bRSL\b", "Dining Out"),
    (r"\bTAVERN\b", "Dining Out"),
    (r"\bBAR AND GRILL\b", "Dining Out"),
    (r"\bBRASSERIE\b", "Dining Out"),
    (r"\bBISTRO\b", "Dining Out"),
    (r"\bROADHOUSE\b", "Dining Out"),
    (r"\bLUNCH BAR\b", "Dining Out"),
    (r"\bTAKEAWAY\b", "Dining Out"),
    (r"\bPIZZA\b", "Dining Out"),
    (r"\bMCDONALD\'?S\b", "Dining Out"),
    (r"\bKFC\b", "Dining Out"),
    (r"\bSUBWAY\b", "Dining Out"),
    (r"\bHUNGRY JACKS\b", "Dining Out"),
    (r"\bDOMINO\'?S\b", "Dining Out"),
    (r"\bGRILL\'?D\b", "Dining Out"),
    (r"\bNANDO\'?S\b", "Dining Out"),
    (r"\bGUZMAN Y GOMEZ\b", "Dining Out"),
    (r"\bZAMBRERO\b", "Dining Out"),
    (r"\bSUSHI\b", "Dining Out"),
    (r"\bNOODLE BOX\b", "Dining Out"),
    # -- Transport --
    (r"\bTRANSPORTFORNSW\b", "Transport"),
    (r"\bDIDIMOBILITY\b", "Transport"),
    (r"\bDIDI\b", "Transport"),
    (r"\bOLAFARE\b", "Transport"),
    (r"\b13CABS\b", "Transport"),
    (r"\bCABCHARGE\b", "Transport"),
    (r"\bTAXI\b", "Transport"),
    (r"\bMYLI\b", "Transport"),
    (r"\bOPAL\b", "Transport"),
    (r"\bUBER(?!\s*EATS)\b", "Transport"),  # Uber ride, not Uber Eats
    # -- Subscription TV / Streaming --
    (r"\bNETFLIX\b", "Subscription TV"),
    (r"\bSTAN\b", "Subscription TV"),
    (r"\bSPOTIFY\b", "Subscription TV"),
    (r"\bDISNEY PLUS\b", "Subscription TV"),
    (r"\bDISNEYPLUS\b", "Subscription TV"),
    (r"\bAMAZON PRIME\b", "Subscription TV"),
    (r"\bAPPLE TV\b", "Subscription TV"),
    (r"\bBINGE\b", "Subscription TV"),
    (r"\bKAYO\b", "Subscription TV"),
    (r"\bFOXTEL\b", "Subscription TV"),
    (r"\bFETCH\b", "Subscription TV"),
    # -- Health --
    (r"\bPHARMACY\b", "Health"),
    (r"\bCHEMIST\b", "Health"),
    (r"\bMEDICAL CENTRE\b", "Health"),
    (r"\bMEDICAL CENTER\b", "Health"),
    (r"\bDENTAL\b", "Health"),
    (r"\bDENTIST\b", "Health"),
    (r"\bOPTICAL\b", "Health"),
    (r"\bOPTOMETRIST\b", "Health"),
    (r"\bPHYSIOTHERAPY\b", "Health"),
    (r"\bCHIROPRACTIC\b", "Health"),
    # -- Insurance --
    (r"\bINSURANCE\b", "Insurance"),
    (r"\bQBE\b", "Insurance"),
    (r"\bAAMI\b", "Insurance"),
    (r"\bALLIANZ\b", "Insurance"),
    (r"\bNRMA\b", "Insurance"),
    (r"\bBUDGET DIRECT\b", "Insurance"),
    # -- Utilities --
    (r"\bENERGY\b", "Utilities"),
    (r"\bELECTRICITY\b", "Utilities"),
    (r"\bWATER CORP\b", "Utilities"),
    (r"\bSYDNEY WATER\b", "Utilities"),
    (r"\bORIGIN ENERGY\b", "Utilities"),
    (r"\bAGL\b", "Utilities"),
    (r"\bENERGYAUSTRALIA\b", "Utilities"),
    # -- Telecommunications --
    (r"\bTELSTRA\b", "Telecommunications"),
    (r"\bOPTUS\b", "Telecommunications"),
    (r"\bVODAFONE\b", "Telecommunications"),
    (r"\bBELONG\b", "Telecommunications"),
    (r"\bAMAYSIM\b", "Telecommunications"),
    (r"\bTPG\b", "Telecommunications"),
    # -- Automotive --
    (r"\bPETROL\b", "Automotive"),
    (r"\bSERVO\b", "Automotive"),
    (r"\bBP\b", "Automotive"),
    (r"\bSHELL\b", "Automotive"),
    (r"\bCALTEX\b", "Automotive"),
    (r"\bAMPOL\b", "Automotive"),
    (r"\b7[ -]?ELEVEN\b", "Automotive"),
    (r"\bUNITED PETROLEUM\b", "Automotive"),
    (r"\bTYRE\b", "Automotive"),
    (r"\bTIRE\b", "Automotive"),
    (r"\bAUTO REPAIR\b", "Automotive"),
    (r"\bMECHANICAL\b", "Automotive"),
    # -- Retail / Department Stores --
    (r"\bKMART\b", "Department Stores"),
    (r"\bTARGET\b", "Department Stores"),
    (r"\bBIG W\b", "Department Stores"),
    (r"\bBIGW\b", "Department Stores"),
    (r"\bMYER\b", "Department Stores"),
    (r"\bBUNNINGS\b", "Department Stores"),
    # -- Donations --
    (r"\bDONATION\b", "Donations"),
    (r"\bCHARITY\b", "Donations"),
    (r"\bSALVATION ARMY\b", "Donations"),
    (r"\bRED CROSS\b", "Donations"),
    # -- Personal Care --
    (r"\bHAIRDRESSER\b", "Personal Care"),
    (r"\bBARBER\b", "Personal Care"),
    (r"\bBEAUTY\b", "Personal Care"),
    (r"\bNAIL\b", "Personal Care"),
    # -- Pet Care --
    (r"\bVETERINARY\b", "Pet Care"),
    (r"\bPETBARN\b", "Pet Care"),
    (r"\bPETSTOCK\b", "Pet Care"),
    # -- Education --
    (r"\bUNIVERSITY\b", "Education"),
    (r"\bTAFE\b", "Education"),
    (r"\bCOLLEGE\b", "Education"),
]

# Patterns that override the general "Dining Out" -> "Entertainment" mapping
# when the matched merchant is clearly a sports/entertainment venue.
_ENTERTAINMENT_REFINEMENT: list[tuple[str, str]] = [
    (r"\bRUGBY LEAGUE\b", "Entertainment"),
    (r"\bFOOTBALL CLUB\b", "Entertainment"),
    (r"\bRACECOURSE\b", "Entertainment"),
    (r"\bCASINO\b", "Entertainment"),
    (r"\bCINEMA\b", "Entertainment"),
    (r"\bHOYTS\b", "Entertainment"),
    (r"\bEVENT\b", "Entertainment"),
    (r"\bTICKET\b", "Entertainment"),
]


def _apply_fallback_classify(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a category to KB-matched rows whose category column was empty.

    Only touches rows where ``matched == True`` and ``finv_category == ""``.
    Checks the matched merchant name first, then the full transaction text.
    """
    out = df.copy()

    fallback_mask = (
        out["matched"].eq(True)
        & (out["finv_category"].eq("") | out["finv_category"].isna())
    )
    if not fallback_mask.any():
        return out

    raw_text = out.get("text", pd.Series("", index=out.index))

    for idx in out[fallback_mask].index:
        text = str(raw_text.get(idx, "")).upper()
        counterparty = str(out.at[idx, "counterparty"] or "").upper()
        category = ""

        # Step 1: check matched counterparty name
        for pattern, cat in _FALLBACK_RULES:
            if re.search(pattern, counterparty):
                category = cat
                break

        # Step 2: fall back to full transaction text
        if not category:
            for pattern, cat in _FALLBACK_RULES:
                if re.search(pattern, text):
                    category = cat
                    break

        if not category:
            continue  # still cannot classify - leave as-is

        # Refine: Dining Out -> Entertainment for sports/cinema venues
        if category == "Dining Out":
            for pattern, cat in _ENTERTAINMENT_REFINEMENT:
                if re.search(pattern, text) or re.search(pattern, counterparty):
                    category = cat
                    break

        out.at[idx, "finv_category"] = category
        out.at[idx, "classification_rule_id"] = "merchant_kb_fallback"
        out.at[idx, "classification_reason"] = format_classification_reason(
            category=category,
            rule="merchant_kb_fallback",
            evidence=[
                f"counterparty={out.at[idx, 'counterparty']}",
            ],
        )

    return out
