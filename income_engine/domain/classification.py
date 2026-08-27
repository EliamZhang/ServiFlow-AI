# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Rule-based income classification pipeline for bank transactions.

Main output fields:
- finv_category: production income category used by downstream FInv output.
- is_wages_pred: whether the transaction is predicted as wages / salary-like income.
- wages_rule_name: the wage detection rule that matched.
- wages_pred_reason: readable reason for wage detection.
- income_type_pred: income classification result.
    Values:
        salary_payg
        centrelink
        self_employed_gig
        salary_packaging
        non_income
- known_non_income_type_pred: optional subtype for recognized non-income credits.
    Values:
        wage_advance
        blank for other rows
- income_type_rule_name: the income classification rule that matched.
- income_type_pred_reason: readable reason for income classification.

Important restrictions:
- Do NOT use third_party as an input feature.
- Do NOT use trx_type / txn_type / txn_type_category as input features.
- Do NOT use category as decision logic. It is only used for optional validation.
- Do NOT use a numeric score. Final result is based on yes/no business rules.

This module is invoked by the unified engine pipeline.
"""

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from classification_core.reasons import format_classification_reason
from classification_core.text import clean_text_with_seams


# =============================================================================
# Paths
# =============================================================================

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
_PATTERN_RULES_FILE = _RESOURCES_DIR / "income_pattern_rules.csv"
_CONFIG_FILE = _RESOURCES_DIR / "income_config.csv"


# =============================================================================
# CSV rule loading (data-driven pattern definitions)
# =============================================================================

def _load_pattern_rules(rules_file: str | Path | None = None) -> Dict[str, List[str]]:
    """Load income pattern rules from CSV, grouped by pattern_group.

    Returns a dict mapping pattern_group name → list of regex pattern strings.
    """
    if rules_file is None:
        rules_file = _PATTERN_RULES_FILE

    grouped: Dict[str, List[str]] = {}
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            group = str(row.get("pattern_group", "")).strip()
            pattern = str(row.get("pattern", "")).strip()
            if not group or not pattern:
                continue
            grouped.setdefault(group, []).append(pattern)
    return grouped


def _load_income_config(config_file: str | Path | None = None) -> Dict[str, float]:
    """Load income classification config values from CSV.

    Returns a dict mapping config_key → numeric value.
    """
    if config_file is None:
        config_file = _CONFIG_FILE

    config: Dict[str, float] = {}
    with open(config_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = str(row.get("config_key", "")).strip()
            value = str(row.get("config_value", "")).strip()
            config_type = str(row.get("config_type", "float")).strip()
            if not key or not value:
                continue
            config[key] = float(value) if config_type == "float" else int(value)
    return config


# Lazy-loaded caches (populated on first use, following liability engine pattern).
_PATTERN_CACHE: Dict[str, List[str]] | None = None
_CONFIG_CACHE: Dict[str, float] | None = None


def _get_patterns() -> Dict[str, List[str]]:
    global _PATTERN_CACHE
    if _PATTERN_CACHE is None:
        _PATTERN_CACHE = _load_pattern_rules()
    return _PATTERN_CACHE


def _get_config() -> Dict[str, float]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _load_income_config()
    return _CONFIG_CACHE


# =============================================================================
# Config — columns and pattern-driven constants
# =============================================================================

# These columns are removed before rule execution to avoid accidental leakage.
RESTRICTED_COLUMNS = ["trx_type", "txn_type", "txn_type_category", "third_party"]


def _make_pattern_list(group: str) -> List[str]:
    """Return pattern strings for *group* from the CSV-backed cache."""
    return _get_patterns().get(group, [])

# Resolve pattern lists (CSV-first, fallback to built-ins).
STRONG_WAGE_PATTERNS = _make_pattern_list("strong_wage")
MEDIUM_INCOME_PATTERNS = _make_pattern_list("medium_income")
REPEAT_EMPLOYER_LIKE_PATTERNS = _make_pattern_list("repeat_employer_like")
REPEAT_EMPLOYER_LIKE_EXCLUSION_PATTERNS = _make_pattern_list("repeat_employer_like_exclusion")
SALARY_PACKAGING_PATTERNS = _make_pattern_list("salary_packaging")
CENTRELINK_PATTERNS = _make_pattern_list("centrelink")
SELF_EMPLOYED_GIG_PATTERNS = _make_pattern_list("self_employed_gig")
WAGE_ADVANCE_PATTERNS = _make_pattern_list("wage_advance")
RETURN_LIKE_PATTERNS = _make_pattern_list("return_like")

# Hard negative = return_like + additional hard_negative patterns
HARD_NEGATIVE_PATTERNS = RETURN_LIKE_PATTERNS + _make_pattern_list("hard_negative")

# TRANSFER FROM texts are flagged separately so the hard gate can waive the
# hard negative when a strong wage keyword is also present: fast-transfer
# payroll texts like "FAST TRANSFER FROM X WAGES" are real wages and must
# flow to rule_transfer_strong_wage_keyword, while bare "TRANSFER FROM X"
# (no wage signal) stays blocked.  NOTE: TRANSFER FROM is no longer part of
# the hard_negative group itself (removed from the CSV): the transfer_from
# group is the single source of that signal, and the high-repeat behavior
# rules gate on it directly (TF texts only flow via the PAY-signal rule).
TRANSFER_FROM_PATTERNS = _make_pattern_list("transfer_from")

# Standalone PAY word: weak wage signal used by rule_transfer_from_pay_signal
# ("FAST TRANSFER FROM X PAY" texts), mirroring rule_transfer_strong_wage_keyword
# but with a standalone PAY instead of a strong keyword.
PAY_SIGNAL_PATTERNS = _make_pattern_list("pay_signal")

# Exclusions for the high-repeat behavior rules only.  Kept separate from the
# hard_negative group so the existing strong/medium keyword rules are
# unaffected; these patterns carry clear non-wage semantics (self-transfers,
# refunds, rent, gambling, invoices, cash deposits, ...) that the behavior
# rules must not claim.
BEHAVIOR_EXCLUSION_PATTERNS = _make_pattern_list("behavior_exclusion")

# Soft negative patterns for wage detection.
SOFT_NEGATIVE_PATTERNS = _make_pattern_list("soft_negative")

# Combined negative view kept for explainability output.
NEGATIVE_PATTERNS = HARD_NEGATIVE_PATTERNS + SOFT_NEGATIVE_PATTERNS

# Extra exclusions used only for self-employed / gig classification.
# Loaded from the CSV-backed ``gig_exclusion_extra`` group (was hardcoded
# before — CSV edits to that group had no effect).
GIG_EXCLUSION_EXTRA_PATTERNS = _make_pattern_list("gig_exclusion_extra")
# SOFT_NEGATIVE (OSKO/TRANSFER) deliberately excluded from the gig gate:
# gig classification already requires a platform/merchant keyword plus the
# hard-negative gate, so the transfer-suspicion signal caused a structural
# deadlock ("Osko + UBER" texts could never classify as gig income).
GIG_EXCLUSION_PATTERNS = (
    HARD_NEGATIVE_PATTERNS + GIG_EXCLUSION_EXTRA_PATTERNS
)

# Gig personal-transfer gates (CSV-backed, loaded by group).
# - gig_family_exclusion: family/living-expense words.  The gate fires only
#   when the payer key consists *entirely* of family words (e.g. "Transfer
#   from Mum and Dad", "Internet Deposit Food" — the payer is a family
#   member / living-expense transfer, not an employer).  Names that merely
#   contain a family word ("NATALIE VAN SCHOOR Mum", "MRS PAULA MINUTOLI
#   Mum") are real people and must NOT be blocked — checked on
#   payer_key_from_text only, not the raw memo text (memo fragments like
#   "food"/"fuel" would otherwise block every named payer).
# - gig_personal_transfer: personal-transfer semantics (Osko/PayID/...).
#   The gig keyword rule defers such texts to rule_income_self_employed_gig_repeat_payer
#   (behavior-based); platform settlements (UBERBV/SQUARE/PAYPAL) carry no
#   transfer word and stay with the keyword rule.
# - gig_personal_exclusion: gambling payouts, self-transfers, acquirer
#   gateways, school fees, car loans, friend-transfer markers, ...
GIG_FAMILY_EXCLUSION_PATTERNS = _make_pattern_list("gig_family_exclusion")
GIG_PERSONAL_TRANSFER_PATTERNS = _make_pattern_list("gig_personal_transfer")
GIG_PERSONAL_EXCLUSION_PATTERNS = _make_pattern_list("gig_personal_exclusion")
# Strong collection keyword: "PAYMENT FROM X ... INVOICE" is invoice income
# (already claimed in production) and is exempt from the personal-transfer
# deferral above.
GIG_INVOICE_PATTERNS = [r"\bINVOICE\b"]
# PSP reference number: standard Osko/NPP personal-transfer reference
# ("PAYMENT FROM X, PSP123456789").  behaviour_exclusion's \bPSP\d+\b was
# intended for PSP financial products but also blocks every Osko personal
# transfer (hundreds of high-frequency payers in sample data).  The gig
# repeat-payer rule waives the behaviour-exclusion gate for PSP references —
# all other gates (repeat count, one-way, family/exclusion words) still apply.
GIG_PSP_REFERENCE_PATTERNS = [r"\bPSP\d+\b"]

PAYER_STOP_WORDS = {
    "DIRECT", "CREDIT", "DIR", "DEPOSIT", "SALARY", "PAYROLL", "WAGE", "WAGES",
    "PAY", "PAYMENT", "PAYMENTS", "TRANSFER", "TRANS", "FROM", "TO", "REF",
    "REFERENCE", "ONLINE", "INTERNET", "EFT", "DEP", "OSKO", "VISA", "CARD",
    "PURCHASE", "DEBIT", "MISCELLANEOUS", "BPAY", "WITHDRAWAL", "ATM",
    "TRNS", "ACC", "ACCOUNT", "LINKED", "AU", "AUS", "THE", "AND", "PTY",
    "LTD", "LIMITED", "PACKAGING", "CENTRELINK", "CENTRE", "LINK",
    "SERVICES", "AUSTRALIA", "GOV", "GOVERNMENT", "RETURN", "VALUE", "DATE",
    "FAST", "NPP", "THANK", "YOU", "RECEIVED", "MAIN", "JAN", "FEB", "MAR",
    "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}

# Config values — loaded from CSV, with built-in defaults as fallback.
_config = _get_config()
MIN_NORMAL_WAGE_AMOUNT = int(_config.get("MIN_NORMAL_WAGE_AMOUNT", 100))
SMALL_WAGE_AMOUNT_MIN = int(_config.get("SMALL_WAGE_AMOUNT_MIN", 50))
COMMON_WAGE_AMOUNT_MIN = int(_config.get("COMMON_WAGE_AMOUNT_MIN", 300))
COMMON_WAGE_AMOUNT_MAX = int(_config.get("COMMON_WAGE_AMOUNT_MAX", 10000))
POSSIBLE_WAGE_AMOUNT_MAX = int(_config.get("POSSIBLE_WAGE_AMOUNT_MAX", 20000))
STABLE_AMOUNT_CV_MAX = float(_config.get("STABLE_AMOUNT_CV_MAX", 0.35))
REGULAR_GAP_RANGES = (
    (int(_config.get("REGULAR_GAP_WEEKLY_MIN", 6)), int(_config.get("REGULAR_GAP_WEEKLY_MAX", 8))),
    (int(_config.get("REGULAR_GAP_FORTNIGHTLY_MIN", 13)), int(_config.get("REGULAR_GAP_FORTNIGHTLY_MAX", 16))),
    (int(_config.get("REGULAR_GAP_MONTHLY_MIN", 27)), int(_config.get("REGULAR_GAP_MONTHLY_MAX", 33))),
)
HIGH_REPEAT_PAYER_COUNT_MIN = int(_config.get("HIGH_REPEAT_PAYER_COUNT_MIN", 4))
VERY_HIGH_REPEAT_PAYER_COUNT_MIN = int(_config.get("VERY_HIGH_REPEAT_PAYER_COUNT_MIN", 8))
STABLE_PAYER_REPEAT_COUNT_MIN = int(_config.get("STABLE_PAYER_REPEAT_COUNT_MIN", 6))
# Gig repeat-payer rule thresholds (user-confirmed: amount floor 20, repeat count 4)
GIG_REPEAT_PAYER_AMOUNT_MIN = int(_config.get("GIG_REPEAT_PAYER_AMOUNT_MIN", 20))
GIG_REPEAT_PAYER_COUNT_MIN = int(_config.get("GIG_REPEAT_PAYER_COUNT_MIN", 4))

IMPORTANT_OUTPUT_COLUMNS = [
    # New income classification fields
    "finv_category",
    "counterparty",
    "stream_id",
    "income_type_pred",
    "known_non_income_type_pred",
    "known_non_income_rule_name",
    "income_type_rule_name",
    "income_type_pred_reason",
    "is_income_pred",

    # Existing wages fields
    "is_wages_pred",
    "wages_rule_name",
    "wages_pred_reason",
    "base_wages_pred",
    "small_amount_wage_history_override",
    "prior_detected_wages_same_payer_count",
    "rule_strong_wage_keyword",
    "rule_transfer_strong_wage_keyword",
    "rule_transfer_with_wage_signal",
    "rule_repeat_employer_like_payment",
    "rule_soft_negative_alias_to_known_wage_payer",
    "rule_small_amount_alias_to_known_wage_payer",
    "rule_direct_credit_with_repeat",
    "rule_medium_income_high_repeat",
    "rule_stable_payer_without_keywords",
    "rule_small_amount_medium_income_high_repeat",
    "rule_small_amount_same_known_wage_payer",
    "rule_recurring_payer_behavior",
    "rule_transfer_from_pay_signal",
    "rule_high_repeat_no_keyword",
    "rule_high_repeat_soft_negative",
    "hard_exclusion",
    "effective_hard_negative",

    # Income rule flags
    "rule_income_salary_packaging",
    "rule_income_centrelink",
    "rule_income_salary_payg",
    "rule_income_self_employed_gig",
    "rule_income_self_employed_gig_repeat_payer",
    "rule_known_non_income_wage_advance",

    # Explainable features
    "text_clean",
    "payer_key_from_text",
    "matched_known_wage_payer_key",
    "known_wage_payer_token_overlap",
    "is_credit",
    "has_strong_wage_keyword",
    "has_medium_income_keyword",
    "has_salary_packaging_keyword",
    "has_centrelink_keyword",
    "has_self_employed_gig_keyword",
    "has_wage_advance_keyword",
    "has_return_like_keyword",
    "has_hard_negative_keyword",
    "has_soft_negative_keyword",
    "has_transfer_from_keyword",
    "has_gig_exclusion_keyword",
    "has_gig_family_exclusion_keyword",
    "has_gig_personal_transfer_pattern",
    "has_gig_personal_exclusion_keyword",
    "has_gig_invoice_keyword",
    "has_gig_psp_reference",
    "has_negative_keyword",
    "has_pay_signal",
    "has_behavior_exclusion",
    "has_payer_debit",
    "is_common_wage_amount",
    "is_possible_wage_amount",
    "same_payer_credit_count",
    "days_since_prev_same_payer",
    "days_to_next_same_payer",
    "has_regular_salary_cycle",
    "has_stable_amount",
    "same_payer_amount_cv",
]


# =============================================================================
# Basic helpers
# =============================================================================

def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(pattern) for pattern in patterns]


STRONG_WAGE_REGEX = compile_patterns(STRONG_WAGE_PATTERNS)
MEDIUM_INCOME_REGEX = compile_patterns(MEDIUM_INCOME_PATTERNS)
REPEAT_EMPLOYER_LIKE_REGEX = compile_patterns(REPEAT_EMPLOYER_LIKE_PATTERNS)
REPEAT_EMPLOYER_LIKE_EXCLUSION_REGEX = compile_patterns(REPEAT_EMPLOYER_LIKE_EXCLUSION_PATTERNS)
SALARY_PACKAGING_REGEX = compile_patterns(SALARY_PACKAGING_PATTERNS)
CENTRELINK_REGEX = compile_patterns(CENTRELINK_PATTERNS)
SELF_EMPLOYED_GIG_REGEX = compile_patterns(SELF_EMPLOYED_GIG_PATTERNS)
WAGE_ADVANCE_REGEX = compile_patterns(WAGE_ADVANCE_PATTERNS)
RETURN_LIKE_REGEX = compile_patterns(RETURN_LIKE_PATTERNS)
HARD_NEGATIVE_REGEX = compile_patterns(HARD_NEGATIVE_PATTERNS)
SOFT_NEGATIVE_REGEX = compile_patterns(SOFT_NEGATIVE_PATTERNS)
NEGATIVE_REGEX = compile_patterns(NEGATIVE_PATTERNS)
GIG_EXCLUSION_REGEX = compile_patterns(GIG_EXCLUSION_PATTERNS)
GIG_FAMILY_EXCLUSION_REGEX = compile_patterns(GIG_FAMILY_EXCLUSION_PATTERNS)
GIG_PERSONAL_TRANSFER_REGEX = compile_patterns(GIG_PERSONAL_TRANSFER_PATTERNS)
GIG_PERSONAL_EXCLUSION_REGEX = compile_patterns(GIG_PERSONAL_EXCLUSION_PATTERNS)
GIG_INVOICE_REGEX = compile_patterns(GIG_INVOICE_PATTERNS)
GIG_PSP_REFERENCE_REGEX = compile_patterns(GIG_PSP_REFERENCE_PATTERNS)
TRANSFER_FROM_REGEX = compile_patterns(TRANSFER_FROM_PATTERNS)
PAY_SIGNAL_REGEX = compile_patterns(PAY_SIGNAL_PATTERNS)
BEHAVIOR_EXCLUSION_REGEX = compile_patterns(BEHAVIOR_EXCLUSION_PATTERNS)


def count_matches(text: str, patterns: List[re.Pattern]) -> int:
    if not text:
        return 0
    return sum(1 for pattern in patterns if pattern.search(text))


def _payer_key_is_family_only(payer_key: str) -> int:
    """Family gate for the gig rules: 1 iff the payer key is *entirely* family
    words ("MUM DAD", "FOOD").  Keys that also contain a name
    ("NATALIE VAN SCHOOR MUM") are real people and return 0."""
    if not payer_key:
        return 0
    tokens = payer_key.split()
    family = {t for t in tokens if any(p.match(t) for p in GIG_FAMILY_EXCLUSION_REGEX)}
    if not family:
        return 0
    return int(len(family) == len(tokens))


def make_payer_key(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", text)
    text = re.sub(r"\b\d{5,}\b", " ", text)

    tokens = re.findall(r"[A-Z]{3,}", text)
    tokens = [token for token in tokens if token not in PAYER_STOP_WORDS]
    if not tokens:
        return ""
    return " ".join(tokens[:4])


def payer_key_tokens(value: str) -> set[str]:
    if not value:
        return set()
    return {token for token in str(value).split() if len(token) >= 3}


def is_regular_gap(days) -> int:
    if pd.isna(days):
        return 0
    return int(any(start <= days <= end for start, end in REGULAR_GAP_RANGES))


def make_group_key(df: pd.DataFrame) -> pd.Series:
    valid = (df["is_credit"] == 1) & (df["has_valid_payer_key"] == 1)
    key = df["bank_account_id"].astype(str) + "||" + df["payer_key_from_text"].astype(str)
    return key.where(valid, np.nan)


# =============================================================================
# Input preparation
# =============================================================================

def merge_unnamed_columns_into_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    unnamed_cols = [col for col in out.columns if str(col).startswith("Unnamed")]
    if not unnamed_cols:
        return out

    if "text" not in out.columns:
        out["text"] = ""
    out["text"] = out["text"].fillna("").astype(str)

    extra_text = out[unnamed_cols].apply(
        lambda row: " ".join(
            str(value).strip()
            for value in row
            if not pd.isna(value) and str(value).strip()
        ),
        axis=1,
    )
    has_extra = extra_text.str.len() > 0
    out.loc[has_extra, "text"] = (
        out.loc[has_extra, "text"].str.rstrip() + " " + extra_text.loc[has_extra]
    ).str.strip()

    return out.drop(columns=unnamed_cols)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    out = merge_unnamed_columns_into_text(df)
    out = out.drop(columns=RESTRICTED_COLUMNS, errors="ignore")

    missing = [col for col in ["amount", "transaction_date"] if col not in out.columns]
    if missing:
        raise ValueError(f"Input must contain required column(s): {', '.join(missing)}")

    defaults = {
        "text": "",
        "dr_cr": "",
        "bank_account_id": "UNKNOWN_ACCOUNT",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    return out


# =============================================================================
# Feature engineering
# =============================================================================

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["txn_date"] = pd.to_datetime(out["transaction_date"], errors="coerce", dayfirst=False)
    out["amount_num"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0)
    out["text_clean"] = out["text"].apply(clean_text_with_seams)
    out["is_credit"] = out["dr_cr"].astype(str).str.lower().str.strip().eq("credit").astype(int)

    for count_col, flag_col, patterns in [
        ("strong_wage_keyword_count", "has_strong_wage_keyword", STRONG_WAGE_REGEX),
        ("medium_income_keyword_count", "has_medium_income_keyword", MEDIUM_INCOME_REGEX),
        ("repeat_employer_like_keyword_count", "has_repeat_employer_like_keyword", REPEAT_EMPLOYER_LIKE_REGEX),
        (
            "repeat_employer_like_exclusion_keyword_count",
            "has_repeat_employer_like_exclusion_keyword",
            REPEAT_EMPLOYER_LIKE_EXCLUSION_REGEX,
        ),
        ("salary_packaging_keyword_count", "has_salary_packaging_keyword", SALARY_PACKAGING_REGEX),
        ("centrelink_keyword_count", "has_centrelink_keyword", CENTRELINK_REGEX),
        ("self_employed_gig_keyword_count", "has_self_employed_gig_keyword", SELF_EMPLOYED_GIG_REGEX),
        ("wage_advance_keyword_count", "has_wage_advance_keyword", WAGE_ADVANCE_REGEX),
        ("return_like_keyword_count", "has_return_like_keyword", RETURN_LIKE_REGEX),
        ("hard_negative_keyword_count", "has_hard_negative_keyword", HARD_NEGATIVE_REGEX),
        ("soft_negative_keyword_count", "has_soft_negative_keyword", SOFT_NEGATIVE_REGEX),
        ("negative_keyword_count", "has_negative_keyword", NEGATIVE_REGEX),
        ("gig_exclusion_keyword_count", "has_gig_exclusion_keyword", GIG_EXCLUSION_REGEX),
        ("transfer_from_keyword_count", "has_transfer_from_keyword", TRANSFER_FROM_REGEX),
        ("pay_signal_keyword_count", "has_pay_signal", PAY_SIGNAL_REGEX),
        ("behavior_exclusion_keyword_count", "has_behavior_exclusion", BEHAVIOR_EXCLUSION_REGEX),
        ("gig_personal_transfer_pattern_count", "has_gig_personal_transfer_pattern", GIG_PERSONAL_TRANSFER_REGEX),
        ("gig_personal_exclusion_keyword_count", "has_gig_personal_exclusion_keyword", GIG_PERSONAL_EXCLUSION_REGEX),
        ("gig_invoice_keyword_count", "has_gig_invoice_keyword", GIG_INVOICE_REGEX),
        ("gig_psp_reference_count", "has_gig_psp_reference", GIG_PSP_REFERENCE_REGEX),
    ]:
        out[count_col] = out["text_clean"].apply(lambda x: count_matches(x, patterns))
        out[flag_col] = (out[count_col] > 0).astype(int)

    out["is_common_wage_amount"] = out["amount_num"].between(
        COMMON_WAGE_AMOUNT_MIN, COMMON_WAGE_AMOUNT_MAX, inclusive="both"
    ).astype(int)
    out["is_possible_wage_amount"] = out["amount_num"].between(
        MIN_NORMAL_WAGE_AMOUNT, POSSIBLE_WAGE_AMOUNT_MAX, inclusive="both"
    ).astype(int)
    out["payer_key_from_text"] = out["text_clean"].apply(make_payer_key)
    out["has_valid_payer_key"] = (out["payer_key_from_text"].str.len() >= 3).astype(int)
    # Family gate: fires only when the payer key is *entirely* family words
    # ("Transfer from Mum and Dad" -> key "MUM DAD").  A named payer whose
    # name happens to contain a family word ("NATALIE VAN SCHOOR Mum") has
    # non-family tokens in the key and is a real person — not blocked.
    out["has_gig_family_exclusion_keyword"] = out["payer_key_from_text"].apply(
        lambda x: _payer_key_is_family_only(x)
    ).astype(int)
    return out


def add_payer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_group_key"] = make_group_key(out)

    # Two-way payer flag: the same payer key also appears on a debit of the
    # same account (money flowed back to that payer).  Personal transfers
    # between people often go both ways; employers virtually never receive
    # money back from the employee's transaction account.  Used by the
    # high-repeat behavior rules to drop person-to-person transfers.
    out["_payer_debit_key"] = np.where(
        (out["is_credit"] == 0)
        & out["payer_key_from_text"].notna()
        & (out["payer_key_from_text"].str.len() > 0),
        out["bank_account_id"].astype(str) + "||" + out["payer_key_from_text"].astype(str),
        np.nan,
    )
    debit_keys = set(out["_payer_debit_key"].dropna().unique())
    out["has_payer_debit"] = (
        out["bank_account_id"].astype(str) + "||" + out["payer_key_from_text"].astype(str)
    ).isin(debit_keys).astype(int)
    out = out.drop(columns="_payer_debit_key")

    # Keep original script behavior: sort by payer group and transaction date.
    out = out.sort_values(["_group_key", "txn_date"], na_position="last").copy()
    group = out.groupby("_group_key", dropna=True)

    out["prev_same_payer_date"] = group["txn_date"].shift(1)
    out["next_same_payer_date"] = group["txn_date"].shift(-1)
    out["days_since_prev_same_payer"] = (out["txn_date"] - out["prev_same_payer_date"]).dt.days
    out["days_to_next_same_payer"] = (out["next_same_payer_date"] - out["txn_date"]).dt.days

    out["regular_gap_prev"] = out["days_since_prev_same_payer"].apply(is_regular_gap)
    out["regular_gap_next"] = out["days_to_next_same_payer"].apply(is_regular_gap)
    out["has_regular_salary_cycle"] = (
        (out["regular_gap_prev"] == 1) | (out["regular_gap_next"] == 1)
    ).astype(int)

    count_col = "transaction_id" if "transaction_id" in out.columns else "amount_num"
    out["same_payer_credit_count"] = group[count_col].transform("count")
    out["same_payer_amount_mean"] = group["amount_num"].transform("mean")
    out["same_payer_amount_std"] = group["amount_num"].transform("std").fillna(0)
    out["same_payer_amount_cv"] = np.where(
        out["same_payer_amount_mean"].abs() > 0,
        out["same_payer_amount_std"] / out["same_payer_amount_mean"].abs(),
        np.nan,
    )
    out["has_stable_amount"] = (
        (out["same_payer_credit_count"].fillna(0) >= 2)
        & (out["same_payer_amount_cv"].fillna(999) <= STABLE_AMOUNT_CV_MAX)
    ).astype(int)

    return out.drop(columns=[
        "_group_key", "prev_same_payer_date", "next_same_payer_date",
        "regular_gap_prev", "regular_gap_next",
        "same_payer_amount_mean", "same_payer_amount_std",
    ])


def add_wages_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_payer_history_features(add_basic_features(df))


# =============================================================================
# Wage rule decision
# =============================================================================

def add_hard_gate_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rule_pass_credit"] = (out["is_credit"] == 1).astype(int)
    out["rule_pass_amount"] = (out["amount_num"] >= MIN_NORMAL_WAGE_AMOUNT).astype(int)
    # The TRANSFER FROM hard negative is waived when the text also carries a
    # strong wage keyword: fast-transfer payroll texts ("FAST TRANSFER FROM
    # X WAGES") are claimed by rule_transfer_strong_wage_keyword, while bare
    # "TRANSFER FROM X" texts with no wage signal stay hard-blocked.
    out["effective_hard_negative"] = (
        (out["has_hard_negative_keyword"] == 1)
        & ~(
            (out["has_transfer_from_keyword"] == 1)
            & (out["has_strong_wage_keyword"] == 1)
        )
    ).astype(int)
    out["rule_pass_no_hard_negative_keyword"] = (out["effective_hard_negative"] == 0).astype(int)
    out["rule_pass_possible_wage_amount"] = (out["is_possible_wage_amount"] == 1).astype(int)

    gate_cols = [
        "rule_pass_credit",
        "rule_pass_amount",
        "rule_pass_no_hard_negative_keyword",
        "rule_pass_possible_wage_amount",
    ]
    out["hard_exclusion"] = (out[gate_cols].min(axis=1) == 0).astype(int)
    return out


def add_base_wage_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eligible = out["hard_exclusion"] == 0
    repeated = out["same_payer_credit_count"].fillna(0) >= 2
    high_repeat = out["same_payer_credit_count"].fillna(0) >= HIGH_REPEAT_PAYER_COUNT_MIN
    stable_repeat = out["same_payer_credit_count"].fillna(0) >= STABLE_PAYER_REPEAT_COUNT_MIN
    no_soft_negative = out["has_soft_negative_keyword"] == 0
    no_wage_advance = out["has_wage_advance_keyword"] == 0

    out["rule_strong_wage_keyword"] = (
        eligible
        & no_soft_negative
        & no_wage_advance
        & (out["has_strong_wage_keyword"] == 1)
    ).astype(int)

    # Strong wage keyword + small amount in [SMALL_WAGE_AMOUNT_MIN, 100).
    # The standard amount gate (MIN_NORMAL_WAGE_AMOUNT = 100) blocks small but
    # genuine wage deposits (e.g. HUMANFORCE part-time pay at <$100); a strong
    # keyword with no soft-negative / wage-advance signal is sufficient alone
    # for this band.  Same guard rails as rule_strong_wage_keyword.
    out["rule_strong_wage_keyword_small_amount"] = (
        (out["is_credit"] == 1)
        & (out["amount_num"] >= SMALL_WAGE_AMOUNT_MIN)
        & (out["amount_num"] < MIN_NORMAL_WAGE_AMOUNT)
        & (out["has_strong_wage_keyword"] == 1)
        & (out["effective_hard_negative"] == 0)
        & no_soft_negative
        & no_wage_advance
    ).astype(int)

    out["rule_transfer_strong_wage_keyword"] = (
        eligible
        & (out["has_soft_negative_keyword"] == 1)
        & no_wage_advance
        & (out["has_strong_wage_keyword"] == 1)
        & (out["has_valid_payer_key"] == 1)
        & (out["is_possible_wage_amount"] == 1)
    ).astype(int)

    # TRANSFER FROM + strong keyword where the payer key cannot be extracted:
    # "TRANSFER FROM PAYROLL SALARY" has every token in PAYER_STOP_WORDS, so
    # has_valid_payer_key=0 and rule_transfer_strong_wage_keyword never fires.
    # A strong keyword + TRANSFER FROM is already the exact pair the hard-gate
    # waiver trusts, so a missing payer key alone should not block it.
    out["rule_transfer_strong_wage_keyword_no_payer_key"] = (
        eligible
        & (out["has_soft_negative_keyword"] == 1)
        & no_wage_advance
        & (out["has_strong_wage_keyword"] == 1)
        & (out["has_transfer_from_keyword"] == 1)
        & (out["has_valid_payer_key"] == 0)
        & (out["is_possible_wage_amount"] == 1)
    ).astype(int)

    out["rule_transfer_with_wage_signal"] = (
        eligible
        & (out["has_soft_negative_keyword"] == 1)
        & no_wage_advance
        & (out["has_strong_wage_keyword"] == 1)
        & (out["has_valid_payer_key"] == 1)
        & repeated
        & ((out["has_regular_salary_cycle"] == 1) | (out["has_stable_amount"] == 1))
    ).astype(int)

    out["rule_repeat_employer_like_payment"] = (
        eligible
        & no_soft_negative
        & no_wage_advance
        & (out["has_valid_payer_key"] == 1)
        & repeated
        & (out["has_stable_amount"] == 1)
        & (out["is_common_wage_amount"] == 1)
        & (out["has_repeat_employer_like_keyword"] == 1)
        & (out["has_repeat_employer_like_exclusion_keyword"] == 0)
    ).astype(int)

    out["rule_direct_credit_with_repeat"] = (
        eligible
        & no_soft_negative
        & no_wage_advance
        & (out["has_medium_income_keyword"] == 1)
        & (out["has_valid_payer_key"] == 1)
        & repeated
        & ((out["has_regular_salary_cycle"] == 1) | (out["has_stable_amount"] == 1))
    ).astype(int)

    out["rule_medium_income_high_repeat"] = (
        eligible
        & no_soft_negative
        & no_wage_advance
        & (out["has_medium_income_keyword"] == 1)
        & (out["has_valid_payer_key"] == 1)
        & high_repeat
        & (out["is_possible_wage_amount"] == 1)
    ).astype(int)

    out["rule_stable_payer_without_keywords"] = (
        eligible
        & no_soft_negative
        & no_wage_advance
        & (out["has_valid_payer_key"] == 1)
        & stable_repeat
        & (out["has_stable_amount"] == 1)
        & (out["is_common_wage_amount"] == 1)
    ).astype(int)

    out["rule_recurring_payer_behavior"] = (
        eligible
        & no_soft_negative
        & no_wage_advance
        & (out["has_valid_payer_key"] == 1)
        & (out["is_common_wage_amount"] == 1)
        & repeated
        & (out["has_regular_salary_cycle"] == 1)
        & (out["has_stable_amount"] == 1)
    ).astype(int)

    rule_cols = [
        "rule_strong_wage_keyword",
        "rule_strong_wage_keyword_small_amount",
        "rule_transfer_strong_wage_keyword",
        "rule_transfer_strong_wage_keyword_no_payer_key",
        "rule_transfer_with_wage_signal",
        "rule_repeat_employer_like_payment",
        "rule_direct_credit_with_repeat",
        "rule_medium_income_high_repeat",
        "rule_stable_payer_without_keywords",
        "rule_recurring_payer_behavior",
    ]
    out["base_wages_pred"] = (out[rule_cols].max(axis=1) == 1).astype(int)
    return out


def add_high_repeat_behavior_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Rescue no-keyword payroll texts with strong behavioral evidence.

    Closes two gaps found by the Wages miss analysis:

    1. rule_transfer_from_pay_signal: "FAST TRANSFER FROM X PAY" texts were
       hard-blocked by the TRANSFER FROM hard negative (the waiver only
       covered strong keywords).  A standalone PAY word now waives it, mirroring
       rule_transfer_strong_wage_keyword with PAY in place of a strong keyword.
    2. rule_high_repeat_no_keyword / rule_high_repeat_soft_negative: texts with
       no strong/medium keyword at all (INTER BANK CREDIT <name>, DEPOSIT OSKO
       PAYMENT <name>, PAYMENT FROM <name>, ...) can still be wages when the
       payer repeats >= STABLE_PAYER_REPEAT_COUNT_MIN times with a regular
       salary cycle or stable amount and a common wage amount.

    Guard rails:
    - TRANSFER FROM texts flow ONLY through rule_transfer_from_pay_signal
      (they must carry the PAY signal); the behavior rules require no TF so
      bare self-transfer texts stay blocked.
    - has_behavior_exclusion / has_repeat_employer_like_exclusion_keyword /
      has_payer_debit drop self-transfers, refunds, rent, gambling, invoices
      and person-to-person transfers with clear non-wage semantics.
    - Deliberately NOT fed into base_wages_pred: these rows must not amplify
      the soft-negative alias / small-amount chains.
    """
    out = df.copy()
    eligible = out["hard_exclusion"] == 0
    transfer_from = out["has_transfer_from_keyword"] == 1
    no_wage_advance = out["has_wage_advance_keyword"] == 0
    no_keyword = (
        (out["has_strong_wage_keyword"] == 0) & (out["has_medium_income_keyword"] == 0)
    )
    high_repeat = out["same_payer_credit_count"].fillna(0) >= STABLE_PAYER_REPEAT_COUNT_MIN
    cycle_or_stable = (
        (out["has_regular_salary_cycle"] == 1) | (out["has_stable_amount"] == 1)
    )
    valid_payer = out["has_valid_payer_key"] == 1
    no_behavior_exclusion = out["has_behavior_exclusion"] == 0
    no_employer_like_exclusion = out["has_repeat_employer_like_exclusion_keyword"] == 0
    no_two_way = out["has_payer_debit"] == 0

    out["rule_transfer_from_pay_signal"] = (
        eligible
        & transfer_from
        & (out["has_pay_signal"] == 1)
        & no_wage_advance
        & valid_payer
        & (out["is_possible_wage_amount"] == 1)
        & no_behavior_exclusion
    ).astype(int)

    out["rule_high_repeat_no_keyword"] = (
        eligible
        & (transfer_from == 0)
        & no_keyword
        & (out["has_soft_negative_keyword"] == 0)
        & no_wage_advance
        & high_repeat
        & cycle_or_stable
        & (out["is_common_wage_amount"] == 1)
        & valid_payer
        & no_behavior_exclusion
        & no_employer_like_exclusion
        & no_two_way
    ).astype(int)

    out["rule_high_repeat_soft_negative"] = (
        eligible
        & (transfer_from == 0)
        & no_keyword
        & (out["has_soft_negative_keyword"] == 1)
        & no_wage_advance
        & high_repeat
        & cycle_or_stable
        & (out["is_common_wage_amount"] == 1)
        & valid_payer
        & no_behavior_exclusion
        & no_employer_like_exclusion
        & no_two_way
    ).astype(int)

    return out


def add_small_amount_history_override(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_history_group_key"] = make_group_key(out)

    ordered = out.sort_values(["_history_group_key", "txn_date"], na_position="last")
    prior_detected = (
        ordered.groupby("_history_group_key")["base_wages_pred"].cumsum()
        - ordered["base_wages_pred"]
    )
    out["prior_detected_wages_same_payer_count"] = prior_detected.reindex(out.index).fillna(0).astype(int)

    out["small_amount_wage_history_override"] = (
        (out["base_wages_pred"] == 0)
        & (out["is_credit"] == 1)
        & (out["amount_num"] < MIN_NORMAL_WAGE_AMOUNT)
        & (out["has_strong_wage_keyword"] == 1)
        & (out["effective_hard_negative"] == 0)
        & (out["prior_detected_wages_same_payer_count"] >= 2)
    ).astype(int)

    return out.drop(columns="_history_group_key")


def add_soft_negative_alias_to_known_wage_payer_rule(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    provisional_wages = (
        (out["base_wages_pred"] == 1) | (out["small_amount_wage_history_override"] == 1)
    )

    matched_known_wage_payer_key = pd.Series("", index=out.index, dtype="object")
    known_wage_payer_token_overlap = pd.Series(0, index=out.index, dtype="int64")
    rule_hits = pd.Series(0, index=out.index, dtype="int64")
    small_amount_rule_hits = pd.Series(0, index=out.index, dtype="int64")

    for _, group_index in out.groupby("bank_account_id", dropna=False).groups.items():
        group = out.loc[group_index]

        known_payers = []
        for payer_key in group.loc[provisional_wages.loc[group.index], "payer_key_from_text"].fillna(""):
            tokens = payer_key_tokens(payer_key)
            if tokens:
                known_payers.append((payer_key, tokens))

        if not known_payers:
            continue

        for idx, row in group.iterrows():
            payer_key = str(row.get("payer_key_from_text", "") or "").strip()
            tokens = payer_key_tokens(payer_key)
            if not tokens:
                continue

            best_overlap = 0
            best_payer = ""
            for known_payer, known_tokens in known_payers:
                overlap = len(tokens & known_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_payer = known_payer

            matched_known_wage_payer_key.at[idx] = best_payer
            known_wage_payer_token_overlap.at[idx] = best_overlap

            # TRANSFER FROM texts stay hard-blocked here even though the
            # effective_hard_negative gate no longer includes TRANSFER FROM
            # (it was removed from the hard_negative group for the PAY-signal
            # rule): known-wage-payer alias rescue must not re-open the
            # self-transfer channel (eval: 16 TF rows, 0 Wages-labelled).
            rule_hits.at[idx] = int(
                (row.get("is_credit", 0) == 1)
                and (row.get("base_wages_pred", 0) == 0)
                and (row.get("small_amount_wage_history_override", 0) == 0)
                and (row.get("effective_hard_negative", 0) == 0)
                and (row.get("has_transfer_from_keyword", 0) == 0)
                and (row.get("has_soft_negative_keyword", 0) == 1)
                and (row.get("has_wage_advance_keyword", 0) == 0)
                and (row.get("has_valid_payer_key", 0) == 1)
                and (row.get("same_payer_credit_count", 0) >= VERY_HIGH_REPEAT_PAYER_COUNT_MIN)
                and (row.get("is_possible_wage_amount", 0) == 1)
                and (best_overlap >= 2)
            )
            small_amount_rule_hits.at[idx] = int(
                (row.get("is_credit", 0) == 1)
                and (row.get("base_wages_pred", 0) == 0)
                and (row.get("small_amount_wage_history_override", 0) == 0)
                and (row.get("effective_hard_negative", 0) == 0)
                and (row.get("has_transfer_from_keyword", 0) == 0)
                and (row.get("has_soft_negative_keyword", 0) == 1)
                and (row.get("has_wage_advance_keyword", 0) == 0)
                and (row.get("has_valid_payer_key", 0) == 1)
                and (row.get("same_payer_credit_count", 0) >= VERY_HIGH_REPEAT_PAYER_COUNT_MIN)
                and (float(row.get("amount_num", 0)) < MIN_NORMAL_WAGE_AMOUNT)
                and (best_overlap >= 2)
            )

    out["matched_known_wage_payer_key"] = matched_known_wage_payer_key
    out["known_wage_payer_token_overlap"] = known_wage_payer_token_overlap
    out["rule_soft_negative_alias_to_known_wage_payer"] = rule_hits
    out["rule_small_amount_alias_to_known_wage_payer"] = small_amount_rule_hits
    out["rule_small_amount_medium_income_high_repeat"] = (
        (out["base_wages_pred"] == 0)
        & (out["small_amount_wage_history_override"] == 0)
        & (out["is_credit"] == 1)
        & (out["amount_num"] < MIN_NORMAL_WAGE_AMOUNT)
        & (out["effective_hard_negative"] == 0)
        & (out["has_soft_negative_keyword"] == 0)
        & (out["has_wage_advance_keyword"] == 0)
        & (out["has_medium_income_keyword"] == 1)
        & (out["has_valid_payer_key"] == 1)
        & (out["same_payer_credit_count"].fillna(0) >= HIGH_REPEAT_PAYER_COUNT_MIN)
        & (out["has_regular_salary_cycle"] == 1)
    ).astype(int)
    out["rule_small_amount_same_known_wage_payer"] = (
        (out["base_wages_pred"] == 0)
        & (out["small_amount_wage_history_override"] == 0)
        & (out["is_credit"] == 1)
        & (out["amount_num"] < MIN_NORMAL_WAGE_AMOUNT)
        & (out["effective_hard_negative"] == 0)
        & (out["has_soft_negative_keyword"] == 0)
        & (out["has_wage_advance_keyword"] == 0)
        & (out["has_medium_income_keyword"] == 1)
        & (out["has_valid_payer_key"] == 1)
        & (out["same_payer_credit_count"].fillna(0) >= HIGH_REPEAT_PAYER_COUNT_MIN)
        & (out["known_wage_payer_token_overlap"] >= 3)
    ).astype(int)
    return out


def choose_rule_name(df: pd.DataFrame) -> pd.Series:
    conditions = [
        df["rule_small_amount_same_known_wage_payer"] == 1,
        df["rule_small_amount_alias_to_known_wage_payer"] == 1,
        df["rule_small_amount_medium_income_high_repeat"] == 1,
        df["small_amount_wage_history_override"] == 1,
        df["rule_strong_wage_keyword"] == 1,
        df["rule_strong_wage_keyword_small_amount"] == 1,
        df["rule_transfer_strong_wage_keyword"] == 1,
        df["rule_transfer_strong_wage_keyword_no_payer_key"] == 1,
        df["rule_transfer_with_wage_signal"] == 1,
        df["rule_repeat_employer_like_payment"] == 1,
        df["rule_soft_negative_alias_to_known_wage_payer"] == 1,
        df["rule_direct_credit_with_repeat"] == 1,
        df["rule_medium_income_high_repeat"] == 1,
        df["rule_stable_payer_without_keywords"] == 1,
        df["rule_recurring_payer_behavior"] == 1,
        df["rule_transfer_from_pay_signal"] == 1,
        df["rule_high_repeat_no_keyword"] == 1,
        df["rule_high_repeat_soft_negative"] == 1,
        df["is_credit"] != 1,
        df["has_hard_negative_keyword"] == 1,
        df["has_soft_negative_keyword"] == 1,
        df["amount_num"] < MIN_NORMAL_WAGE_AMOUNT,
        df["is_possible_wage_amount"] != 1,
    ]
    choices = [
        "small_amount_same_known_wage_payer",
        "small_amount_alias_to_known_wage_payer",
        "small_amount_medium_income_high_repeat",
        "small_amount_wage_history_override",
        "strong_wage_keyword",
        "strong_wage_keyword_small_amount",
        "transfer_strong_wage_keyword",
        "transfer_strong_wage_keyword_no_payer_key",
        "transfer_with_wage_signal",
        "repeat_employer_like_payment",
        "soft_negative_alias_to_known_wage_payer",
        "direct_credit_with_repeat",
        "medium_income_high_repeat",
        "stable_payer_without_keywords",
        "recurring_payer_behavior",
        "transfer_from_pay_signal",
        "high_repeat_no_keyword",
        "high_repeat_soft_negative",
        "not_wages_not_credit",
        "not_wages_hard_negative_keyword",
        "not_wages_soft_negative_keyword",
        "not_wages_amount_too_small",
        "not_wages_amount_out_of_range",
    ]
    return pd.Series(np.select(conditions, choices, default="not_wages_no_matching_rule"), index=df.index)


def build_wages_reason(row: pd.Series) -> str:
    is_wages = row.get("is_wages_pred", 0) == 1
    flags = [
        ("credit", row.get("is_credit", 0) == 1),
        ("strong_wage_keyword", row.get("has_strong_wage_keyword", 0) == 1),
        ("medium_income_keyword", row.get("has_medium_income_keyword", 0) == 1),
        ("pay_signal", row.get("has_pay_signal", 0) == 1),
        ("repeat_payer", row.get("same_payer_credit_count", 0) >= 2),
        ("regular_cycle", row.get("has_regular_salary_cycle", 0) == 1),
        ("stable_amount", row.get("has_stable_amount", 0) == 1),
        ("negative_keyword", row.get("has_hard_negative_keyword", 0) == 1
         or row.get("has_soft_negative_keyword", 0) == 1),
    ]
    return format_classification_reason(
        category="wages" if is_wages else "not_wages",
        rule=row.get("wages_rule_name", ""),
        evidence=[label for label, flag in flags if flag],
    )


def apply_wages_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = add_hard_gate_flags(df)
    out = add_base_wage_rules(out)
    out = add_high_repeat_behavior_rules(out)
    out = add_small_amount_history_override(out)
    out = add_soft_negative_alias_to_known_wage_payer_rule(out)
    out["is_wages_pred"] = (
        (out["base_wages_pred"] == 1)
        | (out["rule_transfer_from_pay_signal"] == 1)
        | (out["rule_high_repeat_no_keyword"] == 1)
        | (out["rule_high_repeat_soft_negative"] == 1)
        | (out["small_amount_wage_history_override"] == 1)
        | (out["rule_soft_negative_alias_to_known_wage_payer"] == 1)
        | (out["rule_small_amount_same_known_wage_payer"] == 1)
        | (out["rule_small_amount_alias_to_known_wage_payer"] == 1)
        | (out["rule_small_amount_medium_income_high_repeat"] == 1)
    ).astype(int)
    out["wages_rule_name"] = choose_rule_name(out)
    out["wages_pred_reason"] = out.apply(build_wages_reason, axis=1)
    return out


# =============================================================================
# Income type classification
# =============================================================================

def add_income_type_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add income classification fields.

    Priority:
    1. salary_packaging
    2. centrelink
    3. salary_payg
    4. self_employed_gig
    5. non_income

    Note:
    - salary_packaging is intentionally evaluated before salary_payg because its text
      often contains the word "salary".
    - centrelink is intentionally separated from wages.
    - self_employed_gig excludes obvious internal transfers, refunds, loans, tax,
      investment income and other non-income credits.
    """
    out = df.copy()
    credit = out["is_credit"] == 1

    out["rule_income_salary_packaging"] = (
        credit & (out["has_salary_packaging_keyword"] == 1)
    ).astype(int)

    out["rule_income_centrelink"] = (
        credit & (out["has_centrelink_keyword"] == 1)
    ).astype(int)

    out["rule_income_salary_payg"] = (
        credit
        & (out["is_wages_pred"] == 1)
        & (out["rule_income_salary_packaging"] == 0)
        & (out["rule_income_centrelink"] == 0)
    ).astype(int)

    out["rule_income_self_employed_gig"] = (
        credit
        & (out["amount_num"] > 0)
        & (out["has_self_employed_gig_keyword"] == 1)
        & (out["has_gig_exclusion_keyword"] == 0)
        & (out["has_wage_advance_keyword"] == 0)
        & (out["rule_income_salary_packaging"] == 0)
        & (out["rule_income_centrelink"] == 0)
        & (out["rule_income_salary_payg"] == 0)
        & (out["has_gig_family_exclusion_keyword"] == 0)
        & (
            (out["has_gig_personal_transfer_pattern"] == 0)
            | (out["has_gig_invoice_keyword"] == 1)
        )
    ).astype(int)

    # Behavioral gig rule: repeated one-way personal transfers from the same
    # payer = personal-employer gig income (same coarse Wages bucket as
    # platform gig income).  Deliberately no cycle/stability requirement —
    # gig payouts are irregular, which is exactly why the wage behavior rules
    # miss them.  Gates mirror the wage rules (amount band, repeat count,
    # one-way, no hard negatives) plus the gig personal-transfer gates above.
    out["rule_income_self_employed_gig_repeat_payer"] = (
        credit
        & (out["amount_num"] > 0)
        & (out["amount_num"] <= POSSIBLE_WAGE_AMOUNT_MAX)
        & (out["amount_num"] >= GIG_REPEAT_PAYER_AMOUNT_MIN)
        & (out["same_payer_credit_count"].fillna(0) >= GIG_REPEAT_PAYER_COUNT_MIN)
        & (out["has_payer_debit"] == 0)
        & (out["has_valid_payer_key"] == 1)
        & (out["effective_hard_negative"] == 0)
        & (out["has_wage_advance_keyword"] == 0)
        & (out["has_transfer_from_keyword"] == 0)
        & ((out["has_behavior_exclusion"] == 0) | (out["has_gig_psp_reference"] == 1))
        & (out["has_gig_family_exclusion_keyword"] == 0)
        & (out["has_gig_personal_transfer_pattern"] == 1)
        & (out["has_gig_personal_exclusion_keyword"] == 0)
        & (out["rule_income_salary_packaging"] == 0)
        & (out["rule_income_centrelink"] == 0)
        & (out["rule_income_salary_payg"] == 0)
        & (out["rule_income_self_employed_gig"] == 0)
    ).astype(int)

    out["rule_known_non_income_wage_advance"] = (
        credit
        & (out["has_wage_advance_keyword"] == 1)
        & (out["rule_income_salary_packaging"] == 0)
        & (out["rule_income_centrelink"] == 0)
        & (out["rule_income_salary_payg"] == 0)
        & (out["rule_income_self_employed_gig"] == 0)
        & (out["rule_income_self_employed_gig_repeat_payer"] == 0)
    ).astype(int)

    conditions = [
        out["rule_income_salary_packaging"] == 1,
        out["rule_income_centrelink"] == 1,
        out["rule_income_salary_payg"] == 1,
        out["rule_income_self_employed_gig"] == 1,
        out["rule_income_self_employed_gig_repeat_payer"] == 1,
    ]
    income_types = [
        "salary_packaging",
        "centrelink",
        "salary_payg",
        "self_employed_gig",
        "self_employed_gig",
    ]
    rule_names = [
        "salary_packaging_text_keyword",
        "centrelink_government_benefit_keyword",
        "salary_payg_wages_rule",
        "self_employed_gig_keyword_without_exclusion",
        "self_employed_gig_repeat_payer",
    ]

    out["income_type_pred"] = np.select(conditions, income_types, default="non_income")
    out["income_type_rule_name"] = np.select(conditions, rule_names, default="no_income_type_rule")
    out["is_income_pred"] = out["income_type_pred"].ne("non_income").astype(int)
    # finv_category is the coarse illion category.  The fine-grained income type
    # (salary_packaging / centrelink / salary_payg / self_employed_gig) stays in
    # income_type_pred for stream grouping and reasoning.
    out["finv_category"] = (
        out["income_type_pred"]
        .map({"salary_packaging": "Wages", "salary_payg": "Wages", "self_employed_gig": "Wages", "centrelink": "Centrelink"})
        .where(out["is_income_pred"].eq(1), "")
    )
    out["known_non_income_type_pred"] = np.select(
        [out["rule_known_non_income_wage_advance"] == 1],
        ["wage_advance"],
        default="",
    )
    out["known_non_income_rule_name"] = np.select(
        [out["rule_known_non_income_wage_advance"] == 1],
        ["wage_advance_keyword"],
        default="",
    )

    out["income_type_pred_reason"] = out.apply(build_income_type_reason, axis=1)
    return out


def build_income_type_reason(row: pd.Series) -> str:
    income_type = row.get("income_type_pred", "non_income")
    rule_name = row.get("income_type_rule_name", "")

    if income_type == "non_income":
        evidence: list[str] = []
        if row.get("is_credit", 0) != 1:
            evidence.append("not_credit")
        kn = row.get("known_non_income_type_pred", "")
        if kn:
            evidence.append(f"known_non_income={kn}")
        if (row.get("has_gig_exclusion_keyword", 0) == 1
                or row.get("has_hard_negative_keyword", 0) == 1
                or row.get("has_soft_negative_keyword", 0) == 1):
            evidence.append("exclusion_keyword")
        return format_classification_reason(
            category=income_type, rule=rule_name, evidence=evidence,
        )

    flags = [
        ("credit", row.get("is_credit", 0) == 1),
        ("salary_packaging_keyword", row.get("has_salary_packaging_keyword", 0) == 1),
        ("centrelink_keyword", row.get("has_centrelink_keyword", 0) == 1),
        ("wages_detector", row.get("is_wages_pred", 0) == 1),
        ("strong_wage_keyword", row.get("has_strong_wage_keyword", 0) == 1),
        ("medium_income_keyword", row.get("has_medium_income_keyword", 0) == 1),
        ("repeat_payer", row.get("same_payer_credit_count", 0) >= 2),
        ("regular_cycle", row.get("has_regular_salary_cycle", 0) == 1),
        ("stable_amount", row.get("has_stable_amount", 0) == 1),
        ("gig_keyword", row.get("has_self_employed_gig_keyword", 0) == 1),
    ]
    return format_classification_reason(
        category=income_type, rule=rule_name,
        evidence=[label for label, flag in flags if flag],
    )


# =============================================================================
# Output and validation
# =============================================================================

def reorder_output_columns(result: pd.DataFrame, original_cols: List[str]) -> pd.DataFrame:
    original_cols = [col for col in original_cols if col in result.columns]
    important_cols = [
        col for col in IMPORTANT_OUTPUT_COLUMNS
        if col in result.columns and col not in original_cols
    ]
    remaining_cols = [
        col for col in result.columns
        if col not in original_cols and col not in important_cols
    ]
    return result[original_cols + important_cols + remaining_cols]
