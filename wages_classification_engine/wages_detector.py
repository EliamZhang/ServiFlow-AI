# -*- coding: utf-8 -*-
"""
Rule-based income detector for bank transactions.

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
- centrelink_payment_type: subtype for Centrelink / government benefit income.
    Values:
        pension
        family_benefit
        youth_allowance
        jobseeker
        parenting_payment
        carer_payment
        disability_support
        other_centrelink
        blank for non-centrelink rows
- income_type_rule_name: the income classification rule that matched.
- income_type_pred_reason: readable reason for income classification.

Important restrictions:
- Do NOT use third_party as an input feature.
- Do NOT use trx_type / txn_type / txn_type_category as input features.
- Do NOT use category as decision logic. It is only used for optional validation.
- Do NOT use a numeric score. Final result is based on yes/no business rules.

The package CLI is exposed by ``wages_classification_engine.model_main``.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IncomeClassificationResult:
    transactions: pd.DataFrame
    summary: pd.DataFrame
    original_columns: tuple[str, ...]


# =============================================================================
# Config
# =============================================================================

# These columns are removed before rule execution to avoid accidental leakage.
RESTRICTED_COLUMNS = ["trx_type", "txn_type", "txn_type_category", "third_party"]

STRONG_WAGE_PATTERNS = [
    r"\bSALARY\b",
    r"\bPAYROLL\b",
    r"\bPAY\s*ROLL\b",
    r"\bWAGES?\b",
    r"\bPAYSLIP\b",
    r"\bPAYRUN\b",
    r"\bEMPLOYER\b",
    r"\bSTAFF\s*PAY\b",
    r"\bEFT\s*SALARY\b",
    r"\bDEPOSIT\s*SALARY\b",
]

MEDIUM_INCOME_PATTERNS = [
    r"\bDIRECT\s*CREDIT\b",
    r"\bDIR\s*CREDIT\b",
    r"\bDEPOSIT[-\s]*SALARY\b",
    r"\bEFT\s*DEP\b",
    r"\bDIRECT\s*DEP(?:OSIT)?\b",
]

REPEAT_EMPLOYER_LIKE_PATTERNS = [
    r"\bPAY\-PACK\b",
    r"\bINTER[-\s]*BANK\s*CREDIT\b",
    r"\bPAY\s+FOR\b",
    r"\bDEPOSIT\s+ONLINE\b.*\bPYMT\b",
    r"\bDEPOSIT\b.*\bPAY\b",
]

REPEAT_EMPLOYER_LIKE_EXCLUSION_PATTERNS = [
    r"\bREPAYMENT\b",
    r"\bPAYPAL\b",
    r"\bJOB\s*SEEKER\b",
    r"\bJOBSEEKER\b",
    r"\bFAMILY\s*PAYMENT\b",
    r"\bMAXXIA\b",
    r"\bCLAIMS?\b",
    r"\bMERCHANT\s*SETTLEMENT\b",
    r"\bDISBURSEMENT\b",
    r"\bINVEST(?:MENT)?\b",
]

SALARY_PACKAGING_PATTERNS = [
    r"\bSALARY\s*PACKAGING\b",
    r"\bSALARY\s*PACKAGE\b",
    r"\bACCESS\s*PAY\b",
    r"\bACCESSPAY\b",
    r"\bSALARY\s*SACRIFICE\b",
    r"\bEMPLOYER\s*BENEFIT\b",
]

CENTRELINK_PATTERNS = [
    r"\bCENTRE\s*LINK\b",
    r"\bCENTRELINK\b",
    r"\bCTRLINK\b",
    r"\bCTR\s*LINK\b",
    r"\bSERVICES\s*AUSTRALIA\b",
    r"\bAUS\s*GOV\b",
    r"\bAUSTRALIAN\s*GOV(?:ERNMENT)?\b",
    r"\bFAMILY\s*ALLOWANCE\b",
    r"\bFAMILY\s*PAYMENT\b",
    r"\bPENSION\b",
    r"\bYOUTH\s*ALLOWANCE\b",
    r"\bJOB\s*SEEKER\b",
    r"\bJOBSEEKER\b",
    r"\bPARENTING\s*PAYMENT\b",
    r"\bCARER\s*PAYMENT\b",
    r"\bDISABILITY\s*SUPPORT\b",
]

CENTRELINK_PAYMENT_TYPE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("pension", [r"\bPENSION\b"]),
    (
        "family_benefit",
        [
            r"\bFAMILY\s*ALLOWANCE\b",
            r"\bFAMILY\s*PAYMENT\b",
            r"\bFAMILY\s*TAX\s*BENEFIT\b",
            r"\bFTB\b",
        ],
    ),
    ("youth_allowance", [r"\bYOUTH\s*ALLOWANCE\b", r"\bYTH\s*ALL\b"]),
    ("jobseeker", [r"\bJOB\s*SEEKER\b", r"\bJOBSEEKER\b"]),
    ("parenting_payment", [r"\bPARENTING\s*PAYMENT\b"]),
    ("carer_payment", [r"\bCARER\s*PAYMENT\b"]),
    ("disability_support", [r"\bDISABILITY\s*SUPPORT\b", r"\bDSP\b"]),
]

SELF_EMPLOYED_GIG_PATTERNS = [
    r"\bUBER\b",
    r"\bUBER\s*PARTNER\b",
    r"\bDOOR\s*DASH\b",
    r"\bDOORDASH\b",
    r"\bMENULOG\b",
    r"\bDELIVEROO\b",
    r"\bAIRTASKER\b",
    r"\bPAYMENT\s*FOR\s*SERVICES?\b",
    r"\bINVOICE\b",
    r"\bCONTRACTOR\b",
    r"\bBUSINESS\s*PAYMENT\b",
    r"\bBUSINESS\s*INCOME\b",
    r"\bSETTLEMENT\b",
    r"\bSTRIPE\b",
    r"\bSQUARE\b",
    r"\bPAYPAL\b",
    r"\bSHOPIFY\b",
]

WAGE_ADVANCE_PATTERNS = [
    r"\bBEFOREPAY\b",
    r"\bMYPAYNOW\b",
    r"\bWAGETAP\b",
    r"\bSTEPPAY\b",
    r"\bPRESS\s*PAY\b",
    r"\bPRESSPAY\b",
    r"\bWAGE\s*PAY\b",
    r"\bWAGEPAY\b",
]

RETURN_LIKE_PATTERNS = [
    r"\bRETURN\b",
    r"\bVALUE\s*DATE\b",
    r"\bDISHONOU?R(?:ED)?\b",
    r"\bREVERSAL\s+OF\s+DEBIT\b",
]

# Hard negative patterns for wage detection.
# These rows should not be treated as wages even if they look repeated.
HARD_NEGATIVE_PATTERNS = RETURN_LIKE_PATTERNS + [
    r"\bINTERNAL\s*TRANSFER\b",
    r"\bLINKED\s*ACC\b",
    r"\bREFUND\b",
    r"\bREVERSAL\b",
    r"\bADJUSTMENT\b",
    r"\bREBATE\b",
    r"\bCASHBACK\b",
    r"\bLOAN\b",
    r"\bADVANCE\b",
    r"\bCENTRE\s*LINK\b",
    r"\bCENTRELINK\b",
    r"\bCTRLINK\b",
    r"\bSERVICES\s*AUSTRALIA\b",
    r"\bATO\b",
    r"\bTAX\b",
    r"\bINTEREST\b",
    r"\bDIVIDEND\b",
    r"\bBNPL\b",
    r"\bZIP\b",
    r"\bAFTERPAY\b",
    r"\bWITHDRAWAL\b",
]

# Soft negative patterns for wage detection.
# These usually indicate non-wage transfers, but can still be overridden by a
# strong wage signal plus repeated salary-like behavior.
SOFT_NEGATIVE_PATTERNS = [
    r"\bTRANSFER\b",
    r"\bOSKO\b",
]

# Combined negative view kept for explainability output.
NEGATIVE_PATTERNS = HARD_NEGATIVE_PATTERNS + SOFT_NEGATIVE_PATTERNS

# Extra exclusions used only for self-employed / gig classification.
GIG_EXCLUSION_PATTERNS = HARD_NEGATIVE_PATTERNS + SOFT_NEGATIVE_PATTERNS + [
    r"\bLOAN\s*DEPOSIT\b",
]

PAYER_STOP_WORDS = {
    "DIRECT", "CREDIT", "DIR", "DEPOSIT", "SALARY", "PAYROLL", "WAGE", "WAGES",
    "PAY", "PAYMENT", "PAYMENTS", "TRANSFER", "TRANS", "FROM", "TO", "REF",
    "REFERENCE", "ONLINE", "INTERNET", "EFT", "DEP", "OSKO", "VISA", "CARD",
    "PURCHASE", "DEBIT", "MISCELLANEOUS", "BPAY", "WITHDRAWAL", "ATM",
    "TRNS", "ACC", "ACCOUNT", "LINKED", "AU", "AUS", "THE", "AND", "PTY",
    "LTD", "LIMITED", "SALARY", "PACKAGING", "CENTRELINK", "CENTRE", "LINK",
    "SERVICES", "AUSTRALIA", "GOV", "GOVERNMENT", "RETURN", "VALUE", "DATE",
    "FAST", "NPP", "THANK", "YOU", "RECEIVED", "MAIN", "JAN", "FEB", "MAR",
    "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}

MIN_NORMAL_WAGE_AMOUNT = 100
COMMON_WAGE_AMOUNT_MIN = 300
COMMON_WAGE_AMOUNT_MAX = 10000
POSSIBLE_WAGE_AMOUNT_MAX = 20000
STABLE_AMOUNT_CV_MAX = 0.35
REGULAR_GAP_RANGES = ((6, 8), (13, 16), (27, 33))
HIGH_REPEAT_PAYER_COUNT_MIN = 4
VERY_HIGH_REPEAT_PAYER_COUNT_MIN = 8
STABLE_PAYER_REPEAT_COUNT_MIN = 6

WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

INCOME_SUMMARY_COLUMNS = [
    "finv_category",
    "stream_id",
    "bank_account_id",
    "account_type",
    "application_id",
    "bank",
    "credit_limit",
    "counterparty",
    "centrelink_payment_type",
    "transaction_start_date",
    "transaction_end_date",
    "status",
    "transaction_count",
    "total_income_amount",
    "average_income_amount",
    "median_income_amount",
    "latest_income_amount",
    "estimated_monthly_income",
    "frequency",
    "frequency_day",
    "predicted_next_income_date",
]

IMPORTANT_OUTPUT_COLUMNS = [
    # New income classification fields
    "finv_category",
    "counterparty",
    "stream_id",
    "income_type_pred",
    "known_non_income_type_pred",
    "known_non_income_rule_name",
    "centrelink_payment_type",
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
    "hard_exclusion",

    # Income rule flags
    "rule_income_salary_packaging",
    "rule_income_centrelink",
    "rule_income_salary_payg",
    "rule_income_self_employed_gig",
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
    "has_gig_exclusion_keyword",
    "has_negative_keyword",
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
CENTRELINK_PAYMENT_TYPE_REGEX: List[Tuple[str, List[re.Pattern]]] = [
    (payment_type, compile_patterns(patterns))
    for payment_type, patterns in CENTRELINK_PAYMENT_TYPE_PATTERNS
]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def count_matches(text: str, patterns: List[re.Pattern]) -> int:
    if not text:
        return 0
    return sum(1 for pattern in patterns if pattern.search(text))


def classify_centrelink_payment_type(text: str) -> str:
    if not text:
        return "other_centrelink"
    for payment_type, patterns in CENTRELINK_PAYMENT_TYPE_REGEX:
        if count_matches(text, patterns) > 0:
            return payment_type
    return "other_centrelink"


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
    out["text_clean"] = out["text"].apply(clean_text)
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
    ]:
        out[count_col] = out["text_clean"].apply(lambda x: count_matches(x, patterns))
        out[flag_col] = (out[count_col] > 0).astype(int)

    out["is_common_wage_amount"] = out["amount_num"].between(
        COMMON_WAGE_AMOUNT_MIN, COMMON_WAGE_AMOUNT_MAX, inclusive="both"
    ).astype(int)
    out["is_possible_wage_amount"] = out["amount_num"].between(
        MIN_NORMAL_WAGE_AMOUNT, POSSIBLE_WAGE_AMOUNT_MAX, inclusive="both"
    ).astype(int)
    out["is_tiny_credit"] = ((out["is_credit"] == 1) & (out["amount_num"] < MIN_NORMAL_WAGE_AMOUNT)).astype(int)

    out["payer_key_from_text"] = out["text_clean"].apply(make_payer_key)
    out["has_valid_payer_key"] = (out["payer_key_from_text"].str.len() >= 3).astype(int)
    return out


def add_payer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_group_key"] = make_group_key(out)

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
    out["has_repeat_same_payer"] = (out["same_payer_credit_count"].fillna(0) >= 2).astype(int)
    out["has_stable_amount"] = (
        (out["same_payer_credit_count"].fillna(0) >= 2)
        & (out["same_payer_amount_cv"].fillna(999) <= STABLE_AMOUNT_CV_MAX)
    ).astype(int)

    return out.drop(columns="_group_key")


def add_wages_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_payer_history_features(add_basic_features(df))


# =============================================================================
# Wage rule decision
# =============================================================================

def add_hard_gate_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rule_pass_credit"] = (out["is_credit"] == 1).astype(int)
    out["rule_pass_amount"] = (out["amount_num"] >= MIN_NORMAL_WAGE_AMOUNT).astype(int)
    out["rule_pass_no_hard_negative_keyword"] = (out["has_hard_negative_keyword"] == 0).astype(int)
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
        eligible & no_soft_negative & (out["has_strong_wage_keyword"] == 1)
    ).astype(int)

    out["rule_transfer_strong_wage_keyword"] = (
        eligible
        & (out["has_soft_negative_keyword"] == 1)
        & no_wage_advance
        & (out["has_strong_wage_keyword"] == 1)
        & (out["has_valid_payer_key"] == 1)
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
        "rule_transfer_strong_wage_keyword",
        "rule_transfer_with_wage_signal",
        "rule_repeat_employer_like_payment",
        "rule_direct_credit_with_repeat",
        "rule_medium_income_high_repeat",
        "rule_stable_payer_without_keywords",
        "rule_recurring_payer_behavior",
    ]
    out["base_wages_pred"] = (out[rule_cols].max(axis=1) == 1).astype(int)
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
        & (out["has_hard_negative_keyword"] == 0)
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

            rule_hits.at[idx] = int(
                (row.get("is_credit", 0) == 1)
                and (row.get("base_wages_pred", 0) == 0)
                and (row.get("small_amount_wage_history_override", 0) == 0)
                and (row.get("has_hard_negative_keyword", 0) == 0)
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
                and (row.get("has_hard_negative_keyword", 0) == 0)
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
        & (out["has_hard_negative_keyword"] == 0)
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
        & (out["has_hard_negative_keyword"] == 0)
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
        df["rule_transfer_strong_wage_keyword"] == 1,
        df["rule_transfer_with_wage_signal"] == 1,
        df["rule_repeat_employer_like_payment"] == 1,
        df["rule_soft_negative_alias_to_known_wage_payer"] == 1,
        df["rule_direct_credit_with_repeat"] == 1,
        df["rule_medium_income_high_repeat"] == 1,
        df["rule_stable_payer_without_keywords"] == 1,
        df["rule_recurring_payer_behavior"] == 1,
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
        "transfer_strong_wage_keyword",
        "transfer_with_wage_signal",
        "repeat_employer_like_payment",
        "soft_negative_alias_to_known_wage_payer",
        "direct_credit_with_repeat",
        "medium_income_high_repeat",
        "stable_payer_without_keywords",
        "recurring_payer_behavior",
        "not_wages_not_credit",
        "not_wages_hard_negative_keyword",
        "not_wages_soft_negative_keyword",
        "not_wages_amount_too_small",
        "not_wages_amount_out_of_range",
    ]
    return pd.Series(np.select(conditions, choices, default="not_wages_no_matching_rule"), index=df.index)


def build_wages_reason(row: pd.Series) -> str:
    reasons = [
        f"matched rule: {row.get('wages_rule_name', '')}"
        if row.get("is_wages_pred", 0) == 1
        else f"not wages: {row.get('wages_rule_name', '')}",
        "credit transaction" if row.get("is_credit", 0) == 1 else "not credit",
    ]

    checks = [
        (row.get("has_strong_wage_keyword", 0) == 1, "salary/payroll/wage keyword"),
        (row.get("has_medium_income_keyword", 0) == 1, "direct credit/deposit keyword"),
        (row.get("is_common_wage_amount", 0) == 1, "common wage amount range"),
        (
            row.get("is_common_wage_amount", 0) != 1 and row.get("is_possible_wage_amount", 0) == 1,
            "possible wage amount range",
        ),
        (
            row.get("is_common_wage_amount", 0) != 1 and row.get("is_possible_wage_amount", 0) != 1,
            "amount outside normal wage range",
        ),
        (row.get("same_payer_credit_count", 0) >= 2, "same payer appears repeatedly"),
        (row.get("has_regular_salary_cycle", 0) == 1, "regular weekly/fortnightly/monthly cycle"),
        (row.get("has_stable_amount", 0) == 1, "stable repeated amount"),
        (
            row.get("rule_transfer_strong_wage_keyword", 0) == 1,
            "transfer/osko style credit but explicit wage keyword and valid payer",
        ),
        (
            row.get("rule_repeat_employer_like_payment", 0) == 1,
            "repeated stable employer-like payment text without negative markers",
        ),
        (row.get("rule_medium_income_high_repeat", 0) == 1, "repeated direct credits within possible wage amount range"),
        (row.get("rule_stable_payer_without_keywords", 0) == 1, "stable repeated payer without explicit wage keyword"),
        (
            row.get("rule_soft_negative_alias_to_known_wage_payer", 0) == 1,
            f"payer overlaps with known wage payer: {row.get('matched_known_wage_payer_key', '')}",
        ),
        (
            row.get("rule_small_amount_alias_to_known_wage_payer", 0) == 1,
            f"small amount but payer overlaps with known wage payer: {row.get('matched_known_wage_payer_key', '')}",
        ),
        (
            row.get("rule_small_amount_medium_income_high_repeat", 0) == 1,
            "small repeated direct credit from high-frequency payer",
        ),
        (
            row.get("rule_small_amount_same_known_wage_payer", 0) == 1,
            "small direct credit from payer already strongly linked to wages",
        ),
        (row.get("small_amount_wage_history_override", 0) == 1, "same payer previously detected as wages"),
        (row.get("has_wage_advance_keyword", 0) == 1, "earned wage advance / short-term credit keyword"),
        (row.get("has_return_like_keyword", 0) == 1, "return / value date / dishonour style credit"),
        (row.get("has_hard_negative_keyword", 0) == 1, "hard negative keyword, e.g. return/loan/refund/interest"),
        (row.get("has_soft_negative_keyword", 0) == 1, "soft negative keyword, e.g. transfer"),
    ]
    reasons.extend(reason for matched, reason in checks if matched)
    return "; ".join(reasons)


def apply_wages_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = add_hard_gate_flags(df)
    out = add_base_wage_rules(out)
    out = add_small_amount_history_override(out)
    out = add_soft_negative_alias_to_known_wage_payer_rule(out)
    out["is_wages_pred"] = (
        (out["base_wages_pred"] == 1)
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

def add_income_type_rules(
    df: pd.DataFrame,
    include_centrelink_payment_type: bool = True,
) -> pd.DataFrame:
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
        & (out["has_self_employed_gig_keyword"] == 1)
        & (out["has_gig_exclusion_keyword"] == 0)
        & (out["has_wage_advance_keyword"] == 0)
        & (out["rule_income_salary_packaging"] == 0)
        & (out["rule_income_centrelink"] == 0)
        & (out["rule_income_salary_payg"] == 0)
    ).astype(int)

    out["rule_known_non_income_wage_advance"] = (
        credit
        & (out["has_wage_advance_keyword"] == 1)
        & (out["rule_income_salary_packaging"] == 0)
        & (out["rule_income_centrelink"] == 0)
        & (out["rule_income_salary_payg"] == 0)
        & (out["rule_income_self_employed_gig"] == 0)
    ).astype(int)

    conditions = [
        out["rule_income_salary_packaging"] == 1,
        out["rule_income_centrelink"] == 1,
        out["rule_income_salary_payg"] == 1,
        out["rule_income_self_employed_gig"] == 1,
    ]
    income_types = [
        "salary_packaging",
        "centrelink",
        "salary_payg",
        "self_employed_gig",
    ]
    rule_names = [
        "salary_packaging_text_keyword",
        "centrelink_government_benefit_keyword",
        "salary_payg_wages_rule",
        "self_employed_gig_keyword_without_exclusion",
    ]

    out["income_type_pred"] = np.select(conditions, income_types, default="non_income")
    out["income_type_rule_name"] = np.select(conditions, rule_names, default="no_income_type_rule")
    out["is_income_pred"] = out["income_type_pred"].ne("non_income").astype(int)
    out["finv_category"] = out["income_type_pred"].where(out["is_income_pred"].eq(1), "")
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

    out["centrelink_payment_type"] = ""
    if include_centrelink_payment_type:
        is_centrelink = out["income_type_pred"].eq("centrelink")
        out.loc[is_centrelink, "centrelink_payment_type"] = out.loc[is_centrelink, "text_clean"].apply(
            classify_centrelink_payment_type
        )

    out["income_type_pred_reason"] = out.apply(build_income_type_reason, axis=1)
    return out


def build_income_type_reason(row: pd.Series) -> str:
    income_type = row.get("income_type_pred", "non_income")
    rule_name = row.get("income_type_rule_name", "")

    if income_type == "non_income":
        reasons = ["not classified as income", f"rule: {rule_name}"]
        if row.get("is_credit", 0) != 1:
            reasons.append("not credit")
        if row.get("known_non_income_type_pred", ""):
            reasons.append(f"known_non_income_type={row.get('known_non_income_type_pred', '')}")
        if row.get("has_gig_exclusion_keyword", 0) == 1:
            reasons.append("contains transfer/refund/loan/tax/investment exclusion keyword")
        if row.get("has_hard_negative_keyword", 0) == 1:
            reasons.append("contains hard negative wage keyword")
        elif row.get("has_soft_negative_keyword", 0) == 1:
            reasons.append("contains soft negative wage keyword")
        return "; ".join(reasons)

    reasons = [f"income_type={income_type}", f"matched rule: {rule_name}", "credit transaction"]

    checks = [
        (row.get("has_salary_packaging_keyword", 0) == 1, "salary packaging / accesspay / salary sacrifice keyword"),
        (row.get("has_centrelink_keyword", 0) == 1, "centrelink / government benefit keyword"),
        (row.get("centrelink_payment_type", "") != "", f"centrelink_payment_type={row.get('centrelink_payment_type', '')}"),
        (row.get("is_wages_pred", 0) == 1, "wages detector matched"),
        (row.get("has_strong_wage_keyword", 0) == 1, "salary/payroll/wage keyword"),
        (row.get("has_medium_income_keyword", 0) == 1, "direct credit/deposit keyword"),
        (row.get("same_payer_credit_count", 0) >= 2, "same payer appears repeatedly"),
        (row.get("has_regular_salary_cycle", 0) == 1, "regular weekly/fortnightly/monthly cycle"),
        (row.get("has_stable_amount", 0) == 1, "stable repeated amount"),
        (row.get("has_self_employed_gig_keyword", 0) == 1, "gig platform / invoice / contractor / business payment keyword"),
    ]
    reasons.extend(reason for matched, reason in checks if matched)
    return "; ".join(reasons)


# =============================================================================
# Income stream summarisation
# =============================================================================

def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def clean_counterparty(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return text[:80]


def derive_counterparty(row: pd.Series) -> str:
    if int(row.get("is_income_pred", 0)) != 1:
        return ""

    income_type = str(row.get("income_type_pred", "")).strip()
    centrelink_payment_type = str(row.get("centrelink_payment_type", "")).strip()
    payer_key = clean_counterparty(row.get("payer_key_from_text", ""))
    text_clean = clean_counterparty(row.get("text_clean", ""))

    if income_type == "centrelink":
        return f"CENTRELINK {centrelink_payment_type.upper()}".strip()
    if payer_key:
        return payer_key
    return text_clean


def build_income_stream_group_key(row: pd.Series) -> Optional[str]:
    if int(row.get("is_income_pred", 0)) != 1:
        return None

    bank_account_id = str(row.get("bank_account_id", "")).strip()
    finv_category = str(row.get("finv_category", "")).strip()
    counterparty = str(row.get("counterparty", "")).strip()
    centrelink_payment_type = str(row.get("centrelink_payment_type", "")).strip()
    if not bank_account_id or not finv_category or not counterparty:
        return None
    return "||".join([bank_account_id, finv_category, counterparty, centrelink_payment_type])


def summarize_frequency(dates: pd.Series) -> tuple[str, Optional[int], str]:
    valid_dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(valid_dates) <= 1:
        weekday_name = WEEKDAY_NAMES.get(valid_dates.iloc[-1].weekday(), "") if len(valid_dates) == 1 else ""
        return "one_off", None, weekday_name

    gaps = valid_dates.diff().dt.days.dropna()
    if gaps.empty:
        weekday_name = WEEKDAY_NAMES.get(valid_dates.iloc[-1].weekday(), "")
        return "one_off", None, weekday_name

    median_gap = float(gaps.median())
    rounded_gap = int(round(median_gap))

    if 6 <= median_gap <= 8:
        frequency = "weekly"
        typical_gap_days = 7
    elif 13 <= median_gap <= 16:
        frequency = "fortnightly"
        typical_gap_days = 14
    elif 27 <= median_gap <= 33:
        frequency = "monthly"
        typical_gap_days = 30
    else:
        frequency = "irregular"
        typical_gap_days = rounded_gap if rounded_gap > 0 else None

    weekday_mode = valid_dates.dt.weekday.mode()
    weekday_name = WEEKDAY_NAMES.get(int(weekday_mode.iloc[0]), "") if not weekday_mode.empty else ""
    return frequency, typical_gap_days, weekday_name


def estimate_monthly_income(frequency: str, typical_amount: float) -> float:
    if pd.isna(typical_amount):
        return np.nan
    if frequency == "weekly":
        return typical_amount * 52.0 / 12.0
    if frequency == "fortnightly":
        return typical_amount * 26.0 / 12.0
    if frequency == "monthly":
        return typical_amount
    return np.nan


def derive_stream_status(
    transaction_count: int,
    frequency: str,
    typical_gap_days: Optional[int],
    last_date: pd.Timestamp,
    global_last_date: pd.Timestamp,
) -> str:
    if pd.isna(last_date):
        return "unknown"
    if transaction_count <= 1:
        return "single_transaction"
    if frequency not in {"weekly", "fortnightly", "monthly"} or not typical_gap_days:
        return "irregular"

    days_since_last = (global_last_date - last_date).days
    allowed_gap = int(round(typical_gap_days * 1.75))
    return "active" if days_since_last <= allowed_gap else "inactive"


def assign_income_stream_ids(result_df: pd.DataFrame) -> pd.DataFrame:
    out = result_df.copy()
    income_mask = out["is_income_pred"].eq(1)
    income_df = out[income_mask].copy()
    if income_df.empty:
        return out

    stream_order = (
        income_df.groupby("_income_stream_group_key", dropna=False)
        .agg(
            finv_category=("finv_category", first_non_null),
            bank_account_id=("bank_account_id", first_non_null),
            counterparty=("counterparty", first_non_null),
            first_txn_date=("txn_date", "min"),
        )
        .sort_values(["finv_category", "bank_account_id", "counterparty", "first_txn_date"], na_position="last")
        .reset_index()
    )

    stream_id_map = {}
    counters: dict[str, int] = {}
    for _, row in stream_order.iterrows():
        income_type = str(row["finv_category"])
        counters[income_type] = counters.get(income_type, 0) + 1
        stream_id_map[row["_income_stream_group_key"]] = f"{income_type}_{counters[income_type]:03d}"

    out.loc[income_mask, "stream_id"] = out.loc[income_mask, "_income_stream_group_key"].map(stream_id_map).fillna("")
    return out


def build_income_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    income_df = result_df[result_df["is_income_pred"].eq(1)].copy()
    if income_df.empty:
        return pd.DataFrame(columns=INCOME_SUMMARY_COLUMNS)

    income_df["txn_date"] = pd.to_datetime(income_df["txn_date"], errors="coerce")
    income_df["amount_num"] = pd.to_numeric(income_df["amount_num"], errors="coerce")
    global_last_date = income_df["txn_date"].max()
    group_column = "_income_stream_group_key" if "_income_stream_group_key" in income_df.columns else "stream_id"

    summary_rows = []
    grouped = income_df.groupby(group_column, dropna=False)

    for _, group in grouped:
        group = group.sort_values("txn_date")
        frequency, typical_gap_days, frequency_day = summarize_frequency(group["txn_date"])
        transaction_count = int(len(group))
        start_date = group["txn_date"].min()
        end_date = group["txn_date"].max()
        latest_row = group.iloc[-1]
        latest_income_amount = latest_row.get("amount_num", np.nan)
        median_income_amount = group["amount_num"].median()
        estimated_monthly_income = estimate_monthly_income(frequency, median_income_amount)
        status = derive_stream_status(
            transaction_count=transaction_count,
            frequency=frequency,
            typical_gap_days=typical_gap_days,
            last_date=end_date,
            global_last_date=global_last_date,
        )

        predicted_next_income_date = pd.NaT
        if (
            transaction_count >= 2
            and frequency in {"weekly", "fortnightly", "monthly"}
            and typical_gap_days
            and not pd.isna(end_date)
        ):
            predicted_next_income_date = end_date + pd.Timedelta(days=int(typical_gap_days))

        summary_rows.append(
            {
                "finv_category": first_non_null(group["finv_category"]),
                "stream_id": first_non_null(group["stream_id"]),
                "bank_account_id": first_non_null(group["bank_account_id"]),
                "account_type": first_non_null(group["account_type"]) if "account_type" in group.columns else np.nan,
                "application_id": first_non_null(group["application_id"])
                if "application_id" in group.columns
                else np.nan,
                "bank": first_non_null(group["bank"]) if "bank" in group.columns else np.nan,
                "credit_limit": first_non_null(group["credit_limit"]) if "credit_limit" in group.columns else np.nan,
                "counterparty": first_non_null(group["counterparty"]),
                "centrelink_payment_type": first_non_null(group["centrelink_payment_type"]),
                "transaction_start_date": start_date.date() if not pd.isna(start_date) else np.nan,
                "transaction_end_date": end_date.date() if not pd.isna(end_date) else np.nan,
                "status": status,
                "transaction_count": transaction_count,
                "total_income_amount": float(group["amount_num"].sum()),
                "average_income_amount": float(group["amount_num"].mean()),
                "median_income_amount": float(median_income_amount) if not pd.isna(median_income_amount) else np.nan,
                "latest_income_amount": float(latest_income_amount) if not pd.isna(latest_income_amount) else np.nan,
                "estimated_monthly_income": float(estimated_monthly_income)
                if not pd.isna(estimated_monthly_income)
                else np.nan,
                "frequency": frequency,
                "frequency_day": frequency_day,
                "predicted_next_income_date": (
                    predicted_next_income_date.date() if not pd.isna(predicted_next_income_date) else np.nan
                ),
            }
        )

    return pd.DataFrame(summary_rows, columns=INCOME_SUMMARY_COLUMNS).sort_values(
        ["finv_category", "stream_id"]
    ).reset_index(drop=True)


def add_income_stream_outputs(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = result_df.copy()
    out["counterparty"] = out.apply(derive_counterparty, axis=1)
    out["_income_stream_group_key"] = out.apply(build_income_stream_group_key, axis=1)
    out["stream_id"] = ""
    out = assign_income_stream_ids(out)
    summary_df = build_income_summary(out)
    out = out.drop(columns=["_income_stream_group_key"])
    return out, summary_df


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


def print_optional_validation(df: pd.DataFrame) -> None:
    if "category" not in df.columns:
        return

    label = df["category"].astype(str).str.lower().str.strip().eq("wages").astype(int)
    pred = df["is_wages_pred"].astype(int)

    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    print("\nOptional validation against category == 'Wages'")
    print("Important: category is only used here for validation, not for prediction.")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision={precision:.4f}" if not np.isnan(precision) else "Precision=NA")
    print(f"Recall={recall:.4f}" if not np.isnan(recall) else "Recall=NA")


def print_income_type_summary(df: pd.DataFrame) -> None:
    if "income_type_pred" not in df.columns:
        return

    print("\nIncome type summary")
    counts = df["income_type_pred"].value_counts(dropna=False)
    for income_type, count in counts.items():
        print(f"{income_type}: {int(count)}")

    if "centrelink_payment_type" in df.columns:
        centrelink = df[df["income_type_pred"].eq("centrelink")]
        if len(centrelink) > 0:
            print("\nCentrelink payment type summary")
            for payment_type, count in centrelink["centrelink_payment_type"].value_counts(dropna=False).items():
                print(f"{payment_type}: {int(count)}")


def classify_income_transactions(
    raw_df: pd.DataFrame,
    include_centrelink_payment_type: bool = False,
) -> IncomeClassificationResult:
    """Classify an in-memory transaction dataframe and build its summary."""
    df = prepare_input(raw_df)
    original_cols = list(df.columns)

    result = add_wages_features(df)
    result = apply_wages_rules(result)
    result = add_income_type_rules(
        result,
        include_centrelink_payment_type=include_centrelink_payment_type,
    )
    result, income_summary = add_income_stream_outputs(result)
    result = reorder_output_columns(result, original_cols)
    return IncomeClassificationResult(
        transactions=result,
        summary=income_summary,
        original_columns=tuple(original_cols),
    )
