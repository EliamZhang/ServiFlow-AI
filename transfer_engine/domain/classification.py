# -*- coding: utf-8 -*-
"""
Rule-based transfer classification for bank transactions.

Classifies transactions into:
- transfer

Uses ~70 regex rules in priority order (high confidence → medium confidence).
Rules support optional dr_cr direction constraints for P0 accuracy.

Important restrictions:
- Do NOT use trx_type / txn_type / txn_type_category as input features.
- Do NOT use third_party as an input feature.
- Classification is based on text + dr_cr only.

This module is invoked by the unified engine pipeline.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from .transfer_counterparty import load_counterparty_rules
from .transfer_rules import (
    ExclusionRule,
    load_exclusion_rules,
)

# Default paths to knowledge-base CSV files.
_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
_DEFAULT_RULES_FILE = _RESOURCES_DIR / "transfer_counterparty_rules.csv"
_EXCLUSION_RULES_FILE = _RESOURCES_DIR / "transfer_pairing_exclusions.csv"
_HIGH_CONFIDENCE_FILE = _RESOURCES_DIR / "transfer_external_high_confidence_rules.csv"
_MEDIUM_CONFIDENCE_FILE = _RESOURCES_DIR / "transfer_external_medium_confidence_rules.csv"
_INTERNAL_REGEX_FILE = _RESOURCES_DIR / "transfer_internal_regex_rules.csv"
_INDICATOR_PATTERNS_FILE = _RESOURCES_DIR / "transfer_indicator_patterns.csv"
_GROUP_EXCLUSION_FILE = _RESOURCES_DIR / "transfer_group_exclusion_patterns.csv"
_ROW_EXCLUSION_FILE = _RESOURCES_DIR / "transfer_row_exclusion_patterns.csv"

# Module-level caches — loaded once on first use.
_COUNTERPARTY_RULES: _KeywordRuleList | None = None
_EXCLUSION_RULES: list[ExclusionRule] | None = None
_HIGH_CONFIDENCE_CACHE: list[_RuleDef] | None = None
_MEDIUM_CONFIDENCE_CACHE: list[_RuleDef] | None = None
_INTERNAL_REGEX_CACHE: list[tuple[str, str, str | None]] | None = None
_INDICATOR_PATTERNS_CACHE: list[re.Pattern] | None = None
_GROUP_EXCLUSION_CACHE: list[re.Pattern] | None = None
_ROW_EXCLUSION_CACHE: list[re.Pattern] | None = None


def _get_counterparty_rules() -> _KeywordRuleList:
    """Return the loaded counterparty rules, loading from CSV if needed."""
    global _COUNTERPARTY_RULES
    if _COUNTERPARTY_RULES is None:
        _COUNTERPARTY_RULES = load_counterparty_rules(str(_DEFAULT_RULES_FILE))
    return _COUNTERPARTY_RULES


def _get_exclusion_rules() -> list[ExclusionRule]:
    """Return the loaded CSV pairing-exclusion rules (lazy)."""
    global _EXCLUSION_RULES
    if _EXCLUSION_RULES is None:
        _EXCLUSION_RULES = load_exclusion_rules(str(_EXCLUSION_RULES_FILE))
    return _EXCLUSION_RULES


# =============================================================================
# CSV rule loading functions (following liability engine pattern)
# =============================================================================

def _load_rules_csv(file_path: str | Path) -> list[_RuleDef]:
    """Load classification rules from a CSV file.

    CSV columns: priority, rule_name, category, pattern, dr_cr, description
    Returns list of (rule_name, category, pattern, dr_cr_or_None) sorted by priority.
    """
    rules: list[tuple[int, str, str, str, str | None]] = []
    with open(file_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_name = str(row.get("rule_name", "")).strip()
            category = str(row.get("category", "")).strip()
            pattern = str(row.get("pattern", "")).strip()
            dr_cr = str(row.get("dr_cr", "")).strip() or None
            priority = int(row.get("priority", 0))
            if not rule_name or not pattern:
                continue
            rules.append((priority, rule_name, category, pattern, dr_cr))
    rules.sort(key=lambda r: r[0])
    return [(name, cat, pat, dr) for _, name, cat, pat, dr in rules]


def _load_pattern_list(file_path: str | Path) -> list[re.Pattern]:
    """Load compiled regex patterns from a CSV file.

    CSV must have a 'pattern' column. Returns compiled patterns in file order.
    """
    patterns: list[re.Pattern] = []
    with open(file_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pattern = str(row.get("pattern", "")).strip()
            if not pattern:
                continue
            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                continue
    return patterns


def _get_high_confidence_rules() -> list[_RuleDef]:
    global _HIGH_CONFIDENCE_CACHE
    if _HIGH_CONFIDENCE_CACHE is None:
        _HIGH_CONFIDENCE_CACHE = _load_rules_csv(_HIGH_CONFIDENCE_FILE)
    return _HIGH_CONFIDENCE_CACHE


def _get_medium_confidence_rules() -> list[_RuleDef]:
    global _MEDIUM_CONFIDENCE_CACHE
    if _MEDIUM_CONFIDENCE_CACHE is None:
        _MEDIUM_CONFIDENCE_CACHE = _load_rules_csv(_MEDIUM_CONFIDENCE_FILE)
    return _MEDIUM_CONFIDENCE_CACHE


def _get_internal_regex_rules() -> list[tuple[str, str, str | None]]:
    global _INTERNAL_REGEX_CACHE
    if _INTERNAL_REGEX_CACHE is None:
        rules = _load_rules_csv(_INTERNAL_REGEX_FILE)
        _INTERNAL_REGEX_CACHE = [(name, pat, dr) for name, _cat, pat, dr in rules]
    return _INTERNAL_REGEX_CACHE


def _get_indicator_patterns() -> list[re.Pattern]:
    global _INDICATOR_PATTERNS_CACHE
    if _INDICATOR_PATTERNS_CACHE is None:
        _INDICATOR_PATTERNS_CACHE = _load_pattern_list(_INDICATOR_PATTERNS_FILE)
    return _INDICATOR_PATTERNS_CACHE


def _get_group_exclusion_patterns() -> list[re.Pattern]:
    global _GROUP_EXCLUSION_CACHE
    if _GROUP_EXCLUSION_CACHE is None:
        _GROUP_EXCLUSION_CACHE = _load_pattern_list(_GROUP_EXCLUSION_FILE)
    return _GROUP_EXCLUSION_CACHE


def _get_row_exclusion_patterns() -> list[re.Pattern]:
    global _ROW_EXCLUSION_CACHE
    if _ROW_EXCLUSION_CACHE is None:
        _ROW_EXCLUSION_CACHE = _load_pattern_list(_ROW_EXCLUSION_FILE)
    return _ROW_EXCLUSION_CACHE


# =============================================================================
# Fixed rule configuration (lazy-loaded from CSV, following liability pattern)
# =============================================================================

# Rules are ordered from more explicit to more ambiguous.
# dr_cr_constraint = None       -> match regardless of dr_cr
# dr_cr_constraint = "debit"    -> only match when dr_cr is debit
# dr_cr_constraint = "credit"   -> only match when dr_cr is credit

# ── Resolve at module level (CSV-first, lazy-loaded on first access) ──────

HIGH_CONFIDENCE_RULES = _get_high_confidence_rules()
MEDIUM_CONFIDENCE_RULES = _get_medium_confidence_rules()
INTERNAL_TRANSFER_RULES = _get_internal_regex_rules()
_TRANSFER_INDICATOR_PATTERNS = _get_indicator_patterns()
_GROUP_EXCLUSION_PATTERNS = _get_group_exclusion_patterns()
_ROW_EXCLUSION_PATTERNS = _get_row_exclusion_patterns()


# =============================================================================
# Text normalization & type aliases
# =============================================================================

from typing import List, Optional, Tuple

_RuleDef = Tuple[str, str, str, Optional[str]]
_KeywordRuleList = List[Tuple[List[str], str]]


def normalize_text(value: object) -> str:
    """Normalize text so rule matching is stable."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


# =============================================================================
# Pipeline entry point
# =============================================================================

# Patterns for account-aware deposit matching.
_WITHDRAWAL_ACCOUNT_RE = re.compile(
    r"\binternet withdrawal\b.*\bto (\d{7,})\b", re.IGNORECASE
)
_DEPOSIT_ACCOUNT_RE = re.compile(
    r"\binternet deposit\b.*\bfrom (\d{7,})\b", re.IGNORECASE
)


def classify_transfers(
    df: pd.DataFrame, *, all_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply transfer classification rules and produce output columns.

    Pipeline:
    1. Internal Transfer — data-driven pairing rule
       (same application_id + transaction_date + amount, debit & credit).
    1.5 Internal Transfer — high-confidence regex rules on remaining rows.
    1.6 CSV knowledge-base rules — Internal/External distinction patterns
        that supplement the hardcoded regex rules.
    2. External Transfers — regex rules on remaining unclassified rows.
    2.5 Personal Osko credit filter — remove informal personal credits from ET.
    3. Known-account deposit matching — extends External Transfers.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate rows the engine may write to.
    all_rows : pd.DataFrame or None
        Full dataset (including rows already claimed by other engines).
        Used for Internal Transfer pairing so pairs are detected even when
        one side has been classified by an earlier pipeline stage.
    """
    output = df.copy()

    # ── initialize columns ──
    raw_text = output.get("text", pd.Series("", index=output.index))
    output["text_norm"] = raw_text.apply(normalize_text)
    output["is_transfer_pred"] = 0
    output["finv_category"] = ""
    output["prediction_confidence"] = ""
    output["prediction_rule"] = ""
    output["prediction_dr_cr_used"] = False

    # ── Step 1: Internal Transfer — pairing logic on ALL rows ──
    pairing_pool = all_rows if all_rows is not None else output
    output = _detect_internal_transfers(output, pairing_pool)

    # ── Step 1.5: Internal Transfer — regex rules ──
    output = _detect_internal_by_regex(output)

    # ── Step 2: External Transfers — regex on remaining rows ──
    output = _detect_external_transfers(output)

    # ── Step 2.5: filter personal Osko credits ──
    output = _filter_personal_osko_credits(output)

    # ── Step 3: extend via known internal accounts (also External Transfers) ──
    output = _match_deposit_to_known_accounts(output)

    # ── counterparty ──
    output["counterparty"] = _derive_counterparty(output)

    # ── rule metadata ──
    output["transfer_pred_reason"] = _build_reason_vectorised(output)

    # ── stream id ──
    output["stream_id"] = output["finv_category"].where(
        output["is_transfer_pred"].eq(1), ""
    )

    return output


def _detect_internal_transfers(
    df: pd.DataFrame, pairing_pool: pd.DataFrame,
) -> pd.DataFrame:
    """Detect Internal Transfers via the pairing rule (fully vectorised).

    Groups by (application_id, transaction_date, amount).  If a group
    contains both ``debit`` and ``credit``, candidate rows whose text
    passes the transfer-indicator check are marked as Internal Transfer.

    Groups are excluded when any row matches gambling / lender keywords.
    """
    output = df.copy()
    if output.empty:
        return output

    # Merge extra rows from pairing_pool for full context.
    extra_rows = pairing_pool[~pairing_pool.index.isin(df.index)]
    combined = df if len(extra_rows) == 0 else pd.concat([df, extra_rows])

    # ── Build group key ──
    pair_key = (
        combined["application_id"].astype(str) + "||"
        + combined["transaction_date"].astype(str) + "||"
        + combined["amount"].astype(str)
    )

    # ── Per-group: has both debit AND credit? ──
    dr_cr_lower = combined["dr_cr"].fillna("").astype(str).str.lower()
    group_has_debit = dr_cr_lower.eq("debit").groupby(pair_key, sort=False).transform("any")
    group_has_credit = dr_cr_lower.eq("credit").groupby(pair_key, sort=False).transform("any")
    valid_pair = group_has_debit & group_has_credit

    if not valid_pair.any():
        return output

    # ── Per-group: any row matches exclusion keywords? ──
    texts = combined.get("text", pd.Series("", index=combined.index)).fillna("").astype(str)
    if _COMBINED_EXCLUSION_RE is not None:
        has_exclusion = texts.str.contains(_COMBINED_EXCLUSION_RE, na=False, regex=True)
        group_excluded = has_exclusion.groupby(pair_key, sort=False).transform("any")
    else:
        group_excluded = pd.Series(False, index=combined.index)

    # CSV exclusion rules: vectorised per-row, then per-group. (only ~5 rules)
    csv_exclusions = _get_exclusion_rules()
    if csv_exclusions:
        for excl in csv_exclusions:
            if excl.match_type == "regex" and excl._compiled is not None:
                hit = texts.str.contains(excl._compiled, na=False, regex=True)
            else:
                # Keyword match: split semicolons, check each
                hit = pd.Series(False, index=combined.index)
                for kw in excl.keyword_raw.split(";"):
                    kw = kw.strip().upper()
                    if kw:
                        hit |= texts.str.contains(kw, na=False, regex=False)
            group_excluded = group_excluded | hit.groupby(pair_key, sort=False).transform("any")

    # ── Per-row: transfer indicator + row exclusion checks ──
    looks_like = _looks_like_transfer(texts)
    row_excluded = _matches_row_exclusion(texts)

    # ── Final mask: only candidate rows in df ──
    is_candidate = combined.index.isin(df.index)
    internal_mask = (
        is_candidate & valid_pair & ~group_excluded & looks_like & ~row_excluded
    )

    if internal_mask.any():
        output.loc[internal_mask[df.index], "is_transfer_pred"] = 1
        output.loc[internal_mask[df.index], "finv_category"] = "Internal Transfer"
        output.loc[internal_mask[df.index], "prediction_confidence"] = "high"
        output.loc[internal_mask[df.index], "prediction_rule"] = "internal_pairing_rule"

    return output


# Pre-compiled combined exclusion pattern for vectorised group checking.
_COMBINED_EXCLUSION_RE = re.compile(
    "|".join(p.pattern for p in _GROUP_EXCLUSION_PATTERNS),
    re.IGNORECASE,
) if _GROUP_EXCLUSION_PATTERNS else None

# Pre-combined patterns for vectorised per-group checks.
_COMBINED_INDICATOR_RE = re.compile(
    "|".join(p.pattern for p in _TRANSFER_INDICATOR_PATTERNS),
    re.IGNORECASE,
) if _TRANSFER_INDICATOR_PATTERNS else None

_COMBINED_ROW_EXCLUSION_RE = re.compile(
    "|".join(p.pattern for p in _ROW_EXCLUSION_PATTERNS),
    re.IGNORECASE,
) if _ROW_EXCLUSION_PATTERNS else None


def _looks_like_transfer(texts: pd.Series) -> pd.Series:
    """Return True for rows containing at least one transfer-indicator pattern."""
    if _COMBINED_INDICATOR_RE is None:
        return pd.Series(False, index=texts.index)
    return texts.fillna("").astype(str).str.contains(
        _COMBINED_INDICATOR_RE, na=False, regex=True
    )


def _matches_row_exclusion(texts: pd.Series) -> pd.Series:
    """Return True for rows matching a row-level P2P exclusion pattern."""
    if _COMBINED_ROW_EXCLUSION_RE is None:
        return pd.Series(False, index=texts.index)
    return texts.fillna("").astype(str).str.contains(
        _COMBINED_ROW_EXCLUSION_RE, na=False, regex=True
    )


def _match_rules(
    df: pd.DataFrame,
    rules: list[tuple[str, str, str | None, str]],  # (name, pattern, dr_cr, confidence)
    category_label: str,
    *,
    per_rule_exclusions: dict[str, re.Pattern] | None = None,
) -> pd.DataFrame:
    """Apply regex rules to unclassified rows (vectorised).

    Only rows with ``is_transfer_pred == 0`` are considered.  Matched rows
    receive *category_label* as their ``finv_category``.
    """
    output = df.copy()
    remaining_mask = output["is_transfer_pred"] == 0
    if not remaining_mask.any():
        return output

    text_col = output.get("text_norm", pd.Series("", index=output.index))
    dr_cr_col = output.get("dr_cr", pd.Series("", index=output.index))

    import warnings
    _warn_msg = "This pattern is interpreted as a regular expression"

    for rule_name, pattern, dr_cr_constraint, confidence in rules:
        if not remaining_mask.any():
            break

        remaining_idx = output.index[remaining_mask]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_warn_msg)
            matched = text_col.loc[remaining_idx].str.contains(pattern, na=False, regex=True)
        if not matched.any():
            continue

        matched_idx = remaining_idx[matched]

        # dr_cr constraint
        if dr_cr_constraint is not None:
            dr_cr_ok = (
                dr_cr_col.loc[matched_idx].astype(str).str.strip().str.lower()
                == dr_cr_constraint
            )
            matched_idx = matched_idx[dr_cr_ok.values]
            if len(matched_idx) == 0:
                continue

        # Per-rule exclusion (e.g. INTL-FEE for internal transfers)
        if per_rule_exclusions and rule_name in per_rule_exclusions:
            exclude = text_col.loc[matched_idx].str.contains(
                per_rule_exclusions[rule_name], na=False, regex=True
            )
            matched_idx = matched_idx[~exclude]
            if len(matched_idx) == 0:
                continue

        # Assign predictions
        output.loc[matched_idx, "is_transfer_pred"] = 1
        output.loc[matched_idx, "finv_category"] = category_label
        output.loc[matched_idx, "prediction_confidence"] = confidence
        output.loc[matched_idx, "prediction_rule"] = rule_name
        output.loc[matched_idx, "prediction_dr_cr_used"] = dr_cr_constraint is not None

        remaining_mask.loc[matched_idx] = False

    return output


def _detect_external_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect External Transfers via regex rules (vectorised)."""
    rules: list[tuple[str, str, str | None, str]] = [
        (name, pat, dr_cr, "high") for name, _cat, pat, dr_cr in HIGH_CONFIDENCE_RULES
    ] + [
        (name, pat, dr_cr, "medium") for name, _cat, pat, dr_cr in MEDIUM_CONFIDENCE_RULES
    ]
    return _match_rules(df, rules, "External Transfers")


def _detect_internal_by_regex(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Internal Transfers via regex rules (vectorised)."""
    rules: list[tuple[str, str, str | None, str]] = [
        (name, pat, dr_cr, "high") for name, pat, dr_cr in INTERNAL_TRANSFER_RULES
    ]
    _intl_fee = re.compile(r"\bINTL[-\s]?FEE\b", re.IGNORECASE)
    exclusions = {
        "internal_anz_funds_tfer": _intl_fee,
        "internal_internet_banking": _intl_fee,
    }
    return _match_rules(df, rules, "Internal Transfer", per_rule_exclusions=exclusions)


def _match_deposit_to_known_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify internet deposit rows whose counterparty account is already
    known from a withdrawal match (vectorised)."""
    output = df.copy()
    classified_mask = output["is_transfer_pred"] == 1
    if not classified_mask.any():
        return output

    bank_col = output.get("bank_account_id", pd.Series("__global__", index=output.index))
    bank_col = bank_col.fillna("__global__").astype(str)
    text_col = output.get("text_norm", pd.Series("", index=output.index))

    # Collect known accounts from withdrawal patterns (vectorised extract).
    withdrawal_texts = text_col[classified_mask]
    extracted = withdrawal_texts.str.extract(_WITHDRAWAL_ACCOUNT_RE, expand=False)
    valid = extracted.notna()
    if not valid.any():
        return output

    known_by_bank: dict[str, set[str]] = {}
    for bank, account in zip(
        bank_col[classified_mask][valid].values,
        extracted[valid].values,
    ):
        known_by_bank.setdefault(bank, set()).add(str(account))

    # Build a bank→account→True lookup for vectorised matching.
    all_accounts: set[str] = set()
    for accounts in known_by_bank.values():
        all_accounts.update(accounts)

    if not all_accounts:
        return output

    # Scan unclassified rows for deposit patterns with known accounts.
    unclassified_mask = output["is_transfer_pred"] == 0
    deposit_texts = text_col[unclassified_mask]
    deposit_extracted = deposit_texts.str.extract(_DEPOSIT_ACCOUNT_RE, expand=False)
    deposit_valid = deposit_extracted.notna()

    if not deposit_valid.any():
        return output

    # Build mask: deposit account is known for this bank (or globally).
    match_mask = pd.Series(False, index=output.index)
    for idx in deposit_valid[deposit_valid].index:
        account = str(deposit_extracted[idx])
        bank = bank_col[idx]
        if account in known_by_bank.get(bank, set()) or account in known_by_bank.get("__global__", set()):
            match_mask[idx] = True

    if match_mask.any():
        output.loc[match_mask, ["is_transfer_pred", "finv_category",
                                 "prediction_confidence", "prediction_rule"]] = [
            1, "External Transfers", "high",
            "internal_internet_deposit_known_account",
        ]
        output.loc[match_mask, "prediction_dr_cr_used"] = False

    return output


def _filter_personal_osko_credits(df: pd.DataFrame) -> pd.DataFrame:
    """Unmark Osko credit rows that look like informal person-to-person payments."""
    output = df.copy()

    et_credit_mask = (
        (output["is_transfer_pred"] == 1)
        & (output["finv_category"] == "External Transfers")
        & (output["dr_cr"].astype(str).str.lower() == "credit")
    )
    if not et_credit_mask.any():
        return output

    text_col = output.get("text_norm", pd.Series("", index=output.index))
    raw_text = output.get("text", pd.Series("", index=output.index))

    # Must contain "osko"
    osko_mask = et_credit_mask & text_col.str.contains(r"\bosko\b", na=False, regex=True, case=False)
    if not osko_mask.any():
        return output

    # Has a 6+ digit reference number?
    has_ref = text_col.str.contains(r"\d{6,}", na=False, regex=True)

    # Has a 6+ digit number followed by a person name (Title Case)?
    has_person_name = raw_text.str.contains(
        r"\d{6,}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", na=False, regex=True
    )

    # Unmark: no ref number at all, OR has ref but with person name
    unmark = osko_mask & (~has_ref | (has_ref & has_person_name))

    if unmark.any():
        output.loc[unmark, ["is_transfer_pred", "finv_category",
                            "prediction_confidence", "prediction_rule"]] = [0, "", "", ""]
        output.loc[unmark, "prediction_dr_cr_used"] = False

    return output


# =============================================================================
# Counterparty extraction
# =============================================================================
#
# Uses a data-driven CSV knowledge base (``transfer_counterparty_rules.csv``)
# to derive standardised counterparty (third_party) labels for ALL transfer
# rows — both Internal and External Transfers follow the same keyword-matching
# logic.  First match wins.
#
#   Any Transfer  →  label from knowledge base, or "Miscellaneous Funds Transfer"
#
# The knowledge base follows the same design as the liability engine's
# ``counterparty_keyword_rules.csv``: keywords are semicolon-separated and
# matched case-insensitively against normalised text.  Rules are applied in
# CSV row order — first match wins.  Unmatched rows fall back to
# ``"Miscellaneous Funds Transfer"``.


def _derive_counterparty(df: pd.DataFrame) -> pd.Series:
    """Extract a counterparty label matching third_party naming conventions.

    Counterparty is determined **exclusively** by keyword matching against
    the CSV knowledge base (``transfer_counterparty_rules.csv``), applied to
    ALL transfer rows regardless of Internal/External category.

    Rows that match no rule fall back to ``"Miscellaneous Funds Transfer"``.

    Optimisation: vectorised ``str.contains`` instead of Python row-by-row loop.
    """
    text_col = df.get("text_norm", df.get("text", pd.Series("", index=df.index)))
    rules = _get_counterparty_rules()

    # Normalise all text once: uppercase + collapse whitespace.
    text_upper = (
        text_col.astype(str).str.strip().str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )

    result = pd.Series("Miscellaneous Funds Transfer", index=df.index)
    matched_mask = pd.Series(False, index=df.index)

    for keywords, counterparty in rules:
        if matched_mask.all():
            break

        # Build a mask for unmatched rows that match any keyword of this rule.
        rule_mask = pd.Series(False, index=df.index)
        unmatched_text = text_upper.loc[~matched_mask]

        for kw in keywords:
            kw_norm = re.sub(r"\s+", " ", kw.strip().upper())
            if not kw_norm:
                continue
            kw_match = unmatched_text.str.contains(kw_norm, na=False, regex=False)
            rule_mask.loc[~matched_mask] = (
                rule_mask.loc[~matched_mask] | kw_match.values
            )

        newly_matched = rule_mask & ~matched_mask
        if newly_matched.any():
            result.loc[newly_matched] = counterparty
            matched_mask |= newly_matched

    return result


def _build_reason_vectorised(df: pd.DataFrame) -> pd.Series:
    """Build ``transfer_pred_reason`` column (vectorised)."""
    is_transfer = df["is_transfer_pred"].astype(int).eq(1)

    result = pd.Series("", index=df.index)

    # Non-transfer rows
    not_transfer = ~is_transfer
    if not_transfer.any():
        result.loc[not_transfer] = "category=not_transfer; rule=no_transfer_rule_matched"

    # Transfer rows
    if is_transfer.any():
        idx = df.index[is_transfer]
        conf = df.loc[idx, "prediction_confidence"].astype(str)
        rule = df.loc[idx, "prediction_rule"].fillna("").astype(str)
        cat = df.loc[idx, "finv_category"].astype(str)
        dr_cr_flag = df.loc[idx, "prediction_dr_cr_used"].fillna(False).astype(bool)

        base = (
            "category=" + cat.str.replace(";", " ").replace("=", " ")
            + "; rule=" + rule.str.replace(";", " ").replace("=", " ")
            + "; evidence=confidence=" + conf
        )
        if dr_cr_flag.any():
            base.loc[dr_cr_flag] += ", dr_cr_used"

        result.loc[idx] = base

    return result
