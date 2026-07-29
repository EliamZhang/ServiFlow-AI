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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from classification_core.reasons import format_classification_reason

from .transfer_counterparty import (
    load_counterparty_rules,
    match_counterparty,
)
from .transfer_rules import (
    ExclusionRule,
    load_exclusion_rules,
    matches_any_exclusion_in_group,
)

# Default paths to knowledge-base CSV files.
_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
_DEFAULT_RULES_FILE = _RESOURCES_DIR / "transfer_counterparty_rules.csv"
_EXCLUSION_RULES_FILE = _RESOURCES_DIR / "transfer_pairing_exclusions.csv"

# Module-level caches — loaded once on first use.
_COUNTERPARTY_RULES: _KeywordRuleList | None = None
_EXCLUSION_RULES: list[ExclusionRule] | None = None


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
# Fixed rule configuration
# =============================================================================

# Rules are ordered from more explicit to more ambiguous.
# dr_cr_constraint = None       -> match regardless of dr_cr
# dr_cr_constraint = "debit"    -> only match when dr_cr is debit
# dr_cr_constraint = "credit"   -> only match when dr_cr is credit

HIGH_CONFIDENCE_RULES = [
    # -------------------------------------------------------------------------
    # Transfer: well-defined merchant/payment patterns (P0 additions)
    # -------------------------------------------------------------------------
    ("external_direct_debit_generic", "transfer", r"^direct debit$", "debit"),
    ("external_transferwise_sydney", "transfer", r"\btransferwise sydney\b", None),
    ("external_debit_card_wise_taptap", "transfer", r"^debit card purchase (wise|taptap send) sydney", None),
    ("external_fast_pymt_in", "transfer", r"^fast pymt in", None),
    ("external_withdrawal_westpac_cho_loan", "transfer", r"^withdrawal mobile \d+ tfr westpac cho loan$", None),
    # IMT = International Money Transfer (common format: "IMT <ref> <name> REF <code> <currency> Rate_...")
    ("external_imt", "transfer", r"^imt \d+", None),
    # ANZ Internet Banking funds transfer with INTL-FEE = international (cross-border).
    # These are explicitly excluded from Internal Transfer by _detect_internal_by_regex.
    ("external_anz_intl_fee", "transfer", r"\banz internet banking funds tfer\b.*\bintl[-\s]?fee\b", None),

    # -------------------------------------------------------------------------
    # Transfer: named mobile/online payments and well-defined syntax
    # -------------------------------------------------------------------------
    ("external_anz_mobile_payment_to_name", "transfer", r"^anz mobile banking payment \d+ to [a-z].*$", None),
    ("external_sav_mobile", "transfer", r"^transfer (to|from) sav \d+ mobile#\d+$", None),
    ("external_sav_ref", "transfer", r"^transfer (to|from) sav \d+ ref#\d+$", None),
    ("external_sav_ref_xxx", "transfer", r"^transfer from sav \d+xxx ref#\d+$", None),
    ("external_anz_mbank_masked", "transfer", r"^anz m-banking funds tfer transfer \d+ to \d+xxxxxxxx\d{4}$", None),
    ("external_statement_or_savings_account", "transfer", r"^(statement account|savings account)$", None),
    ("external_transfer_name_code", "transfer", r"^transfer (debit|credit) [a-z][a-z .'-]*[a-z] [a-z]\d{10,}$", None),
    ("external_credit_adjustment", "transfer", r"^credit adjustment$", None),
    ("external_misc_credit_v", "transfer", r"^miscellaneous credit v\d{4}$", None),
    ("external_payid_osko", "transfer", r"\b(payid|osko|npp)\b", None),
    ("external_fast_transfer", "transfer", r"\bfast transfer\b", None),
    ("external_paypal_worldremit", "transfer", r"\b(paypal australia|worldremit)\b", None),
    ("external_commbank_name", "transfer", r"\btransfer (to|from) [a-z][a-z .'-]+ commbank app\b", None),

    # Transfer: stable merchant/payment patterns confirmed by extraction.
    ("external_paypal_direct_debit", "transfer", r"^direct debit \d+ paypal australia \d+$", None),
    ("external_paypal_authority", "transfer", r"^payment by authority to paypal australia \d+$", None),
    ("external_paypal_payment", "transfer", r"^payment to paypal australia \d+$", None),
    ("external_paypal_account", "transfer", r"^paypal australia \d+$", None),
    ("external_tfr_from_s_mob", "transfer", r"^tfr from \d+s\d+ mob$", None),
    ("external_tfr_from_long_mob", "transfer", r"^tfr from \d+ mob$", None),
    ("external_worldwide_wallet", "transfer", r"^deposit online \d+ \d+ tfr worldwide wallet$", None),
    ("external_wu", "transfer", r"^wu \d+$", None),
    ("external_transfer_funds_trns", "transfer", r"^transfer debit internet transfer funds trns$", None),
    ("external_wise_card", "transfer", r"^wise sydney au aus card xx\d{4} value date: \d+/\d+/\d+$", None),
    ("external_taptap_send", "transfer", r"^taptap send sydney aus card xx\d{4} value date: \d+/\d+/\d+$", None),
    ("external_online_withdrawal_worldwide_wallet", "transfer", r"^withdrawal online \d+ \d+ tfr worldwide wallet$", None),

    # Mobile transfers to/from individuals.
    ("external_mobile_transfer", "transfer", r"\btfr (to|from) .*\b(mob|mobile)\b", None),
    ("external_withdrawal_mobile_pymt", "transfer", r"\bwithdrawal mobile\b.*\bpymt\b", None),

    # ── NAB direct transfer credit/debit (P0: counterparty patterns promoted) ──
    ("external_nab_transfer_credit", "transfer", r"^transfer credit (?!online\b)", None),
    ("external_nab_transfer_debit", "transfer", r"^transfer debit (?!online\b)", None),

    # ── Internet banking transfers ──
    ("external_internet_banking", "transfer", r"^internet (?:withdrawal|deposit|external transfer)\b", None),

    # ── TFR to/from (bank transfers) ──
    ("external_tfr_to_from", "transfer", r"\btfr (?:to|from)\b", None),

    # ── Nabpay (NAB BPAY variant) ──
    ("external_nabpay", "transfer", r"\bnabpay\d+", None),

    # ── Phone/Internet banking transfer ──
    ("external_phone_internet_tfr", "transfer", r"\bphone/internet tfr\b", None),

    # ── Transferred to/from + numeric reference ──
    ("external_transferred_to_from_num", "transfer", r"\btransferred (?:to|from) \d", None),
]

MEDIUM_CONFIDENCE_RULES = [
    # Westpac truncated text (text cut off at "Westpa c" instead of "Westpac Choice").
    ("external_westpac_truncated", "transfer", r"\bwestpa\s", None),

    # Transfer medium patterns.
    ("external_transferred_to_digits", "transfer", r"\btransferred to \d{3,6} \d+\b", None),
    ("external_sav_net", "transfer", r"^transfer (to|from) sav \d+ net#\d+$", None),
    ("external_scheduled_cba_account", "transfer", r"^scheduled payment to a cba account \d+ \d+$", None),
    ("external_anz_mbank_transfer_from", "transfer", r"^anz m-banking transfer \d+ from \d+$", None),

    # Transfer medium patterns (continued).
    ("internal_internet_transfer_credit", "transfer", r"^internet transfer credit from \d+ ref no \d+$", None),
    ("internal_internet_transfer_debit", "transfer", r"^internet transfer debit to \d+ reference no \d+$", None),
    ("internal_ib_transfer_tfd", "transfer", r"^ib transfer \d+ to \d{3}-\d{3}-\d+ \d+:\d+(?:am|pm) tfd$", None),
    ("internal_ib_transfer_tfc", "transfer", r"^ib transfer \d+ from \d{3}-\d{3}-\d+ \d+:\d+(?:am|pm) tfc$", None),
    ("internal_transfer_to_cba_ac", "transfer", r"^transfer to cba a/c commbank app\b", None),
    ("internal_transfer_from_commbank", "transfer", r"^transfer from commbank app\b", None),
    ("internal_masked_commbank", "transfer", r"\btransfer (to|from) xx\d{4}\b", None),
    # Westpac withdrawal patterns: debit direction, still mostly transfer (92-98%).
    # NOTE: moved to INTERNAL_TRANSFER_RULES — Westpac Choice/Life/Maximiser
    # are account types that indicate own-account (internal) transfers.
    ("internal_tfd", "transfer", r"^\. tfd$", None),
    ("internal_tfc", "transfer", r"^\. tfc$", None),
    ("internal_me_tfd", "transfer", r"^me tfd$", None),
    ("internal_x_tfc", "transfer", r"^x tfc$", None),
    ("internal_save_tfc", "transfer", r"^save tfc$", None),
    ("internal_j_tfd", "transfer", r"^j tfd$", None),
    # Letter-prefix TFD/TFC variants (Westpac/St George shorthand).
    # "b TFD" / "b TFC" — common single-letter prefix indicating transfer type.
    ("internal_b_tfd", "transfer", r"^b tfd$", None),
    ("internal_b_tfc", "transfer", r"^b tfc$", None),
    ("internal_n_tfd", "transfer", r"^n tfd$", None),
    ("internal_h_tfd", "transfer", r"^h tfd$", None),
    # Automatic payment / sweep between linked accounts — looks like a transfer.
    ("external_automatic_payment", "transfer", r"^automatic payment$", None),
    # Periodic / scheduled transfers between accounts.
    ("external_periodic_transfer", "transfer", r"^periodic transfer from\b", None),
    # RTGS (Real-Time Gross Settlement) — high-value interbank transfer.
    ("external_rtgs", "transfer", r"^rtgs funds credit$", None),

    # Relaxed name+code: allows extra text after the transaction code.
    # Precision ~95.3% — kept as medium due to some lookalikes.
    ("external_transfer_name_code_relaxed", "transfer", r"^transfer (debit|credit) (?!online )[a-z].*?[a-z]\d{10,}", None),
]

# ── Internal Transfer regex rules ──────────────────────────────────────
# These patterns strongly indicate own-account (internal) transfers.
# Applied after the pairing rule but before external-transfer regex, so
# rows matching these rules are classified as Internal Transfer rather
# than External Transfers.

INTERNAL_TRANSFER_RULES: list[tuple[str, str, str | None]] = [
    # Westpac deposit patterns: always credit, 95-98% transfer.
    ("internal_deposit_westpac_cho", r"^deposit online \d+ tfr westpac cho", "credit"),
    ("internal_deposit_westpac_lif", r"^deposit online \d+ tfr westpac lif", "credit"),
    ("internal_deposit_westpac_esa", r"^deposit online \d+ tfr westpac esa", "credit"),
    # Commbank app masked account with descriptive suffixes.
    ("internal_commbank_from_suffix", r"^transfer from xx\d{4} commbank app (allowance|rent|savings|transfer|food|bill|shop)$", None),
    ("internal_commbank_from_value_date", r"^transfer from xx\d{4} commbank app value date: \d+/\d+/\d+$", None),
    # Standard internal-transfer narratives.
    ("internal_phrase", r"\binternal transfer\b", None),
    ("internal_linked_acc", r"\b(linked acc|linked account|acc trns|acc transfer)\b", None),
    # Westpac account-type indicators: Choice / Life / Maximiser are the
    # user's own accounts — transfers between them are internal.
    ("internal_westpac_family", r"\b(tfr westpac|westpac lif|maximiser)\b", None),
    ("internal_anz_funds_tfer", r"\banz internet banking funds tfer\b", None),
    ("internal_anz_mbank_to_long", r"^anz m-banking funds tfer transfer \d+ to \d+$", None),
    ("internal_anz_mbank_from_long", r"^anz m-banking funds tfer transfer \d+ from \d+$", None),
    ("internal_ib_tfr_to_long", r"^ib tfr \d+ to \d+$", None),
    ("internal_mb_transfer_from", r"^mb transfer from \d+$", None),
    ("internal_mb_transfer_to", r"^mb transfer to \d+$", None),
    ("internal_orange_everyday_to", r"^internal transfer - receipt \d+ - to orange everyday$", None),
    ("internal_orange_everyday_from", r"^internal transfer - receipt \d+ - from orange everyday$", None),
    ("internal_savings_maximiser", r"^internal transfer - internal transfer - receipt \d+ savings maximiser \d+$", None),
    ("internal_ibank_mobile_banking", r"^ibank trf ref: \d+ transferred to \d+ mobile banking$", None),
    # Internet banking withdrawal to own account.
    ("internal_internet_banking", r"\binternet withdrawal\b.*\bto \d{7,}\b", None),
]


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class Prediction:
    category: str | None
    confidence: str | None
    rule_name: str | None
    dr_cr_used: bool = False


# =============================================================================
# Text normalization
# =============================================================================

def normalize_text(value: object) -> str:
    """Normalize text so rule matching is stable."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


# =============================================================================
# Fixed rule classifier
# =============================================================================

_RuleDef = tuple[str, str, str, str | None]
_CompiledRule = tuple[str, str, re.Pattern, str | None]
_KeywordRuleList = list[tuple[list[str], str]]


class FixedRuleClassifier:
    """Apply the permanently confirmed rule list in priority order.

    Rules are 4-tuples: (name, category, pattern, dr_cr_constraint).
    dr_cr_constraint: None (ignore), "debit" (only debit), "credit" (only credit).
    """

    def __init__(self) -> None:
        self.high_rules = self._compile_rules(HIGH_CONFIDENCE_RULES)
        self.medium_rules = self._compile_rules(MEDIUM_CONFIDENCE_RULES)

    @staticmethod
    def _compile_rules(rules: list[_RuleDef]) -> list[_CompiledRule]:
        return [
            (name, category, re.compile(pattern, re.IGNORECASE), dr_cr)
            for name, category, pattern, dr_cr in rules
        ]

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()

        # Normalize text (handle missing column gracefully).
        raw_text = result.get("text", pd.Series("", index=result.index))
        result["text_norm"] = raw_text.apply(normalize_text)

        dr_cr_series = result.get("dr_cr", pd.Series([pd.NA] * len(result)))

        predictions = [
            self._predict(text, dr_cr)
            for text, dr_cr in zip(result["text_norm"], dr_cr_series)
        ]

        result["predicted_category"] = [p.category for p in predictions]
        result["prediction_confidence"] = [p.confidence for p in predictions]
        result["prediction_rule"] = [p.rule_name for p in predictions]
        result["prediction_dr_cr_used"] = [p.dr_cr_used for p in predictions]
        return result

    def _predict(self, text: str, dr_cr: object = None) -> Prediction:
        """Apply rules in priority order, respecting dr_cr constraints."""
        dr_cr_normalized: str | None = None
        if pd.notna(dr_cr):
            dr_cr_normalized = str(dr_cr).strip().lower()

        for rule_name, category, pattern, dr_cr_constraint in self.high_rules:
            if not pattern.search(text):
                continue
            if dr_cr_constraint is not None:
                if dr_cr_normalized != dr_cr_constraint:
                    continue
                return Prediction(category, "high", rule_name, dr_cr_used=True)
            return Prediction(category, "high", rule_name)

        for rule_name, category, pattern, dr_cr_constraint in self.medium_rules:
            if not pattern.search(text):
                continue
            if dr_cr_constraint is not None:
                if dr_cr_normalized != dr_cr_constraint:
                    continue
                return Prediction(category, "medium", rule_name, dr_cr_used=True)
            return Prediction(category, "medium", rule_name)

        return Prediction(None, None, None)


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
    output["predicted_category"] = ""
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
    output["transfer_rule_name"] = _build_transfer_rule_name(output)
    output["transfer_pred_reason"] = _build_reason_vectorised(output)

    # ── stream id ──
    output["stream_id"] = output["finv_category"].where(
        output["is_transfer_pred"].eq(1), ""
    )

    return output


# ── Text patterns that indicate a row is a genuine bank-account transfer ──
# Used to validate Internal Transfer candidates: if a row's text doesn't
# match ANY of these patterns, it is removed from the IT set (likely a
# coincidental amount match — e.g. a gambling debit that happens to share
# the same amount + date as an unrelated credit).

_TRANSFER_INDICATOR_PATTERNS: list[re.Pattern] = [
    re.compile(r"\btransfer\s+(?:debit|credit)\b", re.IGNORECASE),
    re.compile(r"\binternet\s+(?:withdrawal|deposit)\b", re.IGNORECASE),
    re.compile(r"\blinked\s+acc", re.IGNORECASE),
    re.compile(r"\binternal\s+transfer\b", re.IGNORECASE),
    re.compile(r"\b(?:ibank|ib)\s+trf\b", re.IGNORECASE),
    re.compile(r"\bmb\s+transfer\b", re.IGNORECASE),
    re.compile(r"\banz\s+m-banking\s+funds\s+tfer\b", re.IGNORECASE),
    re.compile(r"\b(?:deposit|withdrawal)\s+online\b.*\btfr\b", re.IGNORECASE),
    re.compile(r"\bfast\s+transfer\b", re.IGNORECASE),
    re.compile(r"\bcommbank\s+app\b", re.IGNORECASE),
    re.compile(r"\btfr\s+(?:from|to)\s+\d", re.IGNORECASE),
    re.compile(r"\btransfer\s+(?:to|from)\s+xx\d{4}\b", re.IGNORECASE),
    re.compile(r"\btransferred\s+(?:to|from)\s+\d", re.IGNORECASE),
    re.compile(r"\.[ \t]*tf[cd]\s*$", re.IGNORECASE),
    re.compile(r"\b(?:osko|payid|npp)\b", re.IGNORECASE),
    re.compile(r"\bpaypal\b", re.IGNORECASE),
    re.compile(r"\bbpay\b", re.IGNORECASE),
    re.compile(r"\bdirect\s+(?:debit|credit)\b", re.IGNORECASE),
    re.compile(r"\bscheduled\s+payment\b", re.IGNORECASE),
]


def _looks_like_transfer(text: str) -> bool:
    """Return True if *text* contains at least one transfer-indicator pattern."""
    if not text or pd.isna(text):
        return False
    text_str = str(text)
    return any(p.search(text_str) for p in _TRANSFER_INDICATOR_PATTERNS)


def _matches_row_exclusion(text: str) -> bool:
    """Return True if *text* matches a row-level P2P exclusion pattern.

    Row-level exclusions only exclude the matching row from IT, not the
    entire group — unlike group-level exclusions (gambling, BNPL, etc.)
    which invalidate the entire pairing group.
    """
    if not text or pd.isna(text) or not str(text).strip():
        return False
    text_str = str(text)
    for pattern in _ROW_EXCLUSION_PATTERNS:
        if pattern.search(text_str):
            return True
    return False


def _detect_internal_transfers(
    df: pd.DataFrame, pairing_pool: pd.DataFrame,
) -> pd.DataFrame:
    """Detect Internal Transfers purely by the pairing rule.

    Pairs are detected across *pairing_pool* (typically the full dataset
    including rows already claimed by earlier pipeline engines), but only
    rows in *df* (the current engine's candidates) are marked.

    Groups by (application_id, transaction_date, amount).  If a group
    contains at least one ``debit`` AND at least one ``credit``, every
    candidate row in that group whose text matches a transfer-indicator
    pattern is marked as **Internal Transfer**.

    Pairs are excluded when any row in the group contains known gambling,
    payday-lender, BNPL, EFTPOS, or ATM keywords — those are external
    transactions that happen to match the pairing rule by coincidence.
    """
    output = df.copy()

    # Merge extra rows from pairing_pool that aren't in df for full context.
    extra_rows = pairing_pool[~pairing_pool.index.isin(df.index)]
    if len(extra_rows) == 0:
        combined = df
    else:
        combined = pd.concat([df, extra_rows])

    groups = combined.groupby(
        ["application_id", "transaction_date", "amount"], dropna=True,
    )

    # Only mark rows that belong to *df* (candidates).
    candidate_idx = set(df.index)
    internal_indices: set = set()

    for _key, grp in groups:
        dr_cr_set = set(grp["dr_cr"].dropna().str.lower())
        if dr_cr_set != {"debit", "credit"}:
            continue

        # ── exclude groups containing gambling / lender keywords ──
        if _contains_excluded_keywords(grp):
            continue

        # ── only mark rows whose text looks like a genuine transfer ──
        #     and does NOT match row-level P2P exclusion patterns ──
        group_candidates = grp.index.intersection(candidate_idx)
        text_col = grp.get("text", pd.Series("", index=grp.index))
        transfer_indices = {
            idx for idx in group_candidates
            if _looks_like_transfer(text_col.get(idx, ""))
            and not _matches_row_exclusion(text_col.get(idx, ""))
        }
        internal_indices.update(transfer_indices)

    if internal_indices:
        internal_mask = pd.Series(False, index=output.index)
        internal_mask.loc[list(internal_indices)] = True
        output.loc[internal_mask, "is_transfer_pred"] = 1
        output.loc[internal_mask, "finv_category"] = "Internal Transfer"
        output.loc[internal_mask, "predicted_category"] = "Internal Transfer"
        output.loc[internal_mask, "prediction_confidence"] = "high"
        output.loc[internal_mask, "prediction_rule"] = "internal_pairing_rule"

    return output


# ── Keywords that indicate external transactions, NOT internal transfers ──
# These are checked against the full text of every row in a candidate pair
# group.  If ANY row matches, the entire group is skipped.

# ── Group-level exclusion patterns ───────────────────────────────────────
# These patterns indicate the ENTIRE group is not a genuine internal transfer
# pair (e.g. a gambling debit that happens to share the same amount+date as
# an unrelated credit).  If ANY row in the group matches, the whole group is
# excluded from internal-transfer pairing.

_GROUP_EXCLUSION_PATTERNS: list[re.Pattern] = [
    # ── Gambling / betting operators ──
    re.compile(r"\bsportsbet\b", re.IGNORECASE),
    re.compile(r"\bladbrokes\b", re.IGNORECASE),
    re.compile(r"\balventa\b", re.IGNORECASE),          # Malta gambling processor
    re.compile(r"\bfrvn\b", re.IGNORECASE),             # Limassol gambling (Cyprus)
    re.compile(r"\bvxtrx\b", re.IGNORECASE),            # Limassol gambling (Cyprus)
    re.compile(r"\bbet365\b", re.IGNORECASE),
    re.compile(r"\bbetfair\b", re.IGNORECASE),
    re.compile(r"\bpointsbet\b", re.IGNORECASE),
    re.compile(r"\bunibet\b", re.IGNORECASE),
    re.compile(r"\bplayup\b", re.IGNORECASE),
    re.compile(r"\bbluebet\b", re.IGNORECASE),
    re.compile(r"\bcrownbet\b", re.IGNORECASE),
    re.compile(r"\bdraftkings\b", re.IGNORECASE),
    re.compile(r"\btab\b", re.IGNORECASE),              # TAB sports betting
    re.compile(r"\bclassicbet\b", re.IGNORECASE),
    re.compile(r"\bbetchain\b", re.IGNORECASE),
    re.compile(r"\bpalmerbet\b", re.IGNORECASE),
    re.compile(r"\btopbetta\b", re.IGNORECASE),
    re.compile(r"\bmadbookie\b", re.IGNORECASE),
    re.compile(r"\btatts\b", re.IGNORECASE),            # Tatts Group (lotteries/betting)
    re.compile(r"\btattsbet\b", re.IGNORECASE),
    re.compile(r"\b(?:neds|betr)\b", re.IGNORECASE),    # Australian betting apps
    # ── Payday lenders / wage advance / BNPL (external, not internal transfer) ──
    re.compile(r"\b(?:afterpay|zip\s*pay|zipmoney)\b", re.IGNORECASE),
    re.compile(r"\b(?:wagepay|wagetap|wage\s*advance)\b", re.IGNORECASE),
    re.compile(r"\b(?:mypaynow|nextpayday|beforepay|press\s*pay)\b", re.IGNORECASE),
    re.compile(r"\b(?:cash\s*converters|rapid\s*loans|cash\s*pal|cash\s*train)\b", re.IGNORECASE),
    re.compile(r"\b(?:nimble|moneyme|money3|cash\s*now|cash\s*n\s*go)\b", re.IGNORECASE),
    re.compile(r"\b(?:payday\s*advance|payday\s*loans|sure\s*cash|sunshine\s*loans)\b", re.IGNORECASE),
    re.compile(r"\b(?:credit\s*corp|fair\s*go\s*finance|swoosh\s*finance)\b", re.IGNORECASE),
    re.compile(r"\b(?:spotter\s*loans?|fundo\s*loans?|jacaranda\s*finance)\b", re.IGNORECASE),
    re.compile(r"\b(?:flash\s*money|cash\s*stop|money\s*spot|cigno)\b", re.IGNORECASE),
    re.compile(r"\b(?:wallet\s*wizard|cash\s*converters)\b", re.IGNORECASE),
    # ── Generic lender/phrase patterns that indicate borrowing ──
    re.compile(r"\b(?:loan\s*repaid|loan\s*return|loan\s*repayment|wage\s*advance\s*repayment)\b", re.IGNORECASE),
    re.compile(r"\b(?:pay\s*in\s*4|payin4)\b", re.IGNORECASE),  # PayPal BNPL
    # ── EFTPOS / card purchases (not transfers) ──
    re.compile(r"\beftpos\b", re.IGNORECASE),
    re.compile(r"\bmiscellaneous\s+debit\s+v\d", re.IGNORECASE),
    # ── ATM withdrawals (not transfers) ──
    re.compile(r"\bwithdrawal\s+at\b.*\batm\b", re.IGNORECASE),
    re.compile(r"\batm\s+withdrawal\b", re.IGNORECASE),
    # ── Gaming / gambling operators not covered above ──
    re.compile(r"\bnorhengame\b", re.IGNORECASE),
    re.compile(r"\bhappyfarm\b", re.IGNORECASE),
    re.compile(r"\bwfun\b", re.IGNORECASE),
    re.compile(r"\bsmedge\b", re.IGNORECASE),
    re.compile(r"\bdragfir\b", re.IGNORECASE),
    re.compile(r"\bprezzee\b", re.IGNORECASE),            # gift card (often gambling-adjacent)
]

# ── Row-level exclusion patterns ─────────────────────────────────────────
# These patterns indicate a SPECIFIC row is a P2P / external payment, NOT an
# internal transfer.  Unlike group-level exclusions, only the matching row is
# excluded from IT — other rows in the same group can still be paired as IT.
# This preserves genuine internal transfers that share a group with P2P rows.

_ROW_EXCLUSION_PATTERNS: list[re.Pattern] = [
    # ── Person-to-person (P2P) OSKO / bank transfer patterns ──
    re.compile(r"\bOsko (?:Payment|Deposit)\b", re.IGNORECASE),
    # "IBank Trf Transferred to <account> <PERSON NAME> ..." (Greater Bank P2P)
    re.compile(r"\bIBank Trf\b", re.IGNORECASE),
    # Generic person-title indicators strongly suggest P2P rather than
    # own-account transfers.  Matches both single initials ("MR J") and
    # full names ("MISS KARIN") — [A-Z][A-Za-z]* covers both cases.
    re.compile(r"\b(?:MRS?|MR|MS|MISS)\s+[A-Z][A-Za-z]*\b", re.IGNORECASE),
    # ── PayID / PayTo transfers are person-to-person, never own-account ──
    re.compile(r"\b(?:PayID|PayTo)\b", re.IGNORECASE),
    # ── TRANSFER DEBIT/CREDIT + all-caps person name (NAB-style P2P) ──
    # "TRANSFER DEBIT JOSE VARUGHESE Q8294369047" — name + reference = external.
    # "TRANSFER CREDIT NICHOLAS WYNGAARD Nick & Paige Event" — same pattern.
    # Exclude "TRANSFER ... ONLINE" which is internet banking (genuine internal).
    re.compile(r"^TRANSFER\s+(?:DEBIT|CREDIT)\s+(?!ONLINE\b)[A-Z]{2,}\b", re.IGNORECASE),
    # ── Fast Transfer From/To + person name (P2P, never own-account) ──
    # "Fast Transfer From WAYNE SCOTT BROWNLOWE DAD"
    # "Fast Transfer From HEIKE FOX Hay x 14 rounds"
    # "Fast Transfer From Jordan Brownlowe 6 7"
    re.compile(r"\bFast Transfer (?:From|To)\s+[A-Z][A-Za-z]+", re.IGNORECASE),
    # ── Transfer To [NAME] CommBank App (P2P, not own-account) ──
    # "Transfer To Theresa Salanoa CommBank App misc"
    # "Transfer To Daniel Smith CommBank App Dinner"
    # Distinguished from internal "Transfer to xx\d{4} CommBank App"
    # and "Transfer to cba a/c commbank app" by negative lookahead.
    re.compile(
        r"\bTransfer To\s+(?!xx\d{4}|cba a/c)[A-Za-z0-9].*?\s+CommBank App\b",
        re.IGNORECASE,
    ),
]

# Backward-compatible alias used by _contains_excluded_keywords (group-level only).
_EXCLUDED_PAIRING_PATTERNS = _GROUP_EXCLUSION_PATTERNS


def _contains_excluded_keywords(grp: pd.DataFrame) -> bool:
    """Return True if any row in the group matches excluded keyword patterns.

    Checks both:
    1. Hardcoded ``_EXCLUDED_PAIRING_PATTERNS`` (compiled regex list).
    2. CSV knowledge-base exclusion rules (``transfer_pairing_exclusions.csv``).
    """
    text_col = grp.get("text", pd.Series("", index=grp.index))
    csv_exclusions = _get_exclusion_rules()
    for _, text in text_col.items():
        if pd.isna(text) or not str(text).strip():
            continue
        text_str = str(text)
        # Hardcoded patterns (fast regex check).
        for pattern in _EXCLUDED_PAIRING_PATTERNS:
            if pattern.search(text_str):
                return True
        # CSV knowledge-base exclusion rules.
        if csv_exclusions:
            from .transfer_rules import matches_any_exclusion
            if matches_any_exclusion(text_str, csv_exclusions):
                return True
    return False


def _detect_external_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply regex rules to detect External Transfers (vectorised).

    Only runs on rows that have NOT already been classified (is_transfer_pred == 0).
    Matched rows receive finv_category = "External Transfers".

    Optimisation: uses pandas ``str.contains`` (C-level) instead of a Python
    for-loop + ``re.search`` per row, which is ~50-100× faster on large frames.
    """
    output = df.copy()
    remaining_mask = output["is_transfer_pred"] == 0

    if not remaining_mask.any():
        return output

    text_col = output.get("text_norm", pd.Series("", index=output.index))
    dr_cr_col = output.get("dr_cr", pd.Series("", index=output.index))

    # Pre-compile all rules with their confidence tier.
    _compiled: list[tuple[str, str, "re.Pattern", str | None, str]] = []
    for name, cat, pattern, dr_cr in HIGH_CONFIDENCE_RULES:
        _compiled.append((name, cat, re.compile(pattern, re.IGNORECASE), dr_cr, "high"))
    for name, cat, pattern, dr_cr in MEDIUM_CONFIDENCE_RULES:
        _compiled.append((name, cat, re.compile(pattern, re.IGNORECASE), dr_cr, "medium"))

    for rule_name, _category, pattern, dr_cr_constraint, confidence in _compiled:
        if not remaining_mask.any():
            break

        remaining_idx = output.index[remaining_mask]
        text_subset = text_col.loc[remaining_idx]

        # ── vectorised regex match (C-level) ──
        matched = text_subset.str.contains(pattern, na=False, regex=True)
        if not matched.any():
            continue

        matched_idx = remaining_idx[matched]

        # ── dr_cr constraint ──
        if dr_cr_constraint is not None:
            dr_cr_subset = dr_cr_col.loc[matched_idx]
            dr_cr_ok = (
                dr_cr_subset.astype(str).str.strip().str.lower()
                == dr_cr_constraint
            )
            matched_idx = matched_idx[dr_cr_ok.values]
            if len(matched_idx) == 0:
                continue

        # ── assign predictions ──
        output.loc[matched_idx, "is_transfer_pred"] = 1
        output.loc[matched_idx, "finv_category"] = "External Transfers"
        output.loc[matched_idx, "predicted_category"] = "External Transfers"
        output.loc[matched_idx, "prediction_confidence"] = confidence
        output.loc[matched_idx, "prediction_rule"] = rule_name
        output.loc[matched_idx, "prediction_dr_cr_used"] = (
            dr_cr_constraint is not None
        )

        remaining_mask.loc[matched_idx] = False

    return output


def _match_deposit_to_known_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify internet deposit rows whose counterparty account is already
    known from a withdrawal match (i.e. the user has transferred money TO that
    account before, so it is likely their own account).

    Known accounts are scoped per bank_account_id to avoid cross-user leakage."""

    output = df.copy()

    # Collect known internal accounts per bank_account_id from withdrawal matches.
    bank_account_col = (
        "bank_account_id" if "bank_account_id" in output.columns else None
    )
    known_by_bank: dict[str, set[str]] = {}

    for _, row in output[output["is_transfer_pred"] == 1].iterrows():
        match = _WITHDRAWAL_ACCOUNT_RE.search(str(row.get("text_norm", "")))
        if not match:
            continue
        bank = (
            str(row[bank_account_col])
            if bank_account_col and pd.notna(row.get(bank_account_col))
            else "__global__"
        )
        known_by_bank.setdefault(bank, set()).add(match.group(1))

    if not known_by_bank:
        return output

    # Scan unclassified rows for deposit patterns with known accounts.
    unclassified_mask = output["is_transfer_pred"] == 0
    for idx in output[unclassified_mask].index:
        text = str(output.at[idx, "text_norm"])
        match = _DEPOSIT_ACCOUNT_RE.search(text)
        if not match:
            continue
        account = match.group(1)
        bank = (
            str(output.at[idx, bank_account_col])
            if bank_account_col and pd.notna(output.at[idx, bank_account_col])
            else "__global__"
        )
        known = known_by_bank.get(bank, set()) | known_by_bank.get("__global__", set())
        if account not in known:
            continue

        output.at[idx, "is_transfer_pred"] = 1
        output.at[idx, "finv_category"] = "External Transfers"
        output.at[idx, "predicted_category"] = "External Transfers"
        output.at[idx, "prediction_confidence"] = "high"
        output.at[idx, "prediction_rule"] = (
            "internal_internet_deposit_known_account"
        )
        output.at[idx, "prediction_dr_cr_used"] = False

    return output


def _detect_internal_by_regex(df: pd.DataFrame) -> pd.DataFrame:
    """Apply high-confidence internal-transfer regex rules (vectorised).

    Runs on rows NOT already classified by the pairing rule.  Matched rows
    receive ``finv_category = "Internal Transfer"``, preventing them from
    being claimed by the external-transfer regex step.
    """
    output = df.copy()
    remaining_mask = output["is_transfer_pred"] == 0
    if not remaining_mask.any():
        return output

    compiled = [
        (name, re.compile(pattern, re.IGNORECASE), dr_cr)
        for name, pattern, dr_cr in INTERNAL_TRANSFER_RULES
    ]
    text_col = output.get("text_norm", pd.Series("", index=output.index))
    dr_cr_col = output.get("dr_cr", pd.Series("", index=output.index))

    # Pre-compile the INTL-FEE exclusion pattern used by two rules.
    _intl_fee_re = re.compile(r"\bINTL[-\s]?FEE\b", re.IGNORECASE)

    for rule_name, pattern, dr_cr_constraint in compiled:
        if not remaining_mask.any():
            break

        remaining_idx = output.index[remaining_mask]
        text_subset = text_col.loc[remaining_idx]

        # ── vectorised regex match ──
        matched = text_subset.str.contains(pattern, na=False, regex=True)
        if not matched.any():
            continue

        matched_idx = remaining_idx[matched]

        # ── dr_cr constraint ──
        if dr_cr_constraint is not None:
            dr_cr_subset = dr_cr_col.loc[matched_idx]
            dr_cr_ok = (
                dr_cr_subset.astype(str).str.strip().str.lower()
                == dr_cr_constraint
            )
            matched_idx = matched_idx[dr_cr_ok.values]
            if len(matched_idx) == 0:
                continue

        # ── Per-rule exclusion checks (vectorised) ──
        if rule_name in ("internal_anz_funds_tfer", "internal_internet_banking"):
            # Exclude rows whose text contains INTL-FEE (cross-border).
            exclude = text_col.loc[matched_idx].str.contains(
                _intl_fee_re, na=False, regex=True
            )
            matched_idx = matched_idx[~exclude]
            if len(matched_idx) == 0:
                continue

        # ── assign predictions ──
        output.loc[matched_idx, "is_transfer_pred"] = 1
        output.loc[matched_idx, "finv_category"] = "Internal Transfer"
        output.loc[matched_idx, "predicted_category"] = "Internal Transfer"
        output.loc[matched_idx, "prediction_confidence"] = "high"
        output.loc[matched_idx, "prediction_rule"] = rule_name
        output.loc[matched_idx, "prediction_dr_cr_used"] = (
            dr_cr_constraint is not None
        )

        remaining_mask.loc[matched_idx] = False

    return output


def _filter_personal_osko_credits(df: pd.DataFrame) -> pd.DataFrame:
    """Unmark Osko credit rows that look like informal person-to-person payments.

    Osko deposits / payments that are personal (gifts, reimbursements,
    one-off payments) rather than systematic external transfers are unmarked
    so they can be picked up by other engines or left unclassified.

    Two checks:
    1. No 6+ digit identifier at all → definitely personal.
    2. Has a 6+ digit number BUT followed by a person's name
       (e.g. "DEPOSIT-OSKO PAYMENT 2741681 LAUREN T DUSSIN") → P2P.
    """
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

    for idx in output[et_credit_mask].index:
        text = str(text_col.get(idx, ""))
        if not re.search(r"\bosko\b", text, re.IGNORECASE):
            continue

        has_ref = bool(re.search(r"\d{6,}", text))
        if not has_ref:
            # No reference/account number — informal personal credit.
            pass  # fall through to unmark
        else:
            # Has a 6+ digit number, but check if it looks like a person's
            # name follows in the raw (title-case) text
            # (e.g. "DEPOSIT-OSKO PAYMENT 2741681 LAUREN T DUSSIN").
            raw = str(raw_text.get(idx, ""))
            if re.search(
                r"\d{6,}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
                raw,
            ):
                pass  # P2P with a name — fall through to unmark
            else:
                continue  # Systematic transfer with reference — keep

        # Personal Osko credit — unmark
        output.at[idx, "is_transfer_pred"] = 0
        output.at[idx, "finv_category"] = ""
        output.at[idx, "predicted_category"] = ""
        output.at[idx, "prediction_confidence"] = ""
        output.at[idx, "prediction_rule"] = ""
        output.at[idx, "prediction_dr_cr_used"] = False

    return output


def _build_transfer_rule_name(df: pd.DataFrame) -> pd.Series:
    """Use prediction_rule when available, otherwise transfer_rule_name."""
    if "transfer_rule_name" in df.columns:
        return df["transfer_rule_name"].where(
            df["transfer_rule_name"].notna() & (df["transfer_rule_name"] != ""),
            df["prediction_rule"].fillna(""),
        )
    return df["prediction_rule"].fillna("")


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
        text_col.astype(str)
        .str.strip()
        .str.upper()
        .apply(lambda t: re.sub(r"\s+", " ", t))
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
        rule = df.loc[idx, "transfer_rule_name"].astype(str)
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
