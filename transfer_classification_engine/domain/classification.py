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
from typing import List

import pandas as pd

from classification_core.reasons import format_classification_reason


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

    # -------------------------------------------------------------------------
    # Transfer: high-confidence patterns (P0: promoted from medium)
    # -------------------------------------------------------------------------

    # Westpac deposit patterns: always credit, 95-98% transfer.
    ("internal_deposit_westpac_cho", "transfer", r"^deposit online \d+ tfr westpac cho", "credit"),
    ("internal_deposit_westpac_lif", "transfer", r"^deposit online \d+ tfr westpac lif", "credit"),
    ("internal_deposit_westpac_esa", "transfer", r"^deposit online \d+ tfr westpac esa", "credit"),

    # Commbank app masked account with descriptive suffixes: 92-98% transfer.
    ("internal_commbank_from_suffix", "transfer", r"^transfer from xx\d{4} commbank app (allowance|rent|savings|transfer|food|bill|shop)$", None),
    ("internal_commbank_from_value_date", "transfer", r"^transfer from xx\d{4} commbank app value date: \d+/\d+/\d+$", None),

    # Standard transfer narratives.
    ("internal_phrase", "transfer", r"\binternal transfer\b", None),
    ("internal_linked_acc", "transfer", r"\b(linked acc|linked account|acc trns|acc transfer)\b", None),
    ("internal_anz_funds_tfer", "transfer", r"\banz internet banking funds tfer\b", None),
    ("internal_anz_mbank_to_long", "transfer", r"^anz m-banking funds tfer transfer \d+ to \d+$", None),
    ("internal_anz_mbank_from_long", "transfer", r"^anz m-banking funds tfer transfer \d+ from \d+$", None),
    ("internal_ib_tfr_to_long", "transfer", r"^ib tfr \d+ to \d+$", None),
    ("internal_mb_transfer_from", "transfer", r"^mb transfer from \d+$", None),
    ("internal_mb_transfer_to", "transfer", r"^mb transfer to \d+$", None),
    ("internal_orange_everyday_to", "transfer", r"^internal transfer - receipt \d+ - to orange everyday$", None),
    ("internal_orange_everyday_from", "transfer", r"^internal transfer - receipt \d+ - from orange everyday$", None),
    ("internal_savings_maximiser", "transfer", r"^internal transfer - internal transfer - receipt \d+ savings maximiser \d+$", None),
    ("internal_ibank_mobile_banking", "transfer", r"^ibank trf ref: \d+ transferred to \d+ mobile banking$", None),

    # Internet banking withdrawal to own account (deposit direction excluded — too ambiguous).
    ("internal_internet_banking", "transfer", r"\binternet withdrawal\b.*\bto \d{7,}\b", None),

    # ── NAB direct transfer credit/debit (P0: counterparty patterns promoted) ──
    ("external_nab_transfer_credit", "transfer", r"^transfer credit (?!online\b)", None),
    ("external_nab_transfer_debit", "transfer", r"^transfer debit (?!online\b)", None),

    # ── Internet banking transfers ──
    ("external_internet_banking", "transfer", r"^internet (?:withdrawal|deposit)\b", None),

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
    ("internal_transfer_to_cba_ac", "transfer", r"^transfer to cba a/c commbank app$", None),
    ("internal_transfer_from_commbank", "transfer", r"^transfer from commbank app$", None),
    ("internal_masked_commbank", "transfer", r"\btransfer (to|from) xx\d{4}\b", None),
    # Westpac withdrawal patterns: debit direction, still mostly transfer (92-98%).
    ("internal_westpac_family", "transfer", r"\b(tfr westpac|westpac lif|maximiser)\b", None),
    ("internal_from_account", "transfer", r"\bfrom account \d+ .* internal transfer\b", None),
    ("internal_tfd", "transfer", r"^\. tfd$", None),
    ("internal_tfc", "transfer", r"^\. tfc$", None),
    ("internal_me_tfd", "transfer", r"^me tfd$", None),
    ("internal_x_tfc", "transfer", r"^x tfc$", None),
    ("internal_save_tfc", "transfer", r"^save tfc$", None),
    ("internal_j_tfd", "transfer", r"^j tfd$", None),

    # Relaxed name+code: allows extra text after the transaction code.
    # Precision ~95.3% — kept as medium due to some lookalikes.
    ("external_transfer_name_code_relaxed", "transfer", r"^transfer (debit|credit) (?!online )[a-z].*?[a-z]\d{10,}", None),
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
    1. Internal Transfer — purely data-driven via pairing rule
       (same application_id + transaction_date + amount, both debit & credit).
    2. External Transfers — regex rules on remaining unclassified rows.
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

    # ── Step 2: External Transfers — regex on remaining rows ──
    output = _detect_external_transfers(output)

    # ── Step 3: extend via known internal accounts (also External Transfers) ──
    output = _match_deposit_to_known_accounts(output)

    # ── counterparty ──
    output["counterparty"] = _derive_counterparty(output)

    # ── rule metadata ──
    output["transfer_rule_name"] = _build_transfer_rule_name(output)
    output["transfer_pred_reason"] = output.apply(_build_reason, axis=1)

    # ── stream id ──
    output["stream_id"] = output["finv_category"].where(
        output["is_transfer_pred"].eq(1), ""
    )

    return output


def _detect_internal_transfers(
    df: pd.DataFrame, pairing_pool: pd.DataFrame,
) -> pd.DataFrame:
    """Detect Internal Transfers purely by the pairing rule.

    Pairs are detected across *pairing_pool* (typically the full dataset
    including rows already claimed by earlier pipeline engines), but only
    rows in *df* (the current engine's candidates) are marked.

    Groups by (application_id, transaction_date, amount).  If a group
    contains at least one ``debit`` AND at least one ``credit``, every
    candidate row in that group is marked as **Internal Transfer**.

    Pairs are excluded when any row in the group contains known gambling,
    payday-lender, or BNPL keywords — those are external transactions that
    happen to match the pairing rule by coincidence.
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

        internal_indices.update(grp.index.intersection(candidate_idx))

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

_EXCLUDED_PAIRING_PATTERNS: list[re.Pattern] = [
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
]


def _contains_excluded_keywords(grp: pd.DataFrame) -> bool:
    """Return True if any row in the group matches excluded keyword patterns."""
    text_col = grp.get("text", pd.Series("", index=grp.index))
    for _, text in text_col.items():
        if pd.isna(text) or not str(text).strip():
            continue
        text_str = str(text)
        for pattern in _EXCLUDED_PAIRING_PATTERNS:
            if pattern.search(text_str):
                return True
    return False


def _detect_external_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply regex rules to detect External Transfers.

    Only runs on rows that have NOT already been classified (is_transfer_pred == 0).
    Matched rows receive finv_category = "External Transfers".
    """
    output = df.copy()
    remaining_mask = output["is_transfer_pred"] == 0

    if not remaining_mask.any():
        return output

    remaining = output.loc[remaining_mask]
    classifier = FixedRuleClassifier()
    classified = classifier.predict_frame(remaining)

    # Copy predictions back to output for matched rows.
    matched = classified["predicted_category"].notna()
    if matched.any():
        matched_idx = classified.index[matched]
        output.loc[matched_idx, "is_transfer_pred"] = 1
        output.loc[matched_idx, "finv_category"] = "External Transfers"
        output.loc[matched_idx, "predicted_category"] = classified.loc[matched, "predicted_category"]
        output.loc[matched_idx, "prediction_confidence"] = classified.loc[matched, "prediction_confidence"]
        output.loc[matched_idx, "prediction_rule"] = classified.loc[matched, "prediction_rule"]
        output.loc[matched_idx, "prediction_dr_cr_used"] = classified.loc[matched, "prediction_dr_cr_used"]

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
# Produces standardized counterparty labels matching the third_party naming
# convention observed in the sample data:
#
#   Internal Transfer  →  "Internal Transfer Debit" / "Internal Transfer Credit"
#   External Transfer  →  "<PaymentSystemOrBank> <Category>"
#
# Patterns are ordered by specificity — first match wins.

# ── Payment system / service identifiers ──

_OSKO_RE = re.compile(r"\bosko\b", re.IGNORECASE)
_NPP_RE = re.compile(r"\b(npp|payid)\b", re.IGNORECASE)
_BPAY_RE = re.compile(r"\bbpay\b", re.IGNORECASE)
_PAYPAL_RE = re.compile(r"\bpaypal\b", re.IGNORECASE)
_WISE_RE = re.compile(r"\bwise\b", re.IGNORECASE)
_WORLDREMIT_RE = re.compile(r"\bworldremit\b", re.IGNORECASE)
_WU_RE = re.compile(r"\bwestern\s+union\b", re.IGNORECASE)
_WORLDWIDE_WALLET_RE = re.compile(r"\bworldwide\s+wallet\b", re.IGNORECASE)
_IMT_RTGS_RE = re.compile(r"\b(imt|rtgs|swift)\b", re.IGNORECASE)
_RMTLY_RE = re.compile(r"\brmtly\b", re.IGNORECASE)
_AUTOMATIC_PAYMENT_RE = re.compile(r"^automatic\s+payment$", re.IGNORECASE)
_DIRECT_DEBIT_STANDALONE_RE = re.compile(r"^direct\s+debit$", re.IGNORECASE)

# ── Bank-specific text markers ──

_COMMBANK_APP_RE = re.compile(r"\bcommbank\s+app\b", re.IGNORECASE)
# "Fast Transfer" is CBA's branding for Osko/NPP payments.
_FAST_TRANSFER_RE = re.compile(r"^fast\s+transfer\s+(?:from|to)\b", re.IGNORECASE)
# CommBank internet banking: "Internet Withdrawal/Deposit ... To/From ..."
_INTERNET_BANKING_RE = re.compile(r"^internet\s+(?:withdrawal|deposit)\b", re.IGNORECASE)
_ANZ_BANKING_RE = re.compile(r"\banz\b", re.IGNORECASE)
_WESTPAC_RE = re.compile(r"\bwestpac\b", re.IGNORECASE)
_ING_RE = re.compile(r"\b(?:orange\s+everyday|orange\s+one|savings\s+maximiser)\b", re.IGNORECASE)
# ING internal transfer format: "Internal Transfer - Receipt 123456 - ... Orange Everyday"
_ING_INTERNAL_FMT_RE = re.compile(
    r"internal\s+transfer\s+.*\b(?:orange\s+(?:everyday|one)|savings\s+maximiser)\b",
    re.IGNORECASE,
)
_GREATER_BANK_RE = re.compile(r"\b(?:ibank\s+trf|wdl\s+thpar)\b", re.IGNORECASE)
_BANKWEST_PAYID_RE = re.compile(r"^to\s+(?:phone|email)\b", re.IGNORECASE)
_SUNCORP_RE = re.compile(r"\bsuncorp\b", re.IGNORECASE)
_HERITAGE_RE = re.compile(r"\bheritage\b", re.IGNORECASE)
_BENDIGO_RE = re.compile(r"\bbendigo\s+bank\b", re.IGNORECASE)

# ── Service / merchant identifiers ──

_BIZ_CORE_RE = re.compile(r"\bbiz\s+core\b", re.IGNORECASE)
_DBS_XPLOR_RE = re.compile(r"\bdbs\s+xplor\b", re.IGNORECASE)
_ROCKET_REMIT_RE = re.compile(r"\brocket\s+remit\b", re.IGNORECASE)
_ZEPTO_RE = re.compile(r"\bzepto\b", re.IGNORECASE)
_LIVE_PAYMENTS_RE = re.compile(r"\blive\s+payments\b", re.IGNORECASE)

# ── Short/internal code patterns ("b TFC", "b TFD", "NECTR", etc.) ──

_SHORT_CODE_RE = re.compile(
    r"^(?:[a-z]\s+)?(?:tfc|tfd|nectr)$", re.IGNORECASE,
)

# ── "DEPOSIT-OSKO PAYMENT" → Osko (some banks prefix Osko with "DEPOSIT-") ──

_DEPOSIT_OSKO_RE = re.compile(r"^deposit-osko\s+payment\b", re.IGNORECASE)

# ── NAB direct transfer markers ──

_NAB_TRANSFER_CREDIT_DEBIT_RE = re.compile(
    r"^transfer\s+(?:credit|debit)\s+(?!online\b)",
    re.IGNORECASE,
)

# ── CommBank generic transfer markers ──

_CBA_TRANSFER_TO_FROM_RE = re.compile(
    r"^transfer\s+(?:to|from)\b",
    re.IGNORECASE,
)

# ── Other bank transfer pattern (TFR to/from ... MOB) ──

_TFR_MOB_RE = re.compile(r"\btfr\s+(?:to|from)\b", re.IGNORECASE)

# ── Internal transfer markers (used for fallback detection) ──

_INTERNAL_MARKER_RE = re.compile(
    r"\b(?:linked\s+acc|internal\s+transfer|acc\s+trns|acc\s+transfer)\b",
    re.IGNORECASE,
)


def _is_internal_transfer(row: pd.Series) -> bool:
    """Determine if a row represents an internal transfer for counterparty labelling.

    Uses the finv_category set by _separate_internal_external, which detects
    internal transfers via the pairing rule: same application_id + transaction_date
    + amount, with both debit and credit rows present.
    """
    finv_category = str(row.get("finv_category", "") or "")
    return finv_category == "Internal Transfer"


def _extract_counterparty_from_text(
    text: str, dr_cr: str = "", is_internal: bool = False,
) -> str:
    """Extract a standardized counterparty label matching third_party convention.

    Parameters
    ----------
    text : str
        The raw transaction text.
    dr_cr : str
        Debit/credit direction (for internal transfer labels).
    is_internal : bool
        Whether the row was classified as an internal transfer.

    Returns
    -------
    str
        A counterparty label following third_party naming conventions.
    """
    if not text or pd.isna(text):
        return ""

    text_str = str(text).strip()
    text_lower = text_str.lower()

    # ═══════════════════════════════════════════════════════════════════
    # Internal Transfer
    # ═══════════════════════════════════════════════════════════════════
    if is_internal:
        direction = "Debit" if str(dr_cr).strip().lower() == "debit" else "Credit"
        return f"Internal Transfer {direction}"

    # ═══════════════════════════════════════════════════════════════════
    # External Transfer — payment system / service identifiers (priority)
    # ═══════════════════════════════════════════════════════════════════

    # 1. Osko (including "DEPOSIT-OSKO PAYMENT" variant)
    if _OSKO_RE.search(text_lower) or _DEPOSIT_OSKO_RE.search(text_lower):
        return "Osko"

    # 2. BPAY with specific biller → extract biller name
    m = re.search(r"bill\s*pay\s+(\S+(?:\s+\S+){0,2})", text_lower)
    if m:
        biller = m.group(1).strip()
        biller_lower = biller.lower()
        if "biz core" in biller_lower:
            return "Biz Core"
        if "dbs" in biller_lower or "xplor" in biller_lower:
            return "Debitsuccess Transfer"
        return biller.title()

    # 3. Debitsuccess via BPAY DBS XPLOR (may appear without "BILL PAY" prefix)
    if _DBS_XPLOR_RE.search(text_lower):
        return "Debitsuccess Transfer"

    # 4. Generic BPAY (no recognizable biller name)
    if _BPAY_RE.search(text_lower):
        return "BPAY Transfer"

    # 5. PayPal
    if _PAYPAL_RE.search(text_lower):
        return "Paypal Transfer"

    # 6. Wise
    if _WISE_RE.search(text_lower):
        return "Wise"

    # 7. WorldRemit
    if _WORLDREMIT_RE.search(text_lower):
        return "WorldRemit"

    # 8. Western Union
    if _WU_RE.search(text_lower):
        return "Western Union"

    # 9. Rocket Remit
    if _ROCKET_REMIT_RE.search(text_lower):
        return "Rocket Remit"

    # 10. Zepto
    if _ZEPTO_RE.search(text_lower):
        return "Zepto Payment"

    # 11. IMT / RTGS / SWIFT
    if _IMT_RTGS_RE.search(text_lower):
        return "Money Transfer Services"

    # 12. RMTLY / money transfer services
    if _RMTLY_RE.search(text_lower):
        return "Money Transfers"

    # 13. Live Payments
    if _LIVE_PAYMENTS_RE.search(text_lower):
        return "Live Payments Transfer"

    # 14. Automatic Payment
    if _AUTOMATIC_PAYMENT_RE.search(text_lower):
        return "Automatic Transfers Credit"

    # 15. Standalone Direct Debit
    if _DIRECT_DEBIT_STANDALONE_RE.search(text_lower):
        return "Direct Debit Transfer"

    # ═══════════════════════════════════════════════════════════════════
    # External Transfer — bank identification from text
    # ═══════════════════════════════════════════════════════════════════

    # NOTE: Westpac must precede Worldwide Wallet / Travel Money because
    # some Westpac texts contain "TFR WORLDWIDE WALLET".

    # 17. Westpac (before Travel Money to avoid WORLDWIDE WALLET hijack)
    if _WESTPAC_RE.search(text_lower):
        return "WESTPAC Funds Transfer"

    # 18. CommBank app → CBA Funds Transfer
    if _COMMBANK_APP_RE.search(text_lower):
        return "CBA Funds Transfer"

    # 19. Fast Transfer From/To (CBA's Osko/NPP branding)
    if _FAST_TRANSFER_RE.search(text_lower):
        return "CBA Funds Transfer"

    # 20. ANZ
    if _ANZ_BANKING_RE.search(text_lower):
        return "ANZ Funds Transfer"

    # 21. ING (Orange Everyday / Savings Maximiser) — also matches
    #     "Internal Transfer - Receipt ... - ... Orange Everyday" format
    if _ING_RE.search(text_lower) or _ING_INTERNAL_FMT_RE.search(text_lower):
        return "ING Funds Transfer"

    # 22. Greater Bank
    if _GREATER_BANK_RE.search(text_lower):
        return "GREATER Funds Transfer"

    # 23. BANKWEST (PayID to phone/email)
    if _BANKWEST_PAYID_RE.search(text_lower):
        return "BANKWEST Funds Transfer"

    # 24. SUNCORP
    if _SUNCORP_RE.search(text_lower):
        return "SUNCORP Funds Transfer"

    # 25. Heritage Bank
    if _HERITAGE_RE.search(text_lower):
        return "HERITAGE Funds Transfer"

    # 26. Bendigo Bank
    if _BENDIGO_RE.search(text_lower):
        return "Other Transfers"

    # ═══════════════════════════════════════════════════════════════════
    # External Transfer — generic patterns
    # ═══════════════════════════════════════════════════════════════════

    # 27. Worldwide Wallet (after bank checks to avoid Westpac hijack)
    if _WORLDWIDE_WALLET_RE.search(text_lower):
        return "Travel Money"

    # 28. Short internal codes: "b TFC", "b TFD", "NECTR", etc.
    if _SHORT_CODE_RE.search(text_lower):
        return "Other Transfers"

    # 29. NPP / PayID (without specific bank marker)
    if _NPP_RE.search(text_lower):
        return "Funds Related Transfer"

    # 30. Internet banking withdrawal/deposit (CommBank internet banking)
    if _INTERNET_BANKING_RE.search(text_lower):
        return "Funds Transfer"

    # 31. NAB-style direct transfer credit/debit
    if _NAB_TRANSFER_CREDIT_DEBIT_RE.search(text_lower):
        return "Miscellaneous Funds Transfer"

    # 32. CommBank generic transfer (Transfer To/From... no "CommBank app")
    if _CBA_TRANSFER_TO_FROM_RE.search(text_lower):
        return "Funds Transfer"

    # 33. TFR to/from ... MOB (other bank mobile transfers)
    if _TFR_MOB_RE.search(text_lower):
        return "Other Funds Transfer"

    # 34. "Transferred to" pattern
    if re.search(r"\btransferred\s+to\b", text_lower):
        return "Funds Transfer"

    # 35. Generic fallback for external transfers
    return "External Transfers"


def _derive_counterparty(df: pd.DataFrame) -> pd.Series:
    """Extract a counterparty label matching third_party naming conventions.

    For internal transfers the label is ``"Internal Transfer Debit"`` /
    ``"Internal Transfer Credit"`` based on dr_cr direction.

    For external transfers the label identifies the payment system, bank,
    or service (e.g. ``"Osko"``, ``"CBA Funds Transfer"``, ``"Biz Core"``).
    """
    text_col = df.get("text_norm", df.get("text", pd.Series("", index=df.index)))
    dr_cr_col = df.get(
        "dr_cr", pd.Series("", index=df.index)
    ).fillna("").astype(str)

    results = []
    for idx in df.index:
        text = str(text_col.loc[idx] if idx in text_col.index else "").strip()
        dr_cr = str(dr_cr_col.loc[idx] if idx in dr_cr_col.index else "")
        row = df.loc[idx]
        is_internal = _is_internal_transfer(row)
        counterparty = _extract_counterparty_from_text(
            text, dr_cr, is_internal=is_internal,
        )
        results.append(counterparty)

    return pd.Series(results, index=df.index)


def _build_reason(row: pd.Series) -> str:
    if int(row.get("is_transfer_pred", 0)) != 1:
        return format_classification_reason(
            category="not_transfer",
            rule="no_transfer_rule_matched",
            evidence=[],
        )

    confidence = str(row.get("prediction_confidence", ""))
    rule_name = str(row.get("transfer_rule_name", ""))
    category = str(row.get("finv_category", ""))

    evidence_parts = [f"confidence={confidence}"]
    if row.get("prediction_dr_cr_used", False):
        evidence_parts.append("dr_cr_used")

    return format_classification_reason(
        category=category,
        rule=rule_name,
        evidence=evidence_parts,
    )
