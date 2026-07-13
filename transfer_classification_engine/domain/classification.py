# -*- coding: utf-8 -*-
"""
Rule-based transfer classification for bank transactions.

Classifies transactions into:
- internal transfer
- external transfer

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
    # External transfer: well-defined merchant/payment patterns (P0 additions)
    # -------------------------------------------------------------------------
    ("external_direct_debit_generic", "external transfer", r"^direct debit$", "debit"),
    ("external_eftpos", "external transfer", r"\beftpos\b", None),
    ("external_visa_purchase", "external transfer", r"\bvisa purchase\b", None),
    ("external_bpay", "external transfer", r"\bbpay\b", None),
    ("external_transferwise_sydney", "external transfer", r"\btransferwise sydney\b", None),
    ("external_debit_card_wise_taptap", "external transfer", r"^debit card purchase (wise|taptap send) sydney", None),
    ("external_fast_pymt_in", "external transfer", r"^fast pymt in", None),
    ("external_withdrawal_westpac_cho_loan", "external transfer", r"^withdrawal mobile \d+ tfr westpac cho loan$", None),

    # -------------------------------------------------------------------------
    # External transfer: named mobile/online payments and well-defined syntax
    # -------------------------------------------------------------------------
    ("external_anz_mobile_payment_to_name", "external transfer", r"^anz mobile banking payment \d+ to [a-z].*$", None),
    ("external_sav_mobile", "external transfer", r"^transfer (to|from) sav \d+ mobile#\d+$", None),
    ("external_sav_ref", "external transfer", r"^transfer (to|from) sav \d+ ref#\d+$", None),
    ("external_sav_ref_xxx", "external transfer", r"^transfer from sav \d+xxx ref#\d+$", None),
    ("external_anz_mbank_masked", "external transfer", r"^anz m-banking funds tfer transfer \d+ to \d+xxxxxxxx\d{4}$", None),
    ("external_statement_or_savings_account", "external transfer", r"^(statement account|savings account)$", None),
    ("external_transfer_name_code", "external transfer", r"^transfer (debit|credit) [a-z][a-z .'-]*[a-z] [a-z]\d{10,}$", None),
    ("external_transfer_name_code_compact", "external transfer", r"^transfer (debit|credit) [a-z][a-z .'-]+[a-z]\d{10,}$", None),
    ("external_credit_adjustment", "external transfer", r"^credit adjustment$", None),
    ("external_misc_credit_v", "external transfer", r"^miscellaneous credit v\d{4}$", None),
    ("external_payid_osko", "external transfer", r"\b(payid|osko|npp)\b", None),
    ("external_fast_transfer", "external transfer", r"\bfast transfer\b", None),
    ("external_paypal_worldremit", "external transfer", r"\b(paypal australia|worldremit)\b", None),
    ("external_interbank_credit", "external transfer", r"\binter-bank credit\b", None),
    ("external_commbank_name", "external transfer", r"\btransfer (to|from) [a-z][a-z .'-]+ commbank app\b", None),

    # External transfer: stable merchant/payment patterns confirmed by extraction.
    ("external_paypal_direct_debit", "external transfer", r"^direct debit \d+ paypal australia \d+$", None),
    ("external_paypal_authority", "external transfer", r"^payment by authority to paypal australia \d+$", None),
    ("external_paypal_payment", "external transfer", r"^payment to paypal australia \d+$", None),
    ("external_paypal_account", "external transfer", r"^paypal australia \d+$", None),
    ("external_tfr_from_s_mob", "external transfer", r"^tfr from \d+s\d+ mob$", None),
    ("external_tfr_from_long_mob", "external transfer", r"^tfr from \d+ mob$", None),
    ("external_worldwide_wallet", "external transfer", r"^deposit online \d+ \d+ tfr worldwide wallet$", None),
    ("external_wu", "external transfer", r"^wu \d+$", None),
    ("external_transfer_funds_trns", "external transfer", r"^transfer debit internet transfer funds trns$", None),
    ("external_wise_card", "external transfer", r"^wise sydney au aus card xx\d{4} value date: \d+/\d+/\d+$", None),
    ("external_taptap_send", "external transfer", r"^taptap send sydney aus card xx\d{4} value date: \d+/\d+/\d+$", None),
    ("external_online_withdrawal_worldwide_wallet", "external transfer", r"^withdrawal online \d+ \d+ tfr worldwide wallet$", None),

    # -------------------------------------------------------------------------
    # Internal transfer: high-confidence patterns (P0: promoted from medium)
    # -------------------------------------------------------------------------

    # Westpac deposit patterns: always credit, 95-98% internal.
    ("internal_deposit_westpac_cho", "internal transfer", r"^deposit online \d+ tfr westpac cho", "credit"),
    ("internal_deposit_westpac_lif", "internal transfer", r"^deposit online \d+ tfr westpac lif", "credit"),
    ("internal_deposit_westpac_esa", "internal transfer", r"^deposit online \d+ tfr westpac esa", "credit"),

    # Commbank app masked account with descriptive suffixes: 92-98% internal.
    ("internal_commbank_from_suffix", "internal transfer", r"^transfer from xx\d{4} commbank app (allowance|rent|savings|transfer|food|bill|shop)$", None),
    ("internal_commbank_from_value_date", "internal transfer", r"^transfer from xx\d{4} commbank app value date: \d+/\d+/\d+$", None),

    # Standard internal transfer narratives.
    ("internal_phrase", "internal transfer", r"\binternal transfer\b", None),
    ("internal_linked_acc", "internal transfer", r"\b(linked acc|linked account|acc trns|acc transfer)\b", None),
    ("internal_anz_funds_tfer", "internal transfer", r"\banz internet banking funds tfer\b", None),
    ("internal_anz_mbank_to_long", "internal transfer", r"^anz m-banking funds tfer transfer \d+ to \d+$", None),
    ("internal_anz_mbank_from_long", "internal transfer", r"^anz m-banking funds tfer transfer \d+ from \d+$", None),
    ("internal_ib_tfr_to_long", "internal transfer", r"^ib tfr \d+ to \d+$", None),
    ("internal_mb_transfer_from", "internal transfer", r"^mb transfer from \d+$", None),
    ("internal_mb_transfer_to", "internal transfer", r"^mb transfer to \d+$", None),
    ("internal_orange_everyday_to", "internal transfer", r"^internal transfer - receipt \d+ - to orange everyday$", None),
    ("internal_orange_everyday_from", "internal transfer", r"^internal transfer - receipt \d+ - from orange everyday$", None),
    ("internal_savings_maximiser", "internal transfer", r"^internal transfer - internal transfer - receipt \d+ savings maximiser \d+$", None),
    ("internal_ibank_mobile_banking", "internal transfer", r"^ibank trf ref: \d+ transferred to \d+ mobile banking$", None),
]

MEDIUM_CONFIDENCE_RULES = [
    # External transfer medium patterns.
    ("external_transferred_to_digits", "external transfer", r"\btransferred to \d{3,6} \d+\b", None),
    ("external_sav_net", "external transfer", r"^transfer (to|from) sav \d+ net#\d+$", None),
    ("external_scheduled_cba_account", "external transfer", r"^scheduled payment to a cba account \d+ \d+$", None),
    ("external_anz_mbank_transfer_from", "external transfer", r"^anz m-banking transfer \d+ from \d+$", None),

    # Internal transfer medium patterns.
    ("internal_internet_transfer_credit", "internal transfer", r"^internet transfer credit from \d+ ref no \d+$", None),
    ("internal_internet_transfer_debit", "internal transfer", r"^internet transfer debit to \d+ reference no \d+$", None),
    ("internal_ib_transfer_tfd", "internal transfer", r"^ib transfer \d+ to \d{3}-\d{3}-\d+ \d+:\d+(?:am|pm) tfd$", None),
    ("internal_ib_transfer_tfc", "internal transfer", r"^ib transfer \d+ from \d{3}-\d{3}-\d+ \d+:\d+(?:am|pm) tfc$", None),
    ("internal_transfer_to_cba_ac", "internal transfer", r"^transfer to cba a/c commbank app$", None),
    ("internal_transfer_from_commbank", "internal transfer", r"^transfer from commbank app$", None),
    ("internal_masked_commbank", "internal transfer", r"\btransfer (to|from) xx\d{4}\b", None),
    # Westpac withdrawal patterns: debit direction, still mostly internal (92-98%).
    ("internal_westpac_family", "internal transfer", r"\b(tfr westpac|westpac lif|maximiser)\b", None),
    ("internal_from_account", "internal transfer", r"\bfrom account \d+ .* internal transfer\b", None),
    ("internal_tfd", "internal transfer", r"^\. tfd$", None),
    ("internal_tfc", "internal transfer", r"^\. tfc$", None),
    ("internal_me_tfd", "internal transfer", r"^me tfd$", None),
    ("internal_x_tfc", "internal transfer", r"^x tfc$", None),
    ("internal_save_tfc", "internal transfer", r"^save tfc$", None),
    ("internal_j_tfd", "internal transfer", r"^j tfd$", None),

    # Relaxed name+code: allows extra text after the transaction code.
    # Precision ~95.3% — kept as medium due to some internal lookalikes.
    ("external_transfer_name_code_relaxed", "external transfer", r"^transfer (debit|credit) (?!online )[a-z].*?[a-z]\d{10,}", None),
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

def classify_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transfer classification rules and produce output columns."""
    classifier = FixedRuleClassifier()
    result = classifier.predict_frame(df)

    output = result.copy()

    # ── core classification flags ──
    output["is_transfer_pred"] = (
        output["predicted_category"].notna().astype(int)
    )
    output["finv_category"] = output["predicted_category"].fillna("")

    # ── counterparty ──
    output["counterparty"] = _derive_counterparty(output)

    # ── rule metadata ──
    output["transfer_rule_name"] = output["prediction_rule"].fillna("")
    output["transfer_pred_reason"] = output.apply(_build_reason, axis=1)

    # ── stream id ──
    output["stream_id"] = output["finv_category"].where(
        output["is_transfer_pred"].eq(1), ""
    )

    return output


# =============================================================================
# Helpers
# =============================================================================

def _derive_counterparty(df: pd.DataFrame) -> pd.Series:
    """Extract a short counterparty label from the transaction text."""
    text_col = df.get("text_norm", df.get("text", pd.Series("", index=df.index)))
    return (
        text_col.fillna("").astype(str).str.strip().str.upper().str[:80]
    )


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
