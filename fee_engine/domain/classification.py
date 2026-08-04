# -*- coding: utf-8 -*-
"""
Rule-based fee classification for bank transactions.

Classifies transactions into:
- Overdrawn (checked first — overrides generic fee)
- fee

Uses regex rules in priority order to identify fee transactions from text alone.
Overdrawn-related fees (overdrawn, overlimit, overdraft, overdraw, debit excess
interest) are checked FIRST so they override the generic "Fees" category.

Fee types include: overdrawn/overlimit fees, international transaction fees,
ATM operator fees, bank account fees, dishonour fees, late payment fees,
cash advance fees, and third-party maintenance/membership fees.

This module is invoked by the unified engine pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from classification_core.reasons import format_classification_reason


# =============================================================================
# Fee classification rules (ordered by specificity — first match wins)
# =============================================================================

# Each rule is a 4-tuple: (rule_name, category, pattern, counterparty_label)
# counterparty_label is the extracted counterparty name for the fee type.

FeeRule = tuple[str, str, str, str]

FEE_RULES: list[FeeRule] = [
    # =========================================================================
    # 0. OVERDRAWN / OVERLIMIT FEES — must run FIRST so Overdrawn category
    #    overrides the generic "Fees" category.
    # =========================================================================
    # "HONOUR/OVERDRAWN FEE"
    (
        "honour_overdrawn_fee",
        "Overdrawn",
        r"^HONOUR/OVERDRAWN\s+FEE",
        "Overdrawn Fee",
    ),
    # "OVERLIMIT FEE"
    (
        "overlimit_fee",
        "Overdrawn",
        r"^OVERLIMIT\s+FEE",
        "Overlimit Fee",
    ),
    # "Overdrawn Fee"
    (
        "overdrawn_fee",
        "Overdrawn",
        r"^Overdrawn\s+Fee",
        "Overdrawn Fee",
    ),
    # "Overdraft Usage Fee"
    (
        "overdraft_usage_fee",
        "Overdrawn",
        r"^Overdraft\s+Usage\s+Fee",
        "Overdraft Usage Fee",
    ),
    # "Overdraw Fee For exceeding available funds on 21 Nov"
    (
        "overdraw_fee",
        "Overdrawn",
        r"^Overdraw\s+Fee\s+",
        "Overdrawn Fee",
    ),
    # "OVERDRAWN FEE 19-JANUARY-2026"
    (
        "overdrawn_fee_date",
        "Overdrawn",
        r"^OVERDRAWN\s+FEE\s+\d{1,2}-",
        "Overdrawn Fee",
    ),
    # "Debit Excess Interest" — overdrawn interest charge
    (
        "debit_excess_interest",
        "Overdrawn",
        r"^Debit\s+Excess\s+Interest$",
        "Overdrawn Fee",
    ),
    # "Debit Excess Int Adjusted Value Date: 01/11/2025"
    (
        "debit_excess_int_adjusted",
        "Overdrawn",
        r"^Debit\s+Excess\s+Int\s+Adjusted\s+Value\s+Date:\s+\d{2}/\d{2}/\d{4}",
        "Overdrawn Fee",
    ),

    # =========================================================================
    # 1. INTL TXN FEE — highly specific bank-generated format
    #    "FEES V2965 02/05 INTL TXN FEE-MC 24064666122"
    #    "MISCELLANEOUS CREDIT V2965 01/05 INTL TXN FEE REV-MC 74871156121"
    # =========================================================================
    (
        "intl_txn_fee_standard",
        "Fees",
        r"^FEES\s+V\d{4}\s+\d{2}/\d{2}\s+INTL\s+TXN\s+FEE-(MC|SC)\s+\d+$",
        "International Transaction Fee",
    ),
    (
        "intl_txn_fee_reversal",
        "Fees",
        r"^MISCELLANEOUS\s+CREDIT\s+V\d{4}\s+\d{2}/\d{2}\s+INTL\s+TXN\s+FEE\s+REV-(MC|SC)\s+\d+$",
        "International Transaction Fee Refund",
    ),

    # =========================================================================
    # 2. ATM OPERATOR FEE — various bank/system formats
    # =========================================================================
    # "ATM OPERATOR FEE WITHDRAWAL COOLUM HOTEL"
    (
        "atm_operator_fee_withdrawal",
        "Fees",
        r"^ATM\s+OPERATOR\s+FEE\s+WITHDRAWAL\s+",
        "ATM Operator Fee",
    ),
    # "ATM OPERATOR FEE - WITHDRAWAL AT ETX ATM DICKY BEA ..."
    (
        "atm_operator_fee_dash_withdrawal",
        "Fees",
        r"^ATM\s+OPERATOR\s+FEE\s+-\s+WITHDRAWAL\s+AT\s+",
        "ATM Operator Fee",
    ),
    # "ATM OPERATOR FEE -16:05"
    (
        "atm_operator_fee_time",
        "Fees",
        r"^ATM\s+OPERATOR\s+FEE\s+-\d{2}:\d{2}",
        "ATM Operator Fee",
    ),
    # "Atm Operator Fee Wdl 19Mar22:45 Pioneer Tavern ..."
    # "Atm Operator Fee Inq 09May15:44 Pioneer Tavern ..."
    (
        "atm_operator_fee_wdl",
        "Fees",
        r"^Atm\s+Operator\s+Fee\s+(?:Wdl|Inq)\s+",
        "ATM Operator Fee",
    ),
    # "ATM Operator Fee - Np - Foodworks Sheppartsheppar / 6676"
    (
        "atm_operator_fee_np",
        "Fees",
        r"^ATM\s+Operator\s+Fee\s+-\s+Np\s+-",
        "ATM Operator Fee",
    ),
    # "ATM Operator Fee - Coles Sm - Shepparton Shepp / 3254"
    (
        "atm_operator_fee_coles",
        "Fees",
        r"^ATM\s+Operator\s+Fee\s+-\s+Coles\s+",
        "ATM Operator Fee",
    ),
    # "ATM owner fee of $3.40 charged by ..."
    (
        "atm_owner_fee",
        "Fees",
        r"ATM\s+owner\s+fee\s+of\s+\$",
        "ATM Operator Fee",
    ),
    # "FEES BBL ATM 19th13:28atmx 582 CHAPMAN ROA DIR CHG OTH ATM"
    (
        "fees_bbl_atm",
        "Fees",
        r"^FEES\s+BBL\s+ATM\s+",
        "ATM Operator Fee",
    ),
    # "FEES EFTEX ATM 06th17:42UP RANGEWAY DIR CHG OTH ATM"
    (
        "fees_eftex_atm",
        "Fees",
        r"^FEES\s+EFTEX\s+ATM\s+",
        "ATM Operator Fee",
    ),
    # "FEES SML ATM 16th11:39AMPOL - MAIDSTONE DIR CHG OTH ATM"
    (
        "fees_sml_atm",
        "Fees",
        r"^FEES\s+SML\s+ATM\s+",
        "ATM Operator Fee",
    ),
    # "OFI ATM W/D TRAN FOR $53.50 INCLUDES OFI ATM OPERATOR FEE OF $3.50"
    (
        "ofi_atm_operator_fee",
        "Fees",
        r"OFI\s+ATM\s+.*INCLUDES\s+OFI\s+ATM\s+OPERATOR\s+FEE\s+OF\s+\$",
        "ATM Operator Fee",
    ),
    # "NON-ANZ ATM UP GAGEBROOK ... INCLUDES ATM OPERATOR CHARGE $3.35"
    (
        "non_anz_atm_operator_charge",
        "Fees",
        r"NON-ANZ\s+ATM\s+.*INCLUDES\s+ATM\s+OPERATOR\s+(?:FEE|CHARGE)\s+\$",
        "ATM Operator Fee",
    ),

    # =========================================================================
    # 3. International Transaction / Currency Conversion Fees
    # =========================================================================
    # "International Transaction Fee Value Date: 02/03/2026"
    (
        "international_txn_fee",
        "Fees",
        r"^International\s+Transaction\s+Fee",
        "International Transaction Fee",
    ),
    # "International ATM Withdrawal Fee"
    (
        "international_atm_fee",
        "Fees",
        r"^International\s+ATM\s+Withdrawal\s+Fee",
        "International ATM Fee",
    ),
    # "INTNL TRANSACTION FEE"
    (
        "intnl_txn_fee",
        "Fees",
        r"^INTNL\s+TRANSACTION\s+FEE",
        "International Transaction Fee",
    ),
    # "Foreign Currency Conversn Fee"
    (
        "foreign_currency_conversn_fee",
        "Fees",
        r"^Foreign\s+Currency\s+Convers?n\s+Fee",
        "Foreign Currency Conversion Fee",
    ),
    # "Overseas Currency Conversion Fee"
    (
        "overseas_currency_fee",
        "Fees",
        r"^Overseas\s+Currency\s+Conversion\s+Fee",
        "Foreign Currency Conversion Fee",
    ),
    # "Non CBA ATM Withdrawal Fee"
    (
        "non_cba_atm_fee",
        "Fees",
        r"^Non\s+CBA\s+ATM\s+Withdrawal\s+Fee",
        "Non-CBA ATM Fee",
    ),
    # "IRD Conv Fee Auckland NZL..."
    (
        "ird_conv_fee",
        "Fees",
        r"^IRD\s+Conv\s+Fee\s+",
        "International Transaction Fee",
    ),
    # "RTGS payment fee"
    (
        "rtgs_payment_fee",
        "Fees",
        r"^RTGS\s+payment\s+fee",
        "RTGS Payment Fee",
    ),

    # =========================================================================
    # 4. FOREIGN FEE — statement line items
    #    "FOREIGN FEE AUD 2.51"
    #    "FOREIGN FEE AUD 0.00 FRGN AMT: 15.00 U. S. DOLLAR"
    # =========================================================================
    (
        "foreign_fee_aud",
        "Fees",
        r"^FOREIGN\s+FEE\s+AUD\s+",
        "Foreign Currency Fee",
    ),

    # =========================================================================
    # 5. "Includes Foreign Currency Conversion Fee $X.XX"
    # =========================================================================
    (
        "includes_foreign_currency_fee",
        "Fees",
        r"^Includes\s+Foreign\s+Currency\s+Conversion\s+Fee\s+\$",
        "Foreign Currency Conversion Fee",
    ),

    # =========================================================================
    # 6. "FEES INCLUDED IN TRAN USD/AUD/GBP/EUR... IS WAIVED. FX FEE IS A$..."
    # =========================================================================
    (
        "fees_included_waived",
        "Fees",
        r"^FEES\s+INCLUDED\s+IN\s+TRAN\s+(?:USD|AUD|GBP|EUR|NZD)",
        "Foreign Currency Fee",
    ),

    # =========================================================================
    # 7. Bank Account / Service Fees
    # =========================================================================
    # "MONTHLY FEE"
    (
        "monthly_fee",
        "Fees",
        r"^MONTHLY\s+FEE$",
        "Monthly Account Fee",
    ),
    # "MONTHLY CARD FEE"
    (
        "monthly_card_fee",
        "Fees",
        r"^MONTHLY\s+CARD\s+FEE$",
        "Monthly Card Fee",
    ),
    # "FEES MONTHLY CARD/CREDIT FEE"
    (
        "fees_monthly_card_fee",
        "Fees",
        r"^FEES\s+MONTHLY\s+CARD/CREDIT\s+FEE",
        "Monthly Card Fee",
    ),
    # "LOAN ACCOUNT FEE"
    (
        "loan_account_fee",
        "Fees",
        r"^LOAN\s+ACCOUNT\s+FEE$",
        "Loan Service Fee",
    ),
    # "FEES LOAN SERVICE FEE"
    (
        "fees_loan_service_fee",
        "Fees",
        r"^FEES\s+LOAN\s+SERVICE\s+FEE",
        "Loan Service Fee",
    ),
    # "ACCOUNT SERVICING FEE"
    (
        "account_servicing_fee",
        "Fees",
        r"^ACCOUNT\s+SERVICING\s+FEE",
        "Account Service Fee",
    ),
    # "CARD FEE"
    (
        "card_fee",
        "Fees",
        r"^CARD\s+FEE$",
        "Card Fee",
    ),
    # "Annual Fee"
    (
        "annual_fee",
        "Fees",
        r"^Annual\s+Fee$",
        "Annual Fee",
    ),
    # "Bank@Post Deposit Fee"
    (
        "bankpost_deposit_fee",
        "Fees",
        r"^Bank@Post\s+Deposit\s+Fee",
        "Bank@Post Fee",
    ),
    # "Bank@Post Withdrawal Fee"
    (
        "bankpost_withdrawal_fee",
        "Fees",
        r"^Bank@Post\s+Withdrawal\s+Fee",
        "Bank@Post Fee",
    ),
    # "Fee for servicing your loan"
    (
        "fee_servicing_loan",
        "Fees",
        r"^Fee\s+for\s+servicing\s+your\s+loan",
        "Loan Service Fee",
    ),
    # "LENDING ESTABLISHMENT FEE"
    (
        "lending_establishment_fee",
        "Fees",
        r"^LENDING\s+ESTABLISHMENT\s+FEE",
        "Establishment Fee",
    ),

    # =========================================================================
    # 8. Cash Advance Fees
    # =========================================================================
    # "CBA OTHER CASH ADV FEE"
    (
        "cba_cash_adv_fee",
        "Fees",
        r"^CBA\s+OTHER\s+CASH\s+ADV\s+FEE",
        "Cash Advance Fee",
    ),
    # "CASH ADVANCE FEE"
    (
        "cash_advance_fee_upper",
        "Fees",
        r"^CASH\s+ADVANCE\s+FEE",
        "Cash Advance Fee",
    ),
    # "Cash Advance Fee"
    (
        "cash_advance_fee_title",
        "Fees",
        r"^Cash\s+Advance\s+Fee",
        "Cash Advance Fee",
    ),

    # =========================================================================
    # 10. Late Payment Fees
    # =========================================================================
    # "LATE PAYMENT FEE"
    (
        "late_payment_fee",
        "Fees",
        r"^LATE\s+PAYMENT\s+FEE",
        "Late Payment Fee",
    ),
    # "Late payment fee"
    (
        "late_payment_fee_lower",
        "Fees",
        r"^Late\s+payment\s+fee",
        "Late Payment Fee",
    ),
    # "LATE FEE"
    (
        "late_fee",
        "Fees",
        r"^LATE\s+FEE$",
        "Late Payment Fee",
    ),
    # "STEPPAY LATE FEE"
    (
        "steppay_late_fee",
        "Fees",
        r"^STEPPAY\s+LATE\s+FEE",
        "Late Payment Fee",
    ),
    # "MISSED PAYMENT FEE"
    (
        "missed_payment_fee",
        "Fees",
        r"^MISSED\s+PAYMENT\s+FEE",
        "Late Payment Fee",
    ),

    # =========================================================================
    # 12. CommBank AdvancePay Fee
    # =========================================================================
    (
        "commbank_advancepay_fee",
        "Fees",
        r"^CommBank\s+AdvancePay\s+Fee",
        "AdvancePay Fee",
    ),

    # =========================================================================
    # 13. Raiz Maintenance Fee
    #     "Direct Debit 342120 Raiz Maint Fee a6ba6360c919ece808"
    #     "AUTOMATIC DRAWING 651abb0fe62c40d9d2 Raiz Maint Fee Daniel Iffland"
    # =========================================================================
    (
        "raiz_maint_fee_direct_debit",
        "Fees",
        r"Direct\s+Debit\s+\d+\s+Raiz\s+Maint\s+Fee\s+",
        "Raiz Maintenance Fee",
    ),
    (
        "raiz_maint_fee_automatic",
        "Fees",
        r"^AUTOMATIC\s+DRAWING\s+\w+\s+Raiz\s+Maint\s+Fee",
        "Raiz Maintenance Fee",
    ),

    # =========================================================================
    # 14. UWU Union Fees
    #     "AUTOMATIC DRAWING UWDues924244500004 UWU FEES Jose Varughese"
    #     "PAYMENT BY AUTHORITY TO UWU FEES UWDues923467500168"
    # =========================================================================
    (
        "uwu_fees_automatic",
        "Fees",
        r"^AUTOMATIC\s+DRAWING\s+UWDues\d+\s+UWU\s+FEES",
        "Union Membership Fee",
    ),
    (
        "uwu_fees_authority",
        "Fees",
        r"^PAYMENT\s+BY\s+AUTHORITY\s+TO\s+UWU\s+FEES",
        "Union Membership Fee",
    ),

    # =========================================================================
    # 15. Fee Refunds / Reversals
    # =========================================================================
    # "REFUND OF FEE CHARGED ON 012009 RH *cnfans.com London GBRUSD"
    (
        "refund_of_fee",
        "Fees",
        r"^REFUND\s+OF\s+FEE\s+",
        "Fee Refund",
    ),
    # "LATE FEE REVERSAL"
    (
        "late_fee_reversal",
        "Fees",
        r"^LATE\s+FEE\s+REVERSAL",
        "Fee Refund",
    ),

    # =========================================================================
    # 16. Fee Waivers
    # =========================================================================
    # "MONTHLY FEE WAIVED"
    (
        "monthly_fee_waived",
        "Fees",
        r"^MONTHLY\s+FEE\s+WAIVED",
        "Fee Waived",
    ),
    # "WAIVE: HONOUR/OVERDRAWN FEE"
    (
        "waive_overdrawn_fee",
        "Fees",
        r"^WAIVE:\s+HONOUR/OVERDRAWN\s+FEE",
        "Fee Waived",
    ),

    # =========================================================================
    # 17. ATM Charge — ATM fees described as "Charge" rather than "Fee"
    #     "ATM Charge TOWNSEND GENERAL STORE TOWNSEND AU"
    #     "ATM Charge NP-Mindil Beach Casino6Darwin 08AU"
    # =========================================================================
    (
        "atm_charge",
        "Fees",
        r"^ATM\s+Charge\s+",
        "ATM Operator Fee",
    ),
    # "ATM Owners charge - WDL 005725/0305143659 0200 0305153659"
    (
        "atm_owners_charge",
        "Fees",
        r"^ATM\s+Owners\s+charge\s+-\s+WDL\s+",
        "ATM Operator Fee",
    ),


    # =========================================================================
    # 22. LOAN ADMINISTRATION CHARGE — loan admin fee
    # =========================================================================
    (
        "loan_administration_charge",
        "Fees",
        r"^LOAN\s+ADMINISTRATION\s+CHARGE",
        "Loan Administration Charge",
    ),

    # =========================================================================
    # 23. "FX FEE IS A$X.XX" — embedded FX fee mention in statement lines
    #     (catch-all for FEES INCLUDED IN TRAN lines with non-USD currencies
    #      not covered by fees_included_waived above)
    # =========================================================================
    (
        "fx_fee_is",
        "Fees",
        r"FX\s+FEE\s+IS\s+A\$",
        "Foreign Currency Fee",
    ),

    # =========================================================================
    # 24. INTEREST CHARGES — credit card / loan / overdraft interest.
    #     illion classifies interest as "Fees"; finv follows the same
    #     convention.  Rules are ordered from most specific to most generic
    #     so that explicit patterns (e.g. "INTEREST CHARGES - PUR CH") match
    #     before bare "INTEREST".
    # =========================================================================
    # "INTEREST CHARGES - PUR CH" / "INTEREST CHARGES - PURCH"
    (
        "interest_charges_purch",
        "Fees",
        r"^INTEREST\s+CHARGES\s+-\s+PUR",
        "Interest Charges",
    ),
    # "INTEREST CHARGES - CAS H" / "INTEREST CHARGES - CASH"
    (
        "interest_charges_cash",
        "Fees",
        r"^INTEREST\s+CHARGES\s+-\s+CAS",
        "Interest Charges",
    ),
    # "Interest Charges - Purch" (title case)
    (
        "interest_charges_purch_title",
        "Fees",
        r"^Interest\s+Charges\s+-\s+Purch",
        "Interest Charges",
    ),
    # "Interest Charges - Cash" (title case)
    (
        "interest_charges_cash_title",
        "Fees",
        r"^Interest\s+Charges\s+-\s+Cash",
        "Interest Charges",
    ),
    # "INTEREST ON CASH ADV"
    (
        "interest_on_cash_adv",
        "Fees",
        r"^INTEREST\s+ON\s+CASH\s+ADV",
        "Interest Charges",
    ),
    # "CASH ADVANCE INTEREST"
    (
        "cash_advance_interest",
        "Fees",
        r"^CASH\s+ADVANCE\s+INTEREST",
        "Interest Charges",
    ),
    # "VISA PURCHASE INTEREST"
    (
        "visa_purchase_interest",
        "Fees",
        r"^VISA\s+PURCHASE\s+INTEREST",
        "Interest Charges",
    ),
    # "INSTALMENT PLAN INTEREST"
    (
        "instalment_plan_interest",
        "Fees",
        r"^INSTALMENT\s+PLAN\s+INTEREST",
        "Interest Charges",
    ),
    # "INTEREST CHARGED ON PURCHASES"
    (
        "interest_charged_on_purchases",
        "Fees",
        r"^INTEREST\s+CHARGED\s+ON\s+PURCHASES",
        "Interest Charges",
    ),
    # "INTEREST - BASE PLAN"
    (
        "interest_base_plan",
        "Fees",
        r"^INTEREST\s+-\s+BASE\s+PLAN",
        "Interest Charges",
    ),
    # "INTEREST CHARGED INTEREST CHARGED" (duplicated text from some banks)
    (
        "interest_charged_dup",
        "Fees",
        r"^INTEREST\s+CHARGED\s+INTEREST\s+CHARGED",
        "Interest Charges",
    ),
    # "INTEREST CHARGED" (bare, singular — before the plural "INTEREST CHARGES")
    (
        "interest_charged_bare",
        "Fees",
        r"^INTEREST\s+CHARGED$",
        "Interest Charges",
    ),
    # "DEBIT INTEREST CHARGED"
    (
        "debit_interest_charged",
        "Fees",
        r"^DEBIT\s+INTEREST\s+CHARGED",
        "Interest Charges",
    ),
    # "Debit Interest" (title case, plain — distinct from "Debit Excess Interest"
    # which is already caught as Overdrawn in section 0)
    (
        "debit_interest_title",
        "Fees",
        r"^Debit\s+Interest$",
        "Interest Charges",
    ),
    # "Interest charged" (title case, bare)
    (
        "interest_charged_title",
        "Fees",
        r"^Interest\s+charged$",
        "Interest Charges",
    ),
    # "INTEREST CHARGES" (bare, uppercase)
    (
        "interest_charges_bare",
        "Fees",
        r"^INTEREST\s+CHARGES$",
        "Interest Charges",
    ),
    # "INTEREST DEBIT"
    (
        "interest_debit",
        "Fees",
        r"^INTEREST\s+DEBIT$",
        "Interest Charges",
    ),
    # "INTEREST" (bare, uppercase — most generic, placed last)
    (
        "interest_bare",
        "Fees",
        r"^INTEREST$",
        "Interest Charges",
    ),
    # "Interest" (bare, title case — low-amount residual interest)
    (
        "interest_bare_title",
        "Fees",
        r"^Interest$",
        "Interest Charges",
    ),
]


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class FeePrediction:
    is_fee: bool
    category: str | None
    counterparty: str | None
    rule_name: str | None


# =============================================================================
# Text normalization
# =============================================================================

def normalize_text(value: object) -> str:
    """Normalize text for stable rule matching — preserve original case."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# =============================================================================
# Rules that should be rejected when the transaction amount is $0.00.
#
# These rules match text patterns that describe fees *included* in other
# transactions, *waived* fees, or informational notes — not actual fee
# charges.  When the amount is zero (or very close to it), the row is an
# informational line-item, not a real fee.
# =============================================================================

_AMOUNT_ZERO_REJECT_RULES: set[str] = {
    # "Includes Foreign Currency Conversion Fee $X.XX" — the fee was already
    # included in another transaction; this is a note, not a charge.
    "includes_foreign_currency_fee",
    # "FEES INCLUDED IN TRAN USD11.03 IS WAIVED. FX FEE IS A$0.47"
    # — the fee was waived, not charged.
    "fees_included_waived",
    # "MONTHLY FEE WAIVED" — not actually charged.
    "monthly_fee_waived",
    # "OFI ATM W/D TRAN FOR $X.XX INCLUDES OFI ATM OPERATOR FEE OF $X.XX"
    # — describes a fee embedded in another transaction, not a standalone fee.
    "ofi_atm_operator_fee",
    # "FOREIGN FEE AUD X.XX" — when $0 this is an informational duplicate
    # of a fee line already captured elsewhere.
    "foreign_fee_aud",
    # "INTEREST" / "Interest" — $0 informational interest notes.
    "interest_bare",
    "interest_bare_title",
    # "LOAN ADMINISTRATION CHARGE" — when $0 it's informational.
    "loan_administration_charge",
}


# =============================================================================
# Fee classifier
# =============================================================================

_CompiledFeeRule = tuple[str, str, re.Pattern, str]


class FeeClassifier:
    """Apply fee classification rules in priority order.

    Rules are 4-tuples: (rule_name, category, pattern, counterparty_label).
    """

    def __init__(self) -> None:
        self.rules: list[_CompiledFeeRule] = [
            (name, category, re.compile(pattern), counterparty)
            for name, category, pattern, counterparty in FEE_RULES
        ]

    def predict(self, text: str) -> FeePrediction:
        """Apply rules in priority order — first match wins."""
        for rule_name, category, pattern, counterparty in self.rules:
            if pattern.search(text):
                return FeePrediction(
                    is_fee=True,
                    category=category,
                    counterparty=counterparty,
                    rule_name=rule_name,
                )
        return FeePrediction(
            is_fee=False,
            category=None,
            counterparty=None,
            rule_name=None,
        )


# =============================================================================
# Pipeline entry point
# =============================================================================

def classify_fees(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fee classification rules and produce output columns."""
    classifier = FeeClassifier()

    output = df.copy()
    raw_text = output.get("text", pd.Series("", index=output.index))
    output["_text_original"] = raw_text
    output["text_norm"] = raw_text.apply(normalize_text)

    predictions = [
        classifier.predict(text)
        for text in output["text_norm"]
    ]

    output["is_fee_pred"] = [int(p.is_fee) for p in predictions]
    output["finv_category"] = [
        p.category if p.is_fee else "" for p in predictions
    ]
    output["counterparty"] = [
        p.counterparty if p.is_fee else "" for p in predictions
    ]
    output["fee_rule_name"] = [
        p.rule_name if p.is_fee else "" for p in predictions
    ]

    # ── Post-processing: reject $0 informational fee lines ──────────
    _reject_zero_amount_informational(output)

    output["fee_pred_reason"] = output.apply(_build_reason, axis=1)

    # stream_id (legacy value "fee" — keep as-is for baseline parity)
    output["stream_id"] = output["finv_category"].map(
        {"Fees": "fee"}
    ).where(output["is_fee_pred"].eq(1), "")

    # Drop internal columns
    output = output.drop(columns=["text_norm", "_text_original"])

    return output


def _reject_zero_amount_informational(df: pd.DataFrame) -> None:
    """Unset fee predictions whose amount is $0 and rule is informational.

    Certain fee-rule patterns match informational line-items (e.g. "Includes
    Foreign Currency Conversion Fee $0.81") where the transaction amount is
    $0.00 — these are notes attached to other transactions, NOT real fee
    charges.  This function unsets the prediction so the row can be picked
    up by a later engine or left unclassified.
    """
    if "amount" not in df.columns:
        return

    # Only consider rows currently marked as fee.
    fee_mask = df["is_fee_pred"].eq(1)
    if not fee_mask.any():
        return

    # Find rows where the rule is in our reject set AND amount is zero.
    amount_col = pd.to_numeric(df["amount"], errors="coerce")
    zero_amount_mask = (
        amount_col.abs() < 0.001  # near-zero — informational lines
    )
    reject_rule_mask = df["fee_rule_name"].isin(_AMOUNT_ZERO_REJECT_RULES)

    reject_mask = fee_mask & zero_amount_mask & reject_rule_mask
    if not reject_mask.any():
        return

    # Unset the prediction columns for rejected rows.
    df.loc[reject_mask, "is_fee_pred"] = 0
    df.loc[reject_mask, "finv_category"] = ""
    df.loc[reject_mask, "counterparty"] = ""
    df.loc[reject_mask, "fee_rule_name"] = ""


def _build_reason(row: pd.Series) -> str:
    if int(row.get("is_fee_pred", 0)) != 1:
        return format_classification_reason(
            category="not_fee",
            rule="no_fee_rule_matched",
            evidence=[],
        )

    rule_name = str(row.get("fee_rule_name", ""))
    category = str(row.get("finv_category", ""))
    counterparty = str(row.get("counterparty", ""))

    return format_classification_reason(
        category=category,
        rule=rule_name,
        evidence=[f"counterparty={counterparty}"],
    )
