import re

import pandas as pd


# ---------------------------------------------------------------------------
# Cash Converters: retail vs lending
# ---------------------------------------------------------------------------

# Cash Converters operates two business lines:
#   - Lending (loan repayment): text contains contract IDs like B31470T1949,
#     no store location, no POS terminal markers.
#   - Retail (in-store purchase): text contains store locations (Perth,
#     Noosa, Morningside, etc.), Square (SQ *) or EFTPOS terminals, and/or
#     card tail numbers (Card xx1234).

_AU_LOCATIONS = (
    r"PERTH|NOOSA|MORNINGSIDE|WYNNUM|SHEPPAR|HELENSVALE"
    r"|MELBOURNE|SYDNEY|BRISBANE|ADELAIDE|GOLD\s*COAST"
    r"|NEWCASTLE|CANBERRA|HOBART|DARWIN|GEELONG"
    r"|WOLLONGONG|TOWNSVILLE|CAIRNS|BALLARAT|BENDIGO"
    r"|ALBURY|LAUNCESTON|NOOSAVILLE|MAROOCHYDORE|IPSWICH"
    r"|TOOWOOMBA|ROCKHAMPTON|MACKAY|BUNBURY|MANDURAH"
)

_AU_LOCATION_RE = re.compile(r"(?<!\d)(?:" + _AU_LOCATIONS + r")(?!\d)", re.IGNORECASE)
_LOAN_CONTRACT_RE = re.compile(r"B\d{4,}T\d+", re.IGNORECASE)
_SQ_TERMINAL_RE = re.compile(r"SQ\s*\*\s*CASH\s*CONVERTERS", re.IGNORECASE)
_EFTPOS_TERMINAL_RE = re.compile(r"EFTPOS\s*WDL\s*CASH\s*CONVERTERS", re.IGNORECASE)
_CARD_TAIL_RE = re.compile(r"CARD\s*XX\d+", re.IGNORECASE)


def apply_special_rules(df):
    """Apply special-case rules using vectorised operations.

    The original implementation used ``iterrows()`` to inspect every row
    individually.  This version achieves the identical result with boolean
    masks applied to the full DataFrame in a single pass per rule.
    """
    output = df.copy()

    counterparty_col = output["counterparty"].fillna("").astype(str)
    text_col = output["text"].fillna("").astype(str)

    # ---- Credit Corp sub-product overrides ----
    cc_mask = counterparty_col.str.strip().str.lower().eq("credit corp")
    if cc_mask.any():
        cc_text = text_col.str.lower()

        # wizit / wizitca  → bnpl
        wizit_mask = cc_mask & cc_text.str.contains("wizit", na=False)
        output.loc[wizit_mask, "product_type"] = "bnpl"

        # pup  → loc  (only rows not already claimed by wizit)
        pup_mask = cc_mask & ~wizit_mask & cc_text.str.contains("pup", na=False)
        output.loc[pup_mask, "product_type"] = "loc"

        # ccc  → personal_loan  (only rows not already claimed)
        ccc_mask = (
            cc_mask & ~wizit_mask & ~pup_mask & cc_text.str.contains("ccc", na=False)
        )
        output.loc[ccc_mask, "product_type"] = "personal_loan"

    # ---- Cash Converters retail ----
    cashies_mask = counterparty_col.str.strip().str.lower().eq("cash converters")
    if cashies_mask.any():
        cashies_text = text_col

        # Loan contract IDs (B31470T1949, etc.) indicate lending — NOT retail.
        is_loan_contract = cashies_text.str.contains(
            _LOAN_CONTRACT_RE.pattern, na=False, regex=True, flags=re.IGNORECASE
        )

        # Retail indicators.
        is_retail = (
            cashies_text.str.contains(
                _SQ_TERMINAL_RE.pattern, na=False, regex=True, flags=re.IGNORECASE
            )
            | cashies_text.str.contains(
                _EFTPOS_TERMINAL_RE.pattern, na=False, regex=True, flags=re.IGNORECASE
            )
            | cashies_text.str.contains(
                _CARD_TAIL_RE.pattern, na=False, regex=True, flags=re.IGNORECASE
            )
            | cashies_text.str.contains(
                _AU_LOCATION_RE.pattern, na=False, regex=True, flags=re.IGNORECASE
            )
        )

        retail_mask = cashies_mask & ~is_loan_contract & is_retail
        output.loc[retail_mask, "counterparty"] = "Cash Converters Retail"
        output.loc[retail_mask, "product_type"] = ""
        output.loc[retail_mask, "finv_category"] = "Retail"

    return output
