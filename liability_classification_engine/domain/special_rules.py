import re

import pandas as pd


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def resolve_product_type_override(row):
    counterparty = normalize_text(row.get("counterparty", ""))
    text = normalize_text(row.get("text")).lower()

    if counterparty == "Credit Corp":
        if "wizit" in text or "wizitca" in text:
            return "bnpl"
        if "pup" in text:
            return "loc"
        if "ccc" in text:
            return "personal_loan"

    return None


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

_AU_LOCATION_RE = re.compile(r"(?<!\d)(" + _AU_LOCATIONS + r")(?!\d)", re.IGNORECASE)
_LOAN_CONTRACT_RE = re.compile(r"B\d{4,}T\d+", re.IGNORECASE)
_SQ_TERMINAL_RE = re.compile(r"SQ\s*\*\s*CASH\s*CONVERTERS", re.IGNORECASE)
_EFTPOS_TERMINAL_RE = re.compile(r"EFTPOS\s*WDL\s*CASH\s*CONVERTERS", re.IGNORECASE)
_CARD_TAIL_RE = re.compile(r"CARD\s*XX\d+", re.IGNORECASE)


def _is_cash_converters_retail(text):
    """Return True if the transaction is a retail in-store purchase."""
    if _LOAN_CONTRACT_RE.search(text):
        return False

    if _SQ_TERMINAL_RE.search(text):
        return True
    if _EFTPOS_TERMINAL_RE.search(text):
        return True
    if _CARD_TAIL_RE.search(text):
        return True
    if _AU_LOCATION_RE.search(text):
        return True

    return False


def apply_special_rules(df):
    output = df.copy()
    for row_id, row in output.iterrows():
        counterparty = normalize_text(row.get("counterparty", ""))
        text = normalize_text(row.get("text"))

        # Credit Corp sub-product overrides
        product_type_override = resolve_product_type_override(row)
        if product_type_override is not None:
            output.at[row_id, "product_type"] = product_type_override

        # Cash Converters retail → keep out of personal-loan stream pipeline
        # and classify as Retail instead.
        if counterparty.lower() == "cash converters":
            if _is_cash_converters_retail(text):
                output.at[row_id, "counterparty"] = "Cash Converters Retail"
                output.at[row_id, "product_type"] = ""
                output.at[row_id, "finv_category"] = "Retail"

    return output
