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


def apply_special_rules(df):
    output = df.copy()
    for row_id, row in output.iterrows():
        product_type_override = resolve_product_type_override(row)
        if product_type_override is not None:
            output.at[row_id, "product_type"] = product_type_override

    return output
