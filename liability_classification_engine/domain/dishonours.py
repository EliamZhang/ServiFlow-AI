import csv
import re
from decimal import Decimal, InvalidOperation

import pandas as pd

FIELD_NAME = "is_dishonours"

MATCH_AMOUNT_TOLERANCE = Decimal("0")
MATCH_MAX_DATE_GAP_DAYS = 1


def load_rules(rules_file):
    rules = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rule_type = (row.get("rule_type") or "").strip().lower()
            pattern = row.get("pattern") or ""
            required_terms = [x.strip().lower() for x in (row.get("required_terms") or "").split(";") if x.strip()]
            if rule_type and pattern:
                rules.append((rule_type, pattern, required_terms))
    return rules


def is_dishonour(text, rules):
    text = "" if pd.isna(text) else str(text)
    lower_text = text.lower()
    for rule_type, pattern, required_terms in rules:
        if rule_type == "keyword" and pattern.lower() in lower_text:
            return "Yes"
        if rule_type == "regex" and all(term in lower_text for term in required_terms) and re.search(pattern, text):
            return "Yes"
    return "No"


def _parse_decimal(value):
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _amount_within_tolerance(a, b):
    if a is None or b is None or b == 0:
        return False
    ratio = abs(a) / abs(b)
    return Decimal("1") - MATCH_AMOUNT_TOLERANCE <= ratio <= Decimal("1") + MATCH_AMOUNT_TOLERANCE


def _backfill_dishonour_counterparty(df):
    output = df.copy()
    is_dishonour = output.get(FIELD_NAME, pd.Series("", index=output.index))
    is_dishonour = is_dishonour.astype("string").str.lower().eq("yes")

    cp_col = output.get("counterparty", pd.Series("", index=output.index))
    cp_empty = cp_col.fillna("").astype(str).str.strip().eq("")

    target_mask = is_dishonour & cp_empty
    if not target_mask.any():
        return output

    has_cp = cp_col.fillna("").astype(str).str.strip().ne("")
    reference = output[has_cp].copy()
    reference["_amount_decimal"] = reference["amount"].map(_parse_decimal)
    reference = reference.dropna(subset=["_amount_decimal"])
    reference["_tx_date"] = pd.to_datetime(reference["transaction_date"], errors="coerce")
    reference = reference.dropna(subset=["_tx_date"])

    if reference.empty:
        return output

    max_gap = pd.Timedelta(days=MATCH_MAX_DATE_GAP_DAYS)

    for row_id in output[target_mask].index:
        row = output.loc[row_id]
        amount = _parse_decimal(row.get("amount"))
        tx_date = pd.to_datetime(row.get("transaction_date"), errors="coerce")
        app_id = row.get("application_id")
        acct_id = row.get("bank_account_id")

        if amount is None or pd.isna(tx_date):
            continue

        candidates = reference[
            (reference["application_id"] == app_id)
            & (reference["bank_account_id"] == acct_id)
        ]
        if candidates.empty:
            continue

        candidates = candidates[
            (candidates["_tx_date"] - tx_date).abs() <= max_gap
        ]
        if candidates.empty:
            continue

        candidates = candidates.copy()
        candidates["_amount_diff"] = candidates["_amount_decimal"].map(
            lambda ref_amount: abs(ref_amount - abs(amount))
        )
        candidates = candidates.sort_values(
            ["_amount_diff", "_tx_date"],
            kind="stable",
        )
        best = candidates.iloc[0]
        output.at[row_id, "counterparty"] = best["counterparty"]
        output.at[row_id, "product_type"] = best["product_type"]

    return output


def apply_dishonour_rules(df, rules_file):
    rules = load_rules(rules_file)
    output = df.copy()
    text_values = output.get("text", pd.Series("", index=output.index))
    output[FIELD_NAME] = text_values.map(lambda text: is_dishonour(text, rules))
    output = _backfill_dishonour_counterparty(output)
    return output
