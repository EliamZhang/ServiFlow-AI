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
    """Backfill counterparty on dishonour rows via a merge-based nearest match.

    The original implementation looped row-by-row over every dishonour
    transaction and performed iterative DataFrame filtering inside the loop.
    This version achieves the identical result with a single grouped merge
    followed by a ``sort_values`` / ``drop_duplicates`` pattern to select
    the best (closest-amount, then earliest-date) reference row.
    """
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

    # Preserve the original index so we can write results back to *output*.
    dishonour_rows = output.loc[target_mask].copy()
    dishonour_rows["_orig_idx"] = dishonour_rows.index
    dishonour_rows["_amount_decimal"] = dishonour_rows["amount"].map(_parse_decimal)
    dishonour_rows["_tx_date"] = pd.to_datetime(
        dishonour_rows["transaction_date"], errors="coerce"
    )
    dishonour_rows = dishonour_rows.dropna(subset=["_amount_decimal", "_tx_date"])

    if dishonour_rows.empty:
        return output

    # Merge dishonour rows with reference rows on the group keys.
    merge_keys = ["application_id", "bank_account_id"]
    merged = dishonour_rows[
        merge_keys + ["_tx_date", "_amount_decimal", "_orig_idx"]
    ].merge(
        reference[
            merge_keys
            + ["_tx_date", "_amount_decimal", "counterparty", "product_type"]
        ],
        on=merge_keys,
        suffixes=("_dishonour", "_ref"),
    )

    # Restrict to reference rows within the date window.
    date_diff = (merged["_tx_date_ref"] - merged["_tx_date_dishonour"]).abs()
    merged = merged[date_diff <= max_gap]

    if merged.empty:
        return output

    # For each dishonour row pick the best reference: smallest absolute
    # amount difference, then earliest reference date (stable sort).
    merged["_amount_diff"] = (
        merged["_amount_decimal_ref"].abs() - merged["_amount_decimal_dishonour"].abs()
    ).abs()
    merged = merged.sort_values(
        ["_amount_diff", "_tx_date_ref"],
        kind="stable",
    )
    # Keep the best match per original dishonour row, identified by
    # (_orig_idx, _tx_date_dishonour, _amount_decimal_dishonour).
    best = merged.drop_duplicates(
        subset=["_orig_idx", "_tx_date_dishonour", "_amount_decimal_dishonour"],
        keep="first",
    )

    if best.empty:
        return output

    # Write results back to *output* in one vectorised assignment.
    output.loc[best["_orig_idx"].values, "counterparty"] = best["counterparty"].values
    output.loc[best["_orig_idx"].values, "product_type"] = best["product_type"].values

    return output


def apply_dishonour_rules(df, rules_file):
    """Apply dishonour detection rules using vectorised operations.

    The original implementation called ``is_dishonour`` once per row via
    ``.map()``.  This version lowercases the text column once and then
    applies each rule with vectorised ``str.contains`` scans over the
    full column — identical semantics, far fewer Python-level calls.
    """
    rules = load_rules(rules_file)
    output = df.copy()
    text_col = output["text"].fillna("").astype(str)
    lower_text = text_col.str.lower()

    dishonour_mask = pd.Series(False, index=output.index)

    for rule_type, pattern, required_terms in rules:
        if rule_type == "keyword":
            keyword_lower = pattern.lower()
            mask = lower_text.str.contains(
                re.escape(keyword_lower), na=False, regex=True
            )
            dishonour_mask |= mask
        elif rule_type == "regex":
            # All required terms must be present AND the regex must match.
            term_mask = pd.Series(True, index=output.index)
            for term in required_terms:
                term_mask &= lower_text.str.contains(
                    re.escape(term), na=False, regex=True
                )
            regex_mask = text_col.str.contains(pattern, na=False, regex=True)
            dishonour_mask |= term_mask & regex_mask

    output[FIELD_NAME] = dishonour_mask.map({True: "Yes", False: "No"})
    output = _backfill_dishonour_counterparty(output)
    return output
