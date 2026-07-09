import csv
import os

import pandas as pd


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def parse_amount(row):
    value = row.get("amount")
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_rules(row):
    counterparty = normalize_text(row.get("counterparty", ""))
    text = normalize_text(row.get("text")).lower()
    amount = parse_amount(row)
    is_dishonours = normalize_text(row.get("is_dishonours")).strip().lower()
    dr_cr = normalize_text(row.get("dr_cr")).strip().lower()

    if (
        counterparty == "Cash Converters"
        and is_dishonours != "yes"
        and dr_cr == "credit"
        and amount is not None
        and 50 <= amount <= 200
    ):
        row["product_type"] = "wage_advance"

    if counterparty == "Credit Corp":
        if "wizit" in text or "wizitca" in text:
            row["product_type"] = "bnpl"
        elif "pup" in text:
            row["product_type"] = "loc"
        elif "ccc" in text:
            row["product_type"] = "personal_loan"


def apply_special_rules(df):
    output = df.copy()
    for row_id, row in output.iterrows():
        updated_row = row.to_dict()
        apply_rules(updated_row)
        for column, value in updated_row.items():
            output.at[row_id, column] = value

    required_columns = {
        "counterparty",
        "dr_cr",
        "product_type",
        "transaction_date",
    }
    if not required_columns.issubset(output.columns):
        return output

    group_columns = ["counterparty"]
    if "bank_account_id" in output.columns:
        group_columns.insert(0, "bank_account_id")

    output["_special_rule_transaction_date"] = pd.to_datetime(
        output["transaction_date"],
        errors="coerce",
    )

    for _, group in output.groupby(group_columns, dropna=False, sort=False):
        counterparty = normalize_text(group["counterparty"].iloc[0])
        if counterparty != "Cash Converters":
            continue

        dr_cr = group["dr_cr"].astype("string").str.lower()
        product_type = group["product_type"].astype("string")
        wage_advance_credits = group[
            product_type.eq("wage_advance") & dr_cr.eq("credit")
        ]
        if wage_advance_credits.empty:
            continue

        repayment_rows = group[
            product_type.eq("personal_loan")
            & dr_cr.isin(["credit", "debit"])
        ]
        output.loc[repayment_rows.index, "product_type"] = "wage_advance"

    output.drop(columns=["_special_rule_transaction_date"], inplace=True)
    return output


def process_file(input_file, output_file):
    with open(input_file, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0]) if rows else []

    for row in rows:
        apply_rules(row)

    temp_file = output_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_file, output_file)
