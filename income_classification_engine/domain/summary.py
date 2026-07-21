from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

SUMMARY_COLUMNS = [
    "finv_category",
    "stream_id",
    "bank_account_id",
    "account_type",
    "application_id",
    "bank",
    "credit_limit",
    "counterparty",
    "transaction_start_date",
    "transaction_end_date",
    "status",
    "transaction_count",
    "total_income_amount",
    "average_income_amount",
    "median_income_amount",
    "latest_income_amount",
    "estimated_monthly_income",
    "frequency",
    "frequency_day",
    "predicted_next_income_date",
]


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def clean_counterparty(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()[:80]


def derive_counterparty(row: pd.Series) -> str:
    if int(row.get("is_income_pred", 0)) != 1:
        return ""

    income_type = str(row.get("income_type_pred", "")).strip()

    if income_type == "centrelink":
        return "CENTRELINK"

    # Prefer merchant-KB match over regex-based payer extraction.
    kb_match = row.get("_kb_counterparty")
    if isinstance(kb_match, str) and kb_match.strip():
        return kb_match.strip()

    payer_key = clean_counterparty(row.get("payer_key_from_text", ""))
    text_clean = clean_counterparty(row.get("text_clean", ""))
    return payer_key or text_clean


def build_stream_group_key(row: pd.Series) -> Optional[str]:
    if int(row.get("is_income_pred", 0)) != 1:
        return None

    values = [
        str(row.get("bank_account_id", "")).strip(),
        str(row.get("finv_category", "")).strip(),
        str(row.get("counterparty", "")).strip(),
    ]
    if not all(values[:3]):
        return None
    return "||".join(values)


def summarize_frequency(dates: pd.Series) -> tuple[str, Optional[int], str]:
    valid_dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(valid_dates) <= 1:
        weekday = (
            WEEKDAY_NAMES.get(valid_dates.iloc[-1].weekday(), "")
            if len(valid_dates) == 1
            else ""
        )
        return "one_off", None, weekday

    gaps = valid_dates.diff().dt.days.dropna()
    if gaps.empty:
        return "one_off", None, WEEKDAY_NAMES.get(
            valid_dates.iloc[-1].weekday(), ""
        )

    median_gap = float(gaps.median())
    rounded_gap = int(round(median_gap))
    if 6 <= median_gap <= 8:
        frequency, typical_gap_days = "weekly", 7
    elif 13 <= median_gap <= 16:
        frequency, typical_gap_days = "fortnightly", 14
    elif 27 <= median_gap <= 33:
        frequency, typical_gap_days = "monthly", 30
    else:
        frequency = "irregular"
        typical_gap_days = rounded_gap if rounded_gap > 0 else None

    weekday_mode = valid_dates.dt.weekday.mode()
    weekday = (
        WEEKDAY_NAMES.get(int(weekday_mode.iloc[0]), "")
        if not weekday_mode.empty
        else ""
    )
    return frequency, typical_gap_days, weekday


def estimate_monthly_income(frequency: str, typical_amount: float) -> float:
    if pd.isna(typical_amount):
        return np.nan
    if frequency == "weekly":
        return typical_amount * 52.0 / 12.0
    if frequency == "fortnightly":
        return typical_amount * 26.0 / 12.0
    if frequency == "monthly":
        return typical_amount
    return np.nan


def derive_stream_status(
    transaction_count: int,
    frequency: str,
    typical_gap_days: Optional[int],
    last_date: pd.Timestamp,
    global_last_date: pd.Timestamp,
) -> str:
    if pd.isna(last_date):
        return "unknown"
    if transaction_count <= 1:
        return "single_transaction"
    if frequency not in {"weekly", "fortnightly", "monthly"} or not typical_gap_days:
        return "irregular"

    days_since_last = (global_last_date - last_date).days
    allowed_gap = int(round(typical_gap_days * 1.75))
    return "active" if days_since_last <= allowed_gap else "inactive"


def assign_stream_ids(transactions: pd.DataFrame) -> pd.DataFrame:
    output = transactions.copy()
    income_mask = output["is_income_pred"].eq(1)
    income = output[income_mask].copy()
    if income.empty:
        return output

    stream_order = (
        income.groupby("_income_stream_group_key", dropna=False)
        .agg(
            finv_category=("finv_category", first_non_null),
            bank_account_id=("bank_account_id", first_non_null),
            counterparty=("counterparty", first_non_null),
            first_txn_date=("txn_date", "min"),
        )
        .sort_values(
            ["finv_category", "bank_account_id", "counterparty", "first_txn_date"],
            na_position="last",
        )
        .reset_index()
    )

    stream_ids: dict[str, str] = {}
    counters: dict[str, int] = {}
    for _, row in stream_order.iterrows():
        category = str(row["finv_category"])
        counters[category] = counters.get(category, 0) + 1
        stream_ids[row["_income_stream_group_key"]] = (
            f"{category}_{counters[category]:03d}"
        )

    output.loc[income_mask, "stream_id"] = (
        output.loc[income_mask, "_income_stream_group_key"]
        .map(stream_ids)
        .fillna("")
    )
    return output


def build_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    income = transactions[transactions["is_income_pred"].eq(1)].copy()
    if income.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    income["txn_date"] = pd.to_datetime(income["txn_date"], errors="coerce")
    income["amount_num"] = pd.to_numeric(income["amount_num"], errors="coerce")
    global_last_date = income["txn_date"].max()
    group_column = (
        "_income_stream_group_key"
        if "_income_stream_group_key" in income.columns
        else "stream_id"
    )

    summary_rows = []
    for _, group in income.groupby(group_column, dropna=False):
        group = group.sort_values("txn_date")
        frequency, typical_gap_days, frequency_day = summarize_frequency(
            group["txn_date"]
        )
        transaction_count = int(len(group))
        start_date = group["txn_date"].min()
        end_date = group["txn_date"].max()
        latest_amount = group.iloc[-1].get("amount_num", np.nan)
        median_amount = group["amount_num"].median()
        monthly_income = estimate_monthly_income(frequency, median_amount)
        status = derive_stream_status(
            transaction_count,
            frequency,
            typical_gap_days,
            end_date,
            global_last_date,
        )

        next_date = pd.NaT
        if (
            transaction_count >= 2
            and frequency in {"weekly", "fortnightly", "monthly"}
            and typical_gap_days
            and not pd.isna(end_date)
        ):
            next_date = end_date + pd.Timedelta(days=int(typical_gap_days))

        summary_rows.append(
            {
                "finv_category": first_non_null(group["finv_category"]),
                "stream_id": first_non_null(group["stream_id"]),
                "bank_account_id": first_non_null(group["bank_account_id"]),
                "account_type": first_non_null(group["account_type"])
                if "account_type" in group.columns
                else np.nan,
                "application_id": first_non_null(group["application_id"])
                if "application_id" in group.columns
                else np.nan,
                "bank": first_non_null(group["bank"])
                if "bank" in group.columns
                else np.nan,
                "credit_limit": first_non_null(group["credit_limit"])
                if "credit_limit" in group.columns
                else np.nan,
                "counterparty": first_non_null(group["counterparty"]),
                "transaction_start_date": start_date.date()
                if not pd.isna(start_date)
                else np.nan,
                "transaction_end_date": end_date.date()
                if not pd.isna(end_date)
                else np.nan,
                "status": status,
                "transaction_count": transaction_count,
                "total_income_amount": float(group["amount_num"].sum()),
                "average_income_amount": float(group["amount_num"].mean()),
                "median_income_amount": float(median_amount)
                if not pd.isna(median_amount)
                else np.nan,
                "latest_income_amount": float(latest_amount)
                if not pd.isna(latest_amount)
                else np.nan,
                "estimated_monthly_income": float(monthly_income)
                if not pd.isna(monthly_income)
                else np.nan,
                "frequency": frequency,
                "frequency_day": frequency_day,
                "predicted_next_income_date": next_date.date()
                if not pd.isna(next_date)
                else np.nan,
            }
        )

    return (
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
        .sort_values(["finv_category", "stream_id"])
        .reset_index(drop=True)
    )


def add_income_streams(transactions: pd.DataFrame) -> pd.DataFrame:
    output = transactions.copy()
    output["counterparty"] = output.apply(derive_counterparty, axis=1)
    output["_income_stream_group_key"] = output.apply(build_stream_group_key, axis=1)
    output["stream_id"] = ""
    output = assign_stream_ids(output)
    return output.drop(columns=["_income_stream_group_key"])
