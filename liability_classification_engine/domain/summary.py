"""Build liability summary output rows."""

from __future__ import annotations

from collections import Counter
from datetime import date
from math import ceil
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import re

import pandas as pd

SUMMARY_COLUMNS = [
    "finv_category",
    "stream_id",
    "bank_account_id",
    "bank",
    "account_type",
    "credit_limit",
    "application_id",
    "counterparty",
    "transaction_start_date",
    "transaction_end_date",
    "status",
    "funded_amount",
    "repaid_amount",
    "repayment_amount",
    "recent_fn_repay_amount",
    "frequency",
    "frequency_day",
    "predicted_closing_date",
]

DUE_DATE_COLUMN_CANDIDATES = [
    "scheduled_due_date",
    "repayment_due_date",
    "due_date",
    "scheduled_repayment_date",
    "repayment_date",
]

BNPL_LIMIT_COLUMNS = ["counterparty", "bnpl_max_fn_limit"]
BNPL_LIMIT_SOURCE_COLUMNS = {"rate_type", "counterparty", "value"}
FORTNIGHTLY = "fortnightly"
MONEY_PRECISION = Decimal("0.01")
WAGE_ADVANCE_TOTAL_MULTIPLIER = Decimal("1.05")
WAGE_ADVANCE_RECENT_REPAY_RATE = Decimal("0.05")
WEEKLY = "weekly"
MONTHLY = "monthly"
PERSONAL_LOAN_PRODUCT_TYPES = [
    "Non SACC Loans",
    "SACC Loans",
]
SUMMARY_GROUP_COLUMNS = [
    "application_id",
    "counterparty",
    "finv_category",
    "stream_id",
]
PLACEHOLDER_METRIC_VALUE = "-"


def parse_decimal_amount(value: object) -> Decimal | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_absolute_amount(value: object) -> Decimal | None:
    amount = parse_decimal_amount(value)
    if amount is None:
        return None
    return abs(amount)


def round_money(amount: Decimal) -> Decimal:
    return amount.quantize(MONEY_PRECISION, rounding=ROUND_HALF_UP)


def decimal_to_output(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(round_money(value))


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def joined_unique_text(values: pd.Series) -> str:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        unique_values.append(text)
        seen.add(text)
    return ", ".join(unique_values)


def stream_detail_fields(group: pd.DataFrame) -> dict[str, str]:
    return {
        "bank_account_id": joined_unique_text(group["bank_account_id"]),
        "bank": joined_unique_text(group["bank"]) if "bank" in group.columns else "",
        "account_type": joined_unique_text(group["account_type"]),
        "credit_limit": joined_unique_text(group["credit_limit"]),
    }


def normalize_match_key(value: object) -> str:
    return normalize_text(value).casefold()


def stream_base(stream_id: object) -> str:
    text = normalize_text(stream_id)
    if not text:
        return ""
    return re.sub(r"_\d+$", "", text)


def derive_finv_category(product_type: object, stream_id: object) -> object:
    product = normalize_text(product_type)
    base = stream_base(stream_id)
    if not product or not base:
        return pd.NA

    if base in {"bnpl", "wage_advance", "bank", "loc", "contract_loan", "home_loan"}:
        key = base
    else:
        key = f"{product}_{base}"

    from .streams import FINV_CATEGORY_MAP  # noqa: E402

    return FINV_CATEGORY_MAP.get(key, key)


def ensure_finv_category(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "finv_category" not in output.columns:
        output["finv_category"] = [
            derive_finv_category(product_type, stream_id)
            for product_type, stream_id in zip(
                output.get("product_type", pd.Series(index=output.index)),
                output.get("stream_id", pd.Series(index=output.index)),
            )
        ]
    return output


def load_bnpl_maximum_limits(
    limits_file: str | Path = "resources/bnpl_maximum_limits.csv",
) -> pd.DataFrame:
    path = Path(limits_file)
    if not path.exists():
        return pd.DataFrame(columns=BNPL_LIMIT_COLUMNS)

    limits = pd.read_csv(path, encoding="utf-8-sig")
    if not BNPL_LIMIT_SOURCE_COLUMNS.issubset(limits.columns):
        missing_columns = BNPL_LIMIT_SOURCE_COLUMNS.difference(limits.columns)
        raise ValueError(
            "bnpl_maximum_limits is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    limits = (
        limits[
            limits["rate_type"]
            .astype("string")
            .str.strip()
            .str.casefold()
            .eq("bnpl_max_fn_limit")
        ][["counterparty", "value"]]
        .copy()
        .rename(columns={"value": "bnpl_max_fn_limit"})
    )

    limits["_counterparty_key"] = limits["counterparty"].map(normalize_match_key)
    limits["_limit_amount"] = limits["bnpl_max_fn_limit"].map(parse_decimal_amount)
    limits = limits.drop_duplicates(subset=["_counterparty_key"], keep="first")
    return limits


def is_yes(value: object) -> bool:
    return normalize_text(value).casefold() == "yes"


def sorted_stream_transactions(group: pd.DataFrame) -> pd.DataFrame:
    return group.sort_values(
        ["_transaction_date", "_row_order"],
        kind="stable",
        na_position="last",
    )


def mark_failed_repayments(group: pd.DataFrame) -> pd.Series:
    ordered = sorted_stream_transactions(group)
    failed = pd.Series(False, index=group.index, dtype=bool)

    row_ids = ordered.index.tolist()
    for position, row_id in enumerate(row_ids[:-1]):
        row = ordered.loc[row_id]
        next_row = ordered.loc[row_ids[position + 1]]
        if normalize_text(row.get("dr_cr")).casefold() != "debit":
            continue
        if is_yes(next_row.get("is_dishonours")):
            failed.at[row_id] = True

    return failed


def effective_repayment_date(row: pd.Series, due_date_columns: list[str]) -> pd.Timestamp | pd.NaT:
    for column in due_date_columns:
        due_date = pd.to_datetime(row.get(column), errors="coerce")
        if not pd.isna(due_date):
            return due_date
    return row.get("_transaction_date", pd.NaT)


def calculate_frequency_day(
    valid_debits: pd.DataFrame,
    due_date_columns: list[str],
) -> str | None:
    if valid_debits.empty:
        return None

    weekdays: list[int] = []
    for _, row in valid_debits.iterrows():
        repayment_date = effective_repayment_date(row, due_date_columns)
        if pd.isna(repayment_date):
            continue
        weekdays.append(int(repayment_date.dayofweek))

    if not weekdays:
        return None

    counts = Counter(weekdays)
    top_count = max(counts.values())
    for weekday in weekdays:
        if counts[weekday] == top_count:
            return (pd.Timestamp("2024-01-01") + pd.Timedelta(days=weekday)).day_name()
    return None


def prepare_summary_input(df: pd.DataFrame) -> pd.DataFrame:
    output = ensure_finv_category(df)
    output["_row_order"] = range(len(output))
    output["_transaction_date"] = pd.to_datetime(
        output["transaction_date"],
        errors="coerce",
    )
    if "sample_datetime" in output.columns:
        output["_sample_datetime"] = pd.to_datetime(
            output["sample_datetime"],
            errors="coerce",
        )
    else:
        output["_sample_datetime"] = pd.NaT
    return output


def get_due_date_columns(df: pd.DataFrame) -> list[str]:
    return [
        column for column in DUE_DATE_COLUMN_CANDIDATES if column in df.columns
    ]


def empty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def filter_product_streams(
    df: pd.DataFrame,
    finv_category: str,
) -> pd.DataFrame:
    return df[
        df["finv_category"].astype("string").str.casefold().eq(
            finv_category.casefold()
        )
        & df["stream_id"].notna()
        & df["stream_id"].astype("string").str.strip().ne("")
    ].copy()


def get_valid_credits(group: pd.DataFrame) -> pd.DataFrame:
    ordered_group = sorted_stream_transactions(group)
    return ordered_group[
        ordered_group["dr_cr"].astype("string").str.casefold().eq("credit")
        & ~ordered_group["is_dishonours"].map(is_yes)
    ]


def get_latest_funding(
    group: pd.DataFrame,
) -> tuple[pd.Timestamp | pd.NaT, Decimal]:
    valid_credits = get_valid_credits(group)
    if valid_credits.empty:
        return pd.NaT, Decimal("0")

    funded_row = valid_credits.iloc[-1]
    funded_transaction_date = funded_row["_transaction_date"]
    funded_amount = parse_absolute_amount(funded_row["amount"]) or Decimal("0")
    return funded_transaction_date, round_money(funded_amount)


def get_successful_debits_after_date(
    group: pd.DataFrame,
    start_date: pd.Timestamp | pd.NaT,
) -> pd.DataFrame:
    failed_repayments = mark_failed_repayments(group)
    debit_mask = group["dr_cr"].astype("string").str.casefold().eq("debit")
    valid_debit_mask = debit_mask & ~failed_repayments
    if pd.isna(start_date):
        return sorted_stream_transactions(group.loc[valid_debit_mask].iloc[0:0].copy())

    debits = group.loc[
        valid_debit_mask & group["_transaction_date"].gt(start_date)
    ].copy()
    return sorted_stream_transactions(debits)


def get_successful_debits(group: pd.DataFrame) -> pd.DataFrame:
    failed_repayments = mark_failed_repayments(group)
    debit_mask = group["dr_cr"].astype("string").str.casefold().eq("debit")
    return sorted_stream_transactions(group.loc[debit_mask & ~failed_repayments].copy())


def sum_amounts(rows: pd.DataFrame, column: str = "amount") -> Decimal:
    return round_money(sum(
        (
            amount
            for amount in rows[column].map(parse_absolute_amount)
            if amount is not None
        ),
        Decimal("0"),
    ))


def get_amount_series(rows: pd.DataFrame) -> list[Decimal]:
    return [
        amount
        for amount in rows["amount"].map(parse_absolute_amount)
        if amount is not None
    ]


def median_decimal(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal("2")


def infer_frequency(
    funded_transaction_date: pd.Timestamp | pd.NaT,
    repayment_rows: pd.DataFrame,
) -> str:
    if repayment_rows.empty:
        return FORTNIGHTLY

    repayment_dates = sorted(
        repayment_rows["_transaction_date"].dropna().tolist()
    )
    if not repayment_dates:
        return FORTNIGHTLY

    if len(repayment_dates) == 1:
        if pd.isna(funded_transaction_date):
            approx_freq = 14
        else:
            approx_freq = max(
                0,
                (repayment_dates[0].date() - funded_transaction_date.date()).days,
            )
    else:
        intervals = [
            max(0, (curr.date() - prev.date()).days)
            for prev, curr in zip(repayment_dates, repayment_dates[1:])
        ]
        approx_freq = sorted(intervals)[len(intervals) // 2]

    if approx_freq <= 9:
        return WEEKLY
    if approx_freq <= 18:
        return FORTNIGHTLY
    return MONTHLY


def calculate_personal_loan_repayment_amount(
    repayment_rows: pd.DataFrame,
    funded_transaction_date: pd.Timestamp | pd.NaT,
    transaction_end_date: pd.Timestamp | pd.NaT,
) -> Decimal:
    repayment_amounts = get_amount_series(repayment_rows)
    repayment_count = len(repayment_amounts)
    if repayment_count == 0:
        return Decimal("0")
    if repayment_count == 1:
        return round_money(repayment_amounts[0])
    if repayment_count == 2:
        return round_money(sum(repayment_amounts) / Decimal("2"))

    min_amount = min(repayment_amounts)
    max_amount = max(repayment_amounts)
    consistency_ratio = (
        Decimal("0")
        if max_amount == 0
        else min_amount / max_amount
    )
    recent_three_amounts = repayment_amounts[-3:]
    if consistency_ratio > Decimal("0.90"):
        return round_money(median_decimal(recent_three_amounts))

    if pd.isna(funded_transaction_date) or pd.isna(transaction_end_date):
        return round_money(median_decimal(recent_three_amounts))

    history_days = max(
        0,
        (transaction_end_date.date() - funded_transaction_date.date()).days,
    )
    if history_days >= 90:
        lookback_start = transaction_end_date - pd.Timedelta(days=90)
        lookback_rows = repayment_rows[
            repayment_rows["_transaction_date"].ge(lookback_start)
        ]
        amount_90_days = sum_amounts(lookback_rows)
        return round_money(abs((amount_90_days * Decimal("4")) / Decimal("26")))

    actual_days = max(1, history_days)
    total_repaid = sum_amounts(repayment_rows)
    return round_money(abs((total_repaid / Decimal(actual_days)) * Decimal("14")))


def calculate_recent_fn_repay_amount(
    repayment_amount: Decimal,
    repaid_amount: Decimal,
    frequency: str,
) -> Decimal:
    if repaid_amount == 0:
        return Decimal("0")
    if frequency == MONTHLY:
        return round_money((repayment_amount * Decimal("12")) / Decimal("26"))
    if frequency == WEEKLY:
        return round_money(repayment_amount * Decimal("2"))
    return round_money(repayment_amount)


def calculate_personal_loan_status(
    funded_amount: Decimal,
    repaid_amount: Decimal,
    transaction_end_date: pd.Timestamp | pd.NaT,
    sample_datetime: pd.Timestamp | pd.NaT,
) -> str:
    if funded_amount != 0:
        lower_bound = funded_amount * Decimal("0.75")
        upper_bound = funded_amount * Decimal("1.25")
        if repaid_amount <= lower_bound:
            return "Ongoing"
        if repaid_amount <= upper_bound:
            return "Closing Soon"
        return "Closed"

    if (
        not pd.isna(transaction_end_date)
        and not pd.isna(sample_datetime)
        and transaction_end_date >= (sample_datetime - pd.Timedelta(days=33))
    ):
        return "Ongoing"
    return "Closed"


def calculate_predicted_closing_date(
    stream_id: str,
    status: str,
    funded_amount: Decimal,
    repaid_amount: Decimal,
    repayment_amount: Decimal,
    frequency: str,
    transaction_end_date: pd.Timestamp | pd.NaT,
) -> str:
    if not stream_id.lower().startswith("sacc_"):
        return "NA"
    if status == "Closed":
        return "NA"
    if repayment_amount <= 0 or pd.isna(transaction_end_date):
        return "NA"

    loan_amt_rmning = round_money(
        round_money(funded_amount * Decimal("1.25")) - repaid_amount
    )
    rpmnts_rmning = max(0, ceil(float(loan_amt_rmning / repayment_amount)))

    freq_days = {
        WEEKLY: 7,
        FORTNIGHTLY: 14,
        MONTHLY: 31,
    }.get(frequency, 14)
    predicted_date = transaction_end_date + pd.Timedelta(
        days=freq_days * rpmnts_rmning
    )
    return predicted_date.strftime("%Y-%m-%d")


def build_bnpl_summary(
    df: pd.DataFrame,
    limits: pd.DataFrame | None = None,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Return one BNPL summary row per finv_category + stream_id."""

    bnpl = filter_product_streams(df, "BNPL")

    if bnpl.empty:
        return empty_summary()

    due_date_columns = get_due_date_columns(bnpl)
    limits = limits if limits is not None else load_bnpl_maximum_limits()
    limits_by_counterparty = (
        limits.set_index("_counterparty_key")["_limit_amount"].to_dict()
        if "_counterparty_key" in limits.columns
        else {}
    )

    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in bnpl.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        failed_repayments = mark_failed_repayments(group)

        debit_mask = group["dr_cr"].astype("string").str.casefold().eq("debit")
        valid_debits = group.loc[debit_mask & ~failed_repayments].copy()
        valid_debits["_amount_decimal"] = valid_debits["amount"].map(
            parse_absolute_amount
        )
        valid_debits = valid_debits[
            valid_debits["_amount_decimal"].map(
                lambda amount: amount is not None and amount.is_finite()
            )
        ]

        if valid_debits.empty:
            repaid_amount = Decimal("0")
            frequency_day_date = None
        else:
            repaid_amount = round_money(
                sum(valid_debits["_amount_decimal"], Decimal("0"))
            )
            frequency_day_date = calculate_frequency_day(
                sorted_stream_transactions(valid_debits),
                due_date_columns,
            )

        sample_datetime = group["_sample_datetime"].max()
        transaction_end_date = group["_transaction_date"].max()
        as_of_datetime = (
            sample_datetime
            if not pd.isna(sample_datetime)
            else transaction_end_date
        )
        recent_30_debit_amount = Decimal("0")
        raw_recent_fn = Decimal("0")

        if not valid_debits.empty and not pd.isna(as_of_datetime):
            as_of_day = pd.Timestamp(as_of_datetime).normalize()
            transaction_days = valid_debits["_transaction_date"].dt.normalize()
            recent_30_mask = transaction_days.between(
                as_of_day - pd.Timedelta(days=30),
                as_of_day,
                inclusive="both",
            )
            recent_60_mask = transaction_days.between(
                as_of_day - pd.Timedelta(days=60),
                as_of_day,
                inclusive="both",
            )
            recent_30_debit_amount = round_money(
                sum(
                    valid_debits.loc[recent_30_mask, "_amount_decimal"],
                    Decimal("0"),
                )
            )
            recent_60_debit_amount = round_money(
                sum(
                    valid_debits.loc[recent_60_mask, "_amount_decimal"],
                    Decimal("0"),
                )
            )
            raw_recent_fn = round_money(
                recent_60_debit_amount * Decimal("6") / Decimal("26")
            )

        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": transaction_end_date,
                "status": "Closed",
                "funded_amount": 0,
                "repaid_amount": decimal_to_output(repaid_amount),
                "repayment_amount": None,
                "recent_fn_repay_amount": None,
                "frequency": FORTNIGHTLY,
                "frequency_day": frequency_day_date,
                "predicted_closing_date": None,
                "_sample_datetime": sample_datetime,
                "_recent_30_debit_amount": recent_30_debit_amount,
                "_raw_recent_fn": raw_recent_fn,
            }
        )

    summary = pd.DataFrame(summary_rows)

    for (app_id, counterparty), counterparty_rows in summary.groupby(
        ["application_id", "counterparty"],
        dropna=False,
        sort=False,
    ):
        sorted_rows = counterparty_rows.sort_values(
            ["transaction_end_date", "transaction_start_date", "stream_id"],
            ascending=[False, False, True],
            kind="stable",
            na_position="last",
        )
        latest_index = sorted_rows.index[0]
        end_date = summary.at[latest_index, "transaction_end_date"]
        sample_dt = summary.at[latest_index, "_sample_datetime"]
        if (
            not pd.isna(end_date)
            and not pd.isna(sample_dt)
            and (sample_dt.date() - end_date.date()).days <= 33
        ):
            summary.at[latest_index, "status"] = "Ongoing"

    for row_id, row in summary.iterrows():
        counterparty_key = normalize_match_key(row["counterparty"])
        limit_amount = limits_by_counterparty.get(counterparty_key)

        if (
            row["status"] == "Closed"
            or row["_recent_30_debit_amount"] < Decimal("1.00")
        ):
            recent_fn_repay_amount = Decimal("0")
        else:
            recent_fn_repay_amount = row["_raw_recent_fn"]
            if limit_amount is not None:
                recent_fn_repay_amount = min(recent_fn_repay_amount, limit_amount)

        recent_fn_repay_amount = round_money(recent_fn_repay_amount)

        numeric_recent = decimal_to_output(recent_fn_repay_amount)
        summary.at[row_id, "recent_fn_repay_amount"] = numeric_recent
        summary.at[row_id, "repayment_amount"] = numeric_recent

    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_wage_advance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one wage-advance summary row per finv_category + stream_id."""

    wage_advance = filter_product_streams(df, "Wage Advance")

    if wage_advance.empty:
        return empty_summary()

    due_date_columns = get_due_date_columns(wage_advance)
    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in wage_advance.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        funded_transaction_date, funded_amount = get_latest_funding(group)
        eligible_debits = get_successful_debits_after_date(
            group,
            funded_transaction_date,
        )
        repaid_amount = sum_amounts(eligible_debits)

        total_remaining = round_money(
            funded_amount * WAGE_ADVANCE_TOTAL_MULTIPLIER
        )
        repayment_remaining = round_money(total_remaining - repaid_amount)
        if repayment_remaining <= 0:
            recent_fn_repay_amount = Decimal("0")
        else:
            recent_fn_repay_amount = round_money(
                funded_amount * WAGE_ADVANCE_RECENT_REPAY_RATE
            )

        status = (
            "Closed"
            if (
                recent_fn_repay_amount == 0
                and repaid_amount >= total_remaining
            )
            else "Ongoing"
        )

        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": group["_transaction_date"].max(),
                "status": status,
                "funded_amount": decimal_to_output(funded_amount),
                "repaid_amount": decimal_to_output(repaid_amount),
                "repayment_amount": decimal_to_output(recent_fn_repay_amount),
                "recent_fn_repay_amount": decimal_to_output(
                    recent_fn_repay_amount
                ),
                "frequency": FORTNIGHTLY,
                "frequency_day": calculate_frequency_day(
                    eligible_debits,
                    due_date_columns,
                ),
                "predicted_closing_date": None,
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_personal_loan_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per personal-loan non-sacc/sacc stream."""

    personal_loans = df[
        df["finv_category"].astype("string").isin(
            PERSONAL_LOAN_PRODUCT_TYPES
        )
        & df["stream_id"].notna()
        & df["stream_id"].astype("string").str.strip().ne("")
    ].copy()

    if personal_loans.empty:
        return empty_summary()

    due_date_columns = get_due_date_columns(personal_loans)
    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in personal_loans.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        funded_transaction_date, funded_amount = get_latest_funding(group)
        all_repayment_rows = get_successful_debits(group)
        if pd.isna(funded_transaction_date):
            repayment_rows = all_repayment_rows
        else:
            repayment_rows = get_successful_debits_after_date(
                group,
                funded_transaction_date,
            )
        repaid_amount = sum_amounts(repayment_rows)
        transaction_end_date = group["_transaction_date"].max()
        sample_datetime = group["_sample_datetime"].max()
        status = calculate_personal_loan_status(
            funded_amount,
            repaid_amount,
            transaction_end_date,
            sample_datetime,
        )
        if status == "Closed":
            metric_repayment_rows = repayment_rows
            repayment_amount = Decimal("0")
            recent_fn_repay_amount = Decimal("0")
        else:
            metric_repayment_rows = (
                repayment_rows
                if not repayment_rows.empty
                else all_repayment_rows
            )
            repayment_amount = calculate_personal_loan_repayment_amount(
                metric_repayment_rows,
                funded_transaction_date,
                transaction_end_date,
            )
            recent_fn_repay_amount = calculate_recent_fn_repay_amount(
                repayment_amount,
                sum_amounts(metric_repayment_rows),
                infer_frequency(
                    funded_transaction_date,
                    metric_repayment_rows,
                ),
            )
        frequency = infer_frequency(
            funded_transaction_date,
            metric_repayment_rows,
        )
        predicted_closing_date = calculate_predicted_closing_date(
            normalize_text(stream_id_value),
            status,
            funded_amount,
            repaid_amount,
            repayment_amount,
            frequency,
            transaction_end_date,
        )

        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": transaction_end_date,
                "status": status,
                "funded_amount": decimal_to_output(funded_amount),
                "repaid_amount": decimal_to_output(repaid_amount),
                "repayment_amount": decimal_to_output(repayment_amount),
                "recent_fn_repay_amount": decimal_to_output(
                    recent_fn_repay_amount
                ),
                "frequency": frequency,
                "frequency_day": calculate_frequency_day(
                    repayment_rows,
                    due_date_columns,
                ),
                "predicted_closing_date": predicted_closing_date,
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_bank_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one placeholder summary row per bank stream."""

    bank = filter_product_streams(df, "Credit Card Repayment")

    if bank.empty:
        return empty_summary()

    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in bank.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": group["_transaction_date"].max(),
                "status": PLACEHOLDER_METRIC_VALUE,
                "funded_amount": PLACEHOLDER_METRIC_VALUE,
                "repaid_amount": PLACEHOLDER_METRIC_VALUE,
                "repayment_amount": PLACEHOLDER_METRIC_VALUE,
                "recent_fn_repay_amount": PLACEHOLDER_METRIC_VALUE,
                "frequency": PLACEHOLDER_METRIC_VALUE,
                "frequency_day": PLACEHOLDER_METRIC_VALUE,
                "predicted_closing_date": PLACEHOLDER_METRIC_VALUE,
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_home_loan_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one placeholder summary row per home-loan stream."""

    home_loan = filter_product_streams(df, "Home Loan")

    if home_loan.empty:
        return empty_summary()

    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in home_loan.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": group["_transaction_date"].max(),
                "status": PLACEHOLDER_METRIC_VALUE,
                "funded_amount": PLACEHOLDER_METRIC_VALUE,
                "repaid_amount": PLACEHOLDER_METRIC_VALUE,
                "repayment_amount": PLACEHOLDER_METRIC_VALUE,
                "recent_fn_repay_amount": PLACEHOLDER_METRIC_VALUE,
                "frequency": PLACEHOLDER_METRIC_VALUE,
                "frequency_day": PLACEHOLDER_METRIC_VALUE,
                "predicted_closing_date": PLACEHOLDER_METRIC_VALUE,
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_contract_loan_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one placeholder summary row per contract-loan stream."""

    contract_loan = filter_product_streams(df, "Contract Loans")

    if contract_loan.empty:
        return empty_summary()

    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in contract_loan.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": group["_transaction_date"].max(),
                "status": PLACEHOLDER_METRIC_VALUE,
                "funded_amount": PLACEHOLDER_METRIC_VALUE,
                "repaid_amount": PLACEHOLDER_METRIC_VALUE,
                "repayment_amount": PLACEHOLDER_METRIC_VALUE,
                "recent_fn_repay_amount": PLACEHOLDER_METRIC_VALUE,
                "frequency": PLACEHOLDER_METRIC_VALUE,
                "frequency_day": PLACEHOLDER_METRIC_VALUE,
                "predicted_closing_date": PLACEHOLDER_METRIC_VALUE,
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_loc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one placeholder summary row per LOC stream."""

    loc = filter_product_streams(df, "LOC")

    if loc.empty:
        return empty_summary()

    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in loc.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": group["_transaction_date"].max(),
                "status": PLACEHOLDER_METRIC_VALUE,
                "funded_amount": PLACEHOLDER_METRIC_VALUE,
                "repaid_amount": PLACEHOLDER_METRIC_VALUE,
                "repayment_amount": PLACEHOLDER_METRIC_VALUE,
                "recent_fn_repay_amount": PLACEHOLDER_METRIC_VALUE,
                "frequency": PLACEHOLDER_METRIC_VALUE,
                "frequency_day": PLACEHOLDER_METRIC_VALUE,
                "predicted_closing_date": PLACEHOLDER_METRIC_VALUE,
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_unknown_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one placeholder summary row per unknown personal-loan stream."""

    unknown = filter_product_streams(df, "Personal Loan Unknown")

    if unknown.empty:
        return empty_summary()

    summary_rows: list[dict[str, object]] = []

    for (_, _, finv_category, stream_id_value), group in unknown.groupby(
        SUMMARY_GROUP_COLUMNS,
        dropna=False,
        sort=False,
    ):
        summary_rows.append(
            {
                "finv_category": finv_category,
                "stream_id": stream_id_value,
                **stream_detail_fields(group),
                "application_id": normalize_text(group["application_id"].iloc[0]),
                "counterparty": normalize_text(group["counterparty"].iloc[0]),
                "transaction_start_date": group["_transaction_date"].min(),
                "transaction_end_date": group["_transaction_date"].max(),
                "status": PLACEHOLDER_METRIC_VALUE,
                "funded_amount": PLACEHOLDER_METRIC_VALUE,
                "repaid_amount": PLACEHOLDER_METRIC_VALUE,
                "repayment_amount": PLACEHOLDER_METRIC_VALUE,
                "recent_fn_repay_amount": PLACEHOLDER_METRIC_VALUE,
                "frequency": PLACEHOLDER_METRIC_VALUE,
                "frequency_day": PLACEHOLDER_METRIC_VALUE,
                "predicted_closing_date": PLACEHOLDER_METRIC_VALUE,
            }
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary["transaction_start_date"] = summary["transaction_start_date"].dt.date
    summary["transaction_end_date"] = summary["transaction_end_date"].dt.date
    return summary[SUMMARY_COLUMNS]


def build_summary(
    df: pd.DataFrame,
    limits_file: str | Path = "resources/bnpl_maximum_limits.csv",
) -> pd.DataFrame:
    prepared = prepare_summary_input(df)
    limits = load_bnpl_maximum_limits(limits_file)
    summaries = [
        build_bnpl_summary(prepared, limits=limits),
        build_wage_advance_summary(prepared),
        build_home_loan_summary(prepared),
        build_personal_loan_summary(prepared),
        build_bank_summary(prepared),
        build_contract_loan_summary(prepared),
        build_loc_summary(prepared),
        build_unknown_summary(prepared),
    ]
    summaries = [summary for summary in summaries if not summary.empty]
    if not summaries:
        return empty_summary()

    return pd.concat(summaries, ignore_index=True)[SUMMARY_COLUMNS]
