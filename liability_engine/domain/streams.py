"""Assign loan stream IDs by product type and priority.

Priority is defined in ``PRODUCT_RULES``:
BNPL -> Wage Advance -> Bank -> Personal Loan -> LOC.

The final LOC stage contains a controlled refinement rule: after personal-
loan streams have been created, qualifying ``sacc_*`` streams can be merged
into one ``loc_*`` stream. This refinement depends on personal-loan output
and therefore must run last.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

import numpy as np
import pandas as pd

from classification_core.text import parse_decimal_amount


# ---------------------------------------------------------------------------
# Product rule configuration
# ---------------------------------------------------------------------------

DEFAULT_GROUP_COLUMNS = ["application_id", "counterparty"]
SIMPLE_STREAM_SORT_COLUMNS = [
    "application_id",
    "bank_account_id",
    "counterparty",
]

PERSONAL_LOAN = "personal_loan"
DISHONOUR_COLUMN = "is_dishonours"
AMOUNT_TOLERANCE = Decimal("0.05")
STABLE_AMOUNT_MIN_REPEATS = 2
STABLE_AMOUNT_SPLIT_DELTA = Decimal("0.01")

SACC_PREFIX = "sacc_"
NON_SACC_PREFIX = "non_sacc_"
SPECIAL_LOC_PREFIX = "loc_"
UNKNOWN_PREFIX = "unknown_"
UNKNOWN_REASSIGN_MAX_GAP_DAYS = 31
UNKNOWN_NON_SACC_REPAYMENT_THRESHOLD = Decimal("2000")
UNKNOWN_PERIODIC_MIN_DATES = 3
UNKNOWN_PERIODIC_MEDIAN_GAP_RANGES = (
    (4, 10),
    (11, 17),
    (27, 33),
)
MIN_FUNDING_REPAYMENT_GAP_DAYS = 3
SPECIAL_COUNTERPARTY_STREAM_RULES = {
    "zip money": {
        "target_prefix": "non_sacc",
        "mode": "merge_group",
    },
    "credit corp": {
        "target_prefix": "non_sacc",
        "mode": "convert_sacc_streams",
    },
}
DEFAULT_MIN_SACC_STREAMS = 3
DEFAULT_LOC_CV_THRESHOLD = Decimal("0.2")
ORPHAN_SACC_MIN_AGE_DAYS = 31
LOC_REPAYMENT_CLOSE_RATIO = Decimal("0.05")
LOC_REPAYMENT_HALF_RATIO_LOWER = Decimal("0.45")
LOC_REPAYMENT_HALF_RATIO_UPPER = Decimal("0.55")
LOC_REPAYMENT_CONTINUATION_MAX_GAP_DAYS = 45
SINGLE_FUNDING_LOC_MIN_ANCHOR_DEBITS = 3
SINGLE_FUNDING_LOC_MIN_UNKNOWN_DEBITS = 2
SINGLE_FUNDING_LOC_MIN_TOTAL_DEBITS = 5
REVOLVING_LOC_MIN_ANCHOR_DEBITS = 4
REVOLVING_LOC_MIN_SWITCH_DEBITS = 2
REVOLVING_LOC_UNPAID_RATIO = Decimal("0.8")
REVOLVING_LOC_CLOSE_CREDIT_MAX_GAP_DAYS = 3
REVOLVING_LOC_MIN_FOLLOWING_DEBITS = 2
REVOLVING_LOC_CREDIT_ANCHOR_WINDOW_DAYS = 45
REVOLVING_LOC_COMPANION_MAX_GAP_DAYS = 7
REVOLVING_LOC_MIN_COMPANION_NEARBY_PAIRS = 2
PARALLEL_REPAYMENT_MIN_OVERLAP_DATES = 2
PARALLEL_REPAYMENT_MIN_AMOUNT_STREAMS = 2


@dataclass(frozen=True)
class ProductRule:
    """One product-level stream matching rule."""

    priority: int
    product_type: str
    matcher: Callable[[pd.DataFrame, pd.Series, list[str]], None]


# ---------------------------------------------------------------------------
# Shared stream structures
# ---------------------------------------------------------------------------


@dataclass
class RepaymentStream:
    row_indices: list[int]
    baseline_amount: Decimal
    first_date: pd.Timestamp


@dataclass
class FundingFlow:
    row_indices: list[int]
    transaction_date: pd.Timestamp
    amount: Decimal
    matched: bool = False


@dataclass
class AmountBucket:
    amount: Decimal
    row_indices: list[int]
    dates: set[pd.Timestamp]


@dataclass
class AmountCluster:
    buckets: list[AmountBucket]
    baseline_amount: Decimal

    @property
    def row_indices(self) -> list[int]:
        return [
            row_id
            for bucket in self.buckets
            for row_id in bucket.row_indices
        ]

    @property
    def dates_by_amount(self) -> dict[Decimal, set[pd.Timestamp]]:
        return {bucket.amount: bucket.dates for bucket in self.buckets}


class PersonalLoanStreamIdGenerator:
    """Generate the existing personal-loan stream ID formats."""

    def __init__(self) -> None:
        self._counters = {"sacc": 0, "non_sacc": 0, "unknown": 0}

    def next_for_amount(self, amount: Decimal) -> str:
        prefix = "sacc" if amount <= Decimal("2000") else "non_sacc"
        self._counters[prefix] += 1
        return f"{prefix}_{self._counters[prefix]:03d}"

    def next_unknown(self) -> str:
        self._counters["unknown"] += 1
        return f"unknown_{self._counters['unknown']:03d}"


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def normalize_amount_key(value: object) -> str:
    amount = parse_decimal_amount(value)
    if amount is None:
        return "" if pd.isna(value) else str(value).strip()
    return format(amount.normalize(), "f")


def amount_within_tolerance(
    amount: Decimal,
    baseline: Decimal,
    tolerance: Decimal = AMOUNT_TOLERANCE,
) -> bool:
    lower_bound = baseline * (Decimal("1") - tolerance)
    upper_bound = baseline * (Decimal("1") + tolerance)
    return lower_bound <= amount <= upper_bound


def normalize_group_value(value: object) -> object:
    return "" if pd.isna(value) else value


def ensure_stream_id_column(df: pd.DataFrame, reset: bool = False) -> pd.DataFrame:
    output = df.copy()

    if "stream_id" not in output.columns:
        output["stream_id"] = pd.NA
    elif reset:
        output["stream_id"] = pd.NA
    else:
        output["stream_id"] = output["stream_id"].replace(r"^\s*$", pd.NA, regex=True)

    return output


# ---------------------------------------------------------------------------
# Simple grouped products: BNPL / Wage Advance / Bank / direct LOC
# ---------------------------------------------------------------------------


def assign_grouped_product_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    prefix: str,
) -> None:
    """Assign one globally unique stream per account + counterparty group.

    Stream numbering is based on a stable business sort so IDs do not depend
    on the raw input row order.

    The original implementation used ``iterrows()`` to iterate over every
    eligible row.  This version computes unique group keys from the sorted
    rows and only loops over the (far smaller) set of unique groups.
    """

    sort_columns = [
        column for column in SIMPLE_STREAM_SORT_COLUMNS if column in output.columns
    ]
    eligible = output.loc[eligible_mask]
    if sort_columns:
        eligible = eligible.sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )

    # Exclude rows without a counterparty (original behaviour: ``continue``).
    has_cp = (
        eligible["counterparty"].notna()
        & eligible["counterparty"].astype(str).str.strip().ne("")
    )
    eligible = eligible[has_cp]
    if eligible.empty:
        return

    # Build a string-based group key from the three grouping columns so
    # we can work with vectorised Series instead of per-row tuples.
    app_col = eligible["application_id"].map(normalize_group_value)
    acct_col = eligible["bank_account_id"].map(normalize_group_value)
    cp_col = eligible["counterparty"].map(normalize_group_value)
    group_key = app_col.astype(str) + "||" + acct_col.astype(str) + "||" + cp_col.astype(str)

    # Unique groups in sorted first-appearance order.
    unique_groups = group_key.drop_duplicates()

    # Assign sequential stream numbers per application.
    app_counters: dict[object, int] = {}
    stream_map: dict[str, str] = {}
    for g in unique_groups:
        app = g.split("||", 1)[0]
        app_counters[app] = app_counters.get(app, 0) + 1
        stream_map[g] = f"{prefix}_{app_counters[app]:03d}"

    # Map stream IDs back to every row in the eligible set.
    output.loc[eligible.index, "stream_id"] = group_key.map(stream_map).values


def identify_direct_loc_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    """Assign streams to rows already classified as product_type == loc."""

    assign_grouped_product_streams(output, eligible_mask, "loc")


def identify_bnpl_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    assign_grouped_product_streams(output, eligible_mask, "bnpl")


def identify_wage_advance_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    assign_grouped_product_streams(output, eligible_mask, "wage_advance")


def identify_bank_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    assign_grouped_product_streams(output, eligible_mask, "bank")


def identify_contract_loan_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    assign_grouped_product_streams(output, eligible_mask, "contract_loan")


def identify_generic_loan_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    assign_grouped_product_streams(output, eligible_mask, "generic_loan")


def identify_car_loan_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    assign_grouped_product_streams(output, eligible_mask, "car_loan")


def identify_home_loan_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    _: list[str],
) -> None:
    assign_grouped_product_streams(output, eligible_mask, "home_loan")


# ---------------------------------------------------------------------------
# Personal loan matching
# ---------------------------------------------------------------------------


def is_dishonour_credit(df: pd.DataFrame) -> pd.Series:
    return (
        df["dr_cr"].eq("credit")
        & df[DISHONOUR_COLUMN].astype("string").str.lower().eq("yes")
    )


def buckets_have_parallel_dates(
    candidate: AmountBucket,
    cluster: AmountCluster,
) -> bool:
    for amount, dates in cluster.dates_by_amount.items():
        if amount != candidate.amount and candidate.dates.intersection(dates):
            return True
    return False


def choose_amount_cluster(
    candidate: AmountBucket,
    clusters: list[AmountCluster],
) -> AmountCluster | None:
    matches: list[tuple[Decimal, AmountCluster]] = []

    for cluster in clusters:
        if not amount_within_tolerance(
            candidate.amount,
            cluster.baseline_amount,
        ):
            continue
        if (
            len(cluster.row_indices) >= STABLE_AMOUNT_MIN_REPEATS
            and abs(candidate.amount - cluster.baseline_amount)
            > STABLE_AMOUNT_SPLIT_DELTA
        ):
            continue
        if buckets_have_parallel_dates(candidate, cluster):
            continue

        matches.append(
            (abs(candidate.amount - cluster.baseline_amount), cluster)
        )

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def build_amount_buckets(debits: pd.DataFrame) -> list[AmountBucket]:
    buckets: list[AmountBucket] = []

    for amount, amount_rows in debits.groupby(
        "_amount_decimal",
        dropna=True,
        sort=False,
    ):
        buckets.append(
            AmountBucket(
                amount=amount,
                row_indices=amount_rows.index.tolist(),
                dates=set(amount_rows["_transaction_date"].dropna()),
            )
        )

    buckets.sort(key=lambda bucket: (-len(bucket.row_indices), bucket.amount))
    return buckets


def cluster_repayments(group: pd.DataFrame) -> list[RepaymentStream]:
    debits = group[group["dr_cr"].eq("debit")].copy()
    if debits.empty:
        return []

    debits["_amount_decimal"] = debits["amount"].map(parse_decimal_amount)
    debits = debits.dropna(subset=["_amount_decimal", "_transaction_date"])
    if debits.empty:
        return []

    clusters: list[AmountCluster] = []
    for bucket in build_amount_buckets(debits):
        cluster = choose_amount_cluster(bucket, clusters)
        if cluster is None:
            clusters.append(
                AmountCluster(
                    buckets=[bucket],
                    baseline_amount=bucket.amount,
                )
            )
        else:
            cluster.buckets.append(bucket)

    repayment_streams: list[RepaymentStream] = []
    for cluster in clusters:
        cluster_rows = debits.loc[cluster.row_indices]
        repayment_streams.append(
            RepaymentStream(
                row_indices=cluster.row_indices,
                baseline_amount=cluster.baseline_amount,
                first_date=cluster_rows["_transaction_date"].min(),
            )
        )

    return repayment_streams


def build_funding_flows(group: pd.DataFrame) -> list[FundingFlow]:
    credits = group[
        group["dr_cr"].eq("credit") & ~group["_is_dishonour_credit"]
    ].copy()
    if credits.empty:
        return []

    credits["_amount_key"] = credits["amount"].map(normalize_amount_key)
    funding_flows: list[FundingFlow] = []

    for (transaction_date, _), flow_rows in credits.groupby(
        ["_transaction_date", "_amount_key"],
        dropna=False,
        sort=True,
    ):
        amount = parse_decimal_amount(flow_rows["amount"].iloc[0])
        if amount is None or pd.isna(transaction_date):
            continue

        funding_flows.append(
            FundingFlow(
                row_indices=flow_rows.index.tolist(),
                transaction_date=transaction_date,
                amount=amount,
            )
        )

    return funding_flows


def match_funding_flow(
    repayment_stream: RepaymentStream,
    funding_flows: list[FundingFlow],
) -> FundingFlow | None:
    min_gap = pd.Timedelta(days=MIN_FUNDING_REPAYMENT_GAP_DAYS)
    candidates = [
        funding
        for funding in funding_flows
        if not funding.matched
        and repayment_stream.first_date - funding.transaction_date >= min_gap
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda funding: funding.transaction_date)


def assign_dishonour_credits(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    group_columns: list[str],
) -> None:
    for _, group in output.loc[eligible_mask].groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        dishonour_rows = group[group["_is_dishonour_credit"]].sort_values(
            ["_transaction_date", "_row_id"]
        )
        debit_rows = group[
            group["dr_cr"].eq("debit") & group["stream_id"].notna()
        ].copy()

        if dishonour_rows.empty or debit_rows.empty:
            continue

        debit_rows["_amount_decimal"] = debit_rows["amount"].map(
            parse_decimal_amount
        )

        for row_id, dishonour in dishonour_rows.iterrows():
            amount = parse_decimal_amount(dishonour["amount"])
            candidates = debit_rows[
                debit_rows["_transaction_date"]
                <= dishonour["_transaction_date"]
            ]

            if amount is not None:
                exact_amount_candidates = candidates[
                    candidates["_amount_decimal"].eq(amount)
                ]
                if not exact_amount_candidates.empty:
                    candidates = exact_amount_candidates
                else:
                    tolerance_candidates = candidates[
                        candidates["_amount_decimal"].map(
                            lambda debit_amount: (
                                debit_amount is not None
                                and amount_within_tolerance(
                                    amount,
                                    debit_amount,
                                )
                            )
                        )
                    ]
                    if not tolerance_candidates.empty:
                        candidates = tolerance_candidates

            if candidates.empty:
                continue

            matched = candidates.sort_values(
                ["_transaction_date", "_row_id"]
            ).iloc[-1]
            output.at[row_id, "stream_id"] = matched["stream_id"]


def assign_personal_loan_rule(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    group_columns: list[str],
) -> None:
    """Assign personal-loan streams to rows already claimed by this rule."""

    output["_row_id"] = output.index
    output["_transaction_date"] = pd.to_datetime(
        output["transaction_date"],
        errors="coerce",
    )
    output["_is_dishonour_credit"] = is_dishonour_credit(output)

    stream_ids_by_application: dict[object, PersonalLoanStreamIdGenerator] = {}
    for _, group in output.loc[eligible_mask].groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        application_key = normalize_group_value(group["application_id"].iloc[0])
        stream_ids = stream_ids_by_application.setdefault(
            application_key,
            PersonalLoanStreamIdGenerator(),
        )
        repayment_streams = sorted(
            cluster_repayments(group),
            key=lambda stream: (
                stream.first_date,
                stream.baseline_amount,
            ),
        )
        funding_flows = build_funding_flows(group)

        for repayment_stream in repayment_streams:
            funding = match_funding_flow(repayment_stream, funding_flows)

            if funding is None:
                stream_id = stream_ids.next_unknown()
            else:
                funding.matched = True
                stream_id = stream_ids.next_for_amount(funding.amount)
                output.loc[funding.row_indices, "stream_id"] = stream_id

            output.loc[repayment_stream.row_indices, "stream_id"] = stream_id

        for funding in funding_flows:
            if funding.matched:
                continue

            stream_id = stream_ids.next_for_amount(funding.amount)
            output.loc[funding.row_indices, "stream_id"] = stream_id

    assign_dishonour_credits(
        output,
        eligible_mask,
        group_columns,
    )

    output.drop(
        columns=["_row_id", "_transaction_date", "_is_dishonour_credit"],
        inplace=True,
    )


# ---------------------------------------------------------------------------
# LOC refinement: merge qualifying SACC streams after Personal Loan
# ---------------------------------------------------------------------------


class LocStreamIdGenerator:
    """Generate ``loc_001`` IDs without colliding with existing special LOC IDs."""

    def __init__(self, existing_rows: pd.DataFrame) -> None:
        self._counters = self._find_max_counters(existing_rows)

    @staticmethod
    def _find_max_counters(existing_rows: pd.DataFrame) -> dict[object, int]:
        """Return the max ``loc_NNN`` counter per application (vectorised)."""
        sid_col = existing_rows["stream_id"].astype("string").str.strip().str.lower()
        extracted = sid_col.str.extract(r"loc_(\d+)")
        valid = extracted[0].notna()
        if not valid.any():
            return {}

        app_key_col = existing_rows["application_id"].map(normalize_group_value)
        counters_series = (
            extracted.loc[valid, 0]
            .astype(int)
            .groupby(app_key_col[valid], sort=False)
            .max()
        )
        return counters_series.to_dict()

    def next(self, application_id: object) -> str:
        application_key = normalize_group_value(application_id)
        self._counters[application_key] = self._counters.get(
            application_key,
            0,
        ) + 1
        return f"{SPECIAL_LOC_PREFIX}{self._counters[application_key]:03d}"


def parse_loc_amount(value: object) -> Decimal | None:
    """Return an absolute, non-zero amount for the LOC variability rule."""

    amount = parse_decimal_amount(value)
    if amount is None:
        return None

    amount = abs(amount)
    return amount if amount != 0 else None


def calculate_cv(amounts: list[Decimal]) -> Decimal | None:
    """Calculate population coefficient of variation: stddev / mean."""

    if not amounts:
        return None

    mean = sum(amounts) / Decimal(len(amounts))
    if mean == 0:
        return None

    variance = sum(
        (amount - mean) ** 2
        for amount in amounts
    ) / Decimal(len(amounts))
    return variance.sqrt() / mean


def build_sacc_funding_table(
    output: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Return the earliest valid funding amount for each existing SACC stream."""

    stream_id_text = output["stream_id"].astype("string")
    eligible_mask = (
        output["product_type"].eq(PERSONAL_LOAN)
        & stream_id_text.str.lower().str.startswith(SACC_PREFIX, na=False)
        & output["dr_cr"].astype("string").str.lower().eq("credit")
        & ~output["_is_dishonour_credit"]
    )

    funding_rows = output.loc[eligible_mask].copy()
    result_columns = [*group_columns, "stream_id", "funded_amount"]

    if funding_rows.empty:
        return pd.DataFrame(columns=result_columns)

    funding_rows["_funded_amount"] = funding_rows["amount"].map(
        parse_loc_amount
    )
    funding_rows = funding_rows.dropna(
        subset=["_transaction_date", "_funded_amount"]
    )
    if funding_rows.empty:
        return pd.DataFrame(columns=result_columns)

    funding_rows = funding_rows.sort_values(
        [*group_columns, "stream_id", "_transaction_date", "_row_id"],
        kind="stable",
    )
    first_funding_rows = funding_rows.drop_duplicates(
        subset=[*group_columns, "stream_id"],
        keep="first",
    )

    return first_funding_rows[
        [*group_columns, "stream_id", "_funded_amount"]
    ].rename(columns={"_funded_amount": "funded_amount"})


def build_group_mask(
    output: pd.DataFrame,
    group_columns: list[str],
    group_key: tuple[object, ...],
) -> pd.Series:
    """Build a null-safe mask for one configured personal-loan group."""

    mask = pd.Series(True, index=output.index, dtype=bool)

    for column, value in zip(group_columns, group_key):
        if pd.isna(value):
            mask &= output[column].isna()
        else:
            mask &= output[column].eq(value)

    return mask


def has_fractional_part(amount: Decimal) -> bool:
    return amount != amount.to_integral_value()


def repayment_amounts_are_related(
    anchor_amount: Decimal,
    candidate_amount: Decimal,
) -> bool:
    larger_amount = max(anchor_amount, candidate_amount)
    if larger_amount == 0:
        return False

    difference_ratio = abs(anchor_amount - candidate_amount) / larger_amount
    if difference_ratio <= LOC_REPAYMENT_CLOSE_RATIO:
        return True

    smaller_amount = min(anchor_amount, candidate_amount)
    half_ratio = smaller_amount / larger_amount
    return (
        LOC_REPAYMENT_HALF_RATIO_LOWER
        <= half_ratio
        <= LOC_REPAYMENT_HALF_RATIO_UPPER
    )


def repayment_dates_are_related(
    anchor_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> bool:
    anchor_dates = sorted(anchor_rows["_transaction_date"].dropna().tolist())
    candidate_dates = sorted(candidate_rows["_transaction_date"].dropna().tolist())

    if len(anchor_dates) < SINGLE_FUNDING_LOC_MIN_ANCHOR_DEBITS:
        return False
    if len(candidate_dates) < SINGLE_FUNDING_LOC_MIN_UNKNOWN_DEBITS:
        return False

    if any(anchor_dates[0] <= date <= anchor_dates[-1] for date in candidate_dates):
        if any(
            start_date < candidate_date < end_date
            for start_date, end_date in zip(anchor_dates, anchor_dates[1:])
            for candidate_date in candidate_dates
        ):
            return True

    anchor_start = anchor_dates[0]
    anchor_end = anchor_dates[-1]
    candidate_start = candidate_dates[0]
    candidate_end = candidate_dates[-1]
    max_gap = pd.Timedelta(days=LOC_REPAYMENT_CONTINUATION_MAX_GAP_DAYS)

    return (
        pd.Timedelta(0) <= anchor_start - candidate_end <= max_gap
        or pd.Timedelta(0) <= candidate_start - anchor_end <= max_gap
    )


def is_revolving_loc_funding_stream(value: object) -> bool:
    if pd.isna(value):
        return False
    stream_id = str(value).strip().lower()
    return (
        stream_id.startswith(SACC_PREFIX)
        or stream_id.startswith(NON_SACC_PREFIX)
        or stream_id.startswith(SPECIAL_LOC_PREFIX)
    )


def has_debit_near_dates(
    debit_dates: pd.Series,
    target_dates: pd.Series,
    max_gap_days: int,
) -> bool:
    valid_debit_dates = debit_dates.dropna()
    valid_target_dates = target_dates.dropna()
    if valid_debit_dates.empty or valid_target_dates.empty:
        return False

    max_gap = pd.Timedelta(days=max_gap_days)
    for target_date in valid_target_dates:
        if not (valid_debit_dates.sub(target_date).abs() <= max_gap).any():
            return False

    return True


def count_nearby_debit_pairs(
    anchor_dates: pd.Series,
    candidate_dates: pd.Series,
    max_gap_days: int,
) -> int:
    valid_anchor_dates = anchor_dates.dropna()
    valid_candidate_dates = candidate_dates.dropna()
    if valid_anchor_dates.empty or valid_candidate_dates.empty:
        return 0

    max_gap = pd.Timedelta(days=max_gap_days)
    pairs = 0
    for candidate_date in valid_candidate_dates:
        if (valid_anchor_dates.sub(candidate_date).abs() <= max_gap).any():
            pairs += 1

    return pairs


def get_existing_or_next_loc_id(
    output: pd.DataFrame,
    group_mask: pd.Series,
    loc_id_generator: LocStreamIdGenerator,
    application_id: object,
) -> str:
    existing_loc_ids = sorted(
        output.loc[
            group_mask
            & output["stream_id"]
            .astype("string")
            .str.lower()
            .str.startswith(SPECIAL_LOC_PREFIX, na=False),
            "stream_id",
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    return (
        existing_loc_ids[0]
        if existing_loc_ids
        else loc_id_generator.next(application_id)
    )


def has_parallel_stable_repayment_stream(
    debit_rows: pd.DataFrame,
    candidate_stream_ids: set[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> bool:
    if pd.isna(start_date) or pd.isna(end_date):
        return False

    candidate_stream_ids = {
        stream_id
        for stream_id in candidate_stream_ids
        if stream_id and stream_id != "<NA>"
    }
    other_rows = debit_rows[
        debit_rows["_transaction_date"].between(
            start_date,
            end_date,
            inclusive="both",
        )
        & ~debit_rows["stream_id"].astype("string").isin(candidate_stream_ids)
    ]
    if other_rows.empty:
        return False

    for _, stream_rows in other_rows.groupby(
        ["stream_id", "_amount_decimal"],
        dropna=True,
        sort=False,
    ):
        if len(stream_rows) >= REVOLVING_LOC_MIN_SWITCH_DEBITS:
            return True

    return False


def has_prior_funding_parallel_pattern(
    credit_rows: pd.DataFrame,
    debit_rows: pd.DataFrame,
    candidate_stream_ids: set[str],
    first_candidate_credit_date: pd.Timestamp,
) -> bool:
    if pd.isna(first_candidate_credit_date):
        return False

    candidate_stream_ids = {
        stream_id
        for stream_id in candidate_stream_ids
        if stream_id and stream_id != "<NA>"
    }
    prior_credit_rows = credit_rows[
        credit_rows["_transaction_date"].lt(first_candidate_credit_date)
        & ~credit_rows["stream_id"].astype("string").isin(candidate_stream_ids)
    ]
    if prior_credit_rows.empty:
        return False

    for _, amount_rows in debit_rows.groupby(
        "_amount_decimal",
        dropna=True,
        sort=False,
    ):
        before_count = len(
            amount_rows[
                amount_rows["_transaction_date"].lt(first_candidate_credit_date)
            ]
        )
        after_count = len(
            amount_rows[
                amount_rows["_transaction_date"].gt(first_candidate_credit_date)
            ]
        )
        if (
            before_count > 0
            and after_count >= REVOLVING_LOC_MIN_SWITCH_DEBITS
        ):
            return True

    return False


def has_parallel_repayment_amounts(rows: pd.DataFrame) -> bool:
    debit_rows = rows[
        rows["dr_cr"].astype("string").str.lower().eq("debit")
        & rows["_transaction_date"].notna()
        & rows["_amount_decimal"].notna()
        & rows["_amount_decimal"].gt(Decimal("0"))
    ]
    if debit_rows.empty:
        return False

    dates_by_amount: dict[Decimal, set[pd.Timestamp]] = {}
    for amount, amount_rows in debit_rows.groupby(
        "_amount_decimal",
        dropna=True,
        sort=False,
    ):
        if len(amount_rows) < PARALLEL_REPAYMENT_MIN_OVERLAP_DATES:
            continue
        dates_by_amount[amount] = set(amount_rows["_transaction_date"].dropna())

    if len(dates_by_amount) < PARALLEL_REPAYMENT_MIN_AMOUNT_STREAMS:
        return False

    amounts = list(dates_by_amount)
    for position, amount in enumerate(amounts[:-1]):
        for other_amount in amounts[position + 1:]:
            overlap_dates = dates_by_amount[amount].intersection(
                dates_by_amount[other_amount],
            )
            if len(overlap_dates) >= PARALLEL_REPAYMENT_MIN_OVERLAP_DATES:
                return True

    return False


def build_loc_merge_guard_rows(
    output: pd.DataFrame,
    group_mask: pd.Series,
    candidate_indices: pd.Index,
) -> pd.DataFrame:
    direct_loc_debit_mask = (
        group_mask
        & output["product_type"].eq("loc")
        & output["dr_cr"].astype("string").str.lower().eq("debit")
    )
    guard_indices = candidate_indices.union(output.index[direct_loc_debit_mask])
    return output.loc[guard_indices]


def merge_sacc_streams_into_loc(
    output: pd.DataFrame,
    group_columns: list[str],
    cv_threshold: Decimal = DEFAULT_LOC_CV_THRESHOLD,
    min_sacc_streams: int = DEFAULT_MIN_SACC_STREAMS,
) -> None:
    """Merge qualifying SACC streams into one special LOC stream.

    A group qualifies when it has at least ``min_sacc_streams`` SACC streams
    and the coefficient of variation of their funding amounts is greater than
    ``cv_threshold``. Only the matching ``sacc_*`` stream IDs are replaced.
    """

    if min_sacc_streams < 2:
        raise ValueError("min_sacc_streams must be at least 2.")
    if cv_threshold < 0:
        raise ValueError("cv_threshold cannot be negative.")

    output["_row_id"] = output.index
    output["_transaction_date"] = pd.to_datetime(
        output["transaction_date"],
        errors="coerce",
    )
    output["_amount_decimal"] = output["amount"].map(parse_decimal_amount)
    output["_is_dishonour_credit"] = is_dishonour_credit(output)
    if "sample_datetime" in output.columns:
        output["_sample_datetime"] = pd.to_datetime(
            output["sample_datetime"],
            errors="coerce",
        )
    else:
        output["_sample_datetime"] = pd.NaT

    funding_table = build_sacc_funding_table(output, group_columns)
    loc_id_generator = LocStreamIdGenerator(
        output[["application_id", "stream_id"]]
    )

    if not funding_table.empty:
        for group_key, group_funding in funding_table.groupby(
            group_columns,
            dropna=False,
            sort=True,
        ):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            group_funding = group_funding.drop_duplicates(
                subset=["stream_id"],
                keep="first",
            )
            amounts = group_funding["funded_amount"].tolist()

            if len(amounts) < min_sacc_streams:
                continue

            cv = calculate_cv(amounts)
            if cv is None or cv <= cv_threshold:
                continue

            original_sacc_ids = set(
                group_funding["stream_id"].dropna().astype(str)
            )
            group_mask = build_group_mask(
                output,
                group_columns,
                group_key,
            )
            existing_loc_ids = sorted(
                output.loc[
                    group_mask
                    & output["stream_id"]
                    .astype("string")
                    .str.lower()
                    .str.startswith(SPECIAL_LOC_PREFIX, na=False),
                    "stream_id",
                ]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda value: int(
                    re.fullmatch(r"loc_(\d+)", value.strip().lower()).group(1)
                ),
            )
            sacc_stream_mask = (
                output["product_type"].eq(PERSONAL_LOAN)
                & output["stream_id"].astype("string").isin(
                    original_sacc_ids
                )
            )
            update_mask = group_mask & sacc_stream_mask

            if not update_mask.any():
                continue
            guard_rows = build_loc_merge_guard_rows(
                output,
                group_mask,
                output.index[update_mask],
            )
            if has_parallel_repayment_amounts(guard_rows):
                continue

            stream_id = (
                existing_loc_ids[0]
                if existing_loc_ids
                else loc_id_generator.next(group_key[0])
            )
            output.loc[update_mask, "stream_id"] = stream_id

    revolving_group_columns = list(group_columns)
    if (
        "bank_account_id" in output.columns
        and "bank_account_id" not in revolving_group_columns
    ):
        revolving_group_columns.append("bank_account_id")

    dr_cr_text = output["dr_cr"].astype("string").str.lower()
    revolving_rows = output.loc[
        output["product_type"].isin([PERSONAL_LOAN, "wage_advance"])
        & dr_cr_text.isin(["credit", "debit"])
        & output["_transaction_date"].notna()
    ].copy()

    if not revolving_rows.empty:
        revolving_rows["_amount_decimal"] = revolving_rows["amount"].map(
            parse_decimal_amount
        )

        for group_key, group_rows in revolving_rows.groupby(
            revolving_group_columns,
            dropna=False,
            sort=True,
        ):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            group_dr_cr_text = group_rows["dr_cr"].astype("string").str.lower()
            group_stream_id_text = group_rows["stream_id"].astype("string")
            valid_amount_mask = group_rows["_amount_decimal"].notna() & group_rows[
                "_amount_decimal"
            ].gt(Decimal("0"))
            eligible_credit_mask = group_stream_id_text.map(
                is_revolving_loc_funding_stream
            )
            credit_rows = group_rows[
                group_rows["product_type"].eq(PERSONAL_LOAN)
                & group_dr_cr_text.eq("credit")
                & ~group_rows["_is_dishonour_credit"]
                & valid_amount_mask
                & eligible_credit_mask
            ].sort_values("_transaction_date", kind="stable")

            if len(credit_rows) < 2:
                continue

            debit_rows = group_rows[
                group_rows["product_type"].eq(PERSONAL_LOAN)
                & group_dr_cr_text.eq("debit")
                & valid_amount_mask
            ]

            if debit_rows.empty:
                continue

            group_mask = build_group_mask(
                output,
                revolving_group_columns,
                group_key,
            )
            assigned_revolving_loc = False

            ordered_credit_rows = credit_rows.sort_values(
                "_transaction_date",
                kind="stable",
            )
            credit_records = list(ordered_credit_rows.iterrows())
            for (_, previous_credit), (_, next_credit) in zip(
                credit_records,
                credit_records[1:],
            ):
                previous_date = previous_credit["_transaction_date"]
                next_date = next_credit["_transaction_date"]
                previous_amount = previous_credit["_amount_decimal"]
                previous_stream_id = str(previous_credit["stream_id"])
                next_stream_id = str(next_credit["stream_id"])

                before_next_debits = debit_rows[
                    debit_rows["_transaction_date"].gt(previous_date)
                    & debit_rows["_transaction_date"].lt(next_date)
                    & debit_rows["stream_id"]
                    .astype("string")
                    .eq(previous_stream_id)
                ]
                if before_next_debits.empty:
                    continue

                repaid_before_next = before_next_debits["_amount_decimal"].sum()
                if (
                    previous_amount is None
                    or pd.isna(previous_amount)
                    or repaid_before_next
                    >= previous_amount * REVOLVING_LOC_UNPAID_RATIO
                ):
                    continue

                after_next_debits = debit_rows[
                    debit_rows["_transaction_date"].gt(next_date)
                    & debit_rows["stream_id"]
                    .astype("string")
                    .eq(next_stream_id)
                ]
                if after_next_debits.empty:
                    continue

                previous_stable_amounts = [
                    amount
                    for amount, rows in before_next_debits.groupby(
                        "_amount_decimal",
                        dropna=True,
                        sort=False,
                    )
                    if len(rows) >= REVOLVING_LOC_MIN_SWITCH_DEBITS
                ]
                next_stable_amounts = [
                    amount
                    for amount, rows in after_next_debits.groupby(
                        "_amount_decimal",
                        dropna=True,
                        sort=False,
                    )
                    if len(rows) >= REVOLVING_LOC_MIN_SWITCH_DEBITS
                ]
                if not previous_stable_amounts or not next_stable_amounts:
                    continue

                for previous_repayment_amount in previous_stable_amounts:
                    previous_after_count = len(
                        debit_rows[
                            debit_rows["_transaction_date"].gt(next_date)
                            & debit_rows["stream_id"]
                            .astype("string")
                            .eq(previous_stream_id)
                            & debit_rows["_amount_decimal"].eq(
                                previous_repayment_amount
                            )
                        ]
                    )
                    if previous_after_count > 1:
                        continue

                    stream_ids_to_merge = {previous_stream_id, next_stream_id}
                    stream_ids_to_merge.discard("<NA>")
                    switch_end_date = after_next_debits[
                        "_transaction_date"
                    ].max()
                    if has_parallel_stable_repayment_stream(
                        debit_rows,
                        stream_ids_to_merge,
                        previous_date,
                        switch_end_date,
                    ):
                        continue
                    if has_prior_funding_parallel_pattern(
                        credit_rows,
                        debit_rows,
                        stream_ids_to_merge,
                        previous_date,
                    ):
                        continue

                    update_mask = (
                        group_mask
                        & output["product_type"].eq(PERSONAL_LOAN)
                        & output["stream_id"].astype("string").isin(
                            stream_ids_to_merge
                        )
                    )

                    if not update_mask.any():
                        continue
                    guard_rows = build_loc_merge_guard_rows(
                        output,
                        group_mask,
                        output.index[update_mask],
                    )
                    if has_parallel_repayment_amounts(guard_rows):
                        continue

                    stream_id = get_existing_or_next_loc_id(
                        output,
                        group_mask,
                        loc_id_generator,
                        group_key[0],
                    )
                    output.loc[update_mask, "stream_id"] = stream_id
                    assigned_revolving_loc = True
                    break

                if assigned_revolving_loc:
                    break

            if assigned_revolving_loc:
                continue

            max_close_credit_gap = pd.Timedelta(
                days=REVOLVING_LOC_CLOSE_CREDIT_MAX_GAP_DAYS
            )
            for (_, previous_credit), (_, next_credit) in zip(
                credit_records,
                credit_records[1:],
            ):
                previous_date = previous_credit["_transaction_date"]
                next_date = next_credit["_transaction_date"]
                if (
                    pd.isna(previous_date)
                    or pd.isna(next_date)
                    or next_date - previous_date > max_close_credit_gap
                ):
                    continue

                previous_stream_id = str(previous_credit["stream_id"])
                next_stream_id = str(next_credit["stream_id"])
                stream_ids_to_merge = {previous_stream_id, next_stream_id}
                stream_ids_to_merge.discard("<NA>")
                if len(stream_ids_to_merge) < 2:
                    continue

                between_credit_debits = debit_rows[
                    debit_rows["_transaction_date"].gt(previous_date)
                    & debit_rows["_transaction_date"].lt(next_date)
                ]
                if not between_credit_debits.empty:
                    continue

                following_debits = debit_rows[
                    debit_rows["_transaction_date"].gt(next_date)
                    & debit_rows["stream_id"].astype("string").eq(next_stream_id)
                ]
                if following_debits.empty:
                    continue

                has_stable_following_debit = any(
                    len(rows) >= REVOLVING_LOC_MIN_FOLLOWING_DEBITS
                    for _, rows in following_debits.groupby(
                        "_amount_decimal",
                        dropna=True,
                        sort=False,
                    )
                )
                if not has_stable_following_debit:
                    continue

                close_credit_end_date = following_debits[
                    "_transaction_date"
                ].max()
                if has_parallel_stable_repayment_stream(
                    debit_rows,
                    stream_ids_to_merge,
                    previous_date,
                    close_credit_end_date,
                ):
                    continue
                if has_prior_funding_parallel_pattern(
                    credit_rows,
                    debit_rows,
                    stream_ids_to_merge,
                    previous_date,
                ):
                    continue

                update_mask = (
                    group_mask
                    & output["product_type"].eq(PERSONAL_LOAN)
                    & output["stream_id"].astype("string").isin(
                        stream_ids_to_merge
                    )
                )

                if not update_mask.any():
                    continue
                guard_rows = build_loc_merge_guard_rows(
                    output,
                    group_mask,
                    output.index[update_mask],
                )
                if has_parallel_repayment_amounts(guard_rows):
                    continue

                stream_id = get_existing_or_next_loc_id(
                    output,
                    group_mask,
                    loc_id_generator,
                    group_key[0],
                )
                output.loc[update_mask, "stream_id"] = stream_id
                assigned_revolving_loc = True
                break

            if assigned_revolving_loc:
                continue

            if credit_rows["_amount_decimal"].duplicated().any():
                continue

            for anchor_amount, amount_debit_rows in debit_rows.groupby(
                "_amount_decimal",
                dropna=True,
                sort=False,
            ):
                if len(amount_debit_rows) < REVOLVING_LOC_MIN_ANCHOR_DEBITS:
                    continue

                repayment_start = amount_debit_rows["_transaction_date"].min()
                repayment_end = amount_debit_rows["_transaction_date"].max()
                covered_credit_rows = credit_rows[
                    credit_rows["_transaction_date"].between(
                        repayment_start,
                        repayment_end,
                        inclusive="both",
                    )
                ]

                if len(covered_credit_rows) < 2:
                    continue

                covered_credit_amounts = set(
                    covered_credit_rows["_amount_decimal"].dropna().tolist()
                )
                if len(covered_credit_amounts) < 2:
                    continue
                if covered_credit_rows["_amount_decimal"].duplicated().any():
                    continue

                if not has_debit_near_dates(
                    amount_debit_rows["_transaction_date"],
                    covered_credit_rows["_transaction_date"],
                    REVOLVING_LOC_CREDIT_ANCHOR_WINDOW_DAYS,
                ):
                    continue

                has_repeated_funding_in_stream = False
                for credit_stream_id in (
                    covered_credit_rows["stream_id"].dropna().astype(str).unique()
                ):
                    stream_credit_rows = credit_rows[
                        credit_rows["stream_id"]
                        .astype("string")
                        .eq(str(credit_stream_id))
                    ]
                    if stream_credit_rows["_amount_decimal"].duplicated().any():
                        has_repeated_funding_in_stream = True
                        break

                if has_repeated_funding_in_stream:
                    continue

                debit_indices_to_merge = amount_debit_rows.index
                for companion_amount, companion_rows in debit_rows.groupby(
                    "_amount_decimal",
                    dropna=True,
                    sort=False,
                ):
                    if companion_amount == anchor_amount:
                        continue
                    if len(companion_rows) < REVOLVING_LOC_MIN_SWITCH_DEBITS:
                        continue
                    nearby_pairs = count_nearby_debit_pairs(
                        amount_debit_rows["_transaction_date"],
                        companion_rows["_transaction_date"],
                        REVOLVING_LOC_COMPANION_MAX_GAP_DAYS,
                    )
                    if nearby_pairs < REVOLVING_LOC_MIN_COMPANION_NEARBY_PAIRS:
                        continue

                    companion_independent = False
                    for _, credit_row in credit_rows.drop(
                        index=covered_credit_rows.index,
                        errors="ignore",
                    ).iterrows():
                        credit_date = credit_row["_transaction_date"]
                        if (
                            len(
                                companion_rows[
                                    companion_rows["_transaction_date"].gt(
                                        credit_date
                                    )
                                ]
                            )
                            >= REVOLVING_LOC_MIN_SWITCH_DEBITS
                        ):
                            companion_independent = True
                            break

                    if companion_independent:
                        continue

                    debit_indices_to_merge = debit_indices_to_merge.union(
                        companion_rows.index
                    )

                stream_ids_to_merge = set(
                    covered_credit_rows["stream_id"].dropna().astype(str)
                )
                first_covered_credit_date = covered_credit_rows[
                    "_transaction_date"
                ].min()
                if has_prior_funding_parallel_pattern(
                    credit_rows,
                    debit_rows,
                    stream_ids_to_merge,
                    first_covered_credit_date,
                ):
                    continue

                update_mask = (
                    group_mask
                    & output["product_type"].eq(PERSONAL_LOAN)
                    & output["stream_id"].astype("string").isin(
                        stream_ids_to_merge
                    )
                )
                update_indices = output.index[update_mask].union(
                    debit_indices_to_merge
                )
                if len(update_indices) == 0:
                    continue
                guard_rows = build_loc_merge_guard_rows(
                    output,
                    group_mask,
                    update_indices,
                )
                if has_parallel_repayment_amounts(guard_rows):
                    continue

                stream_id = get_existing_or_next_loc_id(
                    output,
                    group_mask,
                    loc_id_generator,
                    group_key[0],
                )
                output.loc[update_indices, "stream_id"] = stream_id
                break

        for group_key, group_rows in revolving_rows.groupby(
            revolving_group_columns,
            dropna=False,
            sort=True,
        ):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            group_dr_cr_text = group_rows["dr_cr"].astype("string").str.lower()
            group_stream_id_text = (
                group_rows["stream_id"].astype("string").str.lower()
            )
            group_amount = group_rows["_amount_decimal"]
            valid_amount_mask = group_amount.notna() & group_amount.gt(
                Decimal("0")
            )
            fractional_amount_mask = group_amount.map(
                lambda amount: (
                    amount is not None
                    and not pd.isna(amount)
                    and has_fractional_part(amount)
                )
            )
            funding_rows = group_rows[
                group_rows["product_type"].eq(PERSONAL_LOAN)
                & group_dr_cr_text.eq("credit")
                & ~group_rows["_is_dishonour_credit"]
                & valid_amount_mask
                & fractional_amount_mask
            ]

            if funding_rows.empty:
                continue

            funding_stream_ids = set(
                funding_rows["stream_id"].dropna().astype(str)
            )
            if len(funding_stream_ids) != 1:
                continue

            funding_stream_id = next(iter(funding_stream_ids))
            unknown_stream_mask = group_stream_id_text.str.startswith(
                "unknown_",
                na=False,
            )
            anchor_stream_rows = group_rows[
                group_rows["stream_id"].astype("string").eq(funding_stream_id)
            ]
            anchor_debit_rows = anchor_stream_rows[
                anchor_stream_rows["product_type"].eq(PERSONAL_LOAN)
                & anchor_stream_rows["dr_cr"].astype("string").str.lower().eq(
                    "debit"
                )
                & anchor_stream_rows["_amount_decimal"].notna()
                & anchor_stream_rows["_amount_decimal"].gt(Decimal("0"))
            ].copy()
            unknown_debit_rows = group_rows[
                group_rows["product_type"].eq(PERSONAL_LOAN)
                & group_dr_cr_text.eq("debit")
                & valid_amount_mask
                & unknown_stream_mask
            ]

            if anchor_debit_rows.empty or unknown_debit_rows.empty:
                continue

            matched_unknown_stream_ids: set[str] = set()
            anchor_stream_ids_to_merge = {funding_stream_id}

            for anchor_amount, anchor_amount_rows in anchor_debit_rows.groupby(
                "_amount_decimal",
                dropna=True,
                sort=False,
            ):
                if len(anchor_amount_rows) < SINGLE_FUNDING_LOC_MIN_ANCHOR_DEBITS:
                    continue
                if not has_fractional_part(anchor_amount):
                    continue

                for candidate_stream_id, candidate_stream_rows in (
                    unknown_debit_rows.groupby(
                        "stream_id",
                        dropna=True,
                        sort=False,
                    )
                ):
                    candidate_stream_id_text = str(candidate_stream_id)
                    same_stream_rows = group_rows[
                        group_rows["stream_id"]
                        .astype("string")
                        .eq(candidate_stream_id_text)
                    ]
                    has_credit = same_stream_rows["dr_cr"].astype(
                        "string"
                    ).str.lower().eq("credit").any()
                    if has_credit:
                        continue

                    candidate_related_rows = candidate_stream_rows[
                        candidate_stream_rows["_amount_decimal"].map(
                            lambda amount: (
                                amount is not None
                                and not pd.isna(amount)
                                and has_fractional_part(amount)
                                and repayment_amounts_are_related(
                                    anchor_amount,
                                    amount,
                                )
                            )
                        )
                    ]
                    if (
                        len(candidate_related_rows)
                        < SINGLE_FUNDING_LOC_MIN_UNKNOWN_DEBITS
                    ):
                        continue
                    if not repayment_dates_are_related(
                        anchor_amount_rows,
                        candidate_related_rows,
                    ):
                        continue
                    if (
                        len(anchor_amount_rows) + len(candidate_related_rows)
                        < SINGLE_FUNDING_LOC_MIN_TOTAL_DEBITS
                    ):
                        continue

                    matched_unknown_stream_ids.add(candidate_stream_id_text)

            if not matched_unknown_stream_ids:
                continue

            stream_ids_to_merge = anchor_stream_ids_to_merge.union(
                matched_unknown_stream_ids
            )
            group_mask = build_group_mask(
                output,
                revolving_group_columns,
                group_key,
            )
            existing_loc_ids = sorted(
                output.loc[
                    group_mask
                    & output["stream_id"]
                    .astype("string")
                    .str.lower()
                    .str.startswith(SPECIAL_LOC_PREFIX, na=False),
                    "stream_id",
                ]
                .dropna()
                .astype(str)
                .unique()
            )
            stream_id = (
                existing_loc_ids[0]
                if existing_loc_ids
                else loc_id_generator.next(group_key[0])
            )
            update_mask = (
                group_mask
                & output["product_type"].eq(PERSONAL_LOAN)
                & output["stream_id"].astype("string").isin(stream_ids_to_merge)
            )

            if not update_mask.any():
                continue
            guard_rows = build_loc_merge_guard_rows(
                output,
                group_mask,
                output.index[update_mask],
            )
            if has_parallel_repayment_amounts(guard_rows):
                continue

            output.loc[update_mask, "stream_id"] = stream_id

    loc_stream_mask = (
        output["stream_id"]
        .astype("string")
        .str.lower()
        .str.startswith(SPECIAL_LOC_PREFIX, na=False)
    )
    sacc_stream_mask = (
        output["product_type"].eq(PERSONAL_LOAN)
        & output["stream_id"]
        .astype("string")
        .str.lower()
        .str.startswith(SACC_PREFIX, na=False)
    )
    loc_rows = output.loc[loc_stream_mask]

    if not loc_rows.empty:
        for group_key, group_loc_rows in loc_rows.groupby(
            group_columns,
            dropna=False,
            sort=True,
        ):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            target_loc_ids = sorted(
                group_loc_rows["stream_id"]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda value: int(
                    re.fullmatch(r"loc_(\d+)", value.strip().lower()).group(1)
                ),
            )
            if not target_loc_ids:
                continue

            target_loc_id = target_loc_ids[0]
            group_mask = build_group_mask(
                output,
                group_columns,
                group_key,
            )
            group_sacc_rows = output.loc[group_mask & sacc_stream_mask]
            if group_sacc_rows.empty:
                continue

            for stream_id, stream_rows in group_sacc_rows.groupby(
                "stream_id",
                dropna=True,
                sort=False,
            ):
                has_credit = stream_rows["dr_cr"].astype("string").str.lower().eq(
                    "credit"
                ).any()
                has_debit = stream_rows["dr_cr"].astype("string").str.lower().eq(
                    "debit"
                ).any()
                if not has_credit or has_debit:
                    continue

                credit_rows = stream_rows[
                    stream_rows["dr_cr"].astype("string").str.lower().eq(
                        "credit"
                    )
                ]
                last_credit_date = credit_rows["_transaction_date"].max()
                sample_datetime = stream_rows["_sample_datetime"].max()
                if pd.isna(last_credit_date) or pd.isna(sample_datetime):
                    continue
                if (
                    sample_datetime - last_credit_date
                    < pd.Timedelta(days=ORPHAN_SACC_MIN_AGE_DAYS)
                ):
                    continue

                output.loc[stream_rows.index, "stream_id"] = target_loc_id

    loc_stream_mask = (
        output["stream_id"]
        .astype("string")
        .str.lower()
        .str.startswith(SPECIAL_LOC_PREFIX, na=False)
    )
    loc_rows = output.loc[loc_stream_mask]
    if not loc_rows.empty:
        for group_key, group_loc_rows in loc_rows.groupby(
            group_columns,
            dropna=False,
            sort=True,
        ):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            loc_ids = sorted(
                group_loc_rows["stream_id"]
                .dropna()
                .astype(str)
                .unique(),
                key=lambda value: int(
                    re.fullmatch(r"loc_(\d+)", value.strip().lower()).group(1)
                ),
            )
            if len(loc_ids) <= 1:
                continue

            group_mask = build_group_mask(
                output,
                group_columns,
                group_key,
            )
            duplicate_loc_ids = set(loc_ids[1:])
            update_mask = (
                group_mask
                & output["stream_id"].astype("string").isin(duplicate_loc_ids)
            )
            if not update_mask.any():
                continue

            output.loc[update_mask, "stream_id"] = loc_ids[0]

    output.drop(
        columns=[
            "_row_id",
            "_transaction_date",
            "_amount_decimal",
            "_is_dishonour_credit",
            "_sample_datetime",
        ],
        inplace=True,
    )


def identify_loc_streams(
    output: pd.DataFrame,
    eligible_mask: pd.Series,
    group_columns: list[str],
) -> None:
    """Run the final LOC stage.

    1. Assign original ``product_type == loc`` rows using the existing grouped
       rule and ``loc_001`` format.
    2. Refine qualifying personal-loan ``sacc_*`` streams into ``loc_*``.
    """

    identify_direct_loc_streams(
        output,
        eligible_mask,
        group_columns,
    )
    merge_sacc_streams_into_loc(
        output,
        group_columns,
    )


# ---------------------------------------------------------------------------
# Priority dispatcher
# ---------------------------------------------------------------------------


# Lower number = higher priority.
# Once a row matches one product_type, it is added to claimed_mask and cannot
# enter any later rule.
PRODUCT_RULES: tuple[ProductRule, ...] = (
    ProductRule(10, "bnpl", identify_bnpl_streams),
    ProductRule(20, "wage_advance", identify_wage_advance_streams),
    ProductRule(25, "home_loan", identify_home_loan_streams),
    ProductRule(27, "car_loan", identify_car_loan_streams),
    ProductRule(30, "bank", identify_bank_streams),
    ProductRule(35, "contract_loan", identify_contract_loan_streams),
    ProductRule(37, "generic_loan", identify_generic_loan_streams),
    ProductRule(40, PERSONAL_LOAN, assign_personal_loan_rule),
    ProductRule(50, "loc", identify_loc_streams),
)

FINV_CATEGORY_MAP = {
    "bank": "Credit Card Repayments",
    "bnpl": "Non SACC Loans",
    "wage_advance": "Non SACC Loans",
    "personal_loan_sacc": "SACC Loans",
    "personal_loan_unknown": "Unknown Loans",
    "personal_loan_non_sacc": "Non SACC Loans",
    "contract_loan": "Non SACC Loans",
    "generic_loan": "Non SACC Loans",
    "loc": "Non SACC Loans",
    "home_loan": "Non SACC Loans",
    "car_loan": "Non SACC Loans",
}


class _StreamIdCounter:
    """O(1) per-call counter for generating stream IDs with ``{prefix}_{###}`` format.

    Scans *output* once on construction to find existing max counters per
    ``(application_id, prefix)``, then tracks increments in-memory so each
    ``next()`` call is a dict lookup rather than a full DataFrame scan.
    """

    def __init__(self, output: pd.DataFrame) -> None:
        self._counters: dict[tuple[object, str], int] = {}
        if "stream_id" not in output.columns or output["stream_id"].isna().all():
            return

        relevant = output.loc[
            output["stream_id"].notna(), ["application_id", "stream_id"]
        ]
        if relevant.empty:
            return

        # Vectorised parse: extract prefix & counter via str.extract.
        sid_col = relevant["stream_id"].astype("string").str.strip()
        extracted = sid_col.str.extract(r"(.+?)_(\d+)")
        valid = extracted[0].notna() & extracted[1].notna()

        if not valid.any():
            return

        app_key_col = relevant["application_id"].map(normalize_group_value)
        prefix_col = extracted[0][valid]
        counter_col = extracted[1][valid].astype(int)

        # Max counter per (app_key, prefix).
        max_counters = (
            counter_col.groupby(
                [app_key_col[valid], prefix_col], sort=False
            ).max()
        )
        for (app_key, prefix), counter in max_counters.items():
            self._counters[(app_key, prefix)] = counter

    def next(self, application_id: object, prefix: str) -> str:
        key = (normalize_group_value(application_id), prefix)
        current = self._counters.get(key, 0) + 1
        self._counters[key] = current
        return f"{prefix}_{current:03d}"


def median_gap_matches_unknown_frequency(median_gap: float) -> bool:
    return any(
        lower <= median_gap <= upper
        for lower, upper in UNKNOWN_PERIODIC_MEDIAN_GAP_RANGES
    )


def merge_periodic_remaining_unknown_streams(
    output: pd.DataFrame,
    group_columns: list[str],
) -> None:
    """Merge remaining unknown streams when all transaction dates are periodic."""

    stream_text = output["stream_id"].astype("string").str.strip().str.lower()
    unknown_rows = output.loc[
        output["product_type"].eq(PERSONAL_LOAN)
        & stream_text.str.startswith(UNKNOWN_PREFIX, na=False)
    ]
    for _, group in unknown_rows.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        if group["stream_id"].nunique(dropna=True) < 2:
            continue

        debit_rows = group[
            group["dr_cr"].astype("string").str.lower().eq("debit")
        ]
        dates = (
            debit_rows["_transaction_date"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        if len(dates) < UNKNOWN_PERIODIC_MIN_DATES:
            continue

        gaps = dates.diff().dropna().dt.days
        if gaps.empty:
            continue

        median_gap = gaps.median()
        if not median_gap_matches_unknown_frequency(median_gap):
            continue

        target_stream_id = (
            group.sort_values("_transaction_date")["stream_id"]
            .dropna()
            .astype(str)
            .iloc[0]
        )
        update_index = group.index[
            ~group["stream_id"].astype("string").eq(target_stream_id)
        ]
        if len(update_index) == 0:
            continue

        output.loc[update_index, "stream_id"] = target_stream_id


def refine_unknown_personal_loan_streams(
    output: pd.DataFrame,
    group_columns: list[str],
    counter: _StreamIdCounter | None = None,
) -> None:
    """Refine remaining unknown personal-loan streams after LOC refinement."""

    if counter is None:
        counter = _StreamIdCounter(output)

    pl_mask = (
        output["product_type"].eq(PERSONAL_LOAN)
        & output["stream_id"].notna()
    )
    pl_idx = output.index[pl_mask]

    output["_transaction_date"] = pd.NaT
    output.loc[pl_idx, "_transaction_date"] = pd.to_datetime(
        output.loc[pl_idx, "transaction_date"], errors="coerce"
    )
    output["_amount_decimal"] = None
    output.loc[pl_idx, "_amount_decimal"] = (
        output.loc[pl_idx, "amount"].map(parse_decimal_amount)
    )
    output["_sample_datetime"] = pd.NaT
    if "sample_datetime" in output.columns:
        output.loc[pl_idx, "_sample_datetime"] = pd.to_datetime(
            output.loc[pl_idx, "sample_datetime"], errors="coerce"
        )

    max_gap = pd.Timedelta(days=UNKNOWN_REASSIGN_MAX_GAP_DAYS)
    target_prefixes = (
        SACC_PREFIX,
        NON_SACC_PREFIX,
        SPECIAL_LOC_PREFIX,
    )

    personal_loan_rows = output.loc[pl_idx]

    for _, group in personal_loan_rows.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        is_debit = group["dr_cr"].astype("string").str.lower().eq("debit")
        not_dishonour = ~(
            group[DISHONOUR_COLUMN].astype("string").str.lower().eq("yes")
        )
        has_amount = group["_amount_decimal"].notna()
        valid_repayment = is_debit & not_dishonour & has_amount

        multi_row_unknown_stream_ids: set[str] = set()

        for stream_id, stream_rows in group.groupby(
            "stream_id",
            dropna=True,
            sort=False,
        ):
            stream_id_text = str(stream_id).strip().lower()
            if not stream_id_text.startswith(UNKNOWN_PREFIX):
                continue
            if len(stream_rows) == 1:
                continue

            multi_row_unknown_stream_ids.add(stream_id_text)
            stream_repayment = valid_repayment.loc[stream_rows.index]
            repayment_rows = stream_rows[stream_repayment]
            repayment_total = sum(
                abs(amount)
                for amount in repayment_rows["_amount_decimal"].tolist()
            )

            application_id = stream_rows["application_id"].iloc[0]
            if repayment_total > UNKNOWN_NON_SACC_REPAYMENT_THRESHOLD:
                target_stream_id = counter.next(application_id, "non_sacc")
            else:
                latest_transaction_date = stream_rows["_transaction_date"].max()
                sample_datetime = stream_rows["_sample_datetime"].max()
                latest_rows = stream_rows[
                    stream_rows["_transaction_date"].eq(latest_transaction_date)
                ]
                latest_is_dishonour = (
                    latest_rows[DISHONOUR_COLUMN]
                    .astype("string")
                    .str.lower()
                    .eq("yes")
                    .any()
                )
                if (
                    pd.isna(latest_transaction_date)
                    or pd.isna(sample_datetime)
                    or latest_is_dishonour
                    or sample_datetime - latest_transaction_date <= max_gap
                ):
                    continue

                target_stream_id = counter.next(application_id, "sacc")

            output.loc[stream_rows.index, "stream_id"] = target_stream_id

        group = output.loc[group.index]
        stream_text = group["stream_id"].astype("string").str.lower()
        candidate_rows = group[
            stream_text.str.startswith(target_prefixes, na=False)
        ]
        candidate_last_dates = (
            candidate_rows.dropna(subset=["_transaction_date"])
            .groupby("stream_id", dropna=True)["_transaction_date"]
            .max()
        )

        for stream_id, stream_rows in group.groupby(
            "stream_id",
            dropna=True,
            sort=False,
        ):
            stream_id_text = str(stream_id).strip().lower()
            if not stream_id_text.startswith(UNKNOWN_PREFIX):
                continue
            if stream_id_text in multi_row_unknown_stream_ids:
                continue

            transaction_date = stream_rows["_transaction_date"].max()
            if pd.isna(transaction_date) or candidate_last_dates.empty:
                continue

            nearby_gaps = (candidate_last_dates - transaction_date).abs()
            nearby_gaps = nearby_gaps[nearby_gaps <= max_gap]
            if nearby_gaps.empty:
                continue

            target_stream_id = nearby_gaps.sort_values().index[0]
            output.loc[stream_rows.index, "stream_id"] = target_stream_id

    merge_periodic_remaining_unknown_streams(
        output,
        group_columns,
    )

    output.drop(
        columns=["_transaction_date", "_amount_decimal", "_sample_datetime"],
        inplace=True,
        errors="ignore",
    )


def apply_special_counterparty_stream_overrides(
    output: pd.DataFrame,
    group_columns: list[str],
    counter: _StreamIdCounter | None = None,
) -> None:
    if counter is None:
        counter = _StreamIdCounter(output)

    personal_loan_rows = output.loc[
        output["product_type"].eq(PERSONAL_LOAN)
        & output["counterparty"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(SPECIAL_COUNTERPARTY_STREAM_RULES.keys())
    ]

    for counterparty, counterparty_rows in personal_loan_rows.groupby(
        personal_loan_rows["counterparty"].astype("string").str.strip().str.lower(),
        dropna=True,
        sort=True,
    ):
        rule = SPECIAL_COUNTERPARTY_STREAM_RULES.get(str(counterparty))
        if not rule:
            continue

        prefix = rule["target_prefix"]
        mode = rule["mode"]

        for _, group in counterparty_rows.groupby(
            group_columns,
            dropna=False,
            sort=True,
        ):
            if mode == "convert_sacc_streams":
                stream_text = group["stream_id"].astype("string").str.lower()
                sacc_rows = group.loc[
                    stream_text.str.startswith(SACC_PREFIX, na=False)
                ]
                if sacc_rows.empty:
                    continue

                for _, stream_group in sacc_rows.groupby(
                    "stream_id",
                    dropna=False,
                    sort=True,
                ):
                    target_stream_id = counter.next(
                        stream_group["application_id"].iloc[0], prefix
                    )
                    output.loc[stream_group.index, "stream_id"] = target_stream_id
                continue

            if mode != "merge_group":
                continue

            prefixed_rows = group[
                group["stream_id"]
                .astype("string")
                .str.lower()
                .str.startswith(f"{prefix}_", na=False)
            ]
            if prefixed_rows.empty:
                target_stream_id = counter.next(
                    group["application_id"].iloc[0], prefix
                )
            else:
                target_stream_id = (
                    prefixed_rows["stream_id"]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .iloc[0]
                )

            update_index = group.index[
                ~group["stream_id"].astype("string").eq(target_stream_id)
            ]
            if len(update_index) == 0:
                continue

            output.loc[update_index, "stream_id"] = target_stream_id


def renumber_stream_ids_by_application(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber stream IDs so they are consecutive per (application, prefix).

    The original implementation used two ``iterrows()`` passes — one to
    discover the existing stream IDs and their first positions, and a
    second to apply the replacements.  This version uses ``str.extract``
    and ``groupby`` to discover the mapping, then applies it with
    ``Series.map`` so every operation stays inside pandas' C-level loops.
    """
    output = df.copy()
    if "application_id" not in output.columns or "stream_id" not in output.columns:
        return output

    # ---- 1. Parse stream IDs (vectorised str.extract) ----
    sid_col = output["stream_id"].astype("string").str.strip()
    extracted = sid_col.str.extract(r"(.+?)_(\d+)")
    # extracted[0] = prefix,  extracted[1] = counter digits
    valid = extracted[0].notna() & extracted[1].notna()

    if not valid.any():
        return output

    # ---- 2. Build lookup columns ----
    app_key_col = output["application_id"].map(normalize_group_value)
    position_col = pd.Series(range(len(output)), index=output.index)

    # ---- 3. First-position per unique (app_key, stream_id) ----
    sid_clean = sid_col.where(valid, other=pd.NA)
    first_pos = (
        position_col
        .groupby([app_key_col, sid_clean], sort=False)
        .first()
    )

    # ---- 4. Build a DataFrame of unique streams ----
    streams_df = pd.DataFrame({
        "app_key": app_key_col[valid],
        "stream_id": sid_clean[valid],
        "prefix": extracted[0][valid],
        "counter": extracted[1][valid].astype(int),
    }).drop_duplicates(subset=["app_key", "stream_id"])

    # Attach first positions via merge
    streams_df = streams_df.merge(
        first_pos.rename("first_pos"),
        left_on=["app_key", "stream_id"],
        right_index=True,
        how="left",
    )

    # ---- 5. Assign new consecutive counters per (app_key, prefix) ----
    # Sort within each group: (old counter, first position, stream_id)
    streams_df = streams_df.sort_values(
        ["app_key", "prefix", "counter", "first_pos", "stream_id"],
        kind="stable",
    )
    streams_df["new_counter"] = streams_df.groupby(
        ["app_key", "prefix"], sort=False
    ).cumcount() + 1
    streams_df["new_stream_id"] = (
        streams_df["prefix"] + "_" + streams_df["new_counter"].astype(str).str.zfill(3)
    )

    # ---- 6. Build replacement map and apply via Series.map ----
    streams_df["_map_key"] = streams_df["app_key"].astype(str) + "||" + streams_df["stream_id"]
    replacement_map = dict(
        zip(streams_df["_map_key"], streams_df["new_stream_id"])
    )

    if not replacement_map:
        return output

    row_keys = app_key_col.astype(str) + "||" + sid_col
    output["stream_id"] = row_keys.map(replacement_map).fillna(sid_col).values

    return output


def renumber_stream_ids_uniform(output: pd.DataFrame) -> pd.DataFrame:
    """Renumber every assigned stream_id as ``loan_001``-style IDs.

    Runs at the very end of the liability pipeline, after stream
    identification and finv_category derivation, so it is a purely cosmetic
    rename: nothing downstream may read product/type semantics from the
    stream_id prefix anymore (use product_type / finv_category instead).
    Numbering is global across applications, ordered by each stream's
    earliest transaction date so it does not depend on the raw input row
    order.
    """

    result = output.copy()
    sid_text = result["stream_id"].astype("string")
    has_sid = sid_text.notna() & sid_text.str.strip().ne("")
    if not has_sid.any():
        return result

    sid_clean = sid_text.str.strip()
    dated = pd.to_datetime(result["transaction_date"], errors="coerce")

    # One stable ordering key per stream: earliest transaction date, then
    # application_id, then the old stream_id as a tiebreaker.
    stream_meta = (
        pd.DataFrame(
            {
                "stream_id": sid_clean.loc[has_sid],
                "first_date": dated.loc[has_sid],
                "application_id": result.loc[has_sid, "application_id"],
            }
        )
        .groupby("stream_id", sort=False)
        .agg(
            first_date=("first_date", "min"),
            application_id=("application_id", "first"),
        )
        .reset_index()
        .sort_values(
            ["first_date", "application_id", "stream_id"],
            kind="stable",
            na_position="last",
        )
    )

    mapping = {
        old: f"loan_{index:03d}"
        for index, old in enumerate(stream_meta["stream_id"], start=1)
    }
    result.loc[has_sid, "stream_id"] = sid_clean.loc[has_sid].map(mapping)
    return result


def identify_streams(
    df: pd.DataFrame,
    group_columns: list[str] | None = None,
    reset_stream_ids: bool = True,
) -> pd.DataFrame:
    """Run all product rules in priority order."""

    group_columns = group_columns or DEFAULT_GROUP_COLUMNS
    output = ensure_stream_id_column(df, reset=reset_stream_ids)
    validate_columns(output, group_columns)

    claimed_mask = pd.Series(False, index=output.index, dtype=bool)

    for rule in sorted(PRODUCT_RULES, key=lambda item: item.priority):
        eligible_mask = (
            ~claimed_mask
            & output["product_type"].eq(rule.product_type)
        )

        # Claim all rows that match this product rule, including rows that
        # cannot receive a stream_id because key information is missing.
        claimed_mask |= eligible_mask

        rule.matcher(
            output,
            eligible_mask,
            group_columns,
        )

    counter = _StreamIdCounter(output)
    refine_unknown_personal_loan_streams(output, group_columns, counter)
    apply_special_counterparty_stream_overrides(output, group_columns, counter)
    return renumber_stream_ids_by_application(output)


# ---------------------------------------------------------------------------
# Validation and final labels
# ---------------------------------------------------------------------------


def validate_columns(df: pd.DataFrame, group_columns: list[str]) -> None:
    required_columns = {
        "product_type",
        "transaction_date",
        "amount",
        "dr_cr",
        DISHONOUR_COLUMN,
        "stream_id",
        *group_columns,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )


def add_finv_category(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    product_type = output["product_type"].astype("string").str.strip()
    stream_base = (
        output["stream_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"_\d+$", "", regex=True)
    )

    valid_mask = (
        product_type.notna()
        & product_type.ne("")
        & stream_base.notna()
        & stream_base.ne("")
    )
    existing = output.get("finv_category", pd.Series(index=output.index))
    output["finv_category"] = pd.NA

    # Vectorised finv_category assignment (replaces the original per-row
    # list comprehension that called ``zip`` on Series slices).
    if valid_mask.any():
        special_bases = stream_base.loc[valid_mask].isin({
            "bnpl", "wage_advance", "home_loan", "bank", "loc", "contract_loan", "generic_loan", "car_loan",
        })
        output.loc[valid_mask, "finv_category"] = np.where(
            special_bases,
            stream_base.loc[valid_mask],
            product_type.loc[valid_mask] + "_" + stream_base.loc[valid_mask],
        )

    preserved_mask = (
        output["finv_category"].isna()
        & existing.notna()
        & existing.astype("string").str.strip().ne("")
    )
    output.loc[preserved_mask, "finv_category"] = existing.loc[preserved_mask]
    dishonour_mask = (
        output["is_dishonours"].astype("string").str.lower().eq("yes")
    )
    output.loc[dishonour_mask, "finv_category"] = "Dishonours"
    output["finv_category"] = output["finv_category"].replace(FINV_CATEGORY_MAP)
    return output
