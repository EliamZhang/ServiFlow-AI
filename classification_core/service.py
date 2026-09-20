"""Shared model-service business logic: input JSON dict -> run pipeline -> output JSON dict.

Two input/output contracts are supported and picked from the payload itself
(`product`: "fundo" -> illion v1, "wagego" -> wagego v2; a payload carrying no
registered `product` falls back to key presence):
- "illion" (v1, original): `illion_raw_transactions` + `bank_accounts` keyed by
  `bank_account_id`; built by build_transactions_frame / serialized by serialize_result.
- "wagego" (v2): `raw_transactions` + optional `bank_accounts` keyed by
  (institution, account_number), plus unrelated upstream feature keys and a
  `day_end_balances` array that this pipeline ignores; built by build_wagego_frame /
  serialized by serialize_wagego.

Used by two entry points:
- model_main.py: production inference entry point (PredictMain.predict).
- verify_model.py: local verification script (CLI).

The dispatching entry points are build_input_frame / serialize_output / build_failed_output.
Both adapters and the shared serialization primitives live in this one module on purpose:
the contract layer stays in a single file, so the two versions cannot import each other in
a cycle.

Sections below: constants / ModelService / serialization primitives / illion (v1) contract /
wagego (v2) contract / contract dispatch.

A v2 payload carries no transaction id, no account id, no `credit_limit`, and frequently an
empty `applicationId` (the real identifier is the composite `flowId`).  build_wagego_frame
synthesizes the internal columns the engines require:

    application_id   = applicationId when non-blank, else flowId
    transaction_id   = 1-based row index of raw_transactions
    bank_account_id  = f"{institution}-{number}", the row's own when it names one, else
                       the matched account's
    bank             = institution as above (v2's name for v1's bank)
    account_type     = looked up from payload["bank_accounts"], "" when unknown
    credit_limit     = looked up from payload["bank_accounts"], None when unknown

All six synthetic columns are internal only and never reach the output: a v2 row carries
exactly the upstream transaction fields plus the business results (`bscat` /
`counterparty` / `stream_id`), and the summaries drop the account key entirely.
Upstream transaction fields are echoed through as sent (input key order) -- an empty string
stays `""`, a null stays `null`, an array stays an array -- while the columns this pipeline
produces keep normalizing their empty values to null; so `institution` /
`bank_account_number` / `secondary_category` survive to the output automatically.
Synthesized column names deliberately overwrite same-named upstream transaction keys when
the payload happens to carry them (ids and account metadata must be ours, not theirs).

A row is related to its account by its own account number, spelled `bank_account_number` in
the production shape and `account_number` in the payroll sample shape; when the row carries
no `institution` of its own, the matched `bank_accounts` entry supplies it (see
_wagego_row_account).  Rows therefore always resolve their `bank` -- which liability's
counterparty rules are masked on -- from either side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
    load_category_owners,
    load_pipeline_config,
)
from .models import (
    ClassificationRunResult,
    TRANSACTION_KEY_COLUMNS,
)
from .orchestrator import ClassificationOrchestrator

# ── constants ───────────────────────────────────────────────────────────────

# Internal orchestrator output columns that are not business results; excluded when serializing
_INTERNAL_OUTPUT_COLUMNS = frozenset(
    {
        "classification_status",
        "classification_engine",
        "classification_engine_version",
        "classification_priority",
        "classification_rule_id",
        "classification_reason",
    }
)

# Account metadata stays at the top-level bank_accounts only; not repeated per row
_ACCOUNT_METADATA_COLUMNS = frozenset({"account_type", "bank", "credit_limit"})

# Defaults for the account-metadata columns when the payload's bank_accounts does not cover
# a row (or is absent / not a list).  The columns must exist even then: the liability
# counterparty rules read `bank` unguarded, and its summary layer reads `account_type` /
# `credit_limit`.  A blank `bank` matches no rule -- credit_card_rules.csv gives every row
# either a concrete bank or the "*" skip sentinel, never a blank -- so an unknown account
# only degrades bank-aware matching.  Key order is the frame's column order, which is the
# order a row is written to JSON in.
_ACCOUNT_METADATA_DEFAULTS: dict[str, Any] = {
    "account_type": "",
    "bank": "",
    "credit_limit": None,
}

# Top-level input fields echoed back verbatim: the output key keeps the payload's own
# spelling (userId / applicationId / flowTime), so a caller reads back exactly what it
# sent.  Values are carried through unchanged.
_INPUT_ECHO_TOP_KEYS = ("userId", "applicationId", "flowTime")

FORMAT_ILLION = "illion"
FORMAT_WAGEGO = "wagego"

# Business product -> payload contract shape.  A payload's `product` field picks the
# chain; a payload without it (or with a blank value) falls back to key-presence
# detection, so callers that predate `product` keep working.  An unregistered
# `product` value is rejected by build_input_frame.  Registering another product is
# one line here.
PRODUCT_FUNDO = "fundo"
PRODUCT_WAGEGO = "wagego"
PRODUCT_FORMATS = {
    PRODUCT_FUNDO: FORMAT_ILLION,
    PRODUCT_WAGEGO: FORMAT_WAGEGO,
}
# Contract shape -> product reported in bscat_stats when the payload omits `product`.
_FORMAT_PRODUCTS = {format_: product for product, format_ in PRODUCT_FORMATS.items()}

# v2 account metadata keys echoed at the top level (v1-style whitelist depth), in the
# official sample's own order: an entry the payload sent without `bank` echoes it as null,
# exactly like any other key of the list.  application_id is resolved (applicationId or
# flowId) and flow_id echoed after it, so serialize_wagego assembles its top-level keys
# explicitly instead of echoing a map.
_WAGEGO_ACCOUNT_KEYS = (
    "bsb",
    "account_number",
    "bank",
    "institution",
    "account_type",
    "account_holder",
    "account_holder_type",
    "account_name",
)

# v2 rows carry the upstream fields plus the business results only: the resolved
# application id, the synthesized transaction id (an internal key the orchestrator's
# uniqueness check needs) and the synthetic account key all stay out, and
# `bank` / `account_type` / `credit_limit` are already covered by
# _ACCOUNT_METADATA_COLUMNS.
_WAGEGO_ROW_EXCLUDE = (
    _INTERNAL_OUTPUT_COLUMNS
    | _ACCOUNT_METADATA_COLUMNS
    | frozenset({"application_id", "bank_account_id", "transaction_id"})
)

# v1 rows keep `bank_account_id` (their only account reference) but not the resolved
# application id -- the top level already carries it.
_ILLION_ROW_EXCLUDE = (
    _INTERNAL_OUTPUT_COLUMNS
    | _ACCOUNT_METADATA_COLUMNS
    | frozenset({"application_id"})
)

# Stream summaries (income / liability) drop the account identity, the coarse category
# and the counterparty -- all three are already on the transaction rows.
_STREAM_SUMMARY_EXCLUDE = _ACCOUNT_METADATA_COLUMNS | frozenset(
    {"bscat", "bank_account_id", "application_id", "counterparty"}
)
# Per-summary output columns.  category_summary groups by `bscat`, so it keeps that one
# and drops only the account id.  Summaries not listed here fall back to the
# account-metadata exclusion alone.
_SUMMARY_EXCLUDE_COLUMNS: dict[str, frozenset[str]] = {
    "income_summary": _STREAM_SUMMARY_EXCLUDE,
    "liability_summary": _STREAM_SUMMARY_EXCLUDE,
    "category_summary": _ACCOUNT_METADATA_COLUMNS | frozenset({"bank_account_id"}),
}

# Columns that engine code reads without guarding for their absence (liability's
# counterparty rules and special rules read `text` / `dr_cr`, its stream builder and
# summary read `amount`), so each builder must carry them even when the payload's
# transaction objects omit them -- otherwise the frame is missing the column and an
# engine raises the same KeyError that a missing `bank_accounts` used to cause.
# Account metadata is per-row and covered by _ACCOUNT_METADATA_DEFAULTS.
# `amount` defaults to NaN, not None: the rule engine compares it with `.gt()`, and a
# None in an object column makes that comparison raise instead of being False.
_MANDATORY_DEFAULTS: dict[str, Any] = {
    "text": "",
    "dr_cr": "",
    "amount": float("nan"),
    "transaction_date": None,
}

# v1 relates rows to accounts with this key, and liability's per-account summary reads it
# unguarded, so the column has to exist even for a payload that ships no account ids.
_ILLION_MANDATORY_DEFAULTS: dict[str, Any] = {"bank_account_id": None}


class ModelService:
    """Model service entry point: loads configuration once, runs inference per single-application input dict."""

    def __init__(
        self,
        pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
        category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
    ) -> None:
        self.orchestrator = ClassificationOrchestrator(
            config=load_pipeline_config(pipeline_config_path),
            category_owners=load_category_owners(category_catalog_path),
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the pipeline on a single application's input dict; return a JSON-serializable result dict.

        The input contract version (illion / wagego) is detected from the payload.
        A payload the contract layer rejects returns a format-matching failed result;
        a pipeline / engine / serialization failure does too, so a malformed batch is
        answered with a structured result instead of an HTTP 500.  The traceback of the
        latter goes to the log -- the failed envelope only carries the message.
        """
        if not isinstance(payload, dict):
            # A body that is neither contract (a JSON array / string / number) has no
            # identifiers to echo and no shape to match; answering it here keeps the
            # failure path below -- which reads the payload -- from raising itself.
            return build_failed_output({}, "Input payload must be a JSON object.")
        try:
            transactions = build_input_frame(payload)
        except Exception as exc:
            # Expected rejections (wrong shape, bad keys): no traceback worth logging.
            return build_failed_output(payload, str(exc))
        try:
            result = self.orchestrator.run(transactions)
            return serialize_output(result, payload)
        except Exception as exc:
            # An engine or serializer bug used to escape as a 500 *after* the run had
            # done its work; the caller could not tell it apart from a dead service.
            logging.exception("classification run failed")
            return build_failed_output(payload, str(exc))


# ── serialization primitives (shared by both contracts) ─────────────────────
# Record keys are the frame column names verbatim (snake_case) so the JSON output
# field names line up with the Excel report columns written by
# classification_core/reporting.write_report.


def _serialize_value(value: Any, keep_empty_string: bool = False) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return str(value)
    if not pd.api.types.is_scalar(value):
        # A payload field holding a JSON array ("tags": ["a", "b"]) used to reach
        # pd.isna() below and raise -- killing the run *after* a successful
        # classification.  Array-valued fields are echoed as arrays.
        return value.tolist() if hasattr(value, "tolist") else value
    if pd.isna(value):
        return None
    # Engine summary layers turn empty values into empty strings via normalize_text
    # (e.g. credit_limit); serialize those as null.  A field the payload sent itself is
    # echoed as sent instead -- see _payload_row_columns / serialize_wagego.
    if isinstance(value, str) and value == "":
        return "" if keep_empty_string else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (pd.Timedelta, type(pd.NaT))):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def _serialize_records(
    frame: pd.DataFrame,
    exclude: frozenset[str] = frozenset(),
    keep_empty_columns: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """One dict per row, in the frame's column order.

    A column named in `keep_empty_columns` is a field the payload sent itself, so its
    values are echoed as sent and an empty string stays `""` instead of becoming null
    (see _serialize_value); every other column is pipeline output and normalizes as before.
    """
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record: dict[str, Any] = {}
        for column in frame.columns:
            if column in exclude:
                continue
            record[column] = _serialize_value(
                row[column], keep_empty_string=column in keep_empty_columns
            )
        records.append(record)
    return records


def _summary_exclude(name: str) -> frozenset[str]:
    """Output columns to drop for one summary artifact (see _SUMMARY_EXCLUDE_COLUMNS)."""
    return _SUMMARY_EXCLUDE_COLUMNS.get(name, _ACCOUNT_METADATA_COLUMNS)


def _payload_row_columns(payload: dict) -> frozenset[str]:
    """Field names the payload's own `raw_transactions` objects carry.

    Those columns are upstream data echoed back, so serialize_wagego hands them to
    _serialize_records as `keep_empty_columns`: their empty strings stay empty strings.
    A column no transaction object carries is either synthesized by the builder or written
    by an engine, and keeps the pipeline's own normalization (empty string -> null).
    """
    transactions = payload.get("raw_transactions")
    if not isinstance(transactions, list):
        return frozenset()
    columns: set[str] = set()
    for transaction in transactions:
        if isinstance(transaction, dict):
            columns.update(transaction.keys())
    return frozenset(columns)


def _max_date(values: list[Any]) -> str | None:
    """Latest of the payload's dates; anything that is not a date string is ignored.

    Values are echoed straight from the payload, so a batch with one null / numeric
    `transaction_date` used to raise here (str vs float comparison) -- after the whole
    classification had already run.
    """
    dates = [
        value.strip() for value in values if isinstance(value, str) and value.strip()
    ]
    return max(dates) if dates else None


def build_stats(
    transactions: pd.DataFrame,
    product: str,
) -> dict[str, Any]:
    return {
        "txn_raw_input_cnt": len(transactions),
        "transaction_date_max": _max_date(
            transactions["transaction_date"].tolist()
        ),
        "product": product,
    }


def product_name(payload: dict) -> str:
    """Product reported in `bscat_stats`: the payload's `product` verbatim when it
    is a non-blank string, otherwise the resolved contract's default product."""
    product = payload.get("product")
    if isinstance(product, str) and product.strip():
        return product.strip()
    return _FORMAT_PRODUCTS[detect_input_format(payload)]


def unsupported_product(payload: dict) -> str | None:
    """The payload's `product` value when it is not a registered one, else None.

    Absent / blank / non-string `product` is not unsupported: those payloads fall
    back to key-presence detection.
    """
    product = payload.get("product")
    if isinstance(product, str) and product.strip():
        if product.strip().lower() not in PRODUCT_FORMATS:
            return product.strip()
    return None


# ── build-time normalizers (shared by both contracts) ───────────────────────

def _as_text(value: Any) -> str:
    """One payload value as the string the engine text matching assumes it is.

    Presence is not the whole story: `text` is read by Series.str.upper(), which
    maps a non-string to NaN -- and NaN is truthy, so the engines' `if text:`
    guards let it through to len(), killing the whole batch.  A numeric /
    boolean `text` becomes its string form; an array or object `text` becomes
    "" rather than its repr, which would inject payload contents into the
    keyword matching.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not pd.api.types.is_scalar(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _to_text_column(series: pd.Series) -> pd.Series:
    """The `text` column normalized to strings (see _as_text).

    A column that is already all strings is returned untouched, so healthy
    payloads keep their values -- and their dtype -- byte for byte.
    """
    if all(isinstance(value, str) for value in series):
        return series
    return series.map(_as_text)


def _to_scalar_column(series: pd.Series, default: Any) -> pd.Series:
    """A payload column reduced to scalars: a JSON array / object becomes `default`.

    Engine code reads these columns one value at a time (`pd.isna(value)`, `if
    value:`), and both idioms raise on an array -- a single array-valued
    `bank_account_id` also breaks the orchestrator's key uniqueness check with
    "unhashable type".  Such a value names nothing, so it takes the same
    missing-value default as an absent column.  All-scalar columns come back
    untouched (healthy payloads keep their values and dtype).
    """
    if all(pd.api.types.is_scalar(value) for value in series):
        return series
    return series.map(
        lambda value: value if pd.api.types.is_scalar(value) else default
    )


# ── illion (v1) contract ────────────────────────────────────────────────────

def build_transactions_frame(payload: dict) -> pd.DataFrame:
    transactions = payload.get("illion_raw_transactions")
    if not isinstance(transactions, list):
        raise ValueError(
            "Input JSON must contain an 'illion_raw_transactions' list."
        )
    if not transactions:
        # Zero-transaction input (empty bank card / statement arrays) is legal:
        # return an empty frame that still carries the key columns and
        # transaction_date so the orchestrator short-circuit and the stats
        # builder can produce a structurally correct empty success result.
        return pd.DataFrame(
            {
                column: pd.Series(dtype="object")
                for column in (*TRANSACTION_KEY_COLUMNS, "transaction_date")
            }
        )
    frame = pd.DataFrame(transactions)
    application_id = payload.get("applicationId")
    if application_id is not None and "application_id" not in frame.columns:
        frame["application_id"] = application_id
    # An array / object in one of these columns is not a value any engine can read (see
    # _to_scalar_column), so it takes the column's missing-value default; `text` gets its
    # own normalization instead -- a number there is classified as its text form.
    for column, default in {**_MANDATORY_DEFAULTS, **_ILLION_MANDATORY_DEFAULTS}.items():
        if column not in frame.columns:
            frame[column] = default
        elif column != "text":
            frame[column] = _to_scalar_column(frame[column], default)
    frame["text"] = _to_text_column(frame["text"])

    # Account metadata is looked up per row, never merged.  A frame merge fans rows out
    # when `bank_accounts` repeats a `bank_account_id` -- breaking the
    # (application_id, transaction_id) uniqueness the orchestrator enforces -- and it
    # leaves the three columns missing entirely when `bank_accounts` is absent, which the
    # liability counterparty rules then read unguarded.  Rows whose account is unknown
    # fall back to the defaults.  Rows and accounts may disagree on the JSON type of the
    # same id (5116090 vs "5116090"), so both sides go through _illion_account_id.
    accounts = _account_dicts_by_key(payload, _illion_account_key)
    account_ids = [_illion_account_id(value) for value in frame["bank_account_id"]]
    for column, default in _ACCOUNT_METADATA_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = [
                accounts.get(account_id, {}).get(column, default)
                for account_id in account_ids
            ]
    return frame


def _illion_account_id(value: Any) -> str:
    """Canonical v1 account key: the int 5116090, the float 5116090.0 and the string
    "5116090" all name one account, and a missing id becomes "" (matching no entry)."""
    return normalize_account_number(value)


def _illion_account_key(account: dict) -> str:
    """Lookup key of a `bank_accounts` entry (see _illion_account_id)."""
    return _illion_account_id(account.get("bank_account_id"))


def _wagego_account_key(account: dict) -> str:
    # The account list names the number `account_number`, the transaction rows name it
    # `bank_account_number`; both sides carry the same value, so the helpers differ.
    return account_key(account.get("institution"), account.get("account_number"))


def _wagego_account_number_key(account: dict) -> str:
    """Bare-number lookup key of a `bank_accounts` entry (see _wagego_row_account)."""
    return normalize_account_number(account.get("account_number"))


def _wagego_row_account_number(row: dict) -> str:
    """A v2 transaction row's own account number, normalized.

    The production shape names it `bank_account_number`, the payroll sample shape
    `account_number`; a row carrying neither (or a non-scalar) names no account.
    """
    for column in ("bank_account_number", "account_number"):
        value = row.get(column)
        if not pd.api.types.is_scalar(value):
            continue  # an array / object names nothing (see _to_scalar_column)
        number = normalize_account_number(value)
        if number:
            return number
    return ""


def _wagego_row_account(
    row: dict,
    accounts: dict[str, dict],
    accounts_by_number: dict[str, dict],
) -> tuple[str, str, dict]:
    """(internal account key, bank, matched account) for one v2 transaction row.

    A row that carries an `institution` of its own is matched on institution + number only,
    so two banks sharing an account number never borrow each other's metadata.  A row that
    carries none (the payroll sample shape) is matched on its number alone and then takes
    the institution from the account it matched -- `institution`, falling back to the
    entry's `bank` -- because the `bank` column built from it is what liability's
    counterparty rules mask on.  No match leaves both halves empty and the metadata
    defaults in place.
    """
    number = _wagego_row_account_number(row)
    institution = normalize_institution(row.get("institution"))
    account = accounts.get(account_key(institution, number))
    if account is None and not institution and number:
        account = accounts_by_number.get(number)
        if account is not None:
            institution = normalize_institution(
                account.get("institution")
            ) or normalize_institution(account.get("bank"))
    return account_key(institution, number), institution, account or {}


def _account_dicts_by_key(payload: dict, key_of) -> dict[Any, dict]:
    """`key_of(account)` -> account dict, from the payload's `bank_accounts`.

    The one place a payload's account list is turned into a lookup table, shared by both
    contracts' frame builders and by the `bank_accounts` echo below, so the two can never
    disagree about which entry is authoritative.  A repeated key collapses to a single
    entry -- the **last** one wins (a later entry is a correction of an earlier one),
    sitting at the position of its first appearance.  Non-dict entries and entries whose
    key is missing / NaN / blank are skipped: no transaction row can reference them.
    """
    accounts = payload.get("bank_accounts")
    mapping: dict[Any, dict] = {}
    if not isinstance(accounts, list):
        return mapping
    for account in accounts:
        if not isinstance(account, dict):
            continue
        key = key_of(account)
        if key is None or key != key or key == "":  # no id -> matches no row
            continue
        mapping[key] = account
    return mapping


def build_bank_accounts(payload: dict) -> list[dict[str, Any]]:
    """Echo the payload's bank_accounts: one entry per account id (deduplicated and
    resolved exactly like the frame -- see _account_dicts_by_key).

    The echoed `bank_account_id` is the entry's own value, not the canonicalized
    lookup key, so an id the payload sent as a number still echoes as a number.
    """
    return [
        {
            "bank_account_id": account.get("bank_account_id"),
            "account_number": account.get("account_number"),
            "account_type": account.get("account_type"),
            "bank": account.get("bank"),
        }
        for account in _account_dicts_by_key(payload, _illion_account_key).values()
    ]


def serialize_result(
    result: ClassificationRunResult,
    payload: dict,
) -> dict:
    output: dict[str, Any] = {}
    for input_key in _INPUT_ECHO_TOP_KEYS:
        if input_key in payload:
            output[input_key] = payload[input_key]
    output["bscat_stats"] = build_stats(result.transactions, product_name(payload))
    output["bank_accounts"] = build_bank_accounts(payload)
    output["bscat_transactions"] = _serialize_records(
        result.transactions,
        exclude=_ILLION_ROW_EXCLUDE,
    )

    summaries: dict[str, list[dict[str, Any]]] = {}
    for artifact in result.summaries:
        summaries[artifact.name] = _serialize_records(
            artifact.data,
            exclude=_summary_exclude(artifact.name),
        )
    output["bscat_summaries"] = summaries
    return output


def _build_error_output(payload: dict, error: str) -> dict:
    output: dict[str, Any] = {
        "status": "failed",
        "error": error,
        "bscat_stats": {
            "txn_raw_input_cnt": 0,
            "transaction_date_max": None,
            "product": product_name(payload),
        },
    }
    for input_key in _INPUT_ECHO_TOP_KEYS:
        if input_key in payload:
            output[input_key] = payload[input_key]
    output["bscat_transactions"] = []
    output["bscat_summaries"] = {}
    return output


# ── wagego (v2) contract ────────────────────────────────────────────────────

def detect_input_format(payload: dict) -> str:
    """Return the payload's contract version.

    A registered `product` value wins.  Otherwise key presence decides,
    `illion_raw_transactions` winning when both keys exist (any payload that is
    v1 today must stay v1) and payloads with neither key routed to v1 so their
    error message stays unchanged.  An unregistered `product` does not change
    detection here either, so its failed output still gets the closest shape --
    build_input_frame is what rejects it.
    """
    if isinstance(payload, dict):
        product = payload.get("product")
        if isinstance(product, str) and product.strip():
            format_ = PRODUCT_FORMATS.get(product.strip().lower())
            if format_ is not None:
                return format_
        if "illion_raw_transactions" in payload:
            return FORMAT_ILLION
        if "raw_transactions" in payload:
            return FORMAT_WAGEGO
    return FORMAT_ILLION


def application_id_from_payload(payload: dict) -> Any | None:
    """applicationId when non-blank, else flowId; None when neither is usable."""
    for input_key in ("applicationId", "flowId"):
        value = payload.get(input_key)
        if isinstance(value, str):
            value = value.strip()
        if value is None or value == "":
            continue
        return value
    return None


def _clean_text(value: Any) -> str:
    """str + strip that never yields the literal 'nan' for missing values.

    Avoids Series.astype(str), whose NaN -> 'nan' semantics differ between
    pandas 1.3.5 (production) and 3.x (local).
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def normalize_institution(value: Any) -> str:
    return _clean_text(value).lower()


def normalize_account_number(value: Any) -> str:
    """Account numbers are opaque text: never numeric-coerce, never drop leading zeros.

    JSON payloads occasionally carry them as numbers (12345678) or floats
    (12345678.0); those render without a decimal part, everything else verbatim.
    """
    if isinstance(value, bool):
        return _clean_text(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    return _clean_text(value)


def account_key(institution: Any, account_number: Any) -> str:
    """Synthetic internal account id: normalized "institution-account_number".

    A constant institution prefix keeps the internal sort order identical to
    sorting account number strings (liability numbers its streams by that order).
    Falls back to whichever half exists when the other is missing.
    """
    normalized_institution = normalize_institution(institution)
    normalized_account = normalize_account_number(account_number)
    if normalized_institution and normalized_account:
        return f"{normalized_institution}-{normalized_account}"
    return normalized_account or normalized_institution


def build_wagego_frame(payload: dict) -> pd.DataFrame:
    """v2 input dict -> internal transactions frame (see module docstring)."""
    transactions = payload.get("raw_transactions")
    if not isinstance(transactions, list):
        raise ValueError("Input JSON must contain a 'raw_transactions' list.")
    if not transactions:
        # Zero-transaction input is legal; mirror the v1 empty-frame shape so the
        # orchestrator short-circuit and the stats builder work unchanged.
        return pd.DataFrame(
            {
                column: pd.Series(dtype="object")
                for column in (*TRANSACTION_KEY_COLUMNS, "transaction_date")
            }
        )
    rows: list[dict] = []
    for transaction in transactions:
        if not isinstance(transaction, dict):
            raise ValueError("Each 'raw_transactions' item must be a JSON object.")
        rows.append(dict(transaction))
    application_id = application_id_from_payload(payload)
    if application_id is None:
        raise ValueError(
            "Input JSON must contain a non-empty 'applicationId' or 'flowId'."
        )

    accounts = _account_dicts_by_key(payload, _wagego_account_key)
    # Second view of the same account list, keyed by the bare number: rows of the payroll
    # shape name no institution, so only the number can find their account.
    accounts_by_number = _account_dicts_by_key(payload, _wagego_account_number_key)
    resolved = [
        _wagego_row_account(row, accounts, accounts_by_number) for row in rows
    ]
    frame = pd.DataFrame(rows)
    # Appended after the upstream fields, so output rows keep the payload's field order.
    frame["transaction_id"] = list(range(1, len(rows) + 1))
    frame["application_id"] = application_id
    frame["bank_account_id"] = [key for key, _, _ in resolved]
    frame["bank"] = [institution for _, institution, _ in resolved]
    # Dictionary lookup, not a frame merge: duplicate accounts in `bank_accounts`
    # would fan rows out and break the (application_id, transaction_id) uniqueness
    # check; unknown accounts simply fall back to the defaults.
    frame["account_type"] = [
        _clean_text(account.get("account_type")) for _, _, account in resolved
    ]
    frame["credit_limit"] = [
        account.get("credit_limit") for _, _, account in resolved
    ]
    # An array / object in one of these columns takes the missing-value default (see
    # _to_scalar_column); `text` is normalized to its text form instead.
    for column, default in _MANDATORY_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
        elif column != "text":
            frame[column] = _to_scalar_column(frame[column], default)
    frame["text"] = _to_text_column(frame["text"])
    return frame


def build_wagego_bank_accounts(payload: dict) -> list[dict[str, Any]]:
    """Echo the payload's bank_accounts in their own shape: one entry per account
    (deduplicated and resolved exactly like the frame -- see _account_dicts_by_key),
    each carrying _WAGEGO_ACCOUNT_KEYS in that key order, `bank` included, with null for
    a key the payload's entry did not send."""
    return [
        {key: account.get(key) for key in _WAGEGO_ACCOUNT_KEYS}
        for account in _account_dicts_by_key(payload, _wagego_account_key).values()
    ]


def serialize_wagego(result: ClassificationRunResult, payload: dict) -> dict:
    """Run result -> v2 output dict: v1's depth (bscat_transactions / bscat_summaries / bscat_stats), v2's wording.

    A row is the payload's own transaction fields in the payload's own order, then the
    business results; the payload's values are echoed as sent, so an upstream empty string
    stays `""` (only fields this pipeline produced serialize their empty strings as null).
    """
    output: dict[str, Any] = {}
    if "userId" in payload:
        output["userId"] = payload["userId"]
    application_id = application_id_from_payload(payload)
    if application_id is not None:
        output["applicationId"] = application_id
    flow_id = payload.get("flowId")
    if isinstance(flow_id, str) and flow_id.strip():
        output["flow_id"] = flow_id
    if "flowTime" in payload:
        output["flowTime"] = payload["flowTime"]
    output["bscat_stats"] = build_stats(result.transactions, product_name(payload))
    output["bank_accounts"] = build_wagego_bank_accounts(payload)
    output["bscat_transactions"] = _serialize_records(
        result.transactions,
        exclude=_WAGEGO_ROW_EXCLUDE,
        # The payload's own fields are echoed as sent (empty string included); the
        # business columns keep normalizing their empty values to null.
        keep_empty_columns=_payload_row_columns(payload),
    )

    summaries: dict[str, list[dict[str, Any]]] = {}
    for artifact in result.summaries:
        summaries[artifact.name] = _serialize_records(
            artifact.data,
            exclude=_summary_exclude(artifact.name),
        )
    output["bscat_summaries"] = summaries
    return output


def build_wagego_error_output(payload: dict, error: str) -> dict:
    """v2 failed output: v1's failed shape (no bank_accounts key), v2's echo keys."""
    output: dict[str, Any] = {
        "status": "failed",
        "error": error,
        "bscat_stats": {
            "txn_raw_input_cnt": 0,
            "transaction_date_max": None,
            "product": product_name(payload),
        },
    }
    if "userId" in payload:
        output["userId"] = payload["userId"]
    application_id = application_id_from_payload(payload)
    if application_id is not None:
        output["applicationId"] = application_id
    flow_id = payload.get("flowId")
    if isinstance(flow_id, str) and flow_id.strip():
        output["flow_id"] = flow_id
    if "flowTime" in payload:
        output["flowTime"] = payload["flowTime"]
    output["bscat_transactions"] = []
    output["bscat_summaries"] = {}
    return output


# ── contract dispatch (fundo v1 / wagego v2) ────────────────────────────────
# Dispatch rule: the payload's `product` when it names a registered product;
# otherwise key presence, with `illion_raw_transactions` winning when both keys
# exist (a payload that is v1 today must stay v1) and undetectable payloads
# routed to v1 so their error message is unchanged.


def build_input_frame(payload: dict) -> pd.DataFrame:
    """Build the internal transactions frame from either supported input contract."""
    product = unsupported_product(payload)
    if product is not None:
        supported = ", ".join(sorted(PRODUCT_FORMATS))
        raise ValueError(
            f"Unsupported 'product' value {product!r}; supported products: {supported}."
        )
    if detect_input_format(payload) == FORMAT_WAGEGO:
        return build_wagego_frame(payload)
    return build_transactions_frame(payload)


def serialize_output(result: ClassificationRunResult, payload: dict) -> dict:
    """Serialize a run result into the output contract matching the payload's version."""
    if detect_input_format(payload) == FORMAT_WAGEGO:
        return serialize_wagego(result, payload)
    return serialize_result(result, payload)


def build_failed_output(payload: dict, error: str) -> dict:
    """Build the failed-inference output matching the payload's version."""
    if detect_input_format(payload) == FORMAT_WAGEGO:
        return build_wagego_error_output(payload, error)
    return _build_error_output(payload, error)
