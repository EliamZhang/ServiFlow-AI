"""Regression check for the wagego (v2) input/output contract on the tracked anonymous sample.

Usage: python check_wagego_sample.py   (run from the repository root)

Exit code 0 when every invariant holds, 1 otherwise (each mismatch is printed).

wagego_sample.json is an anonymized 58-row subset of a real online payload (built by
output/_make_wagego_sample.py, which is not tracked).  It carries the v2 contract shape
(raw_transactions + bank_accounts keyed by institution/account_number, no transaction or
account ids, empty applicationId) and is the handle for "did the v2 adapter or v2
serialization change?".  The counts below are a snapshot of THIS 58-row subset, not
production numbers.

Both contracts share one output shape: the three echoed input identifiers under their own
(userId / applicationId / flowTime), `bscat_stats` / `bank_accounts` / `bscat_transactions`
/ `bscat_summaries`, and no run envelope on success -- `status` / `error` come back only on
a failed run.  It also pins the `bank_accounts` handling: the echo carries the contract's
own key list (`bank` included, null when the entry sent none), one entry per account
(a duplicated account collapses to its last entry, an entry with no id is dropped) and a
duplicated account must not fan transaction rows out.

A v2 row is the payload's own fields in the payload's own order, echoed as sent (an empty
string stays `""`), then the business results -- none of the synthesized columns
(`transaction_id` / `bank_account_id` / `bank` / `account_type` / `credit_limit`) appears.
A row names its account `bank_account_number`, optionally with a per-row `institution`; a row
carrying no `institution` must still resolve its entry (and so its `bank`) by number alone.
`account_number` is the account *list*'s own key, not a row's: a row spelling it that way
names no account and the field rides out as ordinary upstream data.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from classification_core.service import ModelService, build_wagego_frame

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_ROOT / "wagego_sample.json"

# The sample carries no `product`, so the run falls back to key-presence detection
# and reports the resolved chain's product name.
EXPECTED_STATS = {
    "txn_raw_input_cnt": 58,
    "transaction_date_max": "2026-09-15",
    "product": "wagego",
}

# Row keys = the payload's own field order, then the business results.
# The payload's values are echoed as sent (`""` included -- see the empty-string check
# below); the synthesized transaction_id / bank_account_id / bank / account_type /
# credit_limit never reach a row.
EXPECTED_ROW_KEYS = [
    "secondary_category",
    "transaction_date",
    "institution",
    "amount",
    "bank_account_number",
    "balance",
    "dr_cr",
    "trx_type",
    "text",
    "category",
    "third_party",
    "counterparty",
    "bscat",
    "stream_id",
]

# v2 rows express the account with the input's own vocabulary and never expose the
# synthetic internal columns.
FORBIDDEN_ROW_KEYS = (
    "bank_account_id",
    "application_id",
    "transaction_id",
    "bank",
    "account_type",
    "credit_limit",
)

# A successful run returns only the result and the echoed input identifiers -- no
# run envelope (that is reserved for failed runs).
FORBIDDEN_SUCCESS_KEYS = ("run_id", "status", "error")

EXPECTED_RUN_KEYS = (
    "income_summary",
    "liability_summary",
    "category_summary",
)
# Same column sets as the v1 contract's summaries: neither summary carries the account
# identity, and the two stream summaries carry neither the coarse category nor the
# counterparty (both are already on the transaction rows).
EXPECTED_SUMMARY_KEYS = {
    "income_summary": [
        "stream_id", "income_category", "transaction_start_date",
        "transaction_end_date", "status", "transaction_count",
        "total_income_amount", "average_income_amount", "median_income_amount",
        "latest_income_amount", "estimated_monthly_income", "frequency",
        "frequency_day", "predicted_next_income_date",
    ],
    "liability_summary": [
        "stream_id", "liability_category", "transaction_start_date",
        "transaction_end_date", "status", "funded_amount", "repaid_amount",
        "repayment_amount", "recent_fn_repay_amount", "frequency",
        "frequency_day", "predicted_closing_date",
    ],
    "category_summary": [
        "bscat", "transaction_start_date", "transaction_end_date",
        "transaction_count", "total_amount", "average_amount", "median_amount",
        "latest_amount",
    ],
}
FORBIDDEN_SUMMARY_KEYS = (
    "bank_account_id", "application_id", "institution", "bank_account_number",
    "account_type", "bank", "credit_limit",
)
# `bscat` stays in category_summary, which groups by it.
FORBIDDEN_STREAM_SUMMARY_KEYS = ("bscat", "counterparty")
EXPECTED_SUMMARY_COUNTS = {
    "income_summary": 2,
    "liability_summary": 11,
    "category_summary": 37,
}

EXPECTED_CATEGORY_COUNTS = {
    "All Other Credits": 2,
    "Automotive": 2,
    "Centrelink": 2,
    "Credit Card Repayments": 2,
    "Department Stores": 2,
    "Dining Out": 2,
    "Dishonours": 2,
    "Donations": 2,
    "Education": 2,
    "Entertainment": 2,
    "External Transfers": 2,
    "Fees": 2,
    "Gambling": 2,
    "Groceries": 2,
    "Gyms and other memberships": 2,
    "Health": 2,
    "Home Improvement": 1,
    "Insurance": 1,
    "Non SACC Loans": 2,
    "Personal Care": 2,
    "Pet Care": 1,
    "Rent": 2,
    "Retail": 2,
    "SACC Loans": 1,
    "Subscription TV": 2,
    "Telecommunications": 2,
    "Transport": 2,
    "Unknown Loans": 2,
    "Utilities": 2,
    "Wages": 2,
    None: 2,
}


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, label: str, condition: bool, detail: Any = "") -> None:
        print(f"{'PASS' if condition else 'FAIL'} | {label}" + (f" | {detail}" if detail != "" else ""))
        if not condition:
            self.failures.append(label)


def main() -> int:
    payload = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    output = ModelService().predict(payload)
    checker = Checker()

    # ── run level ───────────────────────────────────────────────────────────
    leaked_top = [key for key in FORBIDDEN_SUCCESS_KEYS if key in output]
    checker.check("success output carries no run envelope", not leaked_top, leaked_top)
    checker.check("bscat_stats unchanged", output.get("bscat_stats") == EXPECTED_STATS,
                  output.get("bscat_stats"))
    checker.check("applicationId falls back to flowId",
                  output.get("applicationId") == payload["flowId"],
                  output.get("applicationId"))
    checker.check("flow_id echoed", output.get("flow_id") == payload["flowId"])
    checker.check("userId echoed", output.get("userId") == payload["userId"])
    checker.check("flowTime echoed",
                  output.get("flowTime") == payload["flowTime"])

    # ── bank_accounts echo ──────────────────────────────────────────────────
    # The echo is the contract's fixed key list (the official sample's own), not the
    # entry's: an account that sent no `bank` echoes it as null.
    account_keys = ("bsb", "account_number", "bank", "institution", "account_type",
                    "account_holder", "account_holder_type", "account_name")
    echoed = output.get("bank_accounts", [])
    checker.check("bank_accounts echoed one entry per account",
                  len(echoed) == len(payload["bank_accounts"]),
                  f"{len(echoed)} vs {len(payload['bank_accounts'])}")
    checker.check("bank_accounts keep the contract's 8 keys",
                  all(list(account) == list(account_keys) for account in echoed))
    checker.check("bank_accounts values unchanged",
                  all(account == {key: source.get(key) for key in account_keys}
                      for account, source in zip(echoed, payload["bank_accounts"])))
    with_bank = ModelService().predict({
        **payload,
        "bank_accounts": [dict(payload["bank_accounts"][0], bank="cba"),
                          *payload["bank_accounts"][1:]],
    })
    checker.check("a `bank` on the account is echoed",
                  with_bank.get("bank_accounts", [{}])[0].get("bank") == "cba",
                  with_bank.get("bank_accounts", [{}])[0].get("bank"))

    # A duplicated account must not fan transaction rows out: a repeated account_number
    # used to duplicate rows in the frame, which broke the orchestrator's
    # (application_id, transaction_id) uniqueness check and failed the whole batch.
    # `account_name` is echo-only, so a corrected duplicate must not move the result.
    duplicated = [*payload["bank_accounts"],
                  dict(payload["bank_accounts"][0], account_name="Correction")]
    duped = ModelService().predict({**payload, "bank_accounts": duplicated})
    duped_accounts = duped.get("bank_accounts", [])
    checker.check("a duplicated account does not fan rows out",
                  duped.get("status") != "failed"
                  and len(duped.get("bscat_transactions", [])) == len(payload["raw_transactions"]),
                  duped.get("error") or len(duped.get("bscat_transactions", [])))
    checker.check("a duplicated account collapses to one echoed entry",
                  len(duped_accounts) == len(payload["bank_accounts"]),
                  f"{len(duped_accounts)} vs {len(payload['bank_accounts'])}")
    checker.check("a duplicated account does not move the result",
                  duped.get("bscat_transactions") == output.get("bscat_transactions")
                  and duped.get("bscat_summaries") == output.get("bscat_summaries"),
                  "" if duped.get("bscat_transactions") == output.get("bscat_transactions")
                  else "rows or summaries moved")
    corrected = next(
        (account.get("account_name") for account in duped_accounts
         if account.get("account_number") == payload["bank_accounts"][0]["account_number"]),
        None,
    )
    checker.check("the last entry for an account wins", corrected == "Correction", corrected)
    keyless = ModelService().predict({
        **payload,
        "bank_accounts": [{"account_type": "no id"}, *payload["bank_accounts"]],
    })
    checker.check("an account entry with no id is dropped from the echo",
                  len(keyless.get("bank_accounts", [])) == len(payload["bank_accounts"]),
                  f"{len(keyless.get('bank_accounts', []))} vs {len(payload['bank_accounts'])}")

    # ── transactions ────────────────────────────────────────────────────────
    rows = output.get("bscat_transactions", [])
    checker.check("row count unchanged", len(rows) == 58, len(rows))
    bad_keys = sorted({
        key for row in rows for key in row if key not in EXPECTED_ROW_KEYS
    })
    checker.check("no unexpected row keys", not bad_keys, bad_keys)
    missing_keys = sorted({
        key for row in rows for key in EXPECTED_ROW_KEYS if key not in row
    })
    checker.check("no missing row keys", not missing_keys, missing_keys)
    checker.check("row key order stable",
                  all(list(row) == EXPECTED_ROW_KEYS for row in rows))
    leaked = [key for key in FORBIDDEN_ROW_KEYS
              if any(key in row for row in rows)]
    checker.check("no synthetic/internal columns in rows", not leaked, leaked)
    checker.check("rows echo the payload's own field values",
                  all(row[key] == source[key]
                      for row, source in zip(rows, payload["raw_transactions"])
                      for key in source),
                  "a payload value changed on the way out")
    empty_fields = [row[key]
                    for row, source in zip(rows, payload["raw_transactions"])
                    for key in ("secondary_category", "trx_type")
                    if source[key] == ""]
    checker.check("empty upstream strings stay empty strings",
                  bool(empty_fields) and all(value == "" for value in empty_fields),
                  f"{len(empty_fields)} empty upstream field(s), "
                  f"{sum(value != '' for value in empty_fields)} turned into null")
    with_null = ModelService().predict({
        **payload,
        "raw_transactions": [dict(payload["raw_transactions"][0], third_party=None),
                             *payload["raw_transactions"][1:]],
    })
    checker.check("an upstream null stays a null",
                  with_null.get("bscat_transactions", [{}])[0].get("third_party", "x")
                  is None,
                  with_null.get("bscat_transactions", [{}])[0].get("third_party", "x"))

    accounts_in_rows = {row["bank_account_number"] for row in rows}
    sample_accounts = {account["account_number"] for account in payload["bank_accounts"]}
    checker.check("row accounts come from the payload vocabulary",
                  accounts_in_rows <= sample_accounts,
                  sorted(accounts_in_rows - sample_accounts))

    # The account list is keyed by institution + number, so a row carrying no `institution`
    # of its own must still find its entry by number alone -- that is where its `bank` comes
    # from, which liability's counterparty rules mask on, so no result may move.
    no_institution_rows = [
        {key: value for key, value in source.items() if key != "institution"}
        for source in payload["raw_transactions"]
    ]
    bare = ModelService().predict({**payload, "raw_transactions": no_institution_rows})
    bare_out = bare.get("bscat_transactions", [])
    checker.check("rows carrying no institution classify the same",
                  bare.get("status") != "failed"
                  and [(row.get("bscat"), row.get("counterparty"), row.get("stream_id"))
                       for row in bare_out]
                  == [(row["bscat"], row["counterparty"], row["stream_id"])
                      for row in rows],
                  bare.get("error"))
    checker.check("no row-institution is invented for them",
                  all("institution" not in row for row in bare_out))
    frame = build_wagego_frame({**payload, "raw_transactions": no_institution_rows})
    checker.check("the account resolves from its number alone",
                  set(frame["bank"])
                  == {account["institution"] for account in payload["bank_accounts"]},
                  sorted(set(frame["bank"])))

    # `account_number` is the account list's key, not a row's: a row spelling it that way
    # (stripped of any per-row institution too, i.e. the retired payroll shape) names no
    # account, so it matches no entry and resolves no `bank`.  Accepted trade-off: such a
    # row's bank-masked liability rules go quiet instead of the batch failing.
    alias_rows: list[tuple[dict[str, Any], str]] = []
    for source in payload["raw_transactions"][:2]:
        row = {key: value for key, value in source.items()
               if key not in ("bank_account_number", "institution")}
        row["account_number"] = source["bank_account_number"]
        alias_rows.append((row, source["bank_account_number"]))
    alias_payload = {**payload, "raw_transactions": [row for row, _ in alias_rows]}
    alias_frame = build_wagego_frame(alias_payload)
    checker.check("a row spelling its account `account_number` matches no account",
                  set(alias_frame["bank_account_id"]) == {""},
                  sorted(set(alias_frame["bank_account_id"])))
    checker.check("... and so resolves no bank",
                  set(alias_frame["bank"]) == {""},
                  sorted(set(alias_frame["bank"])))
    alias = ModelService().predict(alias_payload)
    alias_out = alias.get("bscat_transactions", [])
    checker.check("the `account_number` field echoes as ordinary upstream data",
                  len(alias_out) == len(alias_rows)
                  and all(row.get("account_number") == number
                          and "bank_account_number" not in row
                          for row, (_, number) in zip(alias_out, alias_rows)),
                  alias.get("error") or f"{len(alias_out)} row(s)")

    # ── summaries ───────────────────────────────────────────────────────────
    summaries = output.get("bscat_summaries", {})
    checker.check("summaries present", list(summaries) == list(EXPECTED_RUN_KEYS),
                  list(summaries))
    for name, expected_count in EXPECTED_SUMMARY_COUNTS.items():
        summary = summaries.get(name, [])
        checker.check(f"{name} row count unchanged", len(summary) == expected_count,
                      len(summary))
        checker.check(f"{name} columns unchanged",
                      all(list(record) == EXPECTED_SUMMARY_KEYS[name] for record in summary),
                      next((list(r) for r in summary if list(r) != EXPECTED_SUMMARY_KEYS[name]), None))
        leaked_summary = [key for key in FORBIDDEN_SUMMARY_KEYS
                          if any(key in record for record in summary)]
        checker.check(f"{name} exposes no account identity", not leaked_summary,
                      leaked_summary)
        if name in ("income_summary", "liability_summary"):
            leaked_stream = [key for key in FORBIDDEN_STREAM_SUMMARY_KEYS
                             if any(key in record for record in summary)]
            checker.check(f"{name} drops bscat/counterparty", not leaked_stream,
                          leaked_stream)

    # ── product dispatch ────────────────────────────────────────────────────
    # `product` wins over key presence when it names a registered product.
    labeled = ModelService().predict({**payload, "product": "wagego"})
    checker.check("registered product still routes to the same chain",
                  labeled.get("bscat_stats", {}).get("product") == "wagego"
                  and labeled.get("bscat_transactions") == rows
                  and labeled.get("bscat_summaries") == summaries,
                  labeled.get("error") or labeled.get("bscat_stats"))

    unknown = ModelService().predict({**payload, "product": "nope"})
    checker.check("unregistered product fails loudly",
                  unknown.get("status") == "failed"
                  and "Unsupported 'product'" in str(unknown.get("error")),
                  unknown.get("error"))
    checker.check("failed output keeps the error envelope",
                  unknown.get("status") == "failed" and "error" in unknown
                  and "run_id" not in unknown,
                  sorted(unknown))
    checker.check("failed output keeps the payload's product",
                  unknown.get("bscat_stats", {}).get("product") == "nope",
                  unknown.get("bscat_stats"))

    # `product` wins even when the body belongs to the other chain: a fundo label on
    # a wagego body is sent down the v1 chain, which then reports its own missing key.
    mismatched = ModelService().predict({**payload, "product": "fundo"})
    checker.check("product wins over key presence",
                  mismatched.get("status") == "failed"
                  and "illion_raw_transactions" in str(mismatched.get("error")),
                  mismatched.get("error"))

    # ── classification snapshot ─────────────────────────────────────────────
    counts = Counter(row["bscat"] for row in rows)
    if counts != Counter(EXPECTED_CATEGORY_COUNTS):
        gained = {k: v for k, v in (counts - Counter(EXPECTED_CATEGORY_COUNTS)).items()}
        lost = {k: v for k, v in (Counter(EXPECTED_CATEGORY_COUNTS) - counts).items()}
        checker.check("category counts unchanged", False, f"gained {gained} lost {lost}")
    else:
        checker.check("category counts unchanged", True)

    print()
    if checker.failures:
        print(f"{len(checker.failures)} check(s) failed:")
        for failure in checker.failures:
            print(f"  - {failure}")
        return 1
    print("All wagego (v2) contract checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
