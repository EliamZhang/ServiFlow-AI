# ServiFlow-AI

Australian bank transaction classification service: a multi-engine classification pipeline that classifies Illion raw bank transactions line by line and outputs summaries for downstream business use. Supports CSV batch input producing Excel reports and single-application JSON inference, deployed as a Python service.

## Environment

##### Python version

3.10

Note: the pinned `numpy==1.21.6` / `pandas==1.3.5` install on Python **3.8 – 3.10** only — on 3.11+ neither has a matching distribution (verified with `pip download --only-binary`; the oldest versions published for 3.11 are numpy 1.23.2 / pandas 1.5.0). The code itself uses no version-specific syntax beyond `from __future__ import annotations`.

##### Processor

CPU

##### PIP dependencies

```txt
numpy==1.21.6
pandas==1.3.5
openpyxl==3.1.3
pyahocorasick==2.0.0
typing_extensions==4.7.1
tqdm==4.66.1
```

## Model API

##### Model directory

./

##### Inference object path

model_main.PredictMain

##### Input key list

```
[
  "userId",
  "applicationId",
  "flowTime",
  "bank_accounts",
  "illion_raw_transactions"
]
```

Note: additional keys passed by upstream callers (e.g. `illion_day_end_balances`) are ignored and do not affect inference.

##### Input example

```json
{
  "userId": 484579009,
  "applicationId": 2513560,
  "flowTime": "2026-07-05 23:52:48.0",
  "bank_accounts": [
    {
      "bank_account_id": 1042813323,
      "account_type": "transaction",
      "bank": "cba",
      "credit_limit": null
    },
    {
      "bank_account_id": 1525527792,
      "account_type": "credit card",
      "bank": "cba",
      "credit_limit": 5000
    }
  ],
  "illion_raw_transactions": [
    {
      "amount": -12.32,
      "balance": -126.28,
      "bank_account_id": 1042813323,
      "category": "Transport",
      "dr_cr": "debit",
      "illion_trx_uuid": "0e63679d-ff62-5a55-bb5f-ae03d4dec068",
      "text": "UBER *TRIP HELP.UBER.C 14518236738 AUS",
      "third_party": "UBER",
      "transaction_date": "2026-02-05",
      "transaction_id": 1423884392,
      "trx_type": null
    }
  ]
}
```

Field description:

| Field | Type | Description |
| --- | --- | --- |
| userId | int | User ID |
| applicationId | int | Application ID, echoed back at the transaction row level |
| flowTime | string | Request time, echoed back unchanged |
| bank_accounts | array | Account list providing account metadata (account_type / bank / credit_limit) missing from transaction rows; metadata does not participate in classification and is only written to the `bank_accounts` output |
| illion_raw_transactions | array | Illion raw transactions; each must contain at least transaction_id, transaction_date, amount, dr_cr, text; when application_id is absent it is filled from the top-level applicationId; an **empty array (zero transactions) is legal** and returns an empty success result (see edge-case examples below) |

##### Output example

```json
{
  "user_id": 484579009,
  "application_id": 2513560,
  "sample_datetime": "2026-07-05 23:52:48.0",
  "run_id": "3fce407b-5c10-4dd9-ad29-06dbb7678893",
  "status": "success",
  "error": null,
  "stats": {
    "txn_raw_input_cnt": 1,
    "transaction_date_max": "2026-02-05"
  },
  "bank_accounts": [
    {
      "bank_account_id": 1042813323,
      "account_type": "transaction",
      "bank": "cba",
      "credit_limit": null
    },
    {
      "bank_account_id": 1525527792,
      "account_type": "credit card",
      "bank": "cba",
      "credit_limit": 5000
    }
  ],
  "transactions": [
    {
      "amount": -12.32,
      "balance": -126.28,
      "bank_account_id": 1042813323,
      "category": "Transport",
      "dr_cr": "debit",
      "illion_trx_uuid": "0e63679d-ff62-5a55-bb5f-ae03d4dec068",
      "text": "UBER *TRIP HELP.UBER.C 14518236738 AUS",
      "third_party": "UBER",
      "transaction_date": "2026-02-05",
      "transaction_id": 1423884392,
      "trx_type": null,
      "application_id": 2513560,
      "counterparty": "Uber",
      "finv_category": "Transport",
      "stream_id": null
    }
  ],
  "summaries": {
    "income_summary": [],
    "liability_summary": [],
    "category_summary": [
      {
        "finv_category": "Transport",
        "bank_account_id": 1042813323,
        "transaction_start_date": "2026-02-05",
        "transaction_end_date": "2026-02-05",
        "transaction_count": 1,
        "total_amount": 12.32,
        "average_amount": 12.32,
        "median_amount": 12.32,
        "latest_amount": 12.32
      }
    ]
  }
}
```

`summaries` sub-arrays are empty when the application has no income / liability stream detected. Excerpt of the same fields from an application that does (one row per sub-array shown):

```json
{
  "income_summary": [
    {
      "finv_category": "Wages",
      "stream_id": "wage_001",
      "income_category": "salary_payg",
      "bank_account_id": 4689355,
      "application_id": 2420589,
      "counterparty": "Main Roads",
      "transaction_start_date": "2025-11-20",
      "transaction_end_date": "2026-05-07",
      "status": "active",
      "transaction_count": 13,
      "total_income_amount": 39962.91,
      "average_income_amount": 3074.07,
      "median_income_amount": 3214.0,
      "latest_income_amount": 3314.01,
      "estimated_monthly_income": 6963.666666666667,
      "frequency": "fortnightly",
      "frequency_day": "Thursday",
      "predicted_next_income_date": "2026-05-21"
    }
  ],
  "liability_summary": [
    {
      "finv_category": "Non SACC Loans",
      "stream_id": "loan_004",
      "liability_category": "Non SACC Loans",
      "bank_account_id": "4689355",
      "application_id": "2420589",
      "counterparty": "Humm",
      "transaction_start_date": "2025-11-19",
      "transaction_end_date": "2026-03-11",
      "status": "Closed",
      "funded_amount": 0.0,
      "repaid_amount": 750.48,
      "repayment_amount": null,
      "recent_fn_repay_amount": 0.0,
      "frequency": "fortnightly",
      "frequency_day": "Wednesday",
      "predicted_closing_date": null
    }
  ],
  "category_summary": [
    {
      "finv_category": "Transport",
      "bank_account_id": 1042813323,
      "transaction_start_date": "2026-02-05",
      "transaction_end_date": "2026-02-05",
      "transaction_count": 1,
      "total_amount": 12.32,
      "average_amount": 12.32,
      "median_amount": 12.32,
      "latest_amount": 12.32
    }
  ]
}
```

Notes on the summary fields: `stream_id` for Wages is a coarse label shared by all three Wages subtypes (`wage_001`, `wage_002`, …, the fine subtype stays in `income_category`). Value types are not normalized across summary layers — income emits `bank_account_id` / `application_id` as numbers, liability as strings; cast on the consumer side.

Field description:

All output field names are snake_case, matching the Excel report column names produced by `backfill.py`.

| Field | Type | Description |
| --- | --- | --- |
| user_id | int | Echo of the input `userId` |
| application_id | int | Echo of the input `applicationId` |
| sample_datetime | string | Echo of the input `flowTime` |
| run_id | string | Unique ID of a single inference run (uuid4); null in `failed` outputs |
| status | string | `success` / `failed`; a zero-transaction input (empty `illion_raw_transactions`) is legal and returns `success` with empty transactions / summaries; on malformed input no exception is raised, instead `failed` + error message is returned with empty transactions / summaries |
| error | string/null | Failure reason |
| stats.txn_raw_input_cnt | int | Input transaction count |
| stats.transaction_date_max | string | Latest `transaction_date` among input rows; null when there are no rows |
| bank_accounts | array | Account metadata (bank_account_id / account_type / bank / credit_limit), not repeated at the transaction row level |
| transactions | array | Original transaction fields + classification results; the core new fields are `finv_category` (fine-grained category), `counterparty` (counterparty name), `stream_id` (income/liability stream id, null for rows not belonging to any stream), plus the `application_id` echo |
| summaries | object | Summaries grouped by type: income_summary (income streams, incl. estimated_monthly_income / predicted_next_income_date), liability_summary (liability streams, incl. funded_amount / repaid_amount / predicted_closing_date), category_summary (aggregate stats by finv_category). Empty arrays / `{}` when nothing applies |

The three echo keys (`user_id` / `application_id` / `sample_datetime`) are only present when the corresponding input key is present; they come first in `success` outputs and last in `failed` outputs.

##### Input/output examples by scenario

| Scenario | illion_raw_transactions | bank_accounts | status | Output shape |
| --- | --- | --- | --- | --- |
| Standard application | non-empty | any | success | classified transactions + summaries (input / output examples above) |
| Zero transactions, no accounts | `[]` | `[]` | success | empty result: empty transactions / summaries, `stats.txn_raw_input_cnt` = 0 |
| Zero transactions, accounts present | `[]` | non-empty | success | same empty result, but `bank_accounts` still carries the account list |
| Malformed input | missing / null / non-list | any | failed | empty transactions / summaries + `error` message |

**Scenario: zero transactions, no accounts** — a user profile with no bank cards and no statement rows is still a valid application; it returns `success` with an empty result instead of failing:

Input:

```json
{
  "userId": 484579009,
  "applicationId": 2513560,
  "flowTime": "2026-07-05 23:52:48.0",
  "bank_accounts": [],
  "illion_raw_transactions": []
}
```

Output:

```json
{
  "user_id": 484579009,
  "application_id": 2513560,
  "sample_datetime": "2026-07-05 23:52:48.0",
  "run_id": "abcdf92d-5bda-4f49-abcd-c9261c77fec5",
  "status": "success",
  "error": null,
  "stats": {
    "txn_raw_input_cnt": 0,
    "transaction_date_max": null
  },
  "bank_accounts": [],
  "transactions": [],
  "summaries": {}
}
```

**Scenario: zero transactions, accounts present** — same as above, but the account metadata is still echoed at the top level:

Input:

```json
{
  "userId": 484579009,
  "applicationId": 2513560,
  "flowTime": "2026-07-05 23:52:48.0",
  "bank_accounts": [
    {
      "bank_account_id": 1042813323,
      "account_type": "transaction",
      "bank": "cba",
      "credit_limit": null
    }
  ],
  "illion_raw_transactions": []
}
```

Output:

```json
{
  "user_id": 484579009,
  "application_id": 2513560,
  "sample_datetime": "2026-07-05 23:52:48.0",
  "run_id": "5550bad2-931e-480b-9d82-fd09f0f35546",
  "status": "success",
  "error": null,
  "stats": {
    "txn_raw_input_cnt": 0,
    "transaction_date_max": null
  },
  "bank_accounts": [
    {
      "bank_account_id": 1042813323,
      "account_type": "transaction",
      "bank": "cba",
      "credit_limit": null
    }
  ],
  "transactions": [],
  "summaries": {}
}
```

**Scenario: malformed input** — `illion_raw_transactions` missing, `null`, or not a list is a structural error; no exception is raised, the call returns `failed` (note: no `bank_accounts` key in failed outputs):

Input:

```json
{
  "userId": 484579009,
  "applicationId": 2513560,
  "flowTime": "2026-07-05 23:52:48.0"
}
```

Output:

```json
{
  "run_id": null,
  "status": "failed",
  "error": "Input JSON must contain an 'illion_raw_transactions' list.",
  "stats": {
    "txn_raw_input_cnt": 0,
    "transaction_date_max": null
  },
  "user_id": 484579009,
  "application_id": 2513560,
  "sample_datetime": "2026-07-05 23:52:48.0",
  "transactions": [],
  "summaries": {}
}
```

## Local run

```bash
pip install -r requirements.txt
python verify_model.py                  # reads model_input.json by default, writes output/model_output_{applicationId}_{timestamp}.json
python verify_model.py --input xxx.json # specify input
python backfill.py                      # CSV batch input, writes Excel report (output/classification_report_*.xlsx)
```

Note: the sample transactions in `model_input.json` are for demonstration only and do not cover all classification engine rules; for full validation use `input_converter.py` to extract a real application from `sample.csv` and generate the input.

## Entry points

Three entry scripts serve different purposes but run the **same core pipeline** — `ClassificationOrchestrator` loads the pipeline config + category catalog and runs all engines in priority order. They differ only in input/output shape:

| Script | Purpose | Input | Output |
| --- | --- | --- | --- |
| `model_main.py` | Production inference entry (deployed service, `PredictMain.predict`) | single-application JSON dict | JSON dict (transactions + summaries + stats) |
| `verify_model.py` | Local verification / trial run for a single application | single-application JSON file (default `model_input.json`) | JSON file (`output/model_output_{applicationId}_{timestamp}.json`) |
| `backfill.py` | Batch classification of historical / offline data | CSV (multiple applications) | Excel report (`output/classification_report_{timestamp}.xlsx`) |

**Maintenance note (important):** the three entries differ only in input/output format; classification logic — engine priority order, overwrite rules, income / liability protection — lives in the shared core and behaves identically in all three. Make changes in the core or engine code, never per-entry: a change verified through one entry applies to all. Known difference: `backfill.py` reads CSV directly and does not go through `build_transactions_frame`, so the bank-account metadata merge (`account_type` / `bank` / `credit_limit`) applied by the JSON entries is not applied there — CSV input must carry these columns itself if the engines need them.

## Classification pipeline

10 classification engines run in **ascending `priority` order** (the array order in `configs/pipeline.json` is irrelevant); each engine matches transactions line by line and later engines overwrite the classification of earlier ones:

| priority | engine | responsibility |
| --- | --- | --- |
| 1 | transfer | Internal / external transfer identification |
| 10 | initial | Initial classification via merchant knowledge base (merchant_kb.csv) and basic rules |
| 150 | dishonour | Dishonour identification |
| 180 | gambling | Gambling merchant / keyword identification (gambling_rules.csv) |
| 200 | income | Income stream identification (Wages / Centrelink, etc.) |
| 300 | liability | Liability stream identification (loan / BNPL, etc.) |
| 400 | all_other_credit | Collects remaining credits, only processes credit rows |
| 500 | fee | Fee identification |
| 800 | rent | Rent identification (rent_rules.csv: institution layer + keyword layer) |
| 999 | catch_all | Fallback: only matches transactions not yet classified |

Overwrite (later-wins) exceptions, i.e. where a later engine does **not** take a row:

- `liability` and `rent` skip transactions already classified as income or liability; `all_other_credit` only touches `dr_cr == credit` rows and skips everything already classified — except rows labelled `External Transfers`, which it may re-match; `catch_all` only matches unclassified rows.
- Rows claimed by `gambling` cannot be overwritten by `income` / `liability`; every other later engine still wins them.

Engine rules are externalized as CSV files under each engine's `resources/` directory (rent_engine, gambling_engine, liability_engine, transfer_engine, catch_all_engine, etc.); the pipeline configuration lives in `configs/pipeline.json` and the category catalog in `configs/category_catalog.json`. Add a rule by appending a row to the engine's CSV — no engine code change needed.

## Regression check on code changes

After each change, run the baseline comparison to make sure classification results did not change unexpectedly:

```bash
python baseline.py diff    # compares current pipeline output against the baseline (exit 0 = no differences)
python baseline.py save    # rebuilds the baseline (only when the change is expected; use --replace --reason "<reason>")
```

The baseline has four layers: final output layer (sample_baseline.csv), per-engine claims layer (engine_claims.csv), config/version layer (run_meta.json), summary metrics layer (summaries/). A difference in any layer makes `diff` exit non-zero.

## Contacts

* Model developer & maintainer: Eliam Zhang
* Model service consumer: Eliam Zhang
