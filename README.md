# ServiFlow-AI

Australian bank transaction classification service: a multi-engine classification pipeline that classifies raw bank transactions line by line and outputs summaries for downstream business use. Two input/output contracts are supported — `fundo` (v1) and `wagego` (v2) — and the payload's `product` field selects which one runs. Supports CSV batch input producing Excel reports and single-application JSON inference, deployed as a Python service.

## Environment

##### Python version

3.11

##### Processor

CPU

##### PIP dependencies

```txt
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0
pyahocorasick>=2.0.0
```

## Model API

##### Model directory

./

##### Inference object path

model_main.PredictMain

##### Product dispatch

The payload's `product` field selects the contract chain (matched case-insensitively, surrounding blanks ignored):

| `product` | chain | wording |
| --- | --- | --- |
| `fundo` | v1 | Illion raw transactions |
| `wagego` | v2 | wagego raw transactions |

- An **unregistered** `product` is a hard failure with no silent fallback: `status` = `failed`, `error` = `Unsupported 'product' value 'xxx'; supported products: fundo, wagego.`, and `bscat_stats.product` echoes the offending value.
- A payload **without** a registered `product` (missing / blank / non-string) falls back to key-presence detection, so callers that predate `product` keep working: `illion_raw_transactions` present → `fundo` (it wins when both keys are present), otherwise `raw_transactions` present → `wagego`, otherwise `fundo` (whose failure message stays byte-identical to the legacy one).
- `bscat_stats.product` echoes the payload's `product` verbatim; when the payload omits it, the detected chain's product name (`fundo` / `wagego`) is reported instead.

Both contracts share the same pipeline and the same output shape: the three echoed input identifiers (`userId` / `applicationId` / `flowTime`), `bscat_stats`, `bank_accounts`, `bscat_transactions`, `bscat_summaries` — the summaries and the rows' business columns (`counterparty` / `bscat` / `stream_id`) are identical field-for-field. The contracts differ in account/transaction wording only: a v1 row is that contract's own fixed field set (`bank_account_id` / `transaction_id` included), while a v2 row is the payload's own transaction fields echoed as sent.

##### Failure handling

Two guarantees: inference never raises on a malformed request, and a malformed *field* inside an otherwise valid payload never fails a batch. The first is answered with the failed result, the second is absorbed at build time.

**Contract failures return the failed result** — `status` = `failed`, `error`, empty `bscat_transactions` / `bscat_summaries`, zeroed `bscat_stats`, plus whichever of `userId` / `applicationId` / `flowTime` the payload carried (`bank_accounts` is omitted):

| Rejected input | `error` |
| --- | --- |
| The body is not a JSON object (`[1, 2]` / `"junk"` / `null`) | `Input payload must be a JSON object.` |
| `illion_raw_transactions` missing / `null` / not a list | `Input JSON must contain an 'illion_raw_transactions' list.` |
| `raw_transactions` is `null` / not a list, or absent from a payload dispatched to v2 | `Input JSON must contain a 'raw_transactions' list.` |
| A `raw_transactions` item is not a JSON object | `Each 'raw_transactions' item must be a JSON object.` |
| Neither `applicationId` nor `flowId` usable (v2) | `Input JSON must contain a non-empty 'applicationId' or 'flowId'.` |
| `product` names an unregistered product | `Unsupported 'product' value 'nope'; supported products: fundo, wagego.` |
| A v1 transaction is missing a key column | `Input transactions are missing key column(s): transaction_id` |
| A v1 key column holds a blank value | `Transaction key columns cannot contain blank values.` |
| Two v1 transactions share an `(application_id, transaction_id)` pair | `Each (application_id, transaction_id) pair must be unique.` |
| An unexpected pipeline / engine / serialization error | the exception message |

Everything above runs after dispatch (`product` first, then key detection — see Product dispatch), so which wording a rejected payload gets depends on the contract it was routed to: a payload carrying neither transaction key and no registered `product` is treated as v1. The six dispatch / shape rows are answered by the contract layer before the run starts and write nothing to the log; the three v1 key-column rows are raised inside the run, so each leaves a `classification run failed` line and a traceback in the service log as well. Only the last row means the service itself is at fault — a rule or adapter bug, previously an HTTP 500 *after* the classification had run, so the caller could not tell a bad payload from a dead service. None of these is silently absorbed: the batch has no well-defined result, so the caller gets the failure instead of a partial classification (the duplicated pair means the same transaction arrived twice).

**Malformed fields are absorbed** — the payload still classifies, with the affected value read as missing:

| Malformed input | What happens |
| --- | --- |
| A v1 transaction omits `text` / `dr_cr` / `amount` / `transaction_date` / `bank_account_id` | The column is carried with the missing default the engines expect (`""` for `text` / `dr_cr`, missing for `amount`, `null` for `transaction_date` and the account id), so they read a blank value instead of raising. Those columns are part of a v1 row's own field set, so they still appear in the output. A v2 row's fields are the payload's own and nothing is synthesized into them: an omitted field is simply absent from the output row |
| `text` is a number / boolean | Classified as its text form — `12345` → `"12345"` |
| `text` is an array / object | Read as blank, and never stringified, so payload contents cannot leak into keyword matching |
| An array or object sits where a scalar belongs (`dr_cr` / `amount` / `transaction_date` / an account id) | The row is read as missing that value, so the engines see the column's missing default; a v2 row's account number that is an array / object links to no account. When the field is one of the payload's own v2 fields, the row still echoes it as sent (see the two upstream-field rows below) |
| `bank_accounts` is absent / `null` / not a list / holds entries that are not objects or carry no id | No account metadata: rows classify with blank `account_type` / `bank` and `null` `credit_limit`. Blank `bank` matches no bank-specific rule, so an omitted list costs liability accuracy — in v2 `bank` also comes from the row's own `institution` (or the matched account entry), so an omitted list costs only `account_type` / `credit_limit` there |
| `bank_accounts` repeats an account id | Collapsed to one account: the last entry wins, the position is the first appearance, and transaction rows are never fanned out. (A duplicate used to fan rows out and fail the whole batch on `(application_id, transaction_id)` uniqueness.) |
| A row and its account entry type the same id differently (`5116090` / `5116090.0` / `"5116090"`) | Matched as one account, so the metadata resolves; each side keeps its own value in the output (a number stays a number). Leading zeros are not merged — `"0511"` ≠ `511` |
| `transaction_date` is `null` / numeric / an array | Ignored by `bscat_stats.transaction_date_max`; the key is `null` when no value is a usable date string |
| An extra upstream field holds an array / object | Echoed back as it came (an array stays an array) |
| A v2 upstream field is `""` / `null` / an array | Echoed back as sent: `""` stays `""`, `null` stays `null`, an array stays an array. Only the business columns (`counterparty` / `bscat` / `stream_id`) and the summaries keep turning an empty value into `null` — v1 has no such pass-through, since its rows are a fixed field set |
| `illion_raw_transactions` / `raw_transactions` is an empty array | Success, not failure — an empty result with `txn_raw_input_cnt` = 0 (see the edge-case examples below) |

The tolerance lives in the JSON contract layer (`classification_core/service.py`), shared by both contracts; `verify_model.py` still raises on a bad payload or a pipeline error (local trial-run semantics), and `backfill.py` reads CSV rather than JSON — CSV input must carry the internal key columns and the account metadata itself.

#### fundo (v1)

##### Input key list

```
[
  "product",
  "userId",
  "applicationId",
  "flowTime",
  "bank_accounts",
  "illion_raw_transactions"
]
```

`product` is optional (`fundo` is what key detection resolves to when it is absent). Note: additional keys passed by upstream callers (e.g. `illion_day_end_balances`) are ignored and do not affect inference.

##### Input example

```json
{
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "product": "fundo",
  "bank_accounts": [
    {
      "bank_account_id": 4689355,
      "account_type": "transaction",
      "bank": "cba",
      "credit_limit": null
    }
  ],
  "illion_raw_transactions": [
    {
      "amount": -15.0,
      "balance": 799.45,
      "bank_account_id": 4689355,
      "category": "Overdrawn",
      "dr_cr": "debit",
      "illion_trx_uuid": "3aefa463-9c1b-4f22-8c7a-c153fbf440a4",
      "text": "Overdraw Fee For exceeding available funds on 05 Dec",
      "third_party": "nan",
      "transaction_date": "2025-12-06",
      "transaction_id": 2755538243,
      "trx_type": "Overdrawn Fees"
    },
    {
      "amount": -8.0,
      "balance": 513.98,
      "bank_account_id": 4689355,
      "category": "Non SACC Loans",
      "dr_cr": "debit",
      "illion_trx_uuid": "0ad64844-060d-4bd5-b715-bb88b60419b4",
      "text": "Direct Debit 125202 humm BNPL A00000001910998566",
      "third_party": "Humm",
      "transaction_date": "2026-01-02",
      "transaction_id": 2755538124,
      "trx_type": "Direct Debit"
    }
  ]
}
```

(2 of the 825 transactions of this application are shown.)

Field description:

| Field | Type | Description |
| --- | --- | --- |
| product | string | Selects the contract chain: `fundo` (v1) / `wagego` (v2). Optional — see Product dispatch |
| userId | int | User ID, echoed back unchanged as `userId` |
| applicationId | int | Application ID; a row-level `application_id` that is missing internally is filled from it. Echoed back unchanged as `applicationId` (not repeated per transaction row) |
| flowTime | string | Request time, echoed back unchanged as `flowTime` |
| bank_accounts | array | Account list supplying the account metadata (`account_number` / `account_type` / `bank`) used by rows carrying that `bank_account_id`; rows whose account is not listed get blank metadata, and the key may be omitted entirely. `bank` **does** take part in classification — liability's bank-constrained counterparty rules and its per-bank stream grouping read it, and a blank `bank` matches no bank-specific rule — so omitting the list costs liability accuracy. The metadata appears in the top-level `bank_accounts` output only (one entry per `bank_account_id`, list order = first appearance, `account_number` verbatim and `null` when the caller does not send one); a duplicated `bank_account_id` never fans transaction rows out, the last entry for an id wins and entries without an id are dropped. An account id may be written as a number or a string on either side — `5116090`, `5116090.0` and `"5116090"` all name the same account, so a transaction row and its account entry are matched even when the payload types them differently. The echoed `bank_account_id` is the account entry's own value (a number stays a number); a transaction row echoes its own value too |
| illion_raw_transactions | array | Illion raw transactions; each must contain `transaction_id` (`application_id` + `transaction_id` must be unique across the array). A transaction that omits `text` / `dr_cr` / `amount` / `transaction_date` / `bank_account_id` still classifies: the frame carries those columns with blank values. A `text` that is not a string is classified as its text form (`12345` → `"12345"`); an array or object `text` is treated as blank rather than stringified. **empty array (zero transactions) is legal** and returns an empty success result (see edge-case examples below) |

##### Output example

```json
{
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "bscat_stats": {
    "txn_raw_input_cnt": 825,
    "transaction_date_max": "2026-05-09",
    "product": "fundo"
  },
  "bank_accounts": [
    {
      "bank_account_id": 4689355,
      "account_number": null,
      "account_type": "transaction",
      "bank": "cba"
    }
  ],
  "bscat_transactions": [
    {
      "amount": -112.8,
      "balance": 384.52,
      "bank_account_id": 4689355,
      "category": "Automotive",
      "dr_cr": "debit",
      "illion_trx_uuid": "bad12f69-6758-49d1-9a8c-2a9cfdab9878",
      "text": "ATLAS SAWYERS SAWYERS VALLE AUS Card xx9327 Value Date: 10/01/2026",
      "third_party": "Automotive",
      "transaction_date": "2026-01-13",
      "transaction_id": 2755538076,
      "trx_type": "nan",
      "counterparty": "ATLAS SAWYERS SAWYERS VALLE",
      "bscat": "Automotive",
      "stream_id": null
    }
  ],
  "bscat_summaries": {
    "income_summary": [
      {
        "stream_id": "wage_001",
        "income_category": "salary_payg",
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
        "stream_id": "loan_004",
        "liability_category": "Non SACC Loans",
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
        "bscat": "All Other Credits",
        "transaction_start_date": "2025-11-13",
        "transaction_end_date": "2025-11-13",
        "transaction_count": 1,
        "total_amount": 48.6,
        "average_amount": 48.6,
        "median_amount": 48.6,
        "latest_amount": 48.6
      }
    ]
  }
}
```

(1 of 825 classified transactions and 1 record per summary are shown.)

Field description:

| Field | Type | Description |
| --- | --- | --- |
| userId / applicationId / flowTime | int / int / string | Echoes of the same input keys, values unchanged. **This is the only place the input identifiers appear** — the transaction rows and the summaries do not repeat them |
| status / error | string / string | Present **only on a failed run**: `status` = `failed` plus the `error` reason. A successful run returns neither key (there is no `run_id`; a run is identified by the echoed `applicationId` + `flowTime`). A failed run covers three cases (the full list of rejected inputs is under Failure handling above): the payload does not match either contract, the payload's `product` is unregistered, and an unexpected pipeline / engine / serialization error — the last one is logged with its traceback on the service side and reported as a failed result, so a bug in a rule or an adapter surfaces as a structured failure rather than an HTTP 500 |
| bscat_stats.product | string | The payload's `product` verbatim, or the detected chain's product name when the payload omits it |
| bscat_stats.txn_raw_input_cnt | int | Input transaction count |
| bscat_stats.transaction_date_max | string | Max transaction date in input; values that are not date strings (null, numbers, arrays) are ignored, and the key is `null` when none is usable |
| bank_accounts | array | Account metadata (`bank_account_id` / `account_number` / `account_type` / `bank`) echoed from the input list — one entry per `bank_account_id` (duplicates collapsed to their last entry, id-less entries dropped), not repeated at the transaction row level; the key is **absent** in `failed` outputs |
| bscat_transactions | array | Original transaction fields + classification results; the core new fields are `bscat` (fine-grained category), `counterparty` (counterparty name) and `stream_id` (income/liability stream id, null for rows not belonging to any stream). Rows come out in input order. Field set: `amount` / `balance` / `bank_account_id` / `category` / `dr_cr` / `text` / `third_party` / `transaction_date` / `transaction_id` / `trx_type` / `counterparty` / `bscat` / `stream_id` (+ any extra upstream field the payload carries, e.g. `account_number` / `illion_trx_uuid`) |
| bscat_summaries | object | Summaries grouped by type — income_summary (14 fields: stream_id / income_category / transaction_start_date / transaction_end_date / status / transaction_count / total_income_amount / average_income_amount / median_income_amount / latest_income_amount / estimated_monthly_income / frequency / frequency_day / predicted_next_income_date), liability_summary (12 fields: same shape with liability_category / funded_amount / repaid_amount / repayment_amount / recent_fn_repay_amount / predicted_closing_date), category_summary (8 fields: `bscat` + the two dates + transaction_count + total_/average_/median_/latest_amount) |

##### Input/output examples by scenario

| Scenario | illion_raw_transactions | bank_accounts | Run result | Output shape |
| --- | --- | --- | --- | --- |
| Standard application | non-empty | any | success | classified transactions + summaries (input / output examples above) |
| Zero transactions, no accounts | `[]` | `[]` | success | empty result: empty transactions / summaries, `txn_raw_input_cnt` = 0 |
| Zero transactions, accounts present | `[]` | non-empty | success | same empty result, but `bank_accounts` still carries the account list |
| Malformed input | missing / null / non-list | any | failed (`status` / `error`) | empty transactions / summaries + `error` message |
| Malformed field (a numeric `text`, an array `amount`, a duplicated account id, a missing column) | any | any | success | normal success output; the affected value reads as missing — see Failure handling above |
| Unregistered product | any | any | failed (`status` / `error`) | empty transactions / summaries + `Unsupported 'product' value ...` |
| Unexpected engine error | any | any | failed (`status` / `error`) | empty transactions / summaries + the exception message; the traceback goes to the service log |

**Scenario: zero transactions, no accounts** — a user profile with no bank cards and no statement rows is still a valid application; it returns a successful empty result instead of failing:

Input:

```json
{
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "product": "fundo",
  "bank_accounts": [],
  "illion_raw_transactions": []
}
```

Output:

```json
{
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "bscat_stats": {
    "txn_raw_input_cnt": 0,
    "transaction_date_max": null,
    "product": "fundo"
  },
  "bank_accounts": [],
  "bscat_transactions": [],
  "bscat_summaries": {}
}
```

**Scenario: zero transactions, accounts present** — same as above, but the account metadata is still echoed at the top level:

Input:

```json
{
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "product": "fundo",
  "bank_accounts": [
    {
      "bank_account_id": 4689355,
      "account_number": "12345678",
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
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "bscat_stats": {
    "txn_raw_input_cnt": 0,
    "transaction_date_max": null,
    "product": "fundo"
  },
  "bank_accounts": [
    {
      "bank_account_id": 4689355,
      "account_number": "12345678",
      "account_type": "transaction",
      "bank": "cba"
    }
  ],
  "bscat_transactions": [],
  "bscat_summaries": {}
}
```

**Scenario: malformed input** — `illion_raw_transactions` missing, `null`, or not a list is a structural error; no exception is raised, the call returns the failed shape (`status` / `error`; note: no `bank_accounts` key in failed outputs). A body that is not a JSON object at all (`[1, 2]`, `"junk"`, `null`) returns the same shape with `error` = `Input payload must be a JSON object.`:

Input:

```json
{
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "product": "fundo"
}
```

Output:

```json
{
  "status": "failed",
  "error": "Input JSON must contain an 'illion_raw_transactions' list.",
  "bscat_stats": {
    "txn_raw_input_cnt": 0,
    "transaction_date_max": null,
    "product": "fundo"
  },
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "bscat_transactions": [],
  "bscat_summaries": {}
}
```

**Scenario: unregistered product** — `product` is read before anything else, so an unknown value fails immediately instead of silently falling back to a default chain:

Input:

```json
{
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "product": "nope",
  "illion_raw_transactions": []
}
```

Output:

```json
{
  "status": "failed",
  "error": "Unsupported 'product' value 'nope'; supported products: fundo, wagego.",
  "bscat_stats": {
    "txn_raw_input_cnt": 0,
    "transaction_date_max": null,
    "product": "nope"
  },
  "userId": 484225470,
  "applicationId": 2420589,
  "flowTime": "2026/5/9 12:42.0",
  "bscat_transactions": [],
  "bscat_summaries": {}
}
```

#### wagego (v2)

Same pipeline and the same output shape as `fundo`; the wording follows the wagego payload instead of the Illion one.

##### Input key list

```
[
  "product",
  "userId",
  "applicationId",
  "flowId",
  "flowTime",
  "bank_accounts",
  "raw_transactions"
]
```

- `product` is optional; key detection resolves a payload with `raw_transactions` to `wagego`.
- `bank_accounts` **may be missing entirely** (the online payload has no such key).
- `applicationId` is often an empty string — the real identifier is `flowId`.
- Upstream keys such as `sceneId` / `bank_flow_flow_id`, the `day_end_balances` block and the 80+ feature keys are ignored.

##### Input example

```json
{
  "userId": 9000001,
  "applicationId": "",
  "flowId": "9000000001@9000001@10001@4a7c1f52-9e0b-4a3d-8f61-2b5d7c9e0a13",
  "flowTime": "2026-09-15 15:54:57",
  "product": "wagego",
  "bank_accounts": [
    {
      "bsb": "062692",
      "account_number": "55107357",
      "institution": "cba",
      "account_type": "savings",
      "account_holder": "ALEX JORDAN BLAKE",
      "account_holder_type": "single",
      "account_name": "Everyday Saver"
    },
    {
      "bsb": "062692",
      "account_number": "55107322",
      "institution": "cba",
      "account_type": "transaction",
      "account_holder": "ALEX JORDAN BLAKE",
      "account_holder_type": "single",
      "account_name": "Joint Account"
    }
  ],
  "raw_transactions": [
    {
      "secondary_category": "",
      "transaction_date": "2026-09-15",
      "institution": "cba",
      "amount": 1320.15,
      "bank_account_number": "55107322",
      "balance": 1339.16,
      "dr_cr": "credit",
      "trx_type": "",
      "text": "Direct Credit 421520 Doyle Racing Pty Doyle Racing",
      "category": "External Transfers",
      "third_party": "External Transfers"
    },
    {
      "secondary_category": "",
      "transaction_date": "2026-09-15",
      "institution": "cba",
      "amount": -9.97,
      "bank_account_number": "55107322",
      "balance": 19.01,
      "dr_cr": "debit",
      "trx_type": "",
      "text": "1630-REDDY EXPRESS BROA BROADMEADOW AU",
      "category": "Automotive",
      "third_party": "Reddy Express"
    }
  ]
}
```

(2 of the 5 accounts and 2 of the 58 transactions of this sample are shown.)

A row may instead name its account with `account_number` alone and carry no `institution`, as the payroll sample does — it then takes the institution from the `bank_accounts` entry it matches, so the account metadata (and `bank`) still resolve:

```json
{
  "secondary_category": "",
  "transaction_date": "2026-09-15",
  "amount": 1320.15,
  "account_number": "55107322",
  "balance": 1339.16,
  "dr_cr": "credit",
  "trx_type": "",
  "text": "Direct Credit 421520 Doyle Racing Pty Doyle Racing",
  "category": "External Transfers",
  "third_party": "External Transfers"
}
```

Field description:

| Field | Type | Description |
| --- | --- | --- |
| product | string | Selects the contract chain: `wagego` (v2) |
| userId | int | User ID, echoed back unchanged as `userId` |
| applicationId | string | Application ID; when blank the `flowId` is used instead (internally) and both are echoed back as `applicationId` / `flow_id` |
| flowId | string | Real application identifier of the wagego payload, echoed back as `flow_id` |
| flowTime | string | Request time, echoed back unchanged as `flowTime` |
| bank_accounts | array | Optional account list; supplies `account_type` / `credit_limit` (and the `bank` the liability rules mask on) to the transactions whose account it names. Echoed back at the top level as the contract's key list — `bsb` / `account_number` / `bank` / `institution` / `account_type` / `account_holder` / `account_holder_type` / `account_name`, `null` for a key the entry did not send — one entry per account, duplicates collapsed to their last entry and id-less entries dropped. A duplicate `account_number` never fans out transaction rows |
| raw_transactions | array | wagego raw transactions; each names its account with `bank_account_number` (optionally plus a row-level `institution`) or with `account_number` alone, and carries the rest of the fields the classifier reads (`transaction_date` / `amount` / `dr_cr` / `text`). Transactions carry **no id**; the internal `transaction_id` is the 1-based array position and is not exported. Upstream fields (`institution` / `bank_account_number` / `account_number` / `secondary_category` / `trx_type` …) are echoed back **as sent** — `""` stays `""`, a `null` stays `null` — in the payload's own field order. As in `fundo`, a non-string `text` is classified as its text form and an array / object `text` is treated as blank |

##### Output example

```json
{
  "userId": 9000001,
  "applicationId": "9000000001@9000001@10001@4a7c1f52-9e0b-4a3d-8f61-2b5d7c9e0a13",
  "flow_id": "9000000001@9000001@10001@4a7c1f52-9e0b-4a3d-8f61-2b5d7c9e0a13",
  "flowTime": "2026-09-15 15:54:57",
  "bscat_stats": {
    "txn_raw_input_cnt": 58,
    "transaction_date_max": "2026-09-15",
    "product": "wagego"
  },
  "bank_accounts": [
    {
      "bsb": "062692",
      "account_number": "55107357",
      "bank": null,
      "institution": "cba",
      "account_type": "savings",
      "account_holder": "ALEX JORDAN BLAKE",
      "account_holder_type": "single",
      "account_name": "Everyday Saver"
    },
    {
      "bsb": "None",
      "account_number": "55107365",
      "bank": null,
      "institution": "cba",
      "account_type": "trading",
      "account_holder": "Alex Blake",
      "account_holder_type": "single",
      "account_name": "Share Trading"
    },
    {
      "bsb": "062692",
      "account_number": "55107340",
      "bank": null,
      "institution": "cba",
      "account_type": "transaction",
      "account_holder": "ALEX JORDAN BLAKE",
      "account_holder_type": "single",
      "account_name": "Spending Account"
    },
    {
      "bsb": "062692",
      "account_number": "55107322",
      "bank": null,
      "institution": "cba",
      "account_type": "transaction",
      "account_holder": "ALEX JORDAN BLAKE",
      "account_holder_type": "single",
      "account_name": "Joint Account"
    },
    {
      "bsb": "062424",
      "account_number": "55107373",
      "bank": null,
      "institution": "cba",
      "account_type": "savings",
      "account_holder": "CASEY MORGAN JORDAN ELLIS",
      "account_holder_type": "single",
      "account_name": "Savings Account"
    }
  ],
  "bscat_transactions": [
    {
      "secondary_category": "",
      "transaction_date": "2026-09-15",
      "institution": "cba",
      "amount": 1320.15,
      "bank_account_number": "55107322",
      "balance": 1339.16,
      "dr_cr": "credit",
      "trx_type": "",
      "text": "Direct Credit 421520 Doyle Racing Pty Doyle Racing",
      "category": "External Transfers",
      "third_party": "External Transfers",
      "counterparty": "DOYLE RACING DOYLE RACING",
      "bscat": "Wages",
      "stream_id": "wage_001"
    },
    {
      "secondary_category": "",
      "transaction_date": "2026-09-15",
      "institution": "cba",
      "amount": -9.97,
      "bank_account_number": "55107322",
      "balance": 19.01,
      "dr_cr": "debit",
      "trx_type": "",
      "text": "1630-REDDY EXPRESS BROA BROADMEADOW AU",
      "category": "Automotive",
      "third_party": "Reddy Express",
      "counterparty": "Reddy Express",
      "bscat": "Automotive",
      "stream_id": null
    }
  ],
  "bscat_summaries": {
    "income_summary": [
      {
        "stream_id": "centrelink_001",
        "income_category": "centrelink",
        "transaction_start_date": "2026-09-09",
        "transaction_end_date": "2026-09-09",
        "status": "irregular",
        "transaction_count": 2,
        "total_income_amount": 317.62,
        "average_income_amount": 158.81,
        "median_income_amount": 158.81,
        "latest_income_amount": 138.04,
        "estimated_monthly_income": null,
        "frequency": "irregular",
        "frequency_day": "Wednesday",
        "predicted_next_income_date": null
      }
    ],
    "liability_summary": [
      {
        "stream_id": "loan_002",
        "liability_category": "Dishonours",
        "transaction_start_date": "2026-07-25",
        "transaction_end_date": "2026-07-25",
        "status": "Closed",
        "funded_amount": 0.0,
        "repaid_amount": 0.0,
        "repayment_amount": null,
        "recent_fn_repay_amount": 0.0,
        "frequency": "fortnightly",
        "frequency_day": null,
        "predicted_closing_date": null
      }
    ],
    "category_summary": [
      {
        "bscat": "All Other Credits",
        "transaction_start_date": "2026-07-22",
        "transaction_end_date": "2026-07-22",
        "transaction_count": 1,
        "total_amount": 1698.27,
        "average_amount": 1698.27,
        "median_amount": 1698.27,
        "latest_amount": 1698.27
      }
    ]
  }
}
```

(All 5 accounts are shown; 2 of the 58 classified transactions and 1 record per summary are shown.)

Field description:

| Field | Type | Description |
| --- | --- | --- |
| userId / applicationId / flow_id / flowTime | int / string / string / string | Echoes of `userId` / the resolved application id (`applicationId` unless blank, then `flowId`) / `flowId` / `flowTime`. `applicationId` and `flow_id` are omitted when neither is usable. As in `fundo`, the identifiers appear here only — never per row |
| status / error | string / string | Same as `fundo`: only on a failed run, which also carries these echo keys and the v2 wording |
| bscat_stats | object | Same as `fundo`, `product` = `wagego` |
| bank_accounts | array | The payload's `bank_accounts` echoed in the contract's own key list (bsb / account_number / bank / institution / account_type / account_holder / account_holder_type / account_name), `null` for a key an entry did not send — one entry per account, duplicates collapsed to their last entry and id-less entries dropped; `[]` when the key is missing. Absent in `failed` outputs, like `fundo` |
| bscat_transactions | array | The `raw_transactions` fields in their original order, echoed as sent, then the business results `counterparty` / `bscat` / `stream_id`. None of the synthetic columns (`transaction_id` / `bank_account_id` / `bank` / `account_type` / `credit_limit`) and not the resolved `application_id` is repeated per row |
| bscat_summaries | object | Identical to `fundo`'s summaries, field for field (income_summary 14 / liability_summary 12 / category_summary 8) — the account identity and the resolved `application_id` appear in neither contract's summaries |

##### Contract notes (v2)

- Account identity is expressed in the input's own wording and only on the transaction rows (they are upstream fields echoed through). A row is related to its account by the number it carries: `bank_account_number` (the production shape, usually with a row-level `institution`) or `account_number` (the payroll sample shape). Internally the adapter composes `bank_account_id` as `"{institution}-{number}"` and `bank` as the row's `institution`; a row that names no institution of its own takes it from the account entry it matched (`institution`, falling back to `bank`) — the liability counterparty rules are masked on `bank`, so it must be mapped. A row that names an institution is only ever matched on institution + number, so two banks sharing an account number never borrow each other's metadata. Neither column is exported.
- Upstream fields are echoed as sent: an empty string stays `""`, a `null` stays `null`, an array stays an array. Only the pipeline's own columns (`counterparty` / `bscat` / `stream_id`) and the summaries serialize an empty value as `null` — which is why `secondary_category` / `trx_type` come back as `""` for the sample above while the v1 rows (a fixed field set) would show `null`.
- A v2 payload with neither `applicationId` nor `flowId` usable, or without a `raw_transactions` list, returns the failed shape with the same echo keys.
- Robustness: the field-level rules under Failure handling above (non-string `text`, array-valued fields, omitted fields, duplicated / id-less `bank_accounts` entries, account-id type tolerance) apply to v2 as well — the account number is canonicalised (`12345678`, `12345678.0` and `"12345678"` are one account), so a transaction row and its `bank_accounts` entry match whatever type each side used; and a missing or unusable `bank_accounts` costs `account_type` / `credit_limit` for the rows that named an account it does not list.
- Regression handle: `check_wagego_sample.py` runs the anonymised `wagego_sample.json` through the service and asserts the stats, the echoed field values (empty strings included), the row key set and order, the absence of the synthetic columns, the account echo, the summary column sets, the `account_number`-only row shape and a per-category count snapshot. Run it after any change to the v2 adapter or the serializers (`baseline.py` only covers v1).

## Local run

```bash
pip install -r requirements.txt
python verify_model.py                  # reads model_input.json by default, writes output/model_output_{application_id}_{timestamp}.json
python verify_model.py --input xxx.json # specify input
python backfill.py                      # CSV batch input, writes Excel report (output/classification_report_*.xlsx)
```

Note: the sample transactions in `model_input.json` are for demonstration only and do not cover all classification engine rules; for full validation use `input_converter.py` to extract a real application from `sample.csv` and generate the input.

## Entry points

Three entry scripts serve different purposes but run the **same core pipeline** — `ClassificationOrchestrator` loads the pipeline config + category catalog and runs all engines in priority order. They differ only in input/output shape:

| Script | Purpose | Input | Output |
| --- | --- | --- | --- |
| `model_main.py` | Production inference entry (deployed service, `PredictMain.predict`) | single-application JSON dict (either contract, `product`-dispatched) | JSON dict (`bscat_transactions` + `bscat_summaries` + `bscat_stats`) |
| `verify_model.py` | Local verification / trial run for a single application | single-application JSON file (default `model_input.json`) | JSON file (`output/model_output_{application_id}_{timestamp}.json`) |
| `backfill.py` | Batch classification of historical / offline data | CSV (multiple applications) | Excel report (`output/classification_report_{timestamp}.xlsx`) |

**Maintenance note (important):** the three entries differ only in input/output format; classification logic — engine priority order, overwrite rules, income / liability protection — lives in the shared core and behaves identically in all three. Make changes in the core or engine code, never per-entry: a change verified through one entry applies to all. Known difference: `backfill.py` reads CSV directly and does not go through the JSON contract layer (`build_input_frame` / `serialize_output`), so the bank-account metadata lookup (`account_type` / `bank` / `credit_limit`) applied by the JSON entries is not applied there — CSV input must carry these columns itself if the engines need them.

## Classification pipeline

10 classification engines run in ascending priority order; each engine matches all transactions line by line and later engines overwrite the classification of earlier ones:

| priority | engine | responsibility |
| --- | --- | --- |
| 1 | transfer | Internal / external transfer identification |
| 10 | initial | Initial classification via merchant knowledge base (merchant_kb.csv) and basic rules |
| 150 | dishonour | Dishonour identification |
| 180 | gambling | Gambling identification (keyword rules + migrated institution rows) |
| 200 | income | Income stream identification (Wages / Centrelink, etc.) |
| 300 | liability | Liability stream identification (loan / BNPL, etc.), skips transactions already classified as income |
| 400 | all_other_credit | Collects remaining credits, only processes credit rows |
| 500 | fee | Fee identification |
| 800 | rent | Rent identification (keyword rules + migrated institution rows) |
| 999 | catch_all | Fallback: only matches transactions not yet classified |

Engine rules are externalized as CSV files under each engine's `resources/` directory (liability_engine, transfer_engine, catch_all_engine, etc.); the pipeline configuration lives in `configs/pipeline.json` and the category catalog in `configs/category_catalog.json`. The `priority` column decides execution order only; two documented exceptions to the plain "later engine wins" rule apply (gambling wins are protected from income / liability, and the rent / gambling institution layers yield to fee / dishonour and to longer initial keywords) — see `CLAUDE.md` for the full overwrite rules.

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
