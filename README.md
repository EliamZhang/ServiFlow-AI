# ServiFlow-AI

Australian bank transaction classification service: a multi-engine classification pipeline that classifies Illion raw bank transactions line by line and outputs summaries for downstream business use. Supports CSV batch input producing Excel reports and single-application JSON inference, deployed as a Python service.

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
| bank_accounts | array | Account list providing account metadata (account_type / bank / credit_limit) missing from transaction rows; metadata does not participate in classification and is only written to the `bankAccounts` output |
| illion_raw_transactions | array | Illion raw transactions; each must contain at least transaction_id, transaction_date, amount, dr_cr, text; when application_id is absent it is filled from the top-level applicationId |

##### Output example

```json
{
  "userId": 484579009,
  "applicationId": 2513560,
  "flowTime": "2026-07-05 23:52:48.0",
  "runId": "8f0d3c9e-7a1b-4c2d-9e3f-0a1b2c3d4e5f",
  "status": "success",
  "error": null,
  "stats": {
    "txnRawInputCnt": 881,
    "transactionDateMax": "2026-07-09"
  },
  "bankAccounts": [
    {
      "bankAccountId": 1042813323,
      "accountType": "transaction",
      "bank": "cba",
      "creditLimit": null
    }
  ],
  "transactions": [
    {
      "amount": -12.32,
      "balance": -126.28,
      "bankAccountId": 1042813323,
      "category": "Transport",
      "drCr": "debit",
      "illionTrxUuid": "0e63679d-ff62-5a55-bb5f-ae03d4dec068",
      "text": "UBER *TRIP HELP.UBER.C 14518236738 AUS",
      "thirdParty": "UBER",
      "transactionDate": "2026-02-05",
      "transactionId": 1423884392,
      "trxType": null,
      "applicationId": 2513560,
      "counterparty": "Uber",
      "finvCategory": "Transport",
      "streamId": null
    }
  ],
  "summaries": {
    "income_summary": [
      {
        "finvCategory": "Wages",
        "streamId": "salary_payg_001",
        "incomeCategory": "salary_payg",
        "bankAccountId": 459428115,
        "applicationId": 2513560,
        "counterparty": "DELIVERY SERVICE JOB",
        "transactionStartDate": "2026-02-13",
        "transactionEndDate": "2026-07-09",
        "status": "active",
        "transactionCount": 9,
        "totalIncomeAmount": 2072.44,
        "averageIncomeAmount": 230.2711111111111,
        "medianIncomeAmount": 224.8,
        "latestIncomeAmount": 225.8,
        "estimatedMonthlyIncome": 487.06666666666666,
        "frequency": "fortnightly",
        "frequencyDay": "Thursday",
        "predictedNextIncomeDate": "2026-07-23"
      }
    ],
    "liability_summary": [
      {
        "finvCategory": "Non SACC Loans",
        "streamId": "bnpl_003",
        "liabilityCategory": "Non SACC Loans",
        "bankAccountId": "1534823854",
        "applicationId": "2513560",
        "counterparty": "CBA StepPay",
        "transactionStartDate": "2026-02-07",
        "transactionEndDate": "2026-06-25",
        "status": "Closed",
        "fundedAmount": 0.0,
        "repaidAmount": 2817.35,
        "repaymentAmount": null,
        "recentFnRepayAmount": 0.0,
        "frequency": "fortnightly",
        "frequencyDay": "Wednesday",
        "predictedClosingDate": null
      }
    ],
    "category_summary": [
      {
        "finvCategory": "All Other Credits",
        "bankAccountId": 1534823854,
        "transactionStartDate": "2026-04-15",
        "transactionEndDate": "2026-04-15",
        "transactionCount": 1,
        "totalAmount": 10.0,
        "averageAmount": 10.0,
        "medianAmount": 10.0,
        "latestAmount": 10.0
      }
    ]
  }
}
```

Field description:

| Field | Type | Description |
| --- | --- | --- |
| runId | string | Unique ID of a single inference run (uuid4) |
| status | string | `success` / `failed`; on inference errors no exception is raised, instead `failed` + error message is returned with empty transactions / summaries |
| error | string/null | Failure reason |
| stats.txnRawInputCnt | int | Input transaction count |
| stats.transactionDateMax | string | Max transaction date in input |
| bankAccounts | array | Account metadata (bankAccountId / accountType / bank / creditLimit), not repeated at the transaction row level |
| transactions | array | Original transaction fields + classification results; the core new fields are `finvCategory` (fine-grained category), `counterparty` (counterparty name), `streamId` (income/liability stream id, null for rows not belonging to any stream), plus the `applicationId` echo |
| summaries | object | Summaries grouped by type: income_summary (income streams, incl. estimatedMonthlyIncome / predictedNextIncomeDate), liability_summary (liability streams, incl. fundedAmount / repaidAmount / predictedClosingDate), category_summary (aggregate stats by finvCategory) |

## Local run

```bash
pip install -r requirements.txt
python verify_model.py                  # reads model_input.json by default, writes output/model_output_{applicationId}_{timestamp}.json
python verify_model.py --input xxx.json # specify input
python backfill.py                      # CSV batch input, writes Excel report (output/classification_report_*.xlsx)
```

Note: the sample transactions in `model_input.json` are for demonstration only and do not cover all classification engine rules; for full validation use `input_converter.py` to extract a real application from `sample.csv` and generate the input.

## Classification pipeline

8 classification engines run in ascending priority order; each engine matches all transactions line by line and later engines overwrite the classification of earlier ones:

| priority | engine | responsibility |
| --- | --- | --- |
| 1 | initial | Initial classification via merchant knowledge base (merchant_kb.csv) and basic rules |
| 100 | transfer | Internal / external transfer identification |
| 150 | dishonour | Dishonour identification |
| 200 | income | Income stream identification (Wages / Centrelink, etc.) |
| 300 | liability | Liability stream identification (loan / BNPL, etc.), skips transactions already classified as income |
| 400 | all_other_credit | Collects remaining credits, only processes credit rows |
| 500 | fee | Fee identification |
| 999 | catch_all | Fallback: only matches transactions not yet classified |

Engine rules are externalized as CSV files under each engine's `resources/` directory (liability_engine, transfer_engine, catch_all_engine, etc.); the pipeline configuration lives in `configs/pipeline.json` and the category catalog in `configs/category_catalog.json`.

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
