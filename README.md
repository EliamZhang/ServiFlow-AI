# ServiFlow AI

ServiFlow runs one unified transaction-classification pipeline and produces one
Excel deliverable: `output/classification_report.xlsx`.

Income runs before liability by default. A transaction claimed by an earlier
engine cannot be classified again by a later engine. Engine order is configured
in `configs/pipeline.json`, and category ownership is configured in
`configs/category_catalog.json`.

## Run

```powershell
python main.py
```

Optional paths can be supplied to the same unified entry point:

```powershell
python main.py `
  --input another_input.csv `
  --output output/classification_report.xlsx `
  --config configs/pipeline.json `
  --category-catalog configs/category_catalog.json
```

The workbook contains:

- `transactions`: income and liability transaction flows with unified row-level
  classifications;
- `income_summary`: accepted income classifications; and
- `liability_summary`: accepted liability classifications.

## Architecture

The pipeline runs 7 engines in priority order (configured in `configs/pipeline.json`):

| Priority | Engine | Classification |
|----------|--------|---------------|
| 1 | `initial` | Merchant keyword matching (Aho-Corasick, 2.5M-row KB) |
| 10 | `dishonour` | Dishonoured / reversed transactions |
| 100 | `transfer` | Internal & external transfers |
| 200 | `income` | Salary, Centrelink, gig income |
| 300 | `liability` | BNPL, loans, credit cards, debt collection |
| 400 | `all_other_credit` | Catch-all for remaining credits |
| 500 | `fee` | Overdrawn fees, ATM fees, interest charges |

```text
main.py
`- ClassificationOrchestrator
   |- InitialClassificationEngine
   |- DishonourEngine
   |- TransferEngine
   |- IncomeEngine
   |- LiabilityEngine
   |- AllOtherCreditEngine
   |- FeeEngine
   `- classification_core.reporting.write_report
```

### Matching rules

Each engine stores its matching rules as **CSV files** under `resources/`, following
the liability engine pattern:

- **fee**: `resources/fee_classification_rules.csv` (~80 regex rules)
- **income**: `resources/income_pattern_rules.csv` (~110 patterns) + `income_config.csv`
- **transfer**: 6 CSV files for external/internal transfer regex, indicator patterns, exclusions
- **liability**: 8 CSV files (~700 rules) covering counterparty, credit cards, loans, debt collection
- **initial**: `merchant_kb.csv` (~2.5M rows, 233 MB) — merchant keyword database

Rules support `keyword` (case-insensitive substring) and `regex` match types, ordered
by priority. Domain modules load from CSV at import time with lazy caching.

## Tests

```powershell
python -m unittest discover -s tests -v
```
