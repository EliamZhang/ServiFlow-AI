# Wages Classification Engine

Rule-based income and wages classification for bank transactions.

## Main CLI

Build the production Excel workbook:

```powershell
python -m wages_classification_engine.model_main `
  --input wages_classification_engine/sample.csv `
  --output wages_classification_engine/output/income_report.xlsx
```

Build the full audit workbook:

```powershell
python -m wages_classification_engine.model_main `
  --input wages_classification_engine/sample.csv `
  --output wages_classification_engine/output/income_report_full.xlsx `
  --full
```

Optionally save row-level prediction CSV output:

```powershell
python -m wages_classification_engine.model_main `
  --input wages_classification_engine/sample.csv `
  --output wages_classification_engine/output/income_report.xlsx `
  --predictions-csv wages_classification_engine/output/income_predictions.csv
```

## Report Modes

- `summary`: writes the published transaction sheet plus `income_summary`.
- `full`: writes `income_summary` plus income transaction audit detail and Centrelink payment subtypes.

## Python API

- `classify_income_transactions(...)` returns an explicit
  `IncomeClassificationResult` containing transactions, summary, and original
  columns.
- `build_income_workbook(...)` is the primary Excel report builder.
- `IncomeEngine` implements the project-wide engine protocol directly.
