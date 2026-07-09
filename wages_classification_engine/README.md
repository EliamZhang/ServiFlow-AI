# Wages Classification Engine

Rule-based income and wages classification for bank transactions.

## Main CLI

Build the production Excel workbook:

```powershell
python model_main.py --input sample.csv --output output/income_report.xlsx
```

Build the full audit workbook:

```powershell
python model_main.py --input sample.csv --output output/income_report_full.xlsx --report-mode full
```

Optionally save row-level prediction CSV output:

```powershell
python model_main.py --input sample.csv --output output/income_report.xlsx --predictions-csv output/income_predictions.csv
```

## Report Modes

- `summary`: writes the published transaction sheet plus `income_summary`.
- `full`: writes `income_summary` plus income transaction audit detail and Centrelink payment subtypes.

## Python API

- `detect_income(...)` is the primary detection entry point.
- `detect_wages(...)` is retained as a backward-compatible wrapper.
- `build_income_workbook(...)` is the primary Excel report builder.
- `build_income_report(...)` is retained as a backward-compatible wrapper.

