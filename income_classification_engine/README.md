# Income Classification Engine

Rule-based income and wages classification for bank transactions.

The independent CLI reads the project root `sample.csv` by default.

## Package layout

```text
income_classification_engine/
├─ cli.py
├─ engine.py
├─ pipeline.py
├─ domain/
│  ├─ classification.py
│  └─ summary.py
└─ presentation/
   └─ reporting.py
```

## Main CLI

Build the production Excel workbook:

```powershell
python -m income_classification_engine `
  --output income_classification_engine/output/income_report.xlsx
```

Build the full audit workbook:

```powershell
python -m income_classification_engine `
  --output income_classification_engine/output/income_report_full.xlsx `
  --full
```

Optionally save row-level prediction CSV output:

```powershell
python -m income_classification_engine `
  --output income_classification_engine/output/income_report.xlsx `
  --predictions-csv income_classification_engine/output/income_predictions.csv
```

## Report modes

- Default: writes `transactions` plus `income_summary`.
- `--full`: writes `income_summary` plus income transaction audit detail and
  Centrelink payment subtypes.

## Python API

- `run_pipeline(...)` returns an explicit
  `PipelineResult` containing transactions, diagnostics, and original columns.
- `build_summary(...)` builds `income_summary`.
- `write_report(...)` writes the standalone Excel report.
- `IncomeEngine` implements the project-wide engine protocol directly.
