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

- `transactions`: unified row-level classifications;
- `income_summary`: accepted income classifications;
- `liability_summary`: accepted liability classifications; and
- `run_summary`: per-engine run statistics.

## Architecture

```text
main.py
`- ClassificationOrchestrator
   |- IncomeEngine
   |- LiabilityEngine
   `- classification_core.reporting.write_report
```

The engine packages contain only logic used by this flow: their shared engine
adapter, in-memory pipeline, domain rules, and required resources. They do not
provide separate CLIs or report/dashboard writers.

## Tests

```powershell
python -m unittest discover -s tests -v
```
