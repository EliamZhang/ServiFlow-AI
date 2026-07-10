# ServiFlow AI

ServiFlow classifies bank transactions into two production fields:
`counterparty` and `finv_category`. Engines execute in configured priority
order, so a transaction claimed by a higher-priority engine cannot be
classified again by a later engine.

## Unified pipeline

The default order is income, then liability. Configure it in
`configs/pipeline.json`; category ownership is defined in
`configs/category_catalog.json`.

```powershell
python -m serviflow `
  --input income_classification_engine/sample.csv `
  --output output/serviflow_report.xlsx
```

Optionally write row-level CSV output:

```powershell
python -m serviflow `
  --input income_classification_engine/sample.csv `
  --output output/serviflow_report.xlsx `
  --transactions-csv output/serviflow_transactions.csv
```

The workbook contains `transactions`, one summary sheet per engine, and
`run_summary`.

## Engine package convention

Both engines use the same public structure and API:

| Module | Responsibility |
| --- | --- |
| `cli.py` / `__main__.py` | Independent command-line entry point |
| `engine.py` | Implements the shared engine contract |
| `pipeline.py` | Returns `PipelineResult` from `run_pipeline()` |
| `domain/` | Classification rules, streams, and summary calculations |
| `presentation/` | Excel reporting and optional dashboard |
| `resources/` | External rule/configuration files when required |

## Independent engines

Each engine is a first-class package and can run independently.

Income:

```powershell
python -m income_classification_engine `
  --input income_classification_engine/sample.csv `
  --output income_classification_engine/output/income_report.xlsx
```

Liability:

```powershell
python -m liability_classification_engine
```

## Adding an engine

1. Implement the common engine interface in the engine package's `engine.py`.
2. Register its factory in `serviflow/registry.py`.
3. Add its categories to `configs/category_catalog.json`.
4. Add its priority to `configs/pipeline.json`.
5. Add contract and integration tests.

Engines return proposals. Only `ClassificationOrchestrator` commits the final
core fields, which enforces cross-engine priority and category ownership.

## Tests

```powershell
python -m unittest discover -s tests -v
```
