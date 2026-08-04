# Liability Classification Engine

This package is the liability stage of the unified ServiFlow classification
pipeline. It is not a standalone application.

`classification_core.registry` creates `LiabilityEngine`, then the root
orchestrator passes it only transactions that remain unclaimed by earlier
engines. The engine:

1. applies counterparty, credit-card, dishonour, and special rules;
2. identifies liability streams and categories;
3. returns classification proposals to the orchestrator; and
4. builds the accepted `liability_summary` artifact.

The only runnable entry point and report writer live at the project root:

```powershell
python backfill.py
```

That command writes `output/classification_report.xlsx`.

## Package layout

```text
liability_engine/
|- engine.py       # shared engine contract implementation
|- pipeline.py     # in-memory liability classification flow
|- domain/         # rules, stream identification, and summary logic
`- resources/      # rule tables and summary assumptions
```
