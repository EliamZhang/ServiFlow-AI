# Transfer Classification Engine

This package is the transfer stage of the unified ServiFlow classification
pipeline. It is not a standalone application.

`classification_core.registry` creates `TransferEngine`, and the root
orchestrator uses its proposals and accepted `transfer_summary` artifact when
building the unified result.

The only runnable entry point and report writer live at the project root:

```powershell
python main.py
```

That command writes `output/classification_report.xlsx`.

## Package layout

```text
transfer_engine/
|- engine.py       # shared engine contract implementation
|- pipeline.py     # in-memory transfer classification flow
`- domain/         # classification and summary logic
```
