# ServiFlow-AI

交易分类流水线：多个分类引擎按优先级顺序执行，对交易逐行分类并输出 Excel 报告。

## 引擎覆盖行为

- 执行顺序由 `configs/pipeline.json` 中每台引擎的 `priority` 决定，**按 priority 升序执行**（见 `classification_core/config.py` 的 `PipelineConfig.enabled_engines`，按 priority 排序）。**不是**按 JSON 数组中的书写顺序。
- 所有引擎都看到全部交易（`candidates` 默认全量）。每个引擎跑完后，其结果在行级写入 `output`，**覆盖**该行已有的 `finv_category` 与 `counterparty`（成对覆盖），并记录 `classification_engine` / `classification_priority` 等字段。**后面的引擎永远赢**（覆盖写入点在 `classification_core/orchestrator.py` 的 `_commit`）。
- 特例：`liability` 引擎的候选集会排除已被分为收入类的交易（`salary_payg`、`salary_packaging`、`centrelink`、`self_employed_gig`），因此不会覆盖这些行（`orchestrator.py` 的 `run` 中）。
- `priority` 字段仅决定执行顺序，并记录进 `classification_priority` 列，不参与覆盖判定。

## 基线回归对比（output 变更检查）

- 每次改完代码，必须运行 `python baseline.py diff`（用 `.venv/Scripts/python.exe` 运行），将当前流水线输出与基线 `baseline/sample_baseline.csv` 对比，检查分类结果是否发生变更。
- 若 `diff` 报告有差异（exit 1），必须向用户说明每笔差异的原因（对应改动的规则/逻辑），不能只报告"有差异"就结束。
- 基线由 `python baseline.py save` 生成；仅当用户确认分类结果变更符合预期时，才允许重新保存基线。
- 注意：diff 输出含中文，Windows 控制台需设置 `PYTHONIOENCODING=utf-8`，或用 `.venv` 环境运行。

## 重要约定

**所有涉及引擎顺序调整的改动（新增引擎、删除引擎、调整 priority、启用/禁用引擎、修改覆盖规则），都必须先与用户确认后再实施。**
