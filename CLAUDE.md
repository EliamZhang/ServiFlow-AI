# ServiFlow-AI

交易分类流水线：多个分类引擎按优先级顺序执行，对交易逐行分类并输出 Excel 报告。

## 引擎覆盖行为

- 执行顺序由 `configs/pipeline.json` 中每台引擎的 `priority` 决定，**按 priority 升序执行**（见 `classification_core/config.py` 的 `PipelineConfig.enabled_engines`，按 priority 排序）。**不是**按 JSON 数组中的书写顺序。
- 所有引擎都看到全部交易（`candidates` 默认全量）。每个引擎跑完后，其结果在行级写入 `output`，**覆盖**该行已有的 `finv_category` 与 `counterparty`（成对覆盖），并记录 `classification_engine` / `classification_priority` 等字段。**后面的引擎永远赢**（覆盖写入点在 `classification_core/orchestrator.py` 的 `_commit`）。
- 特例：`liability` 引擎的候选集会排除已被分为收入类的交易（`salary_payg`、`salary_packaging`、`centrelink`、`self_employed_gig`），因此不会覆盖这些行（`orchestrator.py` 的 `run` 中）。
- `priority` 字段仅决定执行顺序，并记录进 `classification_priority` 列，不参与覆盖判定。

## 重要约定

**所有涉及引擎顺序调整的改动（新增引擎、删除引擎、调整 priority、启用/禁用引擎、修改覆盖规则），都必须先与用户确认后再实施。**
