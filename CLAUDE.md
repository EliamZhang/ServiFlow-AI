# ServiFlow-AI

交易分类流水线：多个分类引擎按优先级顺序执行，对交易逐行分类并输出 Excel 报告。

## 引擎覆盖行为

- 执行顺序由 `configs/pipeline.json` 中每台引擎的 `priority` 决定，**按 priority 升序执行**（见 `classification_core/config.py` 的 `PipelineConfig.enabled_engines`，按 priority 排序）。**不是**按 JSON 数组中的书写顺序。
- 所有引擎都看到全部交易（`candidates` 默认全量）。每个引擎跑完后，其结果在行级写入 `output`，**覆盖**该行已有的 `finv_category` 与 `counterparty`（成对覆盖），并记录 `classification_engine` / `classification_priority` 等字段。**后面的引擎永远赢**（覆盖写入点在 `classification_core/orchestrator.py` 的 `_commit`）。
- 特例：`liability` 引擎的候选集会排除已被分为收入类的交易（`salary_payg`、`salary_packaging`、`centrelink`、`self_employed_gig`），因此不会覆盖这些行（`orchestrator.py` 的 `run` 中）。
- `priority` 字段仅决定执行顺序，并记录进 `classification_priority` 列，不参与覆盖判定。

## 基线回归对比（output 变更检查）

- 每次改完代码，必须运行 `python baseline.py diff`（用 `.venv/Scripts/python.exe` 运行），将当前流水线输出与基线对比，检查分类结果是否发生变更。基线为**双层**，由 `save` 一次性生成，缺一不可：
  - `baseline/sample_baseline.csv`（最终输出层）：每笔交易 1 行，只记最终赢家结果（`finv_category` / `counterparty` / `classification_engine` / `stream_id`）。
  - `baseline/engine_claims.csv`（每引擎认领层）：每引擎每交易 1 行，记录所有引擎各自的认领（含 `classification_rule_id` / `stream_id` / `priority`），无论该行最终被哪台引擎赢走。**只有它才能抓到"引擎改了逻辑但没当上最终赢家"的回归**（此类变更在最终层完全不可见）。数据来自 `orchestrator.py` 的 `_archive_claims`，挂在 `EngineExecution.claims` 上。
- `diff` 输出两类差异：最终输出变化（CHANGED/NEW/GONE）+ 引擎认领变化（含 rule_id 计数增减）。若 `diff` 报告有差异（exit 1），必须向用户说明每笔差异的原因（对应改动的规则/逻辑），不能只报告"有差异"就结束。
- 基线由 `python baseline.py save` 生成；仅当用户确认分类结果变更符合预期时，才允许重新保存基线。
- 注意：diff 输出含中文，Windows 控制台需设置 `PYTHONIOENCODING=utf-8`，或用 `.venv` 环境运行。

## 重要约定

**所有涉及引擎顺序调整的改动（新增引擎、删除引擎、调整 priority、启用/禁用引擎、修改覆盖规则），都必须先与用户确认后再实施。**
