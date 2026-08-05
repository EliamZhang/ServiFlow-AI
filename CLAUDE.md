# ServiFlow-AI

交易分类流水线：多个分类引擎按优先级顺序执行，对交易逐行分类并输出 Excel 报告（`backfill.py`）或单应用 JSON（`run_model.py`）。

## 引擎覆盖行为

- **执行顺序**：由 `configs/pipeline.json` 中每台引擎的 `priority` 决定，**按 priority 升序执行**（见 `classification_core/config.py` 的 `PipelineConfig.enabled_engines`），不是按 JSON 数组中的书写顺序。当前顺序（priority）：initial(1) → transfer(100) → dishonour(150) → income(200) → liability(300) → all_other_credit(400) → fee(500) → catch_all(999)。
- **覆盖规则**：所有引擎都看到全部交易（`candidates` 默认全量）。每台引擎跑完后，其结果在行级写入 `output`，**覆盖**该行已有的 `finv_category` 与 `counterparty`（成对覆盖），并记录 `classification_engine` / `classification_priority` 等字段。**后面的引擎永远赢**（覆盖写入点：`classification_core/orchestrator.py` 的 `_commit`）。
- **特例 1（income 保护）**：`liability` 引擎的候选集会排除已被分为收入类的交易（`orchestrator.py` 的 `run`），因此不会覆盖这些行。排除判定基于**当前输出层的 `finv_category`**，即收入引擎映射后的粗类 `Wages` / `Centrelink`（`salary_payg` / `salary_packaging` / `self_employed_gig` 三者在 `income_engine/domain/classification.py` 的 `add_income_type_rules` 中统一映射为 `Wages`）。
- **特例 2（all_other_credit）**：`all_other_credit` 只处理 `dr_cr == credit` 的行，且跳过已被先前引擎分类的行——唯一例外是 `External Transfers` 允许被它重新匹配（`all_other_credit_engine/engine.py` 的 `classify`）。
- **特例 3（catch_all）**：`catch_all` 只匹配未被先前引擎分类的行（`catch_all_engine/engine.py` 的 `classify`）。
- `priority` 仅决定执行顺序，并记录进 `classification_priority` 列，不参与覆盖判定。

## 入口与数据流

- `backfill.py`：CSV 批量输入 → 运行流水线 → 输出 Excel 报告（`output/classification_report_{时间戳}.xlsx`）。
- `run_model.py`：单应用 JSON 输入（`model_input.json`）→ 运行流水线 → 输出 JSON（默认 `output/model_output_{applicationId}_{时间戳}.json`）。序列化时排除引擎内部追踪列（`classification_status` 等），账户元数据（`account_type` / `bank` / `credit_limit`）只保留在顶层 `bankAccounts`，行级不重复输出；字段名统一转 camelCase。
- `baseline.py`：基线回归对比（见下节）。

## 基线回归对比（output 变更检查）

- **每次改完代码必须运行** `python baseline.py diff`（用 `.venv/Scripts/python.exe`），将当前流水线输出与基线对比，检查分类结果是否变更。基线为**双层**，由 `save` 一次性生成，缺一不可：
  - `baseline/sample_baseline.csv`（最终输出层）：每笔交易 1 行，只记最终赢家结果（`finv_category` / `counterparty` / `classification_engine` / `stream_id`）。
  - `baseline/engine_claims.csv`（每引擎认领层）：每引擎每交易 1 行，记录各引擎的认领（含 `classification_rule_id` / `stream_id` / `priority`），无论该行最终被哪台引擎赢走。**只有它才能抓到"引擎改了逻辑但没当上最终赢家"的回归**（此类变更在最终层完全不可见）。数据来自 `orchestrator.py` 的 `_archive_claims`，挂在 `EngineExecution.claims` 上。
- `diff` 输出两类差异：最终输出变化（CHANGED / NEW / GONE）+ 引擎认领变化（含 rule_id 计数增减）。若 `diff` 报有差异（exit 1），必须逐笔向用户说明差异原因（对应改动的规则/逻辑），不能只报告"有差异"就结束。
- 基线由 `python baseline.py save` 生成；仅当用户确认分类结果变更符合预期时，才允许重新保存基线。
- Windows 控制台运行 diff 需设置 `PYTHONIOENCODING=utf-8`（输出含中文），或用 `.venv` 环境运行。

## 重要约定

### 引擎相关改动需先确认

**所有涉及引擎顺序调整的改动（新增引擎、删除引擎、调整 priority、启用/禁用引擎、修改覆盖规则），都必须先与用户确认后再实施。**

### 规则数据外置

引擎的具体规则大多外置在各引擎 `resources/` 目录的 CSV 中（如 `liability_engine/resources/`、`transfer_engine/resources/`、`catch_all_engine/resources/`），引擎代码只负责加载与执行。改规则优先改 CSV，避免动引擎代码；各 CSV 的列约定见对应引擎的 `domain/` 加载函数与 docstring。

### 协作方式

- 先检查问题有没有错误前提、逻辑跳跃和信息缺失；
- 不要迎合我，要独立判断；
- 区分事实、推测和主观观点；
- 涉及数字、人物和结论时尽量核实来源；
- 不同意就直接指出，并给出依据、风险和替代解释；
- 主动提醒我忽略的变量、成本和偏差。
