# ServiFlow-AI

交易分类流水线：多个分类引擎按优先级顺序执行，对交易逐行分类并输出 Excel 报告（`backfill.py`）或单应用 JSON（`verify_model.py`）。

## 引擎覆盖行为

- **执行顺序**：由 `configs/pipeline.json` 中每台引擎的 `priority` 决定，**按 priority 升序执行**（见 `classification_core/config.py` 的 `PipelineConfig.enabled_engines`），不是按 JSON 数组中的书写顺序。当前顺序（priority）：transfer(1) → initial(10) → dishonour(150) → gambling(180) → income(200) → liability(300) → all_other_credit(400) → fee(500) → rent(800) → catch_all(999)。
- **覆盖规则**：所有引擎都看到全部交易（`candidates` 默认全量）。每台引擎跑完后，其结果在行级写入 `output`，**覆盖**该行已有的 `finv_category` 与 `counterparty`（成对覆盖），并记录 `classification_engine` / `classification_priority` 等字段。**后面的引擎永远赢**（覆盖写入点：`classification_core/orchestrator.py` 的 `_commit`），唯一例外见**特例 5**：gambling 认领的行对 income / liability 不可覆盖。
- **特例 1（income 保护）**：`liability` 引擎的候选集会排除已被分为收入类的交易（`orchestrator.py` 的 `run`），因此不会覆盖这些行。排除判定基于**当前输出层的 `finv_category`**，即收入引擎映射后的粗类 `Wages` / `Centrelink`（`salary_payg` / `salary_packaging` / `self_employed_gig` 三者在 `income_engine/domain/classification.py` 的 `add_income_type_rules` 中统一映射为 `Wages`）。
- **特例 2（all_other_credit）**：`all_other_credit` 只处理 `dr_cr == credit` 的行，且跳过已被先前引擎分类的行——唯一例外是 `External Transfers` 允许被它重新匹配（`all_other_credit_engine/engine.py` 的 `classify`）。
- **特例 3（catch_all）**：`catch_all` 只匹配未被先前引擎分类的行（`catch_all_engine/engine.py` 的 `classify`）。
- **特例 4（rent）**：`rent` 引擎的候选集排除已被 income / liability 分类的行（`orchestrator.py` 的 `run`）。引擎内部分两层：机构层（`rent_rules.csv` 中 `source=institution` 的行，Aho-Corasick 匹配，counterparty=商户名）**跳过已被 fee / dishonour 认领的行**，且**跳过已被 initial 认领且其匹配关键词长度 ≥ 本次 rent 命中关键词长度的行**（从 initial 认领的 `classification_reason` 中解析 `keyword=K` 的长度）——这两条是为保持 2026-09 机构迁移前的赢家语义（当时机构在 initial(10)，跨类别最长关键词胜，fee/dishonour 可覆盖）；关键词层（`source=rule` 的行）仍可认领任何行。机构层命中时赢过关键词层（confidence 0.95 > 规则 ≤0.90），counterparty 用商户名。机构层关键词**不做静态过滤**（全量 21,794 机构原样迁入），跨类别冲突全部由运行时的长度比较解决。
- **特例 5（gambling）**：`gambling` 引擎 **priority 180**（dishonour 之后、income/liability 之前，2026-09 用户确认前移；原为 700 且候选集排除 income / liability，现顺序反转后该排除对 gambling 已是空操作，代码保留作防御性守卫——**若未来把 gambling 移回 income/liability 之后，必须重审该守卫与本特例的保护过滤**）。**gambling 认领的行对 income(200) / liability(300) 不可覆盖**（实现：`orchestrator.py` 的 `run` 中 validate 之后、`_archive_claims` 之前，按 key 丢弃 income/liability 对 `classification_engine=="gambling"` 行的预测——放在提交层而非候选集，因 income 的行为特征在全量行上计算，候选过滤会扭曲其余行特征）；fee / rent / catch_all 等其他后续引擎的 later-wins 语义不变（2026-09 全量样本实测：fee 双命中行仍归 fee，仅 fee 的 `unclassified_only` catchall 会向 gambling 让路）。机构层语义同 rent（跳过 fee / dishonour + initial 长度让路，共享代码 `classification_core/merchant_institution.py`；fee 分支在 180 时无 fee 认领、成空操作）；**关键词层与 rent 相反——排除所有先前引擎已认领的行**（`exclude_prior_claimed`，复刻迁移前 Gambling 规则在 catch_all(999) 时只处理未认领行的语义；180 时存在的先前认领只可能来自 transfer / initial / dishonour）。机构层命中时赢过关键词层，counterparty 用商户名；关键词层 counterparty 为 "-"。配置表 `gambling_engine/resources/gambling_rules.csv`（结构同 rent_rules.csv）：16 条规则（原 catch_all 的 POKIES/WAGERING/CASINO 等）+ 1,822 机构行（原 merchant_kb Gambling 类）原样迁入。
- `priority` 仅决定执行顺序，并记录进 `classification_priority` 列，不参与覆盖判定（除特例 5 的 gambling→income/liability 保护依赖 180 < 200/300 这一配置事实外）。

## 入口与数据流

- `backfill.py`：CSV 批量输入 → 运行流水线 → 输出 Excel 报告（默认 `output/classification_report_{YYYYMMDD_HHMMSS}.xlsx`，可用 `--output` 指定）。
- `verify_model.py`：单应用 JSON 输入（默认 `model_input.json`，可用 `--input` 指定）→ 运行流水线 → 输出 JSON（默认 `output/model_output_{applicationId}_{时间戳}.json`）。序列化时排除引擎内部追踪列（`classification_status` 等），账户元数据（`account_type` / `bank` / `credit_limit`）只保留在顶层 `bankAccounts`，行级不重复输出；字段名统一转 camelCase。本质是单应用验证/试跑脚本。
- `baseline.py`：基线回归对比（见下节）。
- `input_converter.py`：从 `sample.csv` 提取指定 application（交互式选择），转换为 `verify_model.py` 的 JSON 入参结构（`--input` / `--output-dir`）。

## 基线回归对比（output 变更检查）

- **每次改完代码必须运行** `python baseline.py diff`（用 `.venv/Scripts/python.exe`），将当前流水线输出与基线对比，检查分类结果是否变更。基线为**四层**，由 `save` 一次性生成，缺一不可：
  - `baseline/sample_baseline.csv`（最终输出层）：每笔交易 1 行，只记最终赢家结果（`finv_category` / `counterparty` / `classification_engine` / `stream_id`）。
  - `baseline/engine_claims.csv`（每引擎认领层）：每引擎每交易 1 行，记录各引擎的认领（含 `classification_rule_id` / `classification_reason` / `stream_id` / `priority`），无论该行最终被哪台引擎赢走。**只有它才能抓到"引擎改了逻辑但没当上最终赢家"的回归**（此类变更在最终层完全不可见）。数据来自 `orchestrator.py` 的 `_archive_claims`，挂在 `EngineExecution.claims` 上。
  - `baseline/run_meta.json`（配置/版本层）：引擎清单（engine_id / priority / enabled）、`on_engine_error`、各引擎 `engine_version`、`baseline_format_version`，以及**输入文件与全部规则资源的 SHA-256 指纹**（各引擎 `resources/` 下的 CSV 与 `initial_engine/merchant_kb.csv`）。用于抓取**行级对比看不到的变更**——例如只调换 priority 而结果恰好相同、引擎启停/增删、引擎版本号被改，或改动了规则 CSV/输入样本。
  - `baseline/summaries/`（汇总指标层）：只比对确定性指标——`category_summary.csv` 全列（最终输出层的分类聚合统计）、`liability_summary.csv` 仅金额列（`funded_amount` / `repaid_amount` / `repayment_amount` / `recent_fn_repay_amount`）。时间敏感字段（`status` / `predicted_closing_date` / `frequency` 等）**不纳入**，避免换样本或日期推移导致的噪音误报。两个工件都按键列（application_id / bank_account_id / finv_category / stream_id）对齐；`category_summary` 存在同键多行时保留首行（提取与加载两层都去重）。
- `diff` 输出差异：最终输出变化（CHANGED / NEW / GONE）+ 引擎认领变化（含 rule_id 计数增减）+ 汇总指标变化 + 配置/版本变化 + 行数检查（输入交易数或认领数增减）。以上任何一类有差异都 exit 1（含**规则认领计数变化**——改规则 CSV 后最常见的信号）。若 `diff` 报有差异，必须逐笔向用户说明差异原因（对应改动的规则/逻辑），不能只报告"有差异"就结束。
- 基线由 `python baseline.py save` 生成（会同时生成所有文件）。若任何基线工件已存在，命令会拒绝覆盖（exit 2）；仅当用户确认分类结果变更符合预期时，才允许用 `python baseline.py save --replace --reason "<确认原因>"` 明确重建。`diff` 会报告输入/规则文件的 SHA-256 指纹变化与 `baseline_format_version` 变化。
- 用 `.venv` 环境运行（`baseline.py` 输出已英文化，Windows 控制台无需再设置 `PYTHONIOENCODING`）。

## 重要约定

### 引擎相关改动需先确认

**所有涉及引擎顺序调整的改动（新增引擎、删除引擎、调整 priority、启用/禁用引擎、修改覆盖规则），都必须先与用户确认后再实施。**删除基线文件或修改 `configs/` 下配置文件同理，均属不可逆操作，需先确认。

### 规则数据外置

引擎的具体规则大多外置在各引擎 `resources/` 目录的 CSV 中（如 `liability_engine/resources/`、`transfer_engine/resources/`、`catch_all_engine/resources/`），引擎代码只负责加载与执行。`initial_engine/merchant_kb.csv` 为商户知识库（已跟踪进 git，非 gitignore），**不含 Rent 类**（2026-09 已迁至 `rent_engine/resources/rent_rules.csv`）**与 Gambling 类**（2026-09 已迁至 `gambling_engine/resources/gambling_rules.csv` 的 `source=institution` 行）。`rent_rules.csv` / `gambling_rules.csv` 结构相同，均混合两类行：`source=rule`（关键词/正则规则，`counterparty` 为空 → 输出 "-"）与 `source=institution`（机构行：`rule_name`=商户名、`pattern` 为竖线分隔的关键词、`confidence`=0.95、`counterparty`=商户名）；加新机构时在表尾追加即可，无需改引擎代码。两个引擎的机构层共用 `classification_core/merchant_institution.py`（加载/自动机构建/整词匹配/让路判定）。改规则优先改 CSV，避免动引擎代码；各 CSV 的列约定见对应引擎的 `domain/` 加载函数与 docstring。注意：改动这些资源文件会改变 SHA-256 指纹，`baseline.py diff` 会报告，但行级结果未必变。

### 双仓库发布流程（GitHub 开发 → GitLab 生产）

- **GitHub（https://github.com/EliamZhang/ServiFlow-AI.git）是个人开发仓库**，日常开发、分支、合并都在这里进行。
- **GitLab（http://git.ppdaicorp.com/intl/risk/AUS/Model/AUS_BS_CAT_Model.git）是线上生产环境**，发布目标分支为 `main` / `release-prep` / `staging`（三个分支通常保持同步）。
- **标准发布顺序：先在 GitHub 完成开发与合并，确认无误后再推送到 GitLab**。不要跳过 GitHub 直接把本地改动推上 GitLab。
- 推送 GitLab 属于影响生产的不可逆操作，**必须先将改动在 GitHub 侧定稿（合并到 main 或至少推上 staging）并经用户确认后**才能推送。

### 协作方式

- 在开始执行前，先检查问题中是否存在错误前提、逻辑跳跃、信息缺失或目标不明确；

* 保持独立判断，不盲目认同或迎合我的观点；
* 明确区分 **已确认事实、合理推测、主观判断和未知信息** ；
* 改代码过程中有存疑问题时，可以向我询问
* 涉及数字、人物、政策、时间及关键结论时，优先核实可靠来源，并说明依据；
* 发现我的判断可能有误时，直接指出，并说明理由、潜在风险及其他可能的解释；
* 主动告诉我可能忽略的变量、边界条件、隐性成本、偏差和长期影响；
* 存在多种方案时，说明各方案的优缺点，并给出明确建议，而不是只罗列选项；
* 信息不足时不要编造；可以基于合理假设继续，但必须明确标注假设；
* 默认先给出结论和建议，再补充分析过程与依据；
* 可以自主进行分析、修改和本地验证，但涉及**提交代码、推送远程仓库、发送消息、发布内容、删除数据或其他不可逆操作**时，必须先征得我的明确确认。
