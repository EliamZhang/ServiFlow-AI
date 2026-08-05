# ServiFlow-AI

交易分类流水线：多个分类引擎按优先级顺序执行，对交易逐行分类并输出分类结果与收入/负债汇总。支持三种调用方式：单应用 JSON 分类（`run_model.py`）、批量 CSV 回溯（`backfill.py`）、回归对比（`baseline.py`）。

## 环境准备

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

运行脚本均使用 `.venv/Scripts/python.exe`；输出含中文时，Windows 控制台需设置 `PYTHONIOENCODING=utf-8`。

## 调用方式

### 1. 单应用分类（推荐，JSON 入参/出参）

```bash
.venv/Scripts/python.exe run_model.py \
  --input model_input.json \
  --output output/model_output_xxx.json \
  [--config configs/pipeline.json] \
  [--category-catalog configs/category_catalog.json]
```

- `--input`：应用 JSON 路径，默认 `model_input.json`。
- `--output`：输出 JSON 路径；不指定时自动生成 `output/model_output_{applicationId}_{YYYYMMDD_HHMMSS}.json`。
- 成功时打印完整输出 JSON，并提示写入路径；失败时打印 `status: "failed"` 的 JSON 并以非零码退出。
- 样例见 [model_input.json](model_input.json) / [model_output.json](model_output.json)。

### 2. 批量回溯（CSV 入参 → Excel 报告）

```bash
.venv/Scripts/python.exe backfill.py \
  --input sample.csv \
  --output output/classification_report.xlsx \
  [--config configs/pipeline.json] \
  [--category-catalog configs/category_catalog.json]
```

### 3. 基线回归对比

```bash
.venv/Scripts/python.exe baseline.py save   # 生成双层基线
.venv/Scripts/python.exe baseline.py diff   # 对比当前输出与基线
```

`diff` 有差异时退出码为 1，需逐笔说明差异原因；仅当确认变更符合预期时才重新 `save`。

## 入参（单应用 JSON）

顶层结构：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `userId` | int | 否 | 用户 ID，原样回显到输出 |
| `applicationId` | int | 否 | 申请 ID，原样回显，并回填到每笔交易 |
| `bank_accounts` | array | 否 | 账户列表，用于回填账户信息 |
| `illion_raw_transactions` | array | **是** | 交易列表，须非空 |
| `illion_day_end_balances` | array | 否 | 日终余额（当前未消费，保留供扩展） |

`bank_accounts[]` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `bank_account_id` | int/string | 账户 ID，与交易中 `bank_account_id` 关联 |
| `account_type` | string | 账户类型，如 `transaction` / `savings` / `credit card` |
| `bank` | string | 银行标识，如 `cba` |
| `credit_limit` | number/null | 信用额度，仅信用卡有值 |

`illion_raw_transactions[]` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `amount` | number | 交易金额，负为支出、正为收入 |
| `transaction_date` | string | 交易日期 `YYYY-MM-DD` |
| `transaction_id` | int/string | 交易 ID |
| `bank_account_id` | int/string | 所属账户 ID |
| `dr_cr` | string | 借贷方向：`debit` / `credit` |
| `text` | string | 交易描述原文 |
| `third_party` | string | 第三方名称 |
| `trx_type` | string | 交易类型，如 `Inbound Transfer` |
| `category` | string | 原始分类（上游提供） |
| `balance` | number | 交易后余额 |
| `illion_trx_uuid` | string | 交易唯一标识 |

## 出参（单应用 JSON）

顶层结构：

```json
{
  "userId": 484579009,
  "applicationId": 2513560,
  "runId": "3f1c9a2b-...",
  "status": "success",
  "error": null,
  "bankAccounts": [...],
  "transactions": [...],
  "summaries": { "income_summary": [...], "liability_summary": [...] }
}
```

### 顶层字段

| 字段 | 说明 |
|---|---|
| `userId` / `applicationId` | 原样回显入参 |
| `runId` | 本次运行 ID（UUID） |
| `status` | `success` 或 `failed`；失败时 `error` 携带异常信息，`transactions` 为空数组 |
| `bankAccounts` | 账户列表（`bankAccountId` / `accountType` / `bank` / `creditLimit`） |

### transactions[]

入参交易字段（转驼峰命名，如 `bank_account_id` → `bankAccountId`），并追加分类结果：

| 字段 | 说明 |
|---|---|
| `finvCategory` | 最终分类类别；未分类时为 `null` |
| `counterparty` | 最终匹配的交易对手 |
| `classificationEngine` | 赢家引擎名称 |
| `classificationEngineVersion` | 引擎版本 |
| `classificationPriority` | 引擎执行优先级 |
| `classificationRuleId` | 命中规则 ID |
| `classificationReason` | 分类依据说明 |
| `streamId` | 归属的收入/负债流 ID |

常见 `finvCategory` 取值（完整清单见 `configs/category_catalog.json`）：

`Wages`、`External Transfers`、`Internal Transfer`、`Non SACC Loans`、`Dining Out`、`Groceries`、`Transport`、`Automotive`、`Dishonours`、`Fees`、`Rent`、`Gambling`、`Health`、`Retail`、`Department Stores`、`Education`、`Telecommunications`、`Information`、`Home Improvement`、`Personal Care`、`Credit Card Repayments`、`Overdrawn`、`All Other Credits` 等。

### summaries

- `income_summary[]`：收入流聚合，按 `streamId` 汇总，含 `incomeCategory`、`counterparty`、`transactionCount`、`totalIncomeAmount`、`averageIncomeAmount`、`medianIncomeAmount`、`latestIncomeAmount`、`estimatedMonthlyIncome`、`frequency`、`frequencyDay`、`predictedNextIncomeDate` 等。
- `liability_summary[]`：负债流聚合，含 `liabilityCategory`、`fundedAmount`、`repaidAmount`、`repaymentAmount`、`recentFnRepayAmount`、`predictedClosingDate` 等。

## 配置

- `configs/pipeline.json`：引擎清单与执行优先级。引擎按 `priority` 升序执行，后面的引擎赢（逐行覆盖 `finvCategory` / `counterparty`）；`liability` 引擎不覆盖收入类交易。
- `configs/category_catalog.json`：类别目录与类别归属。

## 目录结构

```
classification_core/   # 引擎与编排核心（config / orchestrator / models / reporting）
configs/               # pipeline.json、category_catalog.json
baseline/              # 双层回归基线（sample_baseline.csv、engine_claims.csv）
output/                # 默认输出目录
run_model.py           # 单应用 JSON 分类入口
backfill.py            # 批量 CSV 回溯入口
baseline.py            # 基线保存与回归对比
sample.csv             # 批量回溯样例数据
model_input.json       # 单应用入参样例（881 笔交易）
model_output.json      # 单应用出参样例
```
