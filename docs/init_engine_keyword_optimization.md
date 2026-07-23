# Init 引擎关键词匹配优化方案

## 一、现状诊断

### 1.1 总体数据

| 指标 | 值 |
|------|-----|
| KB 总行数 | 2,557,022 |
| KB 有 category 的行 | 0（全部为空） |
| 去重关键词数 | ~347 万 |
| 当前最低关键词长度 | 4 字符 |
| STOPWORDS 数量 | ~80 个 |
| Init 引擎 counterparty 准确率 | ~90%（法律实体名 vs 品牌名差异不计入错误） |

### 1.2 Init 引擎匹配流程

```
交易文本 text
  ↓ _CHANNEL_PREFIX_RE（去支付通道前缀）
  ↓ clean_text（大写 + [^A-Z0-9] 替换为空格 + 合并空格）
  ↓ ahocorasick 自动机全量扫描
  ↓ 全词边界检查
  ↓ purity × position 评分排序
  ↓ 取最高分 → 输出 counterparty 和 finv_category
```

### 1.3 匹配错误分布（~900 条真正错误 / 8,714 条 init 分类）

| 类型 | 占比 | 本质原因 |
|------|------|---------|
| **地名/区名误匹配** | ~60% | KB 关键词包含澳洲城市/区名，匹配到交易文本中的地址信息 |
| **通用词误匹配** | ~25% | KB 关键词包含过于宽泛的通用词（WEST、CITY、STATE 等） |
| **STOPWORDS 覆盖缺口** | ~10% | 大量地名和行业通用词未被屏蔽 |
| **短关键词歧义** | ~5% | 4 字符关键词太短，容易与其他词部分匹配 |

---

## 二、问题关键词分析

### 2.1 地名/区名关键词（最大元凶）

澳洲交易文本格式通常为 `商户名 + 区名 + AUS`，KB 中若将区名作为关键词，会无差别匹配所有在该区消费的交易。

| 问题商户 | 问题关键词 | 误匹配到的交易 | 错误数 |
|---------|-----------|---------------|--------|
| Reho Travel Pty. Limited | **Melbourne** | Netflix.com Melbourne, McDonald's Melbourne, Jetstar Melbourne... | 199 |
| JLM HOME LOANS PTY LTD | **ELEVEN**, **FINANCIAL** | 7-Eleven Spalding, 各种金融交易 | 145 |
| PARKINSON PTY. LTD. | **PARKINSON** | Coles Parkinson（Parkinson 是 QLD 区名） | 104 |
| STANFIELD RE PTY LTD | **VINCENTIA** | Coles Vincentia, EG Group Vincentia | 49 |
| PROPERTY CITI PTY. LTD. | **GREEN VALLEY** | Green Valley Hotel, McDonald's Green Valley | 27 |
| PURE REAL ESTATE GROUP | **IPSWICH** | McDonald's Ipswich, Rebel Ipswich | 7 |
| HIGHRISED PTY LTD | **CORRIMAL, SHELLHARBOUR, DAPTO** | Coles Corrimal, TK Maxx Shellharbour | 10 |
| NORTH COAST REAL ESTATE | **YAMBA, ILUKA** | Coles Yamba, Priceline Yamba | 14 |
| APOLLO REAL ESTATE | **Lidcombe** | KFC Lidcombe | 3 |
| 366 DARLING STREET | **Town Hall** | MYKI TOWN HALL（火车站名） | 4 |
| REAL ES PTY LTD | **City, Kirwan** | Fish City Aquarium | 8 |
| THE AGENCY SALES NSW | 大量地名关键词 | Coles, McDonald's, TK Maxx 等 | 11 |
| BIG BEACH PTY LTD | **&White** | White's IGA, TerryWhite | 10 |
| PERTH AIRPORT PTY LTD | **PERTH AIRPORT** | Perth Airport McDonald's, Parking | 2 |
| FREMANTLEMEDIA | **SPRING** | Jordan Spring, Spring mountain | 3 |

**核心模式**：房产中介公司的 KB 关键词大量包含其服务区域的地名，导致该区域所有交易都被吸入。

### 2.2 通用词关键词

| 问题商户 | 问题关键词 | 为什么有毒 |
|---------|-----------|-----------|
| THE WEST GROUP | **WEST** | 匹配 Westfield、Westpac、Western... |
| METRO TASMANIA | **METRO** | 匹配 WW METRO（Woolworths Metro） |
| STATE PTY LTD | **STATE** | 匹配 State High School、State Emergency... |
| SYSTEMS PTY LTD | **SYSTEMS** | 匹配 CANTALOUPE SYSTEMS、各种 Systems 后缀 |
| JLM HOME LOANS | **FINANCIAL** | 匹配一切金融交易文本 |
| JLM HOME LOANS | **ELEVEN** | 匹配 7-Eleven |
| THE WEST GROUP | **WEST APP, WEST SUPPER CLUB** | WEST 前缀过于宽泛 |

### 2.3 STOPWORDS 当前覆盖 vs 实际需要

当前 STOPWORDS（80 个）主要屏蔽了：
- 支付通道词：CARD、VISA、EFTPOS、BPAY、OSKO...
- 交易类型词：PAYMENT、PURCHASE、TRANSFER、DEPOSIT...
- 费用/利息词：FEE、INTEREST、OVERDRAWN...
- 通用公司后缀：LIMITED、GROUP、HOLDINGS...

**缺失的类别**：

```
1. 澳洲主要城市/区名（~100+ 个）
2. 商业通用词：REAL、ESTATE、PROPERTY、FINANCIAL、HOME、LOANS、AIRPORT...
3. 方位词：WEST、EAST、NORTH、SOUTH、CITY、CENTRAL...
4. 基础设施词：STATION、STREET、ROAD、PARK、PLAZA、MALL...
```

### 2.4 短关键词问题

| 长度 | 唯一关键词数 | 歧义词（>1 商户共享） | 歧义率 |
|------|------------|---------------------|--------|
| 4 字符 | 3,588 | ~100 | 2.8% |
| 5 字符 | 6,633 | ~60 | 0.9% |
| 6 字符 | 11,325 | ~29 | 0.3% |
| 7+ 字符 | 3,455,433 | 少量 | <0.1% |

短关键词本身歧义率很低（<3%），但**长度短的关键词若恰好是地名或通用词则危害极大**（如 `WEST`、`CITY`、`YAMBA`）。

---

## 三、优化方案

### 方案一：扩充 STOPWORDS（P0 🔴）

**目标**：屏蔽地名和通用词，从源头阻止其进入自动机。

**预期收益**：减少 ~800 条匹配错误（80%）

**风险**：极低。STOPWORDS 只过滤**单 token 的独立关键词**，多词短语中包含这些词不受影响（如 `GREEN VALLEY HOTEL` 不会被过滤）。

#### 新增 STOPWORDS 清单

```python
# ── 澳洲主要城市 ──
"SYDNEY", "MELBOURNE", "PERTH", "BRISBANE", "ADELAIDE",
"HOBART", "DARWIN", "CANBERRA", "GOLD COAST", "NEWCASTLE",

# ── 常见区/镇名（交易文本中高频出现） ──
"IPSWICH", "TOOWOOMBA", "CAIRNS", "BALLARAT", "BENDIGO",
"ALBURY", "DUBBO", "ORANGE", "PENRITH", "CAMPBELLTOWN",
"LIVERPOOL", "PARRAMATTA", "CHATSWOOD", "HURSTVILLE",
"BANKSTOWN", "BLACKTOWN", "FAIRFIELD", "CABRAMATTA",
"WOLLONGONG", "DAPTO", "CORRIMAL", "SHELLHARBOUR", "FIGTREE",
"NOWRA", "BATEMANS", "BEGA", "COOMA", "GOULBURN",
"MORUYA", "YASS", "COWRA", "FORBES", "PARKES",
"BROKEN", "GRIFFITH", "LEETON", "NARRANDERA", "WAGGA",
"ALBURY", "WODONGA", "SHEPPARTON", "WANGARATTA", "BENALLA",
"ECHUCA", "SWAN", "MILDURA", "HORSHAM", "ARARAT",
"BAIRNSDALE", "SALE", "TRARALGON", "WARRAGUL", "MOE",
"MORWELL", "DANDENONG", "FRANKSTON", "CRANBOURNE", "BERWICK",
"PAKENHAM", "MORNINGTON", "ROSEDALE", "SUNBURY", "MELTON",
"WERRIBEE", "GEELONG", "TORQUAY", "COLAC", "WARRNAMBOOL",
"HAMILTON", "PORTLAND", "BALLARAT", "CASTLEMAINE", "KYNETON",
"SUNSHINE", "BROADMEADOWS", "CRAIGIEBURN", "EPPING", "BUNDOORA",
"HEIDELBERG", "DONCASTER", "RINGWOOD", "BOX", "BOROONDARA",
"MOONEE", "ESSENDON", "BRUNSWICK", "COBURG", "PRESTON",
"RESERVOIR", "THOMASTOWN", "LALOR", "JACANA", "GLENROY",
"OAKLEIGH", "CLAYTON", "SPRINGVALE", "DINGLEY", "MORDIALLOC",
"MENTONE", "SANDRINGHAM", "BRIGHTON", "ST", "ELSTERNWICK",
"CAULFIELD", "MALVERN", "ARMADALE", "TOORAK", "PRAHRAN",
"SOUTH", "PORT", "ALBERT", "FOOTSCRAY", "WILLIAMSTOWN",
"ASCOT", "HAMILTON", "HENDRA", "CLAYFIELD", "ALBION",
"LUTWYCHE", "CHERMSIDE", "ASPLEY", "ZILLMERE", "GEEBUNG",
"STRATHPINE", "PETRIE", "KALLANGUR", "CABOOLTURE", "MORAYFIELD",
"BURPENGARY", "DECEPTION", "NORTH", "REDCLIFFE", "MARGATE",
"SCARBOROUGH", "WOODY", "CLONTARF", "SANDGATE", "BRACKEN",
"BALD", "FITZGIBBON", "TAIGUM", "BOONDALL", "NUDGEE",
"BANYO", "VIRGINIA", "NUNDAH", "TOOMBUL", "WAVELL",
"KEDRON", "GORDON", "EVERTON", "MCDOWALL", "BRIDGEMAN",
"ALBANY", "CANNING", "FREMANTLE", "JOONDALUP", "MANDURAH",
"MIDLAND", "ROCKINGHAM", "ARMADALE", "GOSNELLS", "KALAMUNDA",
"BELMONT", "VICTORIA", "KEWDALE", "CLOVERDALE", "REDCLIFFE",
"BURSWOOD", "RIVERVALE", "MAYLANDS", "BASSENDEAN", "GUILDFORD",
"MIDVALE", "ELLENBROOK", "AVON", "SWAN", "KALGOORLIE",
"BUNBURY", "BUSSELTON", "GERALDTON", "CARNARVON", "PORT",
"BROOME", "KUNUNURRA", "EAST", "SOUTH", "NORAM",
"LAUNCESTON", "DEVONPORT", "BURNIE", "KINGSTON", "GLENORCHY",
"CLAREMONT", "MOONAH", "LINDISFARNE", "HOWRAH", "SORELL",
"NEW", "ROSETTA", "BERRIEDALE", "CHIGWELL", "MONTROSE",

# ── 商业通用词 ──
"REAL", "ESTATE", "PROPERTY", "FINANCIAL", "FINANCE",
"LOAN", "LOANS", "HOME", "HOMES", "RENTAL",
"RENTALS", "AGENCY", "AGENT", "INVESTMENT", "INVESTMENTS",
"VENTURES", "ENTERPRISE", "ENTERPRISES", "TRADING",
"HOLDING", "MANAGEMENT", "SOLUTIONS", "CONSULTING",
"CONSULTANCY", "CONSULTANTS", "ASSOCIATES", "PARTNERS",
"DISTRIBUTORS", "WHOLESALE", "SUPPLIES", "SUPPLIERS",

# ── 方位/基础设施词 ──
"WEST", "EAST", "NORTH", "SOUTH", "CENTRAL", "CITY",
"TOWN", "VALLEY", "BAY", "BEACH", "PARK",
"STREET", "ROAD", "STATION", "SQUARE", "CENTRE",
"CENTER", "PLAZA", "MALL", "AIRPORT", "HARBOUR",
"HARBOR", "BRIDGE", "HILL", "HILLS", "MOUNT",
"LAKE", "RIVER", "COAST", "POINT", "CREEK",
"ISLAND", "GARDEN", "GARDENS", "HEIGHTS",
"JUNCTION", "CROSSING", "GATE", "GATES",
"VILLAGE", "GROVE", "GLEN", "DALE",
"WOOD", "WOODS", "FIELD", "FIELDS",
"MEADOW", "MEADOWS", "GREEN", "SPRINGS",
```

### 方案二：提高最低关键词长度（P1 🟡）

`_MIN_KEYWORD_LEN` 从 `4` → `5`。

**预期收益**：减少 ~200 条匹配错误

**风险**：低。会丢失 ~3,600 个 4 字符关键词，但其中品牌名（如 `NIKE`、`IKEA`、`CUE`）可通过 KB 清洗恢复（改为多词短语如 `NIKE STORE`、`IKEA FURNITURE`）。

### 方案三：关键词位置惩罚（P2 🟡）

在 purity × position 评分中，对匹配位置靠后的关键词施加额外惩罚。

```
当前: score = purity × (1 - pos / text_len)

优化: if pos > text_len * 0.3:
          score *= 0.5    # 后半段匹配，大概率是地址信息
```

**预期收益**：减少 ~100 条匹配错误

**风险**：中。可能压低一些长商户名中的后半段关键词。

### 方案四：purity 公式优化（P3/实验性）

加入关键词长度因子，让短关键词天然处于劣势：

```
当前:  purity = log(total / merchants_sharing_kw) × completeness

优化:  purity = log(total / merchants_sharing_kw) × completeness × min(1.0, len(kw) / 8)
```

**效果**：4 字符关键词 purity 自动折半，8 字符以上不受影响。

**预期收益**：减少 ~50 条匹配错误

### 方案五：KB 关键词数据清洗（P2/P3）

对 KB 做自动化审核，标记并处理问题关键词：

1. **地名审计**：关键词 ∈ 澳洲地名表 → 低置信度标记
2. **通用词审计**：关键词 ∈ 商业通用词表 → 低置信度标记
3. **长度审计**：单 token 且 ≤ 4 字符 → 人工审核标记
4. **串标审计**：同一个关键词被 >10 家商户共享 → 降权

---

## 四、实施建议

| 优先级 | 方案 | 预计减少错误 | 实施难度 | 副作用 |
|--------|------|------------|---------|--------|
| **P0** | 扩充 STOPWORDS | ~800 (80%) | 低（新增 ~200 行配置） | 几乎为零 |
| **P1** | MIN_KEYWORD_LEN 4→5 | ~200 (20%) | 低（改 1 行代码） | 少量短品牌名需清洗 |
| **P2** | 关键词位置惩罚 | ~100 (10%) | 中（改评分逻辑） | 需要调参 |
| **P3** | purity 公式优化 | ~50 (5%) | 中（改评分逻辑） | 需要 AB 测试 |
| **P3** | KB 数据清洗 | 长期收益 | 高（需要审计 255 万行） | 无 |

**P0 + P1 合计预期**：将 KB 映射错误从 ~900 条降到 ~100 条以内，init 引擎 counterparty 总体错误率从 10% 降到 <2%。

---

## 五、附录：问题商户完整清单

以下 427 个商户中，"真正错误"约 60 个（标 ✅），其余为 franchisee/法律实体名差异（非错误）。

### 严重错误（>20 条）

| # | 商户 | 错误数 | 问题关键词 | illion 正确值 |
|---|------|--------|-----------|-------------|
| 1 | Reho Travel Pty. Limited | 199 | Melbourne | Netflix, Sportsbet, McDonald's, DoorDash... |
| 2 | JLM HOME LOANS PTY LTD | 145 | ELEVEN, FINANCIAL | 7-Eleven |
| 3 | PARKINSON PTY. LTD. | 104 | PARKINSON | Coles |
| 4 | STANFIELD RE PTY LTD | 49 | VINCENTIA | Coles, EG Group |
| 5 | PROPERTY CITI PTY. LTD. | 27 | GREEN VALLEY | Hotels, McDonald's, Coles |
| 6 | JACK WESTLAND COSMETIC INJECTOR | 24 | 未知 | LOVISA, Ticketmaster, Grill'd |
| 7 | THE WEST GROUP PTY LTD | 24 | WEST | Aldi, Hotels, Parking |

### 中等错误（5-20 条）

| # | 商户 | 错误数 |
|---|------|--------|
| 8 | NORTH COAST REAL ESTATE PTY LTD | 14 |
| 9 | THE AGENCY SALES NSW PTY LTD | 11 |
| 10 | HIGHRISED PTY LTD | 10 |
| 11 | BIG BEACH PTY LTD | 10 |
| 12 | P.V.C. INVESTMENTS PTY LTD | 10 |
| 13 | REAL ES PTY LTD | 8 |
| 14 | PURE REAL ESTATE GROUP PTY LTD | 7 |
| 15 | BLUE MOUNTAINS NEPEAN REAL ESTATE | 4 |
| 16 | 366 DARLING STREET PTY LTD | 4 |
| 17 | FREMANTLEMEDIA AUSTRALIA PTY LTD | 3 |
| 18 | APOLLO REAL ESTATE PTY LTD | 3 |

### 匹配规则问题（purity 失衡）

| # | 商户 | 错误数 | 问题关键词 |
|---|------|--------|-----------|
| 19 | METRO TASMANIA PTY LTD | 28 | METRO |
| 20 | STATE PTY LTD | 3 | STATE |
| 21 | SYSTEMS PTY LTD | 29 | SYSTEMS |
| 22 | SALES PTY LTD | 11 | SALES |

---

*生成日期：2026-07-23*
