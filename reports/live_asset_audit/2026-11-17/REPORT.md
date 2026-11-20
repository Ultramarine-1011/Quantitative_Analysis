# `codex/frontend-qa` 全资产类别真实行情批量测试报告 (2026-11-17)

## 一、测试环境与口径

- **分支 / 提交**：`codex/frontend-qa` (`a322f3e`)
- **测试窗口**：`2025-11-17 ~ 2026-11-16`（默认货币 `LOCAL`，全程使用 `quant fetch --refresh` 绕过本地缓存）
- **执行时间**：`2026-11-18T16:24:10.156011+00:00` ~ `2026-11-18T16:26:40.472907+00:00`（总耗时 `74.3s`）
- **运行环境**：Python `3.14.5` | AKShare `1.20.46` | CCXT `4.5.92` | pandas `3.1.1` | PyArrow `25.0.1`
- **判定规则**：`PASS`（抓取成功、无 OHLC/缺失/覆盖/币种异常且无降级告警）；`WARN`（抓取成功但存在质量或回退告警）；`FAIL`（抓取失败或违反硬数据契约）；`UNVERIFIED`（未验证）。

## 二、总体统计与 13 类资产成功率/可用率

- **独立标的总数**：**65**（13 类资产）
- **PASS**：**47 / 65**（**72.31%**）
- **WARN**：**18 / 65**
- **FAIL**：**0 / 65**
- **UNVERIFIED**：**0 / 65**
- **可用率 (`PASS + WARN`)**：**65 / 65**（**100.00%**）

| 资产类别 | 标的数量 | PASS | WARN | FAIL | UNVERIFIED | PASS比例 | 可用率(PASS+WARN) | 平均耗时(s) | OHLC异常行 | 回退次数 | 实际数据源 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `etf` | 5 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 0.89 | 0 | 0 | akshare.stock_zh_a_hist_tx |
| `ashare` | 5 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 0.90 | 0 | 0 | akshare.stock_zh_a_hist_tx |
| `mutual_fund` | 5 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 0.97 | 0 | 0 | akshare.fund_open_fund_info_em |
| `gold` | 5 | 0 | 5 | 0 | 0 | 0.0% | 100.0% | 0.69 | 42 | 0 | akshare.futures_foreign_hist_sina |
| `crypto` | 5 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 2.00 | 0 | 0 | ccxt.binance |
| `us_stock` | 5 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 0.83 | 0 | 0 | akshare.stock_us_daily_sina |
| `hk_stock` | 5 | 4 | 1 | 0 | 0 | 80.0% | 100.0% | 1.09 | 2 | 0 | akshare.stock_hk_daily_sina |
| `global_index` | 6 | 3 | 3 | 0 | 0 | 50.0% | 100.0% | 0.76 | 5 | 1 | akshare.index_global_hist_sina; akshare.index_hk_daily_sina; akshare.index_us_daily_sina; sina_forex_direct |
| `bond_cn` | 5 | 1 | 4 | 0 | 0 | 20.0% | 100.0% | 0.74 | 0 | 0 | akshare.bond_zh_hs_daily_sina |
| `bond_us_yield` | 4 | 4 | 0 | 0 | 0 | 100.0% | 100.0% | 0.77 | 0 | 0 | akshare.bond_zh_us_yield_em |
| `commodity` | 5 | 0 | 5 | 0 | 0 | 0.0% | 100.0% | 0.70 | 85 | 0 | akshare.futures_foreign_hist_sina |
| `creit` | 5 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 0.88 | 0 | 0 | akshare.stock_zh_a_hist_tx |
| `fx` | 5 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 1.17 | 0 | 0 | eastmoney_forex_direct |
| **合计 (TOTAL)** | **65** | **47** | **18** | **0** | **0** | **72.31%** | **100.00%** | **0.95** | **134** | **1** | — |

## 三、数据源稳定性、回退、异常与限流统计

| 数据源 (`data_source`) | 命中次数 | PASS | WARN | FAIL | 触发回退数 | 限流/断连信号 | 平均耗时(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `akshare.stock_zh_a_hist_tx` | 20 | 20 | 0 | 0 | 0 | 0 | 0.98 |
| `akshare.futures_foreign_hist_sina` | 11 | 0 | 11 | 0 | 0 | 0 | 0.69 |
| `akshare.stock_hk_daily_sina` | 6 | 4 | 2 | 0 | 0 | 0 | 1.09 |
| `akshare.bond_zh_hs_daily_sina` | 5 | 1 | 4 | 0 | 0 | 0 | 0.74 |
| `akshare.bond_zh_us_yield_em` | 5 | 5 | 0 | 0 | 0 | 0 | 0.81 |
| `akshare.fund_open_fund_info_em` | 5 | 5 | 0 | 0 | 0 | 0 | 0.97 |
| `akshare.stock_us_daily_sina` | 5 | 5 | 0 | 0 | 0 | 0 | 0.83 |
| `ccxt.binance` | 5 | 5 | 0 | 0 | 0 | 0 | 2.00 |
| `eastmoney_forex_direct` | 5 | 5 | 0 | 0 | 0 | 0 | 1.17 |
| `akshare.index_us_daily_sina` | 4 | 4 | 0 | 0 | 0 | 0 | 0.68 |
| `sina_forex_direct` | 3 | 0 | 3 | 0 | 3 | 3 | 1.28 |
| `akshare.index_global_hist_sina` | 1 | 0 | 1 | 0 | 0 | 0 | 0.69 |
| `akshare.index_hk_daily_sina` | 1 | 0 | 1 | 0 | 0 | 0 | 0.71 |

- **失败错误分类统计（全量 76 次抓取含别名）**：无
- **限流/断连/超时信号统计**：`push2_rate_limit`: 3
- **触发多源回退的标的（3 项）**：`global_index:DXY` -> `sina_forex_direct`, `fx:USD/JPY` -> `sina_forex_direct`, `fx:USD/CNY` -> `sina_forex_direct`

## 四、OHLC 异常最严重的标的及具体异常样本

| 类别 | 代码 | 数据源 | 总行数 | 异常行数 | 异常占比 |
| --- | --- | --- | ---: | ---: | ---: |
| `commodity` | `CL` | `akshare.futures_foreign_hist_sina` | 258 | 31 | 12.02% |
| `commodity` | `NG` | `akshare.futures_foreign_hist_sina` | 258 | 25 | 9.69% |
| `gold` | `SI` | `akshare.futures_foreign_hist_sina` | 258 | 18 | 6.98% |
| `gold` | `GC` | `akshare.futures_foreign_hist_sina` | 258 | 17 | 6.59% |
| `commodity` | `HG` | `akshare.futures_foreign_hist_sina` | 258 | 12 | 4.65% |
| `commodity` | `OIL` | `akshare.futures_foreign_hist_sina` | 258 | 10 | 3.88% |
| `gold` | `XPT` | `akshare.futures_foreign_hist_sina` | 258 | 7 | 2.71% |
| `commodity` | `ZSD` | `akshare.futures_foreign_hist_sina` | 253 | 7 | 2.77% |
| `global_index` | `HSI` | `akshare.index_hk_daily_sina` | 246 | 5 | 2.03% |
| `hk_stock` | `00700` | `akshare.stock_hk_daily_sina` | 246 | 2 | 0.81% |

### `commodity:CL` (`akshare.futures_foreign_hist_sina`, 共 31 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-12-01 | 58.48 | 59.97 | 58.83 | 59.5 | 0 | `low_gt_open` |
| 2025-12-17 | 55.07 | 56.81 | 55.08 | 56.73 | 0 | `low_gt_open` |
| 2025-12-22 | 56.53 | 58.13 | 56.6 | 57.95 | 0 | `low_gt_open` |
| 2026-02-02 | 65.74 | 64.74 | 61.39 | 62.32 | 0 | `high_lt_open` |
| 2026-03-02 | 67.29 | 75.33 | 69.2 | 70.99 | 0 | `low_gt_open` |

### `commodity:NG` (`akshare.futures_foreign_hist_sina`, 共 25 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-11-24 | 3.88 | 3.838 | 3.77 | 3.772 | 0 | `high_lt_open` |
| 2025-12-08 | 4.234 | 4.177 | 3.82 | 3.821 | 0 | `high_lt_open` |
| 2025-12-11 | 3.723 | 3.722 | 3.54 | 3.57 | 0 | `high_lt_open` |
| 2026-01-02 | 3.159 | 3.149 | 3.033 | 3.06 | 0 | `high_lt_open` |
| 2026-01-05 | 3.06 | 2.98 | 2.825 | 2.884 | 0 | `high_lt_open` |

### `gold:SI` (`akshare.futures_foreign_hist_sina`, 共 18 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-12-22 | 67.39 | 69.525 | 67.47 | 69.085 | 0 | `low_gt_open` |
| 2025-12-26 | 71.875 | 79.7 | 72.72 | 79.66 | 0 | `low_gt_open` |
| 2026-01-05 | 72.245 | 77.835 | 72.505 | 76.495 | 0 | `low_gt_open` |
| 2026-01-12 | 79.765 | 86.34 | 79.9 | 85.165 | 0 | `low_gt_open` |
| 2026-01-19 | 89.935 | 94.705 | 90.5 | 94.275 | 0 | `low_gt_open` |

### `gold:GC` (`akshare.futures_foreign_hist_sina`, 共 17 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-12-26 | 4,505.10 | 4,584.00 | 4,518.00 | 4,561.90 | 0 | `low_gt_open` |
| 2026-01-05 | 4,341.90 | 4,467.60 | 4,354.60 | 4,459.50 | 0 | `low_gt_open` |
| 2026-01-12 | 4,518.10 | 4,640.50 | 4,520.80 | 4,608.40 | 0 | `low_gt_open` |
| 2026-01-19 | 4,600.90 | 4,698.00 | 4,622.20 | 4,676.30 | 0 | `low_gt_open` |
| 2026-01-26 | 5,019.90 | 5,145.20 | 5,022.90 | 5,042.90 | 0 | `low_gt_open` |

### `commodity:HG` (`akshare.futures_foreign_hist_sina`, 共 12 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-12-26 | 557.25 | 588.80 | 566.95 | 585.05 | 0 | `low_gt_open` |
| 2026-01-05 | 569.70 | 600.70 | 569.95 | 598.95 | 0 | `low_gt_open` |
| 2026-01-12 | 589.05 | 606.90 | 589.70 | 600.70 | 0 | `low_gt_open` |
| 2026-01-14 | 600.30 | 615.40 | 600.40 | 609.45 | 0 | `low_gt_open` |
| 2026-02-02 | 596.50 | 590.50 | 556.40 | 583.15 | 0 | `high_lt_open` |

### `commodity:OIL` (`akshare.futures_foreign_hist_sina`, 共 10 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-03-03 | 77.95 | 85.12 | 78.38 | 81.97 | 1,199,567 | `low_gt_open` |
| 2026-03-12 | 93.66 | 101.85 | 95.2 | 101.71 | 650,839.00 | `low_gt_open` |
| 2026-03-27 | 97.93 | 106.95 | 99.01 | 106.59 | 421,112.00 | `low_gt_open` |
| 2026-05-06 | 110.25 | 109.02 | 96.75 | 101.85 | 585,638.00 | `high_lt_open` |
| 2026-05-08 | 103.48 | 102.92 | 99.55 | 100.25 | 297,352.00 | `high_lt_open` |

### `gold:XPT` (`akshare.futures_foreign_hist_sina`, 共 7 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-04-08 | 1,965.50 | 2,105.00 | 1,966.70 | 2,046.30 | 0 | `low_gt_open` |
| 2026-05-25 | 1,931.40 | 1,984.50 | 1,939.90 | 1,976.80 | 0 | `low_gt_open` |
| 2026-06-15 | 1,720.10 | 1,824.20 | 1,730.60 | 1,773.10 | 0 | `low_gt_open` |
| 2026-07-27 | 1,599.00 | 1,655.00 | 1,605.30 | 1,630.60 | 0 | `low_gt_open` |
| 2026-09-17 | 1,779.80 | 1,831.60 | 1,781.00 | 1,799.10 | 0 | `low_gt_open` |

### `commodity:ZSD` (`akshare.futures_foreign_hist_sina`, 共 7 条异常，展示前 5 条)

| 日期 | Open | High | Low | Close | Volume | 异常原因 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025-12-09 | 3,128.00 | 3,127.00 | 3,086.00 | 3,086.50 | 9,094.00 | `high_lt_open` |
| 2025-12-19 | 3,055.50 | 3,084.00 | 3,057.00 | 3,075.00 | 8,576.00 | `low_gt_open` |
| 2025-12-31 | 3,127.00 | 3,126.00 | 3,095.00 | 3,123.00 | 5,809.00 | `high_lt_open` |
| 2026-01-08 | 3,246.50 | 3,186.00 | 3,123.00 | 3,128.50 | 15,413.00 | `high_lt_open` |
| 2026-03-18 | 3,230.50 | 3,227.00 | 3,130.00 | 3,132.50 | 16,651.00 | `high_lt_open` |


## 五、代码别名测试结果（不计入 65 个独立标的）

| ID | 类别 | 输入别名 | 规范代码 (`canonical_symbol`) | 预期映射 | 状态 | 数据源 | 行数 | 末日收盘 | 与规范标的一致性 |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| A01 | `ashare` | `sh600519` | `600519` | `ashare:600519` | `PASS` | `akshare.stock_zh_a_hist_tx` | 242 | 9,985.53 | 完全一致 |
| A02 | `etf` | `sh510300` | `510300` | `etf:510300` | `PASS` | `akshare.stock_zh_a_hist_tx` | 242 | 1.83 | 完全一致 |
| A03 | `gold` | `AU9999` | `GC` | `gold:GC` | `WARN` | `akshare.futures_foreign_hist_sina` | 258 | 3,858.90 | 完全一致 |
| A04 | `creit` | `sh508097` | `508097` | `creit:508097` | `PASS` | `akshare.stock_zh_a_hist_tx` | 242 | 1.978 | 完全一致 |
| A05 | `fx` | `USD/JPY` | `USDJPY` | `fx:USDJPY` | `WARN` | `sina_forex_direct` | 261 | 159.41 | 差异(rows=261 vs 261, end=2026-11-16 vs 2026-11-16, close=159.41 vs 159.43) |
| A06 | `ashare` | `600519.SH` | `600519` | `ashare:600519` | `PASS` | `akshare.stock_zh_a_hist_tx` | 242 | 9,985.53 | 完全一致 |
| A07 | `ashare` | `sz000001` | `000001` | `ashare:000001` | `PASS` | `akshare.stock_zh_a_hist_tx` | 242 | 2,417.00 | 完全一致 |
| A08 | `hk_stock` | `700` | `00700` | `hk_stock:00700` | `WARN` | `akshare.stock_hk_daily_sina` | 246 | 2,230.93 | 完全一致 |
| A09 | `fx` | `USD/CNY` | `USDCNH` | `fx:USDCNH` | `WARN` | `sina_forex_direct` | 261 | 6.6646 | 差异(rows=261 vs 261, end=2026-11-16 vs 2026-11-16, close=6.6646 vs 6.6647) |
| A10 | `bond_us_yield` | `美国国债收益率10年` | `10Y` | `bond_us_yield:10Y` | `PASS` | `akshare.bond_zh_us_yield_em` | 250 | 5.12 | 完全一致 |
| A11 | `global_index` | `SPX` | `.INX` | `global_index:.INX` | `PASS` | `akshare.index_us_daily_sina` | 251 | 7,416.59 | 完全一致 |

## 六、65 个独立标的逐项测试明细

| ID | 类别 | 代码 | 状态 | 数据源 | 样本区间 | 行数 | 币种 | 复权 | OHLC异常 | 零成交量行 | 最大断档(天) | 耗时(s) | 问题/说明 |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| P01 | `etf` | `510300` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.89 | — |
| P02 | `etf` | `510500` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.85 | — |
| P03 | `etf` | `159915` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.89 | — |
| P04 | `etf` | `511010` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.89 | — |
| P05 | `etf` | `518880` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.90 | — |
| P06 | `ashare` | `600519` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.89 | — |
| P07 | `ashare` | `000001` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.88 | — |
| P08 | `ashare` | `300750` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.89 | — |
| P09 | `ashare` | `688981` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.97 | — |
| P10 | `ashare` | `601318` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `hfq` | 0 | 0 | 11 | 0.86 | — |
| P11 | `mutual_fund` | `009691` | **PASS** | `akshare.fund_open_fund_info_em` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 242 | 11 | 1.04 | zero_volume_expected |
| P12 | `mutual_fund` | `161725` | **PASS** | `akshare.fund_open_fund_info_em` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 242 | 11 | 0.86 | zero_volume_expected |
| P13 | `mutual_fund` | `110011` | **PASS** | `akshare.fund_open_fund_info_em` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 242 | 11 | 1.10 | zero_volume_expected |
| P14 | `mutual_fund` | `005827` | **PASS** | `akshare.fund_open_fund_info_em` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 242 | 11 | 1.02 | zero_volume_expected |
| P15 | `mutual_fund` | `110022` | **PASS** | `akshare.fund_open_fund_info_em` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 242 | 11 | 0.83 | zero_volume_expected |
| P16 | `gold` | `GC` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 17 | 258 | 4 | 0.76 | cli_status_degraded; ohlc_anomaly:17; all_zero_volume |
| P17 | `gold` | `SI` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 18 | 258 | 4 | 0.76 | cli_status_degraded; ohlc_anomaly:18; all_zero_volume |
| P18 | `gold` | `XAU` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 0 | 258 | 4 | 0.60 | all_zero_volume |
| P19 | `gold` | `XAG` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 0 | 258 | 4 | 0.61 | all_zero_volume |
| P20 | `gold` | `XPT` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 7 | 258 | 4 | 0.70 | cli_status_degraded; ohlc_anomaly:7; all_zero_volume |
| P21 | `crypto` | `BTC/USDT` | **PASS** | `ccxt.binance` | 2025-11-17..2026-11-16 | 365 | `USDT` | `raw` | 0 | 0 | 1 | 2.27 | — |
| P22 | `crypto` | `ETH/USDT` | **PASS** | `ccxt.binance` | 2025-11-17..2026-11-16 | 365 | `USDT` | `raw` | 0 | 0 | 1 | 1.67 | — |
| P23 | `crypto` | `SOL/USDT` | **PASS** | `ccxt.binance` | 2025-11-17..2026-11-16 | 365 | `USDT` | `raw` | 0 | 0 | 1 | 1.92 | — |
| P24 | `crypto` | `BNB/USDT` | **PASS** | `ccxt.binance` | 2025-11-17..2026-11-16 | 365 | `USDT` | `raw` | 0 | 0 | 1 | 1.61 | — |
| P25 | `crypto` | `XRP/USDT` | **PASS** | `ccxt.binance` | 2025-11-17..2026-11-16 | 365 | `USDT` | `raw` | 0 | 0 | 1 | 2.51 | — |
| P26 | `us_stock` | `AAPL` | **PASS** | `akshare.stock_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `qfq` | 0 | 0 | 4 | 1.14 | — |
| P27 | `us_stock` | `MSFT` | **PASS** | `akshare.stock_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `qfq` | 0 | 0 | 4 | 0.78 | — |
| P28 | `us_stock` | `NVDA` | **PASS** | `akshare.stock_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `qfq` | 0 | 0 | 4 | 0.69 | — |
| P29 | `us_stock` | `AMZN` | **PASS** | `akshare.stock_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `qfq` | 0 | 0 | 4 | 0.69 | — |
| P30 | `us_stock` | `GOOGL` | **PASS** | `akshare.stock_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `qfq` | 0 | 0 | 4 | 0.88 | — |
| P31 | `hk_stock` | `00700` | **WARN** | `akshare.stock_hk_daily_sina` | 2025-11-17..2026-11-16 | 246 | `HKD` | `hfq` | 2 | 0 | 6 | 0.89 | cli_status_degraded; ohlc_anomaly:2 |
| P32 | `hk_stock` | `09988` | **PASS** | `akshare.stock_hk_daily_sina` | 2025-11-17..2026-11-16 | 246 | `HKD` | `hfq` | 0 | 0 | 6 | 1.18 | — |
| P33 | `hk_stock` | `01810` | **PASS** | `akshare.stock_hk_daily_sina` | 2025-11-17..2026-11-16 | 246 | `HKD` | `hfq` | 0 | 0 | 6 | 1.19 | — |
| P34 | `hk_stock` | `03690` | **PASS** | `akshare.stock_hk_daily_sina` | 2025-11-17..2026-11-16 | 246 | `HKD` | `hfq` | 0 | 0 | 6 | 0.98 | — |
| P35 | `hk_stock` | `00941` | **PASS** | `akshare.stock_hk_daily_sina` | 2025-11-17..2026-11-16 | 246 | `HKD` | `hfq` | 0 | 0 | 6 | 1.21 | — |
| P36 | `global_index` | `.INX` | **PASS** | `akshare.index_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `raw` | 0 | 0 | 4 | 0.69 | — |
| P37 | `global_index` | `.DJI` | **PASS** | `akshare.index_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `raw` | 0 | 0 | 4 | 0.71 | — |
| P38 | `global_index` | `.IXIC` | **PASS** | `akshare.index_us_daily_sina` | 2025-11-17..2026-11-16 | 251 | `USD` | `raw` | 0 | 0 | 4 | 0.67 | — |
| P39 | `global_index` | `HSI` | **WARN** | `akshare.index_hk_daily_sina` | 2025-11-17..2026-11-16 | 246 | `HKD` | `raw` | 5 | 0 | 6 | 0.71 | cli_status_degraded; ohlc_anomaly:5 |
| P40 | `global_index` | `N225` | **WARN** | `akshare.index_global_hist_sina` | 2025-11-17..2026-11-16 | 242 | `JPY` | `raw` | 0 | 86 | 6 | 0.69 | partial_zero_volume:86/242 |
| P41 | `global_index` | `DXY` | **WARN** | `sina_forex_direct` | 2025-11-17..2026-11-16 | 261 | `UNKNOWN` | `raw` | 0 | 261 | 3 | 1.10 | cli_status_degraded; vendor_fallback_triggered; unknown_output_currency; index_no_volume_from_vendor |
| P42 | `bond_cn` | `sh019547` | **PASS** | `akshare.bond_zh_hs_daily_sina` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 0 | 11 | 0.74 | — |
| P43 | `bond_cn` | `sh019743` | **WARN** | `akshare.bond_zh_hs_daily_sina` | 2025-11-19..2026-11-06 | 111 | `CNY` | `raw` | 0 | 0 | 28 | 0.68 | stale_end_date:lag_10d; abnormal_calendar_gap:max_28d |
| P44 | `bond_cn` | `sh019780` | **WARN** | `akshare.bond_zh_hs_daily_sina` | 2025-11-17..2026-11-16 | 184 | `CNY` | `raw` | 0 | 1 | 14 | 0.77 | abnormal_calendar_gap:max_14d; bond_illiquid_zero_vol_days:1/184 |
| P45 | `bond_cn` | `sh019788` | **WARN** | `akshare.bond_zh_hs_daily_sina` | 2025-11-17..2026-11-09 | 169 | `CNY` | `raw` | 0 | 0 | 11 | 0.78 | stale_end_date:lag_7d; abnormal_calendar_gap:max_11d |
| P46 | `bond_cn` | `sh019795` | **WARN** | `akshare.bond_zh_hs_daily_sina` | 2025-11-20..2026-11-10 | 189 | `CNY` | `raw` | 0 | 0 | 11 | 0.76 | stale_end_date:lag_6d; abnormal_calendar_gap:max_10d |
| P47 | `bond_us_yield` | `10Y` | **PASS** | `akshare.bond_zh_us_yield_em` | 2025-11-17..2026-11-16 | 250 | `USD` | `raw` | 0 | 250 | 4 | 0.86 | zero_volume_expected |
| P48 | `bond_us_yield` | `美国国债收益率2年` | **PASS** | `akshare.bond_zh_us_yield_em` | 2025-11-17..2026-11-16 | 250 | `USD` | `raw` | 0 | 250 | 4 | 0.76 | zero_volume_expected |
| P49 | `bond_us_yield` | `美国国债收益率5年` | **PASS** | `akshare.bond_zh_us_yield_em` | 2025-11-17..2026-11-16 | 250 | `USD` | `raw` | 0 | 250 | 4 | 0.72 | zero_volume_expected |
| P50 | `bond_us_yield` | `美国国债收益率30年` | **PASS** | `akshare.bond_zh_us_yield_em` | 2025-11-17..2026-11-16 | 250 | `USD` | `raw` | 0 | 250 | 4 | 0.73 | zero_volume_expected |
| P51 | `commodity` | `CL` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 31 | 258 | 4 | 0.84 | cli_status_degraded; ohlc_anomaly:31; all_zero_volume |
| P52 | `commodity` | `HG` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 12 | 258 | 4 | 0.67 | cli_status_degraded; ohlc_anomaly:12; all_zero_volume |
| P53 | `commodity` | `NG` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 25 | 258 | 4 | 0.63 | cli_status_degraded; ohlc_anomaly:25; all_zero_volume |
| P54 | `commodity` | `OIL` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 258 | `USD` | `raw` | 10 | 0 | 4 | 0.63 | cli_status_degraded; ohlc_anomaly:10 |
| P55 | `commodity` | `ZSD` | **WARN** | `akshare.futures_foreign_hist_sina` | 2025-11-17..2026-11-16 | 253 | `USD` | `raw` | 7 | 0 | 5 | 0.73 | cli_status_degraded; ohlc_anomaly:7 |
| P56 | `creit` | `508097` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 0 | 11 | 0.89 | — |
| P57 | `creit` | `180101` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-18..2026-11-16 | 241 | `CNY` | `raw` | 0 | 0 | 11 | 0.93 | — |
| P58 | `creit` | `508000` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 0 | 11 | 0.82 | — |
| P59 | `creit` | `180201` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 0 | 11 | 0.91 | — |
| P60 | `creit` | `508028` | **PASS** | `akshare.stock_zh_a_hist_tx` | 2025-11-17..2026-11-16 | 242 | `CNY` | `raw` | 0 | 0 | 11 | 0.84 | — |
| P61 | `fx` | `USDCNH` | **PASS** | `eastmoney_forex_direct` | 2025-11-17..2026-11-16 | 261 | `CNY` | `raw` | 0 | 261 | 3 | 1.11 | zero_volume_expected |
| P62 | `fx` | `EURUSD` | **PASS** | `eastmoney_forex_direct` | 2025-11-17..2026-11-16 | 261 | `USD` | `raw` | 0 | 261 | 3 | 0.85 | zero_volume_expected |
| P63 | `fx` | `USDJPY` | **PASS** | `eastmoney_forex_direct` | 2025-11-17..2026-11-16 | 261 | `JPY` | `raw` | 0 | 261 | 3 | 1.54 | zero_volume_expected |
| P64 | `fx` | `GBPUSD` | **PASS** | `eastmoney_forex_direct` | 2025-11-17..2026-11-16 | 260 | `USD` | `raw` | 0 | 260 | 3 | 0.96 | zero_volume_expected |
| P65 | `fx` | `AUDUSD` | **PASS** | `eastmoney_forex_direct` | 2025-11-17..2026-11-16 | 261 | `USD` | `raw` | 0 | 261 | 3 | 1.40 | zero_volume_expected |

## 七、特殊序列语义与成交量单位审计

1. **公募基金净值 (`mutual_fund`)**：5 只测试基金均来自 `akshare.fund_open_fund_info_em`，全部满足 `Open == High == Low == Close`（累计净值伪 OHLCV，0 条不一致）、`Volume == 0`、样本区间完整（`2025-11-17..2026-11-16`，242 行），无缺失与异常跳变。
2. **美债收益率水平 (`bond_us_yield`)**：4 个期限（`10Y` / 2年 / 5年 / 30年）及别名 `美国国债收益率10年`均来自 `akshare.bond_zh_us_yield_em`，`series_kind='yield_level'`、`return_basis='yield_level'`，满足 `Open == High == Low == Close` 与 `Volume == 0` 语义，收益率水平位于 `3.38% ~ 5.67%` 合理区间（250 行）。
3. **成交量单位差异**：
   - A 股 / ETF / C-REITs 主源 `akshare.stock_zh_a_hist_tx`（腾讯）返回成交量单位为 **股 / 份**（已乘以手股换算倍率），而东财源（`fund_etf_hist_em` / `stock_zh_a_hist` / `eastmoney_direct`）返回 **手（100 股/份）**，发生回退时存在 **100 倍** 量纲差。
   - 加密货币 (`ccxt.binance`) 为基础币数量，美股/港股为股数。
   - 外盘期货 (`akshare.futures_foreign_hist_sina`) 存在数据源内部差异：`GC/SI/XAU/XAG/XPT/CL/HG/NG` 成交量全为 `0`，仅 `OIL` 与 `ZSD` 提供真实成交量。
   - 全球指数 (`N225` / `DXY`)、中国国债活跃券外部分债券及外汇即期 (`fx`) 存在无成交量或流动性稀疏现象。

## 八、发现的程序缺陷及修复优先级

| 优先级 | 缺陷位置 | 现象与实测证据 | 影响 | 建议修复方向 |
| --- | --- | --- | --- | --- |
| **P1 (高)** | `currency.py` (`GLOBAL_INDEX_CURRENCIES`) | `global_index:DXY` 实测返回 `native_currency='UNKNOWN'`、`output_currency='UNKNOWN'` | 在 `--base-currency CNY/USD` 或多资产组合换算中会因未知币种报错中断 | 为 `DXY`/`UDI`/`美元指数`/`DINIW` 补充显式指数点位/非单币种资产保护或币种映射 |
| **P1 (高)** | `data_sources/forex.py` (`_fetch_forex_hist_em_robust`) | 东财限流触发时 `eastmoney_forex_direct` 将空 `data` 响应误判为 `EmptySymbolDataError` 并提前 `break` 全部节点，紧随其后的 `akshare.fx_hist_em` 抛出 `Push2RateLimitError`（见 `A05 USD/JPY`、`A09 USD/CNY`） | 误把限流/风控当成“标的无数据”，阻断节点轮询并在新浪兜底失败时误报空数据 | 解析 JSON `rc`/`rt` 及 `data is None`，区分 `VendorUpstreamError`（限流/异常）与真实空 K 线 |
| **P1 (高)** | `data_fetcher.py` (`load_reit_data`) / `asset_catalog.py` | `creit` 实际通过 `stock_zh_a_hist_tx(..., adjust='qfq')` 与 `eastmoney_direct(..., 'qfq')` 拉取前复权数据（如 `508097` 从 `2.645` 降至 `1.978`），但元数据却错误标记 `adjust=''`、`return_basis='adjusted_or_price'` | 下游/Agent 无法区分前复权与新浪兜底的不复权价格 | 在 `load_reit_data` 中按实际路径写入 `attrs['adjust']='qfq'` 或 `''`，并将 `creit` 纳入复权规范 |
| **P2 (中)** | `ohlcv.py` (`_compute_ohlcv_quality`) | Sina 港股/恒指解码器（`klc2_kl.js`）产生 float32 精度微差（如 `00700` `2505.93` vs `2505.9299`、`HSI` `25540.78` vs `25540.78125`），零容差比较触发 `ohlc_anomaly` 与 `status='degraded'` | 干净港股/指数被误标记为降级，掩盖真正的期货脏数据 | 对 OHLC 关系校验引入亚 tick 浮点容差（如 `1e-4` 或相对 `1e-6`），区分舍入噪声与真实破界 |
| **P2 (中)** | `data_sources/china_equity.py` / `contracts.py` | 腾讯 `stock_zh_a_hist_tx` 返回成交量为“股/份”，东财源返回“手（100 股）”，多源回退未统一单位也未在 `SeriesMetadata` 标明 `volume_unit` | 触发东财回退时成交量出现 100 倍突变 | 统一规范为“股/份”或在 `SeriesMetadata` 新增 `volume_unit` 元数据字段 |
| **P2 (中)** | `data_fetcher.py` (`load_futures_foreign_ohlcv`) | `akshare.futures_foreign_hist_sina` 在黄金/商品上存在大量周一开盘跳空型 OHLC 异常（最高 `CL` `31/258=12.02%`、`NG` `25/258=9.69%`、`SI` `6.98%`、`GC` `6.59%`），且多数合约成交量为 `0`、`XAU/XAG` 实为现货 | 外盘期货 K 线结构性破界导致技术分析/风控信号失真 | 增加跨源交叉校验、对 `low_gt_open`/`high_lt_open` 提供显著告警提示并区分现货与连续期货 |
| **P2 (中)** | `asset_catalog.py` (`normalize_symbol_alias`) | `normalize_symbol_alias('global_index', 'SPX')` 返回 `'SPX'` 而实际拉取返回 `canonical_symbol='.INX'`（`N225` 同理）；`gold:AU9999` 静默代理为美元盎司期货 `GC` | 多资产组合去重无法在拉取前识别 `.INX` 与 `SPX` 重复；`AU9999` 存在跨币种/跨市场语义偏差 | 统一 `asset_catalog` 与 `data_fetcher` 的指数别名映射，并对 `AU9999->GC` 输出显式代理告警 |
| **P3 (低)** | `data_fetcher.py` / `contracts.py` | `global_index:N225`（`index_global_hist_sina`）在 `2026-06-23~2026-10-28` 连续 `86/242` 日成交量为 `0`；非活跃国债 `sh019743` 仅 `111` 行且最大断档 `28` 天、尾部滞后 `10` 天 | 局部成交量缺失或债券低流动性未在质量报告中区分 | 在 `QualityReport` / `warnings` 中补充区间覆盖率、尾部滞后天数与分段零成交量告警 |
| **P3 (低)** | `data_sources/eastmoney.py` / `forex.py` | 连续多标的访问东财触发 `Push2RateLimitError`（本轮实测触发 `3` 次：`DXY`、`USD/JPY`、`USD/CNY`），虽由新浪成功兜底但缺少请求间隔节流 | 批量抓取场景易加速触发 IP 限流 | 为东财直连与 AKShare 东财源增加轻量域名级节流与退避冷却 |
