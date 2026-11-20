# 全资产真实行情审计失败与告警清单 (2026-11-17)

- 独立标的 FAIL 数量：**0 / 65**
- 独立标的 WARN 数量：**18 / 65**
- 别名测试 FAIL / WARN 数量：**0 / 4**（共 11 项）

## 一、失败标的明细（FAIL）

本轮无 FAIL 标的。

## 二、质量告警明细（WARN）

| ID | 分组 | 类别 | 代码 | 数据源 | 行数 | 样本区间 | OHLC异常 | 告警代码 |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |
| P16 | primary | `gold` | `GC` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 17 | `cli_status_degraded`, `ohlc_anomaly:17`, `all_zero_volume` |
| P17 | primary | `gold` | `SI` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 18 | `cli_status_degraded`, `ohlc_anomaly:18`, `all_zero_volume` |
| P18 | primary | `gold` | `XAU` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 0 | `all_zero_volume` |
| P19 | primary | `gold` | `XAG` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 0 | `all_zero_volume` |
| P20 | primary | `gold` | `XPT` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 7 | `cli_status_degraded`, `ohlc_anomaly:7`, `all_zero_volume` |
| P31 | primary | `hk_stock` | `00700` | `akshare.stock_hk_daily_sina` | 246 | 2025-11-17..2026-11-16 | 2 | `cli_status_degraded`, `ohlc_anomaly:2` |
| P39 | primary | `global_index` | `HSI` | `akshare.index_hk_daily_sina` | 246 | 2025-11-17..2026-11-16 | 5 | `cli_status_degraded`, `ohlc_anomaly:5` |
| P40 | primary | `global_index` | `N225` | `akshare.index_global_hist_sina` | 242 | 2025-11-17..2026-11-16 | 0 | `partial_zero_volume:86/242` |
| P41 | primary | `global_index` | `DXY` | `sina_forex_direct` | 261 | 2025-11-17..2026-11-16 | 0 | `cli_status_degraded`, `vendor_fallback_triggered`, `unknown_output_currency` |
| P43 | primary | `bond_cn` | `sh019743` | `akshare.bond_zh_hs_daily_sina` | 111 | 2025-11-19..2026-11-06 | 0 | `stale_end_date:lag_10d`, `abnormal_calendar_gap:max_28d` |
| P44 | primary | `bond_cn` | `sh019780` | `akshare.bond_zh_hs_daily_sina` | 184 | 2025-11-17..2026-11-16 | 0 | `abnormal_calendar_gap:max_14d` |
| P45 | primary | `bond_cn` | `sh019788` | `akshare.bond_zh_hs_daily_sina` | 169 | 2025-11-17..2026-11-09 | 0 | `stale_end_date:lag_7d`, `abnormal_calendar_gap:max_11d` |
| P46 | primary | `bond_cn` | `sh019795` | `akshare.bond_zh_hs_daily_sina` | 189 | 2025-11-20..2026-11-10 | 0 | `stale_end_date:lag_6d`, `abnormal_calendar_gap:max_10d` |
| P51 | primary | `commodity` | `CL` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 31 | `cli_status_degraded`, `ohlc_anomaly:31`, `all_zero_volume` |
| P52 | primary | `commodity` | `HG` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 12 | `cli_status_degraded`, `ohlc_anomaly:12`, `all_zero_volume` |
| P53 | primary | `commodity` | `NG` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 25 | `cli_status_degraded`, `ohlc_anomaly:25`, `all_zero_volume` |
| P54 | primary | `commodity` | `OIL` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 10 | `cli_status_degraded`, `ohlc_anomaly:10` |
| P55 | primary | `commodity` | `ZSD` | `akshare.futures_foreign_hist_sina` | 253 | 2025-11-17..2026-11-16 | 7 | `cli_status_degraded`, `ohlc_anomaly:7` |
| A03 | alias | `gold` | `AU9999` | `akshare.futures_foreign_hist_sina` | 258 | 2025-11-17..2026-11-16 | 17 | `cli_status_degraded`, `ohlc_anomaly:17`, `all_zero_volume` |
| A05 | alias | `fx` | `USD/JPY` | `sina_forex_direct` | 261 | 2025-11-17..2026-11-16 | 0 | `cli_status_degraded`, `vendor_fallback_triggered` |
| A08 | alias | `hk_stock` | `700` | `akshare.stock_hk_daily_sina` | 246 | 2025-11-17..2026-11-16 | 2 | `cli_status_degraded`, `ohlc_anomaly:2` |
| A09 | alias | `fx` | `USD/CNY` | `sina_forex_direct` | 261 | 2025-11-17..2026-11-16 | 0 | `cli_status_degraded`, `vendor_fallback_triggered` |
