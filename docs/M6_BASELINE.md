# M6 后端行为与依赖基线（M6-0）

> 记录分支：`refactor/m6-backend`
> 基线提交：`33afdcb90a6544c7ac198711fc6bfef342e58429`（在 `codex/frontend-qa` @ `826d0a8caaacbbabb198ba99b397cd546f84bdb3` 上新增 `docs/M6_BACKEND_REFACTOR_PLAN.md`）
> 采集日期：2026-11-20

---

## 1. 运行环境与验证矩阵

| 检查项 | 实际状态 | 说明 |
| :--- | :--- | :--- |
| **本地 Python** | `3.14.5`（`.venv/bin/python`） | 本地完整执行测试与静态检查 |
| **Python 3.11 / 3.12 本地运行时测试** | `NOT RUN` | 本机未安装 `python3.11`；`/Users/ultramarine/.local/bin/python3.12` 无项目依赖虚拟环境；CI 矩阵（`.github/workflows/tests.yml`）覆盖 `["3.11", "3.12", "3.14"]`，本地通过 `ruff` `target-version = "py311"` 做静态兼容检查 |
| **核心依赖版本（实测）** | `numpy 2.5.4`, `pandas 3.1.1`, `scipy 1.18.1`, `pyarrow 25.0.1`, `streamlit 1.67.0`, `plotly 7.1.0`, `matplotlib 3.11.2`, `akshare 1.20.46`, `ccxt 4.5.92`, `requests 2.34.2`, `httpx 0.28.1` | 依赖树经 `pip check` 验证无破损 |
| **基线离线 `pytest -q`** | `284 passed in 38.40s` | `MPLBACKEND=Agg QUANT_DISABLE_CACHE=1`，`tests/conftest.py` 强制阻断外网访问与磁盘缓存 |
| **M6-0 黄金测试补充后** | `292 passed, 3 xfailed`（新增 8 项黄金/审计数 + 3 项 strict `xfail`） | 3 项 `xfail(strict=True)` 精确刻画 M6-1 待修的 P0 问题（A1/A2/A3） |
| **`ruff check .`** | `PASS`（`All checks passed!`） | 覆盖 `E9, F, I, B, UP` 及 `target-version = "py311"` |
| **`ruff format --check .`** | `FAIL`（历史存量 2 个文件未格式化，其余 69 个文件已格式化） | 历史存量未格式化文件：`data_sources/network.py`（L54）、`scripts/live_asset_audit.py`（L374/929/1269/1293/1633）；所有 M6-0 新增/修改文件均需通过 `ruff format --check` |
| **`.venv/bin/python -m pip check`** | `PASS`（`No broken requirements found.`） | — |

---

## 2. 源码规模与依赖方向审计

### 2.1 当前源码行数分布（共 33 个非测试源码文件，22,817 行）

| 模块 | 行数 | 当前职责概况 |
| :--- | ---: | :--- |
| `reporting/report_renderer.py` | 2,384 | Markdown / 自包含 HTML 报告渲染 |
| `app.py` | 2,335 | Streamlit Web UI（M6 禁止主动视觉/交互重构） |
| `quant_models.py` | 2,046 | 协方差收缩、SLSQP 优化、有效前沿、回测、Walk-Forward、GBM |
| `services/multi_asset_service.py` | 1,872 | 多资产加载、统计比较、组合优化、基准对齐、敏感性扫描 |
| `charts.py` | 1,447 | Plotly 自适应主题图表构建 |
| `reporting/chart_renderer.py` | 1,390 | 独立图表导出调度（Plotly HTML / Matplotlib PNG & SVG） |
| `quant_cli.py` | 1,379 | `quant` CLI 解析、事务性落盘与报告/图表触发 |
| `analysis_metrics.py` | 1,002 | 单资产风险收益、分布检验、基准归因、Memmel 检验、滚动统计 |
| `contracts.py` | 965 | 请求/结果契约（当前 `SCHEMA_VERSION = "0.1.0"`） |
| `data_fetcher.py` | 914 | 多市场行情抓取路由与格式转换 |
| `asset_loader.py` | 735 | 13 类资产统一入口与多资产日历对齐 |
| `currency.py` | 621 | 币种推断、防前视 FX 折算 |
| `data_sources/cache.py` | 543 | Parquet + JSON 元信息共享缓存与文件锁 |
| `reporting/formatting.py` | 514 | 指标格式化 + 磁盘产物反序列化（`load_result_from_artifacts`） |
| 其余模块（20 个） | 3,669 | `asset_catalog.py`(459), `services/analysis_service.py`(443), `main.py`(401), `ohlcv.py`(315), `feature_engineering.py`(290), `data_sources/*`, `visualizer.py`(367), `asset_resolver.py`(113) 等 |

### 2.2 实际依赖方向与真实耦合记录

经 AST 静态扫描（`tests/test_m6_0_baseline_golden.py::test_module_dependency_direction_audit`）验证：

1. **已保持清洁的边界**：
   - `asset_catalog` 零内部依赖。
   - `contracts` 仅依赖 `asset_catalog`、`ohlcv`（不反向依赖 `services`、`quant_cli`、`app`、`reporting` 或数据源）。
   - `analysis_metrics`、`feature_engineering`、`quant_models` 不依赖 `streamlit`、`app`、`main`、`quant_cli`、`reporting`、`services`、`data_fetcher`、`asset_loader`。
   - 顶层模块无 `ImportError` 级循环导入。
2. **已验证的跨层真实耦合（M6-1 至 M6-4 待治理）**：
   - **A1（P0）**：`max_drawdown` 符号语义跨命令分裂——`analysis_metrics.get_quant_metrics` 与 `quant_models.compute_backtest_metrics` 输出非负深度（`>= 0`），而 `services/multi_asset_service.py::_compute_aligned_asset_metrics` 输出非正水下收益（`<= 0`）；`reporting/formatting.py::format_drawdown_percent` 用 `-abs(...)` 掩盖差异。
   - **A2（P0）**：`contracts.py` 的 `metrics` / `parameters` / `methodology` 为自由字典，且 `build_series_metadata` 依赖 `DataFrame.attrs`；当 `ashare`/`etf`/`hk_stock`/`us_stock` 缺失 `attrs["adjust"]` 时，`build_series_metadata` 会同时输出矛盾的 `adjust=""` 与 `return_basis="adjusted_price"`。
   - **A3（P0）**：`quant_cli.py::_write_outputs` 将绝对路径写入 `result.json["artifacts"]`；`reporting/formatting.py::_resolve_artifact_path` 优先检查该绝对路径是否存在，复制目录后若旧目录仍存在会读取旧目录数据，且缺少路径越界检查与完整性校验。
   - **后续阶段（仅记录，不在 M6-0/M6-1 改造）**：`services.multi_asset_service` 依赖 `services.analysis_service._slice_to_requested_window`（M6-2）；`reporting/formatting.py` 承担产物读取职责（M6-1 迁出至 `artifacts/`，`reporting` 保留兼容重导出）；`app.py` 仍保留少数旧模块导入（M7 再处理）。

---

## 3. 金融计算与 JSON 契约基线

### 3.1 当前数值计算口径（M6 重构保持不变）

| 计算项 | 所在实现 | 基线口径 |
| :--- | :--- | :--- |
| 单资产累计收益 / CAGR | `analysis_metrics.py` | 累计收益 `P_T / P_0 - 1`；CAGR 按自然日跨度 `(P_T / P_0)^(365.25 / days) - 1` |
| 单资产波动率 / Sharpe / Sortino | `analysis_metrics.py` | 默认 `return_convention="log"`（可选 `"simple"`），样本标准差 `ddof=1`，Sortino 下行偏差为对 `0` 截断的总体 RMS；年化周期取 `ASSET_TRADING_DAYS`（`crypto=365`, `fx=260`, 其余 `252`） |
| 单资产 VaR(95%) / CVaR(95%) | `analysis_metrics.py` | 日简单收益率分位数（`q=0.05`）及尾部均值，返回非正损失幅度 |
| 组合与回测收益/风险 | `quant_models.py` | `return_convention="simple"`，`rf_per_period = rf / periods_per_year`；多资产对齐后采用实际有效年化周期 `effective_periods_per_year` / `backtest_periods_per_year` |
| 协方差与凸优化 | `quant_models.py` | `ledoit_wolf` / `sample` / `diagonal`；SciPy SLSQP 精确前沿与目标组合求解；Euler 风险贡献满足 $\sum RC_i = \sigma_p$ |
| 特殊序列拦截 | `analysis_metrics.py` / `currency.py` / `asset_loader.py` | `yield_level` 阻断全部 9 项收益/风险指标（服务层为 `NaN`、JSON 严格序列化为 `null` + `metric_reasons="yield_level_not_return"`）并禁止 FX 换汇与组合纳入；`fx_rate` 禁止非 `LOCAL` 换汇 |

### 3.2 黄金测试数值容差与契约迁移策略

- **数值容差**：
  - 解析/确定性指标、Service ↔ CLI 一致性、跨模式同口径对齐：`rel=1e-12, abs=1e-12`（跨模块边界如 Euler 风险加总 `rel=1e-10, abs=1e-10`）。
  - SLSQP 优化权重及有效前沿：给定固定样本和参数，同环境可复现容差 `abs=1e-12`（跨平台浮点优化器容差 `1e-6`）。
  - 随机可行域抽样：固定 `random_seed=42` 时要求严格逐点可复现（`abs=1e-12`）。
- **Schema 迁移策略（M6-1：`0.1.0 → 0.2.0`）**：
  - M6-1 统一所有命令的 `max_drawdown` 为**非负深度（`>= 0`）**，并将 `SCHEMA_VERSION` 从 `"0.1.0"` 升级为 `"0.2.0"`。
  - 读取历史 `"0.1.0"` 产物时，按 `schema_version + command` 显式迁移：仅当 `command == "compare"` 时将非正 `max_drawdown <= 0` 转为正深度（若出现意外正值报错而非盲目 `abs()`）；`analyze`/`portfolio` 校验为 `>= 0`；同时将历史绝对产物路径受限迁移为目录内安全相对路径，绝不回落读取旧目录文件。
  - 所有对外 `result.json` 仍严格执行 `allow_nan=False`，服务内部浮点 `NaN` 在 JSON 中统一为 `null`。

---

## 4. 基线测试清单

详见 [tests/test_m6_0_baseline_golden.py](../tests/test_m6_0_baseline_golden.py)：

- `test_single_asset_analytic_golden_and_cli_parity[log|simple]`：单资产解析黄金值与 Service/CLI 严格一致性。
- `test_compare_cross_market_golden_and_cli_parity[common_trading_dates|mtm_outer]`：跨市场 FX 对齐、Memmel/JB 检验与 Service/CLI 一致性。
- `test_portfolio_train_test_walk_forward_custom_and_seed_reproducibility`：SLSQP、`train_test`、Walk-Forward、`custom` 权重、Euler 风险加总与固定随机种子可复现性。
- `test_special_series_and_edge_cases_golden`：`yield_level`、`fx_rate`、常数序列、单调上涨序列、3 条极短序列。
- `test_offline_parquet_and_report_replay_without_loaders`：`analyze`/`compare`/`portfolio` 离线 Parquet + 报告重放。
- `test_module_dependency_direction_audit`：AST 模块依赖方向与无循环导入检查。
- 3 项 strict `xfail` P0 刻画测试：`test_p0_a1_cross_command_max_drawdown_sign_consistency`、`test_p0_a2_series_metadata_rejects_missing_adjust_contradiction`、`test_p0_a3_copied_artifact_dir_reads_local_parquet_and_uses_relative_paths`。
