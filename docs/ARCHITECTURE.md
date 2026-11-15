# 工程架构、数据口径与兼容边界

> 更新日期：2026-11-15（M3 / V0.3 基线对齐）  
> 详细技术债清单、V0.1/V0.2/V0.3 数据契约与模块拆分方案见 [M0 审计与契约文档](M0_AUDIT_AND_CONTRACTS.md)；长期阶段规划见 [产品定位与开发路线图](ROADMAP.md)。

---

## 1. 职责分层与模块依赖

当前工程由三个面向用户/Agent 的入口、统一服务与契约层、资产目录与路由层、特征/指标/模型计算层、图表可视化层以及数据源与缓存基础设施组成：

- **入口层**：
  - [quant_cli.py](../quant_cli.py)：面向 AI Agent 的命令行入口（已注册 CLI 脚本 `quant`，提供 `quant fetch`、`quant analyze`、`quant compare` 与 `quant portfolio`）。`stdout` 严格输出 JSON，`stderr` 输出诊断日志与结构化 JSON 错误；严格校验资产类型输入（裸代码无 `-t/--asset-type` 及非法/空 `TYPE:` / `TYPE|` 前缀直接以退出码 `2` 失败，不静默猜为 `etf`）；`--output <dir>` 事务性写入 `result.json` 与类型化 Parquet（如 `data.parquet`、`prices.parquet`、`returns.parquet`、`correlation.parquet`、`covariance.parquet`、`rolling_metrics.parquet`、`weights.parquet`、`frontier.parquet`、`backtest.parquet`、`weight_history.parquet`，可选 companion CSV）及分级退出码（`0`/`2`/`3`/`4`/`5`/`1`），不触发任何图表渲染，为 M4 报告/图表渲染器保留完整文件契约。
  - [main.py](../main.py)：原有全 13 类资产命令行入口（已注册 CLI 脚本 `quantitative-analysis`），保留交互/无头文本诊断报告、`mplfinance` 图表与 CSV 导出兼容。
  - [app.py](../app.py)：Streamlit 三标签页 Web 应用（`量化诊断看板`、`数学原理`、`进阶金融算法`），维护会话结果、侧边栏代理状态及 `@st.cache_data` 内存缓存。
- **契约与分析服务层**：
  - [contracts.py](../contracts.py)：版本化统一数据契约（`SCHEMA_VERSION = "0.1.0"`、`SOFTWARE_VERSION` 对应 `0.3.0`）：单资产契约 `AnalysisRequest`、`SeriesMetadata`、`QualityReport`、`AnalysisResult`，与多资产契约 `AssetSpec`、`CompareRequest`、`PortfolioRequest`、`MultiAssetResult`（均支持 `to_dict(include_data=...)` 与 strict JSON `allow_nan=False` 序列化）。
  - [services/analysis_service.py](../services/analysis_service.py)：解耦 UI/CLI 的单资产服务（`fetch_single_asset` 与 `analyze_single_asset` 共用 `_prepare_single_asset_frame` 完成窗口裁剪、元信息补全与 FX 折算）。
  - [services/multi_asset_service.py](../services/multi_asset_service.py)：解耦 UI/CLI 的多资产研究服务（`compare_assets` 与 `analyze_portfolio` 复用 `_fetch_multi_asset_frames`、`align_multi_asset_frames`、`analysis_metrics` 与 `quant_models`），统一输出跨市场可比指标、FX 折算元信息与基准对照。
- **资产与数据准备层**：
  - [asset_catalog.py](../asset_catalog.py)：13 类资产稳定 key、标签、默认值、交易日年化因子（`ASSET_TRADING_DAYS`）、默认币种、9 类收益语义 `ASSET_RETURN_BASIS` 及复权支持能力矩阵 `SUPPORTED_ADJUSTMENTS`。
  - [asset_resolver.py](../asset_resolver.py)（[china_equity_entry.py](../china_equity_entry.py) 保留兼容重导出）：分析窗口校验与境内 ETF/A 股统一错误信息格式化。
  - [asset_loader.py](../asset_loader.py)：全 13 类资产统一入口 `load_asset_data`、多线程并发拉取，以及 Stage 1 多市场对齐引擎 `align_multi_asset_frames` / `load_and_align_multi_asset_portfolio`（支持 `common_trading_dates`、`weekly_friday`、`mtm_outer`）。
  - [currency.py](../currency.py)：跨资产计价币种推断、向后防前视泄露 `merge_asof` FX 对齐（`max_staleness_days=4`）、`CNY`/`USD`/`LOCAL` 基准货币折算及显式 USDT/USD 锚定开关 `allow_usdt_usd_peg`。
  - [ohlcv.py](../ohlcv.py)：日期校验与统一 OHLCV 清洗、缺失策略（`raise`/`drop`/`ffill`）及 9 项质量诊断 `_compute_ohlcv_quality`。
  - [data_fetcher.py](../data_fetcher.py) 与 [data_sources/](../data_sources/)：各外部行情适配器（`china_equity.py`、`crypto.py`、`eastmoney.py`、`forex.py`、`funds.py`）、`RequestBudget` 与错误分类（`network.py`、`errors.py`）、以及 Web/CLI 共享 Parquet + JSON 元信息磁盘缓存（`cache.py`）。

**M1 已完成解耦与兼容保留边界**：
- 已抽离 [data_sources/crypto.py](../data_sources/crypto.py) 供 `asset_loader.py` 与 `currency.py` 共同依赖，消除 `asset_loader <-> currency` 循环导入；`asset_loader` 保留旧私有别名以兼容现有单测。
- 已将 OHLCV 换汇收敛至 [currency.py](../currency.py) `convert_ohlcv_to_base_currency`（`main._convert_ohlcv_to_base_currency` 保留重导出别名），并在 `data_fetcher.py` / `data_sources/*` 清理跨模块私有导入。
- [YFinanceFetcher](../data_sources/base.py) 作为历史 API 保留占位，调用会发出 `DeprecationWarning` 并抛出 `NotImplementedError`；美股日线通过 AKShare 路由加载。

---

## 2. 数据与计算口径

- **OHLCV 契约（`ohlcv.py`）**：固定列为 `Open`、`High`、`Low`、`Close`、`Volume`（`float64`）；索引 `Date` 为无时区、`ns` 时间单位升序唯一 `DatetimeIndex`；去重保留最后一条。`missing_policy="drop"` 剔除无效价格行但**不前向拷贝昨日成交量**（缺成交量补 `0.0` 并计入 `missing_volume_rows`）。
- **特征工程（`feature_engineering.py`）**：计算 SMA、对数收益滚动年化波动率（`ddof=1`）、布林带、最大回撤（`1 - Close / cummax(Close)`）以及可选 EMA/RSI(14)/MACD；保留并透传 `df.attrs`。
- **单资产风险收益诊断（`analysis_metrics.py`）**：
  - CAGR 按自然日跨度（`days / 365.25`）与起止价格比计算；
  - 年化波动率、Sharpe、Sortino 使用**日对数收益率**、`daily_rf_log = log1p(rf) / trading_days`；年化因子按 `ASSET_TRADING_DAYS` 自适应（加密货币 `365`、外汇 `260`、其余默认 `252`）；
  - 95% 日 VaR / CVaR 使用日简单收益率分位数；常数序列或单日样本 Sharpe 安全返回 `NaN`。
- **多资产统计、风险分解与优化回测（[analysis_metrics.py](../analysis_metrics.py) / [quant_models.py](../quant_models.py)）**：
  - 使用 **日简单收益率**（`quant compare` 支持切换 `--return-convention simple|log`）、`rf_per_period = rf / periods_per_year`；分布统计（偏度、超额峰度、Jarque-Bera 正态性检验）与 Memmel (2003) 夏普差异检验通过延迟导入 `scipy.stats` 实现（保持纯核心模块不硬依赖可选包）。
  - `mtm_outer` 模式仅用于连续净值展示与回测中基于 `tradable_mask` 的非交易日顺延调仓；相关性、协方差及波动率/夏普统计仍严格基于共同真实报价区间 `correlation_returns_base`，避免非交易日零收益膨胀。
  - 多资产组合支持 6 类策略（`max_sharpe`、`min_variance`、`risk_parity`、`inverse_volatility`、`equal_weight`、`custom`）与 Euler 风险贡献分解（`compute_risk_contributions`，保障 $\sum RC_i = \sigma_p$），协方差收缩可选 `ledoit_wolf`、`sample`、`diagonal`；精确有效前沿明确以 `algorithm="slsqp_exact_frontier"` / `"dirichlet_random_sampling"` 区分约束优化与随机抽样。
  - 样本外验证支持静态 `train_test` 与滚动/扩展 `walk_forward_rolling` / `walk_forward_expanding`（参数严禁使用样本外测试数据，外接基准严格裁剪至样本外重叠区间再锚定对比；`in_sample` 显式标注 `has_lookahead_bias=True`），支持 `rebalance_freq` 与双边 `transaction_cost_bps` 及多维敏感性扫描。
  - 为兼顾 M1/Streamlit（`total_return`）与 M2/M3 统一契约（`cumulative_return`），`compute_backtest_metrics` 同时输出两字段且保持值一致；`LOCAL` 跨币种混排通过 `mixed_local_currencies=True` 触发 `degraded` 告警而非隐式按 1:1 折算。
- **M1 语义拦截与口径对齐（[analysis_metrics.py](../analysis_metrics.py) / [asset_catalog.py](../asset_catalog.py)）**：
  - `get_quant_metrics` 与单资产风险/波动/Sharpe/Sortino 函数支持 `return_convention="log" | "simple"`（默认 `"log"`），与 [quant_models.py](../quant_models.py) `compute_backtest_metrics` 的 `"simple"` 口径可在同一 `periods_per_year` 下严格对齐，结果元数据显式声明 `return_convention` 与 `periods_per_year`。
  - `yield_level` 序列硬阻止无效收益/回撤/波动/Sharpe/Sortino/Calmar/VaR/CVaR（输出 `NaN` + `metric_reasons["..."] = "yield_level_not_return"`，改输出 `latest_level` / `level_change` / `level_min` / `level_max`），并在单/多资产转换层禁止基准货币换算；`fx_rate` 禁止非 `LOCAL` 换算并附无 Carry 警示。

---

## 3. 安全修复与向后兼容边界

- **向后兼容保留清单**：
  - CLI `quantitative-analysis`（`python main.py`）、Streamlit 入口 `streamlit run app.py`、`data_fetcher` / `asset_resolver` / `china_equity_entry` 既有导入路径、OHLCV CSV UTF-8 BOM 六列格式均保持兼容。
  - `main.py` 保留历史测试对 `load_china_equity_ohlcv` 的 monkeypatch 兼容路径；`app.py` 保留 `load_multi_leg_close_wide` `@st.cache_data` 包装以兼容旧测试清理调用。
- **服务与缓存边界加固（M2）**：
  - `services/analysis_service.py` 在分析与拉取前置阶段严格按 `[request.start_date, request.end_date]` 裁剪传入样本（空重叠直接抛出 `EmptySymbolDataError`），且 `build_quality_report` 与 `data_summary` 始终以实际交付样本与 FX 对齐后行数推导起止日期与质量范围。
  - 共享 Parquet + JSON 元信息缓存默认写入 `~/.cache/quantitative_analysis/ohlcv`（TTL 3600 秒）；M0 已修复禁止持久化 `proxy_url` 到 `.meta.json` 并在空切片时触发重拉；M1/M2 将进程内线程锁键细化至 `(cache_dir, key)`，在 POSIX `fcntl` 之外增加 Windows `msvcrt` 字节锁回退，`clear_ohlcv_cache()` 按 key 加锁且保留 `.locks/` 句柄避免破坏 POSIX inode 锁；M3 进一步加固 `save_cached_ohlcv` 覆盖已有缓存时的旧文件同盘备份（`_backup_existing_file`）与按替换标志精准回滚（`_restore_or_unlink_file`），确保中途失败时原有有效 `(parquet, meta.json)` 成对保留且不残余 `.tmp`/`.bak` 文件。
- **Agent CLI 事务性落盘与严格标的解析（M3）**：
  - [quant_cli.py](../quant_cli.py) `_write_outputs` 采用三阶段事务提交（全量临时暂存 -> 同文件系统旧文件/陈旧产物备份 -> 原子 `os.replace` 提交并按受管文件名集合清理上一次残留的陈旧产物，如从 `portfolio` 切换为 `compare` 时删除 `weights.parquet`/`frontier.parquet`/`backtest.parquet`），任意文件失败时完整回滚并还原覆盖或已删文件，绝不触碰用户非受管文件。
  - [asset_catalog.py](../asset_catalog.py) `normalize_symbol_alias` 在 `hk_stock`、`fx`、`gold` 之外新增 `ashare`/`etf` 交易所前缀/后缀（如 `sh600519` / `600519.SH` -> `600519`）规范化，配合 `quant_cli` 严格前缀校验消除多资产重复腿漏判与裸代码猜错类别风险。
  - Web Tab 1 `app.py` 在 M0 已对齐 `ohlcv.py` 真实的 `dropped_invalid_rows` / `filled_price_rows` / `missing_volume_rows` / `ohlc_anomaly_rows` / `non_positive_price_rows` 质量字段，确保清洗提示真实可见。
  - `data_sources/network.py` 的 `classify_network_exception` 在 M2 支持解析 `ChinaEquityPipelineError.network_related` 与 `DataFetchError` 包裹链，`tests/conftest.py` 的离线隔离门禁覆盖 `socket.connect`/`connect_ex`、`socket.getaddrinfo` 与 `requests`/`urllib3`/`httpx`/`urllib.request` 非回环外连（包括本地环境变量代理路径）。

---

## 4. 环境依赖与验证边界

- **Python 版本要求**：[pyproject.toml](../pyproject.toml) 指定 `requires-python >= 3.11`；本地主要开发与验证环境为 **Python 3.14.5**。
- **关键依赖约束**：[requirements.txt](../requirements.txt) 由 `pyproject.toml` 动态读取，包含 `scipy>=1.11.0`、`streamlit>=1.50.0`、`plotly>=5.18.0`、`pyarrow<26.0.0`、`ccxt>=4.0.0`、`akshare>=1.13.0`、`httpx>=0.27.0` 等；`pip install -e ".[dev]"` 同时注册 `quant` 与 `quantitative-analysis` 命令及 `pytest`、`ruff`。
- **离线测试与外部数据源边界**：
  - 自动化测试全部采用确定性样本或 monkeypatch 模拟数据源（默认 `QUANT_DISABLE_CACHE=1` 隔离），覆盖 OHLCV 清洗、数学公式、FX 折算、网络预算与代理策略、SLSQP 优化与 OOS 回测、共享缓存、CLI 与 Streamlit `AppTest`。
  - 真实行情接口（AKShare / 东财 / 新浪 / 腾讯 / CCXT）的可用性、限流、字段变化与复权精度取决于第三方上游，**无法通过离线测试保证**；实盘与浏览器端手工验收请核对 [CHINA_EQUITY_VERIFICATION.md](CHINA_EQUITY_VERIFICATION.md) 与 [FRONTEND_VERIFICATION.md](FRONTEND_VERIFICATION.md)。
