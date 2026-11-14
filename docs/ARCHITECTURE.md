# 工程架构、数据口径与兼容边界

> 更新日期：2026-11-14（M0 基线对齐）  
> 详细技术债清单、V0.1/V0.2 数据契约与模块拆分方案见 [M0 审计与契约文档](M0_AUDIT_AND_CONTRACTS.md)；长期阶段规划见 [产品定位与开发路线图](ROADMAP.md)。

---

## 1. 职责分层与模块依赖

当前工程由两个用户入口、资产目录与路由层、特征/指标/模型计算层、图表可视化层以及数据源与缓存基础设施组成：

- **入口层**：
  - [app.py](../app.py)：Streamlit 三标签页 Web 应用（`量化诊断看板`、`数学原理`、`进阶金融算法`），维护会话结果、侧边栏代理状态及 `@st.cache_data` 内存缓存。
  - [main.py](../main.py)：全 13 类资产命令行入口（已注册 CLI 脚本 `quantitative-analysis`），负责参数解析、文本诊断报告、`mplfinance` 图表与 CSV 导出。
- **资产与数据准备层**：
  - [asset_catalog.py](../asset_catalog.py)：13 类资产稳定 key、标签、默认值、交易日年化因子（`ASSET_TRADING_DAYS`）、默认币种、9 类收益语义 `ASSET_RETURN_BASIS` 及复权支持能力矩阵 `SUPPORTED_ADJUSTMENTS`。
  - [asset_resolver.py](../asset_resolver.py)（[china_equity_entry.py](../china_equity_entry.py) 保留兼容重导出）：分析窗口校验与境内 ETF/A 股统一错误信息格式化。
  - [asset_loader.py](../asset_loader.py)：全 13 类资产统一入口 `load_asset_data`、多线程并发拉取，以及 Stage 1 多市场对齐引擎 `align_multi_asset_frames` / `load_and_align_multi_asset_portfolio`（支持 `common_trading_dates`、`weekly_friday`、`mtm_outer`）。
  - [currency.py](../currency.py)：跨资产计价币种推断、向后防前视泄露 `merge_asof` FX 对齐（`max_staleness_days=4`）、`CNY`/`USD`/`LOCAL` 基准货币折算及显式 USDT/USD 锚定开关 `allow_usdt_usd_peg`。
  - [ohlcv.py](../ohlcv.py)：日期校验与统一 OHLCV 清洗、缺失策略（`raise`/`drop`/`ffill`）及 9 项质量诊断 `_compute_ohlcv_quality`。
  - [data_fetcher.py](../data_fetcher.py) 与 [data_sources/](../data_sources/)：各外部行情适配器（`china_equity.py`、`eastmoney.py`、`forex.py`、`funds.py`）、`RequestBudget` 与错误分类（`network.py`、`errors.py`）、以及 Web/CLI 共享 Parquet + JSON 元信息磁盘缓存（`cache.py`）。

**当前已知跨模块耦合边界（M1 解耦目标）**：
- `asset_loader.py` 顶层导入 `currency`，而 `currency._default_usdt_usd_loader` 在函数内部延迟导入 `asset_loader` 私有 CCXT 辅助函数，存在循环依赖。
- `data_fetcher.py` 与 `data_sources/forex.py` 直接导入部分底层私有符号；`main.py` 私有 `_convert_ohlcv_to_base_currency` 尚未下沉为共享模块能力。
- [YFinanceFetcher](../data_sources/base.py) 作为历史 API 保留占位，调用会发出 `DeprecationWarning` 并抛出 `NotImplementedError`；美股日线通过 AKShare 路由加载。

---

## 2. 数据与计算口径

- **OHLCV 契约（`ohlcv.py`）**：固定列为 `Open`、`High`、`Low`、`Close`、`Volume`（`float64`）；索引 `Date` 为无时区、`ns` 时间单位升序唯一 `DatetimeIndex`；去重保留最后一条。`missing_policy="drop"` 剔除无效价格行但**不前向拷贝昨日成交量**（缺成交量补 `0.0` 并计入 `missing_volume_rows`）。
- **特征工程（`feature_engineering.py`）**：计算 SMA、对数收益滚动年化波动率（`ddof=1`）、布林带、最大回撤（`1 - Close / cummax(Close)`）以及可选 EMA/RSI(14)/MACD；保留并透传 `df.attrs`。
- **单资产风险收益诊断（`analysis_metrics.py`）**：
  - CAGR 按自然日跨度（`days / 365.25`）与起止价格比计算；
  - 年化波动率、Sharpe、Sortino 使用**日对数收益率**、`daily_rf_log = log1p(rf) / trading_days`；年化因子按 `ASSET_TRADING_DAYS` 自适应（加密货币 `365`、外汇 `260`、其余默认 `252`）；
  - 95% 日 VaR / CVaR 使用日简单收益率分位数；常数序列或单日样本 Sharpe 安全返回 `NaN`。
- **多资产组合、有效前沿与样本外回测（`quant_models.py`）**：
  - 使用 **日简单收益率**、`rf_per_period = rf / periods_per_year`；多资产组合采用协方差收缩可选 `ledoit_wolf`（内置解析公式）与 SciPy `SLSQP` 精确优化 GMV / 最大夏普切线 / 精确有效前沿（支持权重上下界与最小持仓校验）；
  - 样本外验证支持静态 `Train/Test` 切分与滚动/扩展 `Walk-Forward` 回测（严禁前视偏差，支持再平衡频次、`tradable_mask` 停牌隔离与双边 `transaction_cost_bps`）。
- **已知口径差异（见 M0 审计文档）**：单资产诊断（对数收益 + 静态交易日）与组合优化/回测（简单收益 + 对齐期数）公式尚未统一；`yield_level` 序列仅禁止进入组合优化及 CLI 换汇，单资产诊断层目前仍只做文案警示。

---

## 3. 安全修复与向后兼容边界

- **向后兼容保留清单**：
  - CLI `quantitative-analysis`（`python main.py`）、Streamlit 入口 `streamlit run app.py`、`data_fetcher` / `asset_resolver` / `china_equity_entry` 既有导入路径、OHLCV CSV UTF-8 BOM 六列格式均保持兼容。
  - `main.py` 保留历史测试对 `load_china_equity_ohlcv` 的 monkeypatch 兼容路径；`app.py` 保留 `load_multi_leg_close_wide` `@st.cache_data` 包装以兼容旧测试清理调用。
- **代理与缓存边界**：
  - 导入模块不改写全局环境变量；境内直连与系统代理回退受线程隔离 `RequestBudget` / `_without_http_proxy_env` 控制，设 `QUANT_USE_SYSTEM_PROXY=1` 时强制走系统/环境代理，加密货币仅使用显式传入的 CCXT `proxy_url`。
  - 共享 Parquet + JSON 元信息缓存默认写入 `~/.cache/quantitative_analysis/ohlcv`（TTL 3600 秒）；M0 已修复禁止持久化 `proxy_url` 到 `.meta.json`（并在内存属性对代理凭证脱敏），且覆盖范围内切片为空时视为缓存未命中并重新触发上游拉取，避免抛 `ValueError`。
  - Web Tab 1 `app.py` 在 M0 已对齐 `ohlcv.py` 真实的 `dropped_invalid_rows` / `filled_price_rows` / `missing_volume_rows` / `ohlc_anomaly_rows` / `non_positive_price_rows` 质量字段，确保清洗提示真实可见。

---

## 4. 环境依赖与验证边界

- **Python 版本要求**：[pyproject.toml](../pyproject.toml) 指定 `requires-python >= 3.11`；本地主要开发与验证环境为 **Python 3.14.5**，GitHub Actions（[.github/workflows/tests.yml](../.github/workflows/tests.yml)）在 Python `3.11` / `3.12` / `3.14` 矩阵上执行回归。
- **关键依赖约束**：[requirements.txt](../requirements.txt) 由 `pyproject.toml` 动态读取，包含 `scipy>=1.11.0`、`streamlit>=1.50.0`、`plotly>=5.18.0`、`pyarrow<26.0.0`、`ccxt>=4.0.0`、`akshare>=1.13.0`、`httpx>=0.27.0` 等；`pip install -e ".[dev]"` 同时安装 `pytest`、`ruff` 与 `quantitative-analysis` 命令。
- **离线测试与外部数据源边界**：
  - 自动化测试全部采用确定性样本或 monkeypatch 模拟数据源（默认 `QUANT_DISABLE_CACHE=1` 隔离），覆盖 OHLCV 清洗、数学公式、FX 折算、网络预算与代理策略、SLSQP 优化与 OOS 回测、共享缓存、CLI 与 Streamlit `AppTest`。
  - 真实行情接口（AKShare / 东财 / 新浪 / 腾讯 / CCXT）的可用性、限流、字段变化与复权精度取决于第三方上游，**无法通过离线测试保证**；实盘与浏览器端手工验收请核对 [CHINA_EQUITY_VERIFICATION.md](CHINA_EQUITY_VERIFICATION.md) 与 [FRONTEND_VERIFICATION.md](FRONTEND_VERIFICATION.md)。
