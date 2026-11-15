# M0 现状审计、技术债清单与 V0.1/V0.2 架构契约

> 阶段：**M0（V0.1 准备工作）** · 基线分支：`codex/frontend-qa`（已合入 `origin/main` 路线图）  
> 主开发与验证环境：Python 3.14.5（[pyproject.toml](../pyproject.toml) 声明 `>=3.11`，CI 覆盖 `3.11` / `3.12` / `3.14`）

本文档固定 M0 阶段的全部审计事实、历史兼容边界、技术债优先级清单、`ROADMAP.md` 已实现能力重估以及 V0.1/V0.2 核心契约与模块拆分方案，作为 M1–M5 渐进式重构的单一设计依据。

---

## 1. 仓库实际能力盘点（Implemented / Limited / Planned）

### 1.1 能力成熟度总览

| 能力域 | 现状评级 | 已实现能力 | 有限支持 / 已知缺陷 | 后续计划（M1–M5） |
| :--- | :--- | :--- | :--- | :--- |
| **13 类资产数据接入** | **已实现（基础完善）** | [asset_catalog.py](../asset_catalog.py) 注册 `etf`, `ashare`, `mutual_fund`, `gold`, `crypto`, `us_stock`, `hk_stock`, `global_index`, `bond_cn`, `bond_us_yield`, `commodity`, `creit`, `fx`；[asset_loader.py](../asset_loader.py) `load_asset_data` 覆盖全部分支并支持境内多源回退、外盘外汇直连镜像与 CCXT Binance 分页。 | 全部依赖第三方公共行情与交易所接口；`YFinanceFetcher` 为 `NotImplementedError` 占位；各适配器在 `attrs["series_kind"]` 上尚未与 9 种 `ASSET_RETURN_BASIS` 一一对齐。 | **M1**：统一 `SeriesMetadata` 与 `series_kind`/`return_basis` 契约，完善降级诊断透传。 |
| **OHLCV 清洗与质量检查** | **已实现（骨架成型）** | [ohlcv.py](../ohlcv.py) `prepare_ohlcv` / `normalize_ohlcv` / `_compute_ohlcv_quality` 强制 float64 OHLCV 与无时区 ns 升序唯一日期索引，输出 9 项质量与异常计数（`dropped_invalid_rows`, `filled_price_rows`, `missing_volume_rows`, `zero_volume_rows`, `non_positive_price_rows`, `negative_volume_rows`, `ohlc_anomaly_rows`, `true_zero_return_rows`, `max_calendar_gap_days`）。 | 多资产并存旧宽表路径（`get_multiple_assets_close` / `load_multi_leg_close_wide`）；缓存切片非空返回时直接透传全历史 `data_quality` 计数而未按切片窗口重算。 | **M1**：输出强类型 `QualityReport`，规范切片质量重算并下线旧宽表。 |
| **跨市场日历与换汇** | **已实现（核心完备）** | [asset_loader.py](../asset_loader.py) Stage 1 `align_multi_asset_frames` 支持 `common_trading_dates` / `weekly_friday` / `mtm_outer` 并生成 `tradable_mask` 与禁止 `yield_level` 进入组合；[currency.py](../currency.py) 支持币种推断、防前视泄露 `merge_asof`（`max_staleness_days=4`）、`CNY`/`USD`/`LOCAL` 折算与显式 `allow_usdt_usd_peg`。 | CLI `_convert_ohlcv_to_base_currency` 尚未拦截 `fx_rate`（可被当成现金价换汇）且未把 `conv.warnings` 打印进报告；Web Tab 1 尚不支持基准货币折算。 | **M1**：将 OHLCV 换汇收敛到 `currency.py`/服务层，拦截 `fx_rate`，Web/CLI 共用转换与警告输出。 |
| **指标与特征计算** | **已实现（口径待统一）** | [feature_engineering.py](../feature_engineering.py) 计算 SMA、滚动波动率、布林带、回撤、EMA、RSI、MACD；[analysis_metrics.py](../analysis_metrics.py) 计算累计收益、自然日 CAGR、对数收益 Sharpe/Sortino、Calmar、95% 日 VaR/CVaR（按 `ASSET_TRADING_DAYS` 自适应 252/365/260）。 | ① 单资产（Log 收益）与组合（Simple 收益）两套年化/Sharpe 公式并存；② `yield_level` 仅改展示文案未阻止几何 CAGR/Sharpe 误算；③ CLI `main.py` 触发双次 `transform()`（第二次会重置自定义窗口）。 | **M1–M3**：明确并统一或区分 Log/Simple 口径；对 `yield_level`/`fx_rate` 硬阻止不适用的投资回报指标。 |
| **组合优化与 OOS 回测** | **已实现（超出早期规划）** | [quant_models.py](../quant_models.py)（Stage 3）已实现 GBM、相关性矩阵、Ledoit-Wolf 收缩协方差、SciPy SLSQP GMV/Tangency/精确有效前沿、权重上下界与 `split_train_test_returns` / 滚动与扩展 `run_walk_forward_backtest`（含调仓周期与 bps 费率）。 | 仅通过 Web Tab 3 与 Python API 暴露，CLI 尚无 `compare` / `portfolio`；保留了纯随机采样 `generate_efficient_frontier`；缺乏分布检验与敏感性分析。 | **M3**：通过 `quant compare` / `quant portfolio` 暴露并在报告层严格区分模拟、样本内拟合与样本外回测。 |
| **共享持久化缓存与网络韧性** | **已实现（M0 补齐安全修复）** | [data_sources/cache.py](../data_sources/cache.py) 实现 Parquet + `.meta.json` 跨端共享缓存；[data_sources/network.py](../data_sources/network.py) 提供 `RequestBudget` 与错误分类；M0 已修复 `.meta.json` 剔除 `proxy_url`、缓存空切片返回 `None` 触发重拉。 | 写入 `.parquet` 与 `.meta.json` 间无并发文件锁；标的别名键（如 `"700"` vs `"00700"`）未全归一化；`china_equity.py` 静默吞掉 Sina/Tencent 异常；`network.py` 依赖全局代理 monkeypatch。 | **M1/M4**：完善键规范化与结构化降级日志，按需评估更轻量并发锁。 |
| **Web UI（Streamlit）** | **已实现（业务耦合待解耦）** | [app.py](../app.py) 提供 3 Tab、参数记忆、自适应主题（[charts.py](../charts.py) 11 个 Plotly 构建器）、CSV 导出；M0 修复 Tab 1 `data_quality` 键名失配导致的警告静默失效。 | 仍直接编排加载、指标、优化与旧缓存函数；Tab 1 口径说明漏掉 `adjusted_price`/`cumulative_nav`/`spot_price`/`exchange_bond_price`；Tab 3 GBM 默认 `gbm_days` 硬编码 `252`（未按 `crypto=365` 自适应）。 | **M5**：变为薄 UI 层，直接调用 `AnalysisService`，按单资产/多资产比较/组合研究重组。 |
| **CLI（`main.py`）** | **有限支持** | 注册 `quantitative-analysis` 命令，支持全部 13 类资产、`--adjust`、`--base-currency`、`--refresh`、`--no-show`、`--save-chart`、`--export-csv` 与文本诊断报告。 | 缺少 `--rf`、均线/布林/动量参数与 `--allow-usdt-peg`；报错与结果全写 `stdout`、退出码全为 `1`；无 `quant` 子命令与 JSON/Parquet 输出；即使 `--no-show` 且未保存图片也无条件渲染 matplotlib 图表。 | **M2（V0.2 优先）**：新增面向 Agent 的 `quant fetch` / `quant analyze` CLI（保留 `quantitative-analysis` 向后兼容）。 |

---

## 2. `ROADMAP.md` 实际完成度重估（防止 M1–M5 重复开发）

由于 `origin/main:docs/ROADMAP.md` 基于旧基线 `1accbfa` 编写，而 `codex/frontend-qa` 已合入 Stages 1–5 成果，后续里程碑需按以下重估边界裁剪工作量：

| 路线图阶段 | 原计划中**已由 Stages 1–5 / M0 提前实现**的能力 | 实际**仍需开发**的核心增量 |
| :--- | :--- | :--- |
| **M1 · 统一金融数据层（V0.1）** | OHLCV 9 项质量诊断（`ohlcv.py`）、`align_multi_asset_frames` 3 种跨市场日历对齐（`asset_loader.py`）、防前视 `merge_asof` FX 转换（`currency.py`）、`yield_level` 组合拦截、Parquet + JSON 磁盘缓存（`data_sources/cache.py`）、M0 缓存防泄漏与空切片回退修复。 | 1. 落地强类型 `SeriesMetadata` 与 `QualityReport` 契约及统一 `AnalysisRequest`/`AnalysisResult` 模型。<br>2. 统一适配器 `series_kind` 与 `ASSET_RETURN_BASIS`；对 `yield_level` / `fx_rate` 硬阻止不可用指标与无效换汇。<br>3. 下沉 CCXT 拉取至 `data_sources/crypto.py`，切断 `asset_loader <-> currency` 循环耦合与 `data_fetcher` 私有导入。<br>4. 清理或弃用历史 `load_multi_leg_close_wide` / `get_multiple_assets_close`。 |
| **M2 · Agent CLI（V0.2 优先）** | 全 13 类资产终端运行链、`--adjust` / `--base-currency` / `--refresh` 与文本报告。 | 1. 抽出与 UI 无关的 `services/analysis_service.py`。<br>2. 注册新入口 `quant`（`quant fetch` / `quant analyze`），保留 `quantitative-analysis` 兼容层。<br>3. 输出版本化 `result.json` + `data.parquet`，分离 `stdout`（结构化结果）与 `stderr`（进度/错误），退出码分级（`0/2/3/4/5`），按需跳过绘图与重复 `transform()`。 |
| **M3 · 多资产与组合（V0.3）** | Ledoit-Wolf 收缩协方差、SciPy SLSQP 精确有效前沿（GMV/Tangency/权重约束）、Walk-Forward 滚动与扩展 OOS 回测、`tradable_mask` 与交易费率 bps、图表清晰区分“精确有效前沿”与“随机可行组合”。 | 1. 增加 `quant compare` 与 `quant portfolio` CLI，复用 `asset_loader.load_and_align_multi_asset_portfolio` 与 `quant_models`。<br>2. 对齐单资产（Log）与组合（Simple）年化/Sharpe/Sortino 计算口径或显式标注参数。<br>3. 清理遗留纯随机前沿入口 `generate_efficient_frontier`；按需补强分布检验、滚动相关性与参数敏感性输出。 |
| **M4 · 报告与图表生成（V0.4）** | `charts.py` 11 个自适应主题 Plotly 构建器、`visualizer.py` 静态 K 线/折线图导出。 | 1. 实现独立的 Markdown / HTML Report Renderer，仅消费 `AnalysisResult`，不重复拉数/重算。<br>2. 扩展独立 Chart Renderer 导出 HTML/PNG/SVG（复用 `charts.py`/`visualizer.py`）并由 CLI `--report` / `--charts` 控制。 |
| **M5 · Web UI 重构（V1.0）** | Tab 1/2/3 结构、参数持久化、亮暗主题自适应、WCAG 对比度热图与 AppTest 基础回归。 | 1. `app.py` 重构为调用 `AnalysisService` 的薄交互层，去除旧多资产包装及重复规则。<br>2. Tab 1 补齐基准货币折算、完整 `ASSET_RETURN_BASIS_NOTES` 展示、Tab 3 GBM 默认 `trading_days` 资产自适应。<br>3. Web/CLI 同快照同参数一致性测试。 |

---

## 3. 计算口径、金融语义与技术债详查

### 3.1 共享口径与语义偏差

1. **单资产（Log） vs 组合（Simple）两套年化风险收益公式**
   - [analysis_metrics.py](../analysis_metrics.py)：`compute_annualized_volatility`、`compute_sharpe_ratio`、`compute_sortino_ratio` 基于 **日对数收益率**，`daily_rf_log = np.log1p(risk_free_rate) / trading_days`，分母使用样本标准差（注意 Sortino 使用对 `0` 截断的总体均方根下行偏差）。
   - [quant_models.py](../quant_models.py)：`compute_backtest_metrics`、`_portfolio_stats` 与 `generate_efficient_frontier` 基于 **日简单收益率**，`rf_per_period = risk_free_rate / ppy`，且多资产对齐后 `periods_per_year` 为实际日历折算年化期数（`effective_periods_per_year`），而非单资产静态 `ASSET_TRADING_DAYS`（252 / 365 / 260）。
   - **M0 处理**：保留现状不做破坏性改动，写入口径边界并在 M1/M3 显式对齐或通过元数据字段 `return_convention` / `periods_per_year` 声明。
2. **CLI 双重特征计算与参数覆盖风险**
   - [main.py](../main.py) `run_analysis` 先调用 `engineer.transform(raw_df, trading_days=...)`，随后 `build_diagnostic_report` 调用 `get_quant_metrics(featured_df)`，后者内部会再按默认 `sma_short_window=20, sma_long_window=60, bollinger_window=20` 重算并覆盖列（已有单测 `test_metrics_recompute_stale_metadata_and_accept_features` 依赖“重算陈旧 `attrs`”行为）。M1 服务化后应由单一入口接收窗口参数并一次完成 `transform`。
3. **Web 与 CLI 参数与提示不对称**
   - Web 支持 `rf_pct`、`sma_short`、`sma_long`、`bb_window`、`include_momentum`、`allow_usdt_peg`；CLI 当前缺少这些参数（默认 `rf=0.016` 且无法开启 RSI/MACD/USDT 锚定）。
   - CLI 支持 `--base-currency CNY|USD` 换汇，但 Web Tab 1 不支持；且 CLI 未把 `conv.warnings` 写入报告文本，也未禁止 `--asset-type fx --base-currency USD` 的无效换汇。
   - Web Tab 1 仅对 `("futures_continuous", "price_index", "fx_rate", "adjusted_or_price")` 显示 `ASSET_RETURN_BASIS_NOTES`，遗漏了 `adjusted_price`、`cumulative_nav`、`spot_price`、`exchange_bond_price`（而 CLI 报告均会展示）。
   - Web Tab 3 单资产 GBM 默认预测天数 `gbm_days` 取常量 `TRADING_DAYS`（252），未对 `crypto` 采用 365。
4. **`yield_level` 语义与 `series_kind` 对齐不完整**
   - `align_multi_asset_frames` 与 CLI 换汇正确拒绝 `yield_level`，但 `analysis_metrics.get_quant_metrics` 仅修改展示标签（`cumulative_return_label` / `cagr_label` / `metric_disclaimer`），数值字段仍对收益率百分比做几何增长率和夏普计算；`data_fetcher.py` 与 `data_sources/funds.py` 多处把基金/外汇/商品/期货 `series_kind` 仍笼统标为 `"price"`。

### 3.2 待办技术债清单（P0 / P1 / P2）

| 编号 | 级别 | 位置与问题描述 | 当前状态 / 归属里程碑 |
| :--- | :--- | :--- | :--- |
| **TD-01** | **P0** | [data_sources/cache.py](../data_sources/cache.py) 持久化 `.meta.json` 时未过滤 `attrs["proxy_url"]`，可能明文保存含账号密码的代理 URL；[asset_loader.py](../asset_loader.py) 内存中也未对 `proxy_url` 凭证脱敏。 | **已在 M0 修复**（落盘过滤 `proxy_url` + 内存凭证脱敏 + 回归测试）。 |
| **TD-02** | **P0** | [app.py](../app.py) 读取 `data_quality["dropped_rows"]` / `ffilled_cells`，与 [ohlcv.py](../ohlcv.py) 实际输出的 `dropped_invalid_rows` / `filled_price_rows` / `missing_volume_rows` 不匹配，导致 Web 数据质量警告静默失效。 | **已在 M0 修复**（键名对齐并兼容旧键 + AppTest 回归测试）。 |
| **TD-03** | **P0** | [data_sources/cache.py](../data_sources/cache.py) `load_cached_ohlcv` 在覆盖范围内切片结果为空时直接返回空 DataFrame，导致下游 `OHLCVFeatureEngineer` 抛 `ValueError` 而非缓存未命中回退重拉。 | **已在 M0 修复**（空切片返回 `None` 视作未命中并重新拉取 + 回归测试）。 |
| **TD-04** | **P1** | `asset_loader.py`（顶层导入 `currency`）与 `currency._default_usdt_usd_loader`（函数内导入 `asset_loader` 私有 CCXT 函数）存在循环耦合；`data_fetcher.py` 与 `forex.py` 直接导入跨模块私有函数/常量。 | **已在 M1 修复**：抽离 [data_sources/crypto.py](../data_sources/crypto.py) 与各数据源公开契约函数，保留原模块别名兼容既有单测 `monkeypatch`。 |
| **TD-05** | **P1** | `cache.py` 切片未按区间窗口重算结构化质量字段（`total_rows` / `max_calendar_gap_days` 等）；`ohlcv.prepare_ohlcv` 在二次进入特征工程（默认 `missing_policy="raise"`）时会覆盖上游原有的 `data_quality["missing_policy"]`；缓存双写无进程间锁；同标的别名（如 `"700"` vs `"00700"`、`"USD/CNY"` vs `"USDCNH"`）未在缓存键阶段统一归一化。注：含权价格（`qfq`/`hfq`）不得跨批次直接拼接条数，仍应保持整块一致刷新。 | **已在 M1 修复**：`cache.py` 补齐切片重算（`quality_scope`）、`normalize_symbol_alias` 别名归一化与 `.locks/` 文件/线程锁；`ohlcv.py` 二次校验保留上游 `missing_policy`。 |
| **TD-06** | **P1** | `get_quant_metrics` 对 `yield_level` 仍输出几何 CAGR/Sharpe 数值；CLI `_convert_ohlcv_to_base_currency` 允许对 `fx_rate` 换汇且吞掉 `fx_warnings`。 | **已在 M1 修复**：`yield_level` 全部 9 项收益/风险指标硬阻止（`NaN` + `metric_reasons`）并禁用换汇；`currency.convert_ohlcv_to_base_currency` 与 `asset_loader.align_multi_asset_frames` 拦截非 `LOCAL` `fx_rate` 并去重保留 `fx_warnings`。 |
| **TD-07** | **P1** | `main.py` 仍包含针对旧单测的 `load_china_equity_ohlcv` monkeypatch 特判分支（会绕过 `load_asset_data` 缓存与元数据补全）；报错与报告均写 `stdout`，退出码恒为 `1`；无条件调用 `visualizer.plot()` 带来不必要渲染开销。 | **部分已在 M1 完成，余项归 M2**：已抽取 [services/analysis_service.py](../services/analysis_service.py) 供 `main.run_analysis` 单次串联执行（消除双次特征计算）；M2 将落地 `quant` CLI、结构化错误退出码与按需绘图。 |
| **TD-08** | **P1** | `data_sources/network.py` 全局 monkeypatch `requests`/`urllib` 代理函数并在线程局部暂改环境；`data_sources/china_equity.py` `except Exception: return pd.DataFrame()` 静默吞掉 Sina/Tencent 异常。 | **部分已在 M1 修复，余项归 M4**：`china_equity.py` 已记录 `_LOGGER.warning` 与 `fallback_warnings` 透传契约；`network.py` `Session` 隔离替代全局补丁留待 M4。 |
| **TD-09** | **P2** | 遗留并存 API：`get_multiple_assets_close`、`load_multi_leg_close_wide`（及 `app.py` 兼容缓存包装）、`generate_efficient_frontier`（纯随机采样）、`YFinanceFetcher`（占位）；`pyproject.toml` 配有 `[tool.mypy]` 但未接入 `dev` 依赖或 CI，`per-file-ignores` 忽略 `F401`。 | **待 M1/M3/M5 分批清理**（清理前须迁出 `tests/test_application_flows.py` 等既有单测）。 |

---

## 4. V0.1 / V0.2 核心数据契约与模块拆分方案

### 4.1 统一数据契约（M1 落地，M2 CLI 直接消费）

```python
@dataclass(frozen=True)
class AnalysisRequest:
    asset_type: str  # asset_catalog.SUPPORTED_ASSET_TYPES 之一
    symbol: str  # 原始用户输入符号
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    adjust: str = "qfq"  # resolve_adjustment 校验结果
    base_currency: str = "LOCAL"  # "LOCAL" | "CNY" | "USD"
    risk_free_rate: float = 0.016  # 年化无风险利率（小数）
    trading_days: int | None = None  # 默认 ASSET_TRADING_DAYS[asset_type]
    sma_short: int = 20
    sma_long: int = 60
    bb_window: int = 20
    include_momentum: bool = False
    allow_usdt_peg: bool = False
    force_refresh: bool = False


@dataclass(frozen=True)
class SeriesMetadata:
    asset_type: str
    symbol: str
    canonical_symbol: str
    adjust: str
    return_basis: str  # ASSET_RETURN_BASIS 9 种值之一
    series_kind: str  # 与语义契约对齐的序列类型
    native_currency: str | None
    output_currency: str | None
    fx_converted: bool
    fx_pair: str | None
    trading_days: int
    data_source: str | None
    fetched_at: str | None  # ISO-8601 UTC 时间戳
    cache_hit: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityReport:
    missing_policy: str
    total_rows: int
    start_date: str
    end_date: str
    dropped_invalid_rows: int = 0
    filled_price_rows: int = 0
    missing_volume_rows: int = 0
    zero_volume_rows: int = 0
    non_positive_price_rows: int = 0
    negative_volume_rows: int = 0
    ohlc_anomaly_rows: int = 0
    true_zero_return_rows: int = 0
    max_calendar_gap_days: int = 0


@dataclass(frozen=True)
class AnalysisResult:
    schema_version: str  # "0.1.0"
    status: str  # "ok" | "degraded" | "error"
    asset: SeriesMetadata
    quality: QualityReport
    parameters: dict[str, Any]
    metrics: dict[str, Any]  # 结构化数值（不适用/不足时为 None + metric_reasons）
    metric_reasons: dict[str, str]  # 机读缺失或不适用原因（如 yield_level 禁算 CAGR）
    latest_snapshot: dict[str, Any]
    data_frame: pd.DataFrame | None  # 服务层内存传递；CLI 按需导出 data.parquet
    warnings: tuple[str, ...] = ()
    disclaimers: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
```

### 4.2 渐进式模块拆分边界（不引入重型框架，零破坏）

1. **`data_sources/crypto.py`（M1）**：迁入 CCXT 配置与分页逻辑（`_build_ccxt_proxy_config`、`_fetch_crypto_ohlcv_rows`、`_normalize_crypto_data`），让 `asset_loader.py` 与 `currency.py` 同时依赖 `data_sources/crypto.py`，消除循环导入（`asset_loader.py` 保留别名重导出以兼容现有单测）。
2. **`currency.py`（M1）**：接收原 `main._convert_ohlcv_to_base_currency` 为公开 `convert_ohlcv_to_base_currency(...)`，补充 `fx_rate` 拦截与 `warnings` 合并到 `df.attrs["fx_warnings"]`。
3. **`services/analysis_service.py`（M1–M2）**：串联 `asset_resolver` -> `asset_loader` -> `currency` -> `OHLCVFeatureEngineer`（一次计算，避免参数覆盖） -> `analysis_metrics` -> `AnalysisResult`。`main.py` 保持 `quantitative-analysis` 旧参数与退出码完全兼容；M2 新增 `quant` CLI（`quant fetch` / `quant analyze`）输出 `result.json` + `data.parquet` 到 `output` 目录，日志写 `stderr`，退出码映射 `data_sources/errors.py`（`0` 成功、`2` 参数/语义校验错误、`3` 无数据 `EmptySymbolDataError`、`4` 网络/代理故障、`5` 上游接口结构/协议异常）。
4. **Web UI 兼容过渡（M1–M5）**：`app.py` 继续保留 `@st.cache_data` 会话缓存与 3-Tab 外壳，待服务层稳定后再切换调用。

---

## 5. 测试基线与外部行情接口边界

### 5.1 离线自动化基线

- **运行方式**：`MPLBACKEND=Agg QUANT_DISABLE_CACHE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
- **检查工具**：`ruff check .`、`ruff format --check .`、`python -m pip check`
- **CI 矩阵**：[.github/workflows/tests.yml](../.github/workflows/tests.yml) 在 Python `3.11` / `3.12` / `3.14` 上全量执行测试、Lint、格式与依赖完整性检查。
- **测试分布**：覆盖 `tests/` 下的契约与数学公式、网络预算与代理隔离、SLSQP/OOS 组合模型、Web AppTest 与状态持久、Plotly/mplfinance 图表、共享缓存与 CLI 端到端。

### 5.2 离线测试无法覆盖的外部边界

自动化测试全部基于确定性合成数据与 monkeypatch 模拟，以下边界**不纳入离线 CI 保证**，变更数据源或升级依赖后需按 [CHINA_EQUITY_VERIFICATION.md](CHINA_EQUITY_VERIFICATION.md) 与 [FRONTEND_VERIFICATION.md](FRONTEND_VERIFICATION.md) 做实盘/浏览器验证：

1. AKShare 上游函数名、返回列名、反爬策略与东财/新浪/腾讯多节点实时可用性及复权历史数值一致性。
2. CCXT Binance 分页、Kraken/Coinbase/Bitfinex USDT-USD 汇率接口在不同区域代理下的连通性。
3. 真实桌面/移动端浏览器对最新 Streamlit 与 Plotly 版本的 DOM/Canvas 交互渲染细节。
