# 产品定位与开发路线图（Roadmap）

> 状态：**M3（V0.3）已完成** · 更新：2026-11-15  
> V0.1（M0–M1）、V0.2（M2）与 V0.3（M3：`quant fetch` / `analyze` / `compare` / `portfolio`）已落地；M4（按需报告/图表生成）与 M5（Web UI 全面迁移至统一服务层）仍为规划中里程碑。现有功能与口径细节以 [README](../README.md)、[工程架构](ARCHITECTURE.md) 与 [M0 审计与契约](M0_AUDIT_AND_CONTRACTS.md) 为准。
>
> **M0 重估提示**：因近期代码分支（Stages 1–5）已提前落地了多市场日历对齐、FX 换汇、9 项 OHLCV 质量诊断、SLSQP 精确有效前沿、Ledoit-Wolf 收缩、Walk-Forward 样本外回测与 Parquet 共享缓存，以下 M1/M3/M5 中的部分基础工作已具备实现，**后续开发须先对照 [M0 审计与契约](M0_AUDIT_AND_CONTRACTS.md#2-roadmapmd-实际完成度重估防止-m1m5-重复开发) 避免重复造轮子**。

## 1. 产品定位

**Quantitative Analysis** 是一个面向人类与 AI Agent 的多资产金融数据处理和量化分析工具箱。项目有两个平行目标：

1. **面向人类的资产可视化平台**：通过 Web 界面搜索、查看和比较资产的历史表现、风险收益指标、统计特征及组合表现，支持交互式图表和参数探索。
2. **面向 Agent 的金融数据与指标工具模块**：Agent 通过简单的 CLI 命令提交标的、时间范围和分析参数，由模块自动完成**资产识别与数据检索 → 清洗与标准化 → 数据质量校验 → 指标计算 → 结果封装**。Agent 直接读取结构化结果开展进一步推理；按需生成供人类阅读的报告和图表。

**职责边界**：工具负责可验证的数据准备、计算、模型执行与结果呈现；Agent 负责结合财报、新闻、宏观背景等信息进行解释、推理与判断。工具不把技术指标或模型模拟结果自动当作投资建议。

## 2. 设计原则

- **单一分析核心、双入口**：Web UI 和 CLI 调用同一套与界面无关的分析服务，避免两份指标实现。
- **CLI-first for Agents**：优先提供稳定、可组合的 CLI 与机器可读结果；Python 核心库作为内部基础，MCP 等协议适配层按实际需要再添加。
- **金融语义优先**：明确区分股票价格、复权价格、基金净值、利率/收益率水平、期货合约等。指标的适用范围由数据契约与校验逻辑保障，而非仅通过 UI 提示。
- **可追溯与可复现**：结果包含数据源、获取时间、实际样本区间、资产标识、币种、复权/转换口径、缺失处理、指标定义、计算参数、软件版本及警告；固定输入数据和配置应得到一致的计算结果。
- **明确失败，不静默修复**：对缺失、跨市场日历、币种不一致、潜在前视偏差等做显式检查。不能可靠处理时给出可机读的错误或警告，禁止无说明地生成貌似可信的指标。
- **渐进式重构**：优先复用现有模块，保留当前可用 Web/CLI 功能；通过回归测试保障改造。
- **输出按需生成**：结构化结果为主要产物，报告和图表为可选产物；不强制每次执行耗时渲染。

## 3. 目标架构

```text
Human / Browser                  AI Agent / Terminal
       |                                  |
  Streamlit Web UI                     CLI (quant)
       |                                  |
       +---------- Analysis Service ------+
                         |
           +-------------+-------------+
           |             |             |
       Data Layer   Analytics Layer  Model Layer
           |             |             |
      Data Quality   Metrics/Stats   Simulation/
      & Provenance   & Comparison    Portfolio
           +-------------+-------------+
                         |
                 Structured Result
                 /       |        \
          JSON + data  Report    Charts
          (Parquet)    MD/HTML   HTML/PNG/SVG
```

建议实现统一的“分析请求 → 分析结果”契约：分析服务接收明确的资产、日期、计算参数，返回结构化指标、元数据、质量检查、警告和可选文件引用。图表与报告只消费分析结果，不重复拉数或计算。Streamlit 不再承担业务逻辑。

### 现有代码的复用方向

| 当前模块 | 后续职责 |
| --- | --- |
| `data_fetcher.py`, `data_sources/` | 数据源与检索适配 |
| `asset_catalog.py`, `asset_loader.py`, `asset_resolver.py` | 资产解析与统一加载 |
| `ohlcv.py` | 清洗、校验；扩展金融语义与元数据契约 |
| `feature_engineering.py`, `analysis_metrics.py` | 指标与统计计算 |
| `quant_models.py` | 可选模拟、组合及回测能力 |
| `charts.py`, `visualizer.py` | 可视化渲染 |
| `main.py` | 从当前有限 CLI 向统一命令入口演进 |
| `app.py` | 仅处理交互、展示和调用分析服务 |

## 4. 分阶段开发目标

实施顺序按 **M0 → M1 → M2 → M3 → M4 → M5**。各阶段应形成独立、可测试的增量交付；必要时先实现满足下一阶段的最小数据契约，再迭代扩展。

### M0 · 现状审计与基线（V0.1 的准备工作 — 已完成）

**目标**：确认仓库的实际能力、技术债与兼容边界，避免边开发边改变设计。

**工作内容**
- [x] 盘点 Web、CLI、数据源、指标、图表及模型的实际功能，区分已实现、有限支持与计划功能。
- [x] 审查 Web/CLI 计算是否共享口径；检查资产语义、复权、缺失值填充、跨市场日历、币种与模拟/回测标签。
- [x] 固定现有测试基线，记录外部行情接口无法由离线测试保证的边界，并修复 3 项向后兼容的 P0 缺陷（Web 数据质量键名失配、缓存落盘泄露 `proxy_url`、缓存空切片未按未命中重拉）。
- [x] 形成核心分析请求/结果契约和后续模块拆分方案（见 [M0 审计与契约](M0_AUDIT_AND_CONTRACTS.md)）。

**验收标准**：已达成一致的架构说明、待办问题清单（TD-01 至 TD-09）与全量通过的 Python 3.14 离线测试基线；`app.py` 与 `quantitative-analysis` 现有入口保持可用。

> **M1/M3/M5 现状重估（基于 M0 代码审计）**：
> - **M1 已具备骨架**：`ohlcv.py` 已输出 9 项 `data_quality` 统计，`asset_loader.py` 已实现 3 种跨市场日历对齐及 `yield_level` 组合拦截，`currency.py` 已实现防泄露向后 FX 转换，`data_sources/cache.py` 已实现共享缓存；M1 重点转为落地 `SeriesMetadata`/`QualityReport` 强契约、消除 `asset_loader <-> currency` 循环依赖、单资产侧拦截 `yield_level` / `fx_rate` 无效计算及对齐 `series_kind`。
> - **M3 算法核心已先行落地**：`quant_models.py` 已通过 SLSQP 精确有效前沿与 `run_walk_forward_backtest` 实现约束优化、样本内外划分与调仓/费率控制，`charts.py` 已同图区分精确有效前沿与随机组合；M3 重点转为提供 `quant compare`/`quant portfolio` CLI 及对齐 Log/Simple 年化与夏普口径。
> - **M5 前端基础已优化**：`app.py` 与 `charts.py` 已完成自适应主题与会话结果保存；M5 重点转为调用统一 `AnalysisService` 与补齐单资产基准货币选项。

### M1 · 统一金融数据层（V0.1 — 已完成）

**目标**：让数据不仅格式统一，而且来源明确、经济意义清楚、质量可检查。

**工作内容（已在 M1 落地）**
- [x] 建立强类型统一契约 [contracts.py](../contracts.py)（`SCHEMA_VERSION = "0.1.0"`）：`AnalysisRequest`、`SeriesMetadata`、`QualityReport`、`AnalysisResult`（支持 strict JSON `to_dict()`），统一输出资产/源/口径/质量/诊断原因与告警。
- [x] 在 [asset_catalog.py](../asset_catalog.py) 统一 `normalize_symbol_alias`、`resolve_return_basis`（覆盖动态复权 `adjusted_price` / `unadjusted_price`）与 `resolve_semantic_series_kind`。
- [x] 修复金融语义与口径：
  - `yield_level` 在 `analysis_metrics.get_quant_metrics` 硬阻止 `cumulative_return` / `cagr` / `max_drawdown` / 波动率 / Sharpe / Sortino / Calmar / VaR / CVaR（输出 `NaN` + `metric_reasons="yield_level_not_return"`，保留描述性 `latest_level` / `level_change`），并在 `currency.py` / `asset_loader.py` 禁止基准货币折算。
  - `fx_rate` 在 `currency.convert_ohlcv_to_base_currency` 与 `asset_loader.align_multi_asset_frames` 禁止非 `LOCAL` 基准货币换汇，并在单资产统计附带无 Carry 警示；有效 FX 折算后去重透传 `fx_warnings` 并在删行时重算 `data_quality`。
  - 对齐单资产（`return_convention="log" | "simple"`、`periods_per_year`）与 `quant_models.py` 组合简单收益口径；`backtest_portfolio` / `run_walk_forward_backtest` 元数据显式声明 `return_convention="simple"`。
- [x] 加固 OHLCV 质量与共享缓存：`ohlcv.prepare_ohlcv` / `validate_ohlcv` 保留上游 `missing_policy`；`data_sources/cache.py` 统一别名键、引入 `.locks/` 进程/线程读写锁、按返回切片重算 `data_quality`（`quality_scope="slice" | "full"`）。
- [x] 模块解耦：抽离 CCXT 至 `data_sources/crypto.py` 切断 `asset_loader <-> currency` 循环依赖；收敛 `convert_ohlcv_to_base_currency` 至 `currency.py`；清理跨模块私有导入并记录 `china_equity.py` 回退降级告警；落地共享分析服务 [services/analysis_service.py](../services/analysis_service.py)（`analyze_single_asset`），由 `main.run_analysis` 单次调用串联（避免双次特征重算）并为 M2 `quant` CLI 奠基。

**验收标准**：加载与分析单资产可得到标准化序列、`SeriesMetadata`、`QualityReport`、`AnalysisResult` 与结构化警告；`yield_level` 与 `fx_rate` 误用可靠拒绝；M1 与离线网络回退全量测试通过。

### M2 · Agent CLI 与统一分析工作流（V0.2 — 已完成）

**目标**：Agent 用一条命令得到可直接分析的资产数据和指标，无需手写数据抓取与计算脚本。

**工作内容（已在 M2 落地）**
- [x] 在 [services/analysis_service.py](../services/analysis_service.py) 抽取统一预处理骨架 `_prepare_single_asset_frame`，新增 `fetch_single_asset` 并与 `analyze_single_asset` 共享日期窗口裁剪、实际样本边界重算、元信息规范化与基准货币折算，避免重复拉取与重复计算；在 [contracts.py](../contracts.py) 新增 `metric_units`、`data_summary`（显式区分 `requested_*` 与 `actual_*`）、`software_version` 与 `command` 字段。
- [x] 新增 Agent 专用命令行入口 [quant_cli.py](../quant_cli.py)（在 [pyproject.toml](../pyproject.toml) 注册脚本命令 `quant`），提供 `quant fetch` 与 `quant analyze`，支持全 13 类资产（含 `TYPE:SYMBOL` 简写）、代码、日期区间、复权方式、`LOCAL`/`CNY`/`USD` 基准货币、`--allow-usdt-peg`、`--proxy`、`--refresh` 及全部单资产指标计算参数（`--rf`、`--trading-days`、`--return-convention`、`--sma-short`、`--sma-long`、`--bb-window`、`--include-momentum`），不触发任何图表渲染。
- [x] 标准化输出与自动化协议：`--output <dir>` 原子落盘 `result.json` 与 `data.parquet`（可选 `--csv` 输出 `data.csv`），重复执行幂等安全覆盖；`stdout` 严格仅输出可解析 JSON（`allow_nan=False`），`stderr` 输出精简进度、告警与结构化 JSON 错误对象，并映射稳定退出码（`0` 正常/降级、`2` 调用参数/校验错误、`3` 空数据、`4` 网络传输错误、`5` 上游协议错误、`1` 内部/I/O 异常）。
- [x] 保留历史兼容性与加固边界：保留 `quantitative-analysis`（[main.py](../main.py)）与 Streamlit `app.py` 兼容；加固缓存线程/进程跨平台锁（POSIX `fcntl` + Windows `msvcrt` 回退、目录作用域锁键、`clear_ohlcv_cache` 保留 `.lock`、Parquet+Meta 成对回滚），强化 `tests/conftest.py` DNS/Socket/HTTP 客户端离线隔离与异常分类，并建立 `tests/test_m2_agent_cli_and_hardening.py` 端到端及边界回归测试。

**接口示例**

```bash
quant fetch ashare:600519 --start 2024-01-01 --end 2026-11-13 -o ./results/600519_raw
quant analyze -t us_stock -s NVDA --start 2024-01-01 --end 2026-11-13 -o ./results/nvda
```

**验收标准**：编程 Agent 仅凭终端命令 `quant` 与退出码即可无交互完成单资产检索、校验、指标计算与 `result.json` / `data.parquet` 读取；在固定数据和参数下结果稳定可复现。

### M3 · 多资产比较、统计与组合分析（V0.3 — 已完成）

**目标**：从单资产诊断扩展到常用金融研究任务。

**工作内容（已在 M3 落地）**
- [x] 新增 `quant compare` 与 [services/multi_asset_service.py](../services/multi_asset_service.py) `compare_assets`：标准化归一化净值比较（`data.parquet`）、统一基准货币收益风险指标、Pearson/Spearman 相关性与 Ledoit-Wolf/样本/对角协方差、收益分布统计（偏度、超额峰度、Jarque-Bera 正态性检验、自相关）、Memmel (2003) 渐近夏普差异检验、相对基准指标（Tracking Error、Information Ratio、CAPM Beta/Jensen Alpha、R²、Up/Down Capture）及滚动统计。
- [x] 新增 `quant portfolio` 与 [services/multi_asset_service.py](../services/multi_asset_service.py) `analyze_portfolio`：支持 6 类策略（`max_sharpe`、`min_variance`、`risk_parity`、`inverse_volatility`、`equal_weight`、`custom`），输出权重、Euler 风险贡献（`compute_risk_contributions`、分散化比率、有效资产数）、外部基准对照（严格裁剪至样本外重叠区间）与历史回测。
- [x] 严格区分优化与验证边界：在 `frontier.parquet` 与 JSON 中通过 `frontier_type` / `algorithm` 明确分离约束优化 `slsqp_exact`（SLSQP 精确有效前沿）与 `dirichlet_random`（Dirichlet 随机采样散点）；默认采用严格无前视偏差的 `train_test` 与 `walk_forward_rolling` / `walk_forward_expanding` 样本外验证（`in_sample` 显式标记 `has_lookahead_bias=True` 并发出警告），明确调仓频次（含 `tradable_mask` 停牌顺延）、双边交易成本（`--tx-bps`）与协方差/费率/再平衡敏感性网格（`--no-sensitivity` 可跳过）。
- [x] 明确多市场对齐、非同步交易与汇率处理：`mtm_outer` 仅用于连续净值展示与非交易日隔离，相关性/协方差严格采用共同真实报价区间（`correlation_returns_base`）；`LOCAL` 多币种混排显式标注 `mixed_local_currencies=True` 并降级告警；`yield_level`（任何基准货币）与 `fx_rate`（非 `LOCAL` 基准货币）继续在比较/组合前置阶段拦截。
- [x] 修复 M2 遗留边界：完善 [data_sources/cache.py](../data_sources/cache.py) 覆盖已有缓存时的双文件备份与失败回滚（不留残留 `.tmp`/`.bak`）；升级 [quant_cli.py](../quant_cli.py) 多文件事务性提交（先 staged 全量暂存、同盘备份后原子替换、按 `failed/rolled-back` 回滚且仅清理不再产出的受管文件名，绝不误删用户文件）；收紧 CLI 资产类型解析规则，禁止裸代码静默默认猜为 `etf`，规范 A 股/ETF 交易所前缀/后缀别名防重复腿；扩展 [contracts.py](../contracts.py) `AssetSpec` / `CompareRequest` / `PortfolioRequest` / `MultiAssetResult` 并保留向后兼容（`SOFTWARE_VERSION = "0.3.0"`）。

**接口示例**

```bash
quant compare etf:510300 us_stock:SPY --benchmark etf:510500 --start 2024-01-01 --end 2026-11-13 --base-currency CNY -o ./results/compare
quant portfolio etf:510300 etf:510500 etf:518880 --strategy risk_parity --validation-mode walk_forward_rolling --train-window 60 --tx-bps 10 -o ./results/portfolio
```

**验收标准**：可用 CLI 完成典型多资产比较和组合研究，结果包含可核对的样本、参数、基准、质量与方法说明；模拟不冒充实际预测或有效策略表现；离线回归测试全量通过。

### M4 · 可阅读报告与可视化生成（V0.4）

**目标**：同一份分析结果既便于 Agent 消费，也便于人类审阅。

**工作内容**
- 建立独立 Report Renderer：按分析任务生成 Markdown / HTML 报告，包含数据说明、统计结果、方法和假设、质量警告与图表引用。
- 建立独立 Chart Renderer：按需输出交互式 HTML、PNG 或 SVG，覆盖价格/净值、回撤、滚动统计、多资产对比和组合等适用场景。
- 支持通过 CLI 控制是否生成报告、生成格式及图表集合；不重复获取数据或重新计算指标。
- 报告明确区分历史统计、统计推断、模型模拟和回测结果；避免由技术指标直接推出买卖结论。

**目标接口示意（尚未实现）**

```bash
quant analyze NVDA --start 2021-01-01 --report md --charts html,png
```

**验收标准**：一次命令可按需得到 JSON、标准化数据、Markdown/HTML 报告及图表；报告数值与结构化结果一致，图表不改变计算口径。

### M5 · Web UI 重构与统一验收（V1.0）

**目标**：形成可长期维护的 Web + CLI 双入口金融分析平台。

**工作内容**
- Streamlit 改为调用统一分析服务，界面围绕**单资产探索、多资产比较、投资组合研究**组织。
- 数学原理成为指标帮助或独立文档；模型模拟及高级选项按需展开，避免功能横向堆积。
- 验证 Web 与 CLI 在相同数据快照和参数下输出一致；保留数据导出与交互图表。
- 完善跨平台安装与运行文档、错误提示、CI、测试、数据源故障与缓存策略；使用真实 Agent 场景进行端到端验收。

**验收标准**：Web 和 CLI 共用同一计算口径；能够从资产查询到图表、数据、指标与报告完成全流程；自动化测试与文档能够支持持续维护。

## 5. 版本与交付重点

| 版本 | 覆盖阶段 | 可验证交付 |
| --- | --- | --- |
| **V0.1** | M0 + M1 | **已完成**：统一的 `contracts.py` 数据契约、`AnalysisService` 基础服务、质量诊断、语义拦截与离线测试基线 |
| **V0.2** | M2 | **已完成**：Agent CLI（`quant fetch` / `quant analyze`）、`result.json` + `data.parquet` 产物、退出码契约与缓存/离线加固 |
| **V0.3** | M3 | **已完成**：多资产比较（`quant compare`）、组合与无前视偏差回测（`quant portfolio`）、分布/Memmel/风险贡献统计及 M2 事务性落盘与解析加固 |
| **V0.4** | M4 | **下一步**：消费统一 JSON + Parquet 产物的按需 Markdown / HTML 报告与图表渲染 |
| **V1.0** | M5 | Web/CLI 共享分析核心的完整平台 |

**近期开发重心将转向 V0.4（M4 按需报告与图表生成）。** 继续优先保证计算准确性、口径一致性与结构化输出可读性；报告与图表严格仅消费 `result.json` 与 Parquet 产物。

## 6. 成功标准与暂不实施的内容

项目的核心成功标准：

1. 数据有明确来源、语义、质量记录与计算口径。
2. Agent 能通过简单 CLI 命令完成单资产数据准备和指标计算。
3. 支持严谨的多资产比较及投资组合分析。
4. 可按需生成一致、可阅读的报告和可视化结果。
5. Web 与 CLI 对同一数据及参数产生一致的分析结果。

**暂不优先实施**：MCP Server、独立数据库服务、复杂 Web 后端、交易执行系统、自动交易决策或单纯为数量而扩展算法/资产覆盖。在 CLI 和核心分析契约稳定之后，再以实际使用场景决定是否引入这些能力。
