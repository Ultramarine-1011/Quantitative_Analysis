# 产品定位与开发路线图（Roadmap）

> 状态：**规划中（已完成 M0 现状重估）** · 更新：2026-11-14  
> 本文描述目标架构和未来里程碑；所列 `quant` 新 CLI 命令、结构化 JSON/Parquet 服务接口与独立报告渲染**尚未实现**，现有功能和运行方法以 [README](../README.md)、[工程架构](ARCHITECTURE.md) 与 [M0 审计与契约](M0_AUDIT_AND_CONTRACTS.md) 为准。
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

### M2 · Agent CLI 与统一分析工作流（V0.2，首个核心 MVP）

**目标**：Agent 用一条命令得到可直接分析的资产数据和指标，无需手写数据抓取与计算脚本。

**工作内容**
- 建立与 Streamlit 解耦的分析服务，串联解析、加载、质量校验、特征与指标。
- 首批命令：`quant fetch`（检索/标准化数据）、`quant analyze`（单资产指标）；后续兼容现有 `quantitative-analysis` 入口的迁移策略需明确。
- 定义版本化的结果 Schema：资产与数据元信息、样本范围、指标数值及单位、计算参数、警告、错误和生成文件位置。
- 输出 `result.json` 和可供二次分析的标准数据文件（优先 Parquet；必要时 CSV）。stdout 适合机器读取；进度和诊断日志写 stderr；不同错误类别有稳定退出码。
- 建立离线固定样本的 CLI 端到端与指标回归测试。

**目标接口示意（尚未实现）**

```bash
quant analyze NVDA --start 2021-01-01 --end 2026-11-13 --output ./results/nvda
```

最小产物示例：`result.json`、`data.parquet`。JSON 应提供 `schema_version`、`asset`、`data`、`metrics`、`parameters`、`warnings` 和 `artifacts` 等稳定字段；数值不足或不适用时要保留机器可辨识的缺失原因。

**验收标准**：编程 Agent 仅凭终端命令和读取输出文件，即可完成单资产数据准备与基本量化分析；在固定数据和参数下结果可复现。

### M3 · 多资产比较、统计与组合分析（V0.3）

**目标**：从单资产诊断扩展到常用金融研究任务。

**工作内容**
- 增加 `quant compare`：标准化净值/价格比较、同口径收益风险、相对基准、相关性与协方差、滚动统计。
- 增加 `quant portfolio`：权重组合、收益风险与风险贡献、基准对照、历史回测。
- 加强收益分布、统计检验、稳健性及参数敏感性分析；按功能适用性逐步引入。
- 明确有效前沿中“随机权重采样”和“约束优化”是不同算法；区分样本内回放与样本外回测，并明确调仓、交易成本和前视信息约束。
- 明确多市场对齐、非同步交易和汇率处理逻辑，避免夸大相关性或组合收益结论。

**验收标准**：可用 CLI 完成典型多资产比较和组合研究，结果有可核对的样本、参数、基准和方法说明；模拟不冒充实际预测或有效策略表现。

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
| **V0.2** | M2 | **Agent 可直接使用的单资产 CLI MVP** |
| **V0.3** | M3 | 多资产比较、组合与严谨统计能力 |
| **V0.4** | M4 | 按需生成报告与图表 |
| **V1.0** | M5 | Web/CLI 共享分析核心的完整平台 |

**近期开发重心是 V0.2。** 在此之前，优先修复会影响数据准确性、指标解释和 CLI 稳定性的缺陷；暂缓纯粹增加资产种类、装饰性图表和算法演示。

## 6. 成功标准与暂不实施的内容

项目的核心成功标准：

1. 数据有明确来源、语义、质量记录与计算口径。
2. Agent 能通过简单 CLI 命令完成单资产数据准备和指标计算。
3. 支持严谨的多资产比较及投资组合分析。
4. 可按需生成一致、可阅读的报告和可视化结果。
5. Web 与 CLI 对同一数据及参数产生一致的分析结果。

**暂不优先实施**：MCP Server、独立数据库服务、复杂 Web 后端、交易执行系统、自动交易决策或单纯为数量而扩展算法/资产覆盖。在 CLI 和核心分析契约稳定之后，再以实际使用场景决定是否引入这些能力。
