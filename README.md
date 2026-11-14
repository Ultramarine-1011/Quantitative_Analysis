# Quantitative Analysis（量化诊断）

基于 Python 3.11+ 的量化分析工具集：提供 **Streamlit Web 应用** 与 **命令行入口** 统一覆盖 13 类跨市场资产的历史行情诊断、多币种折算、精确有效前沿优化与无前视偏差样本外回测。数据经统一 **OHLCV**（开高低收量）契约清洗后，复用同一套特征工程、风险指标与跨端持久化缓存。

---

## 功能概览

### Web 应用（`app.py`）

- **多资产类型**：A 股 ETF/个股、公募基金、贵金属与**外盘商品期货**、加密货币（Binance + CCXT）、**美股/港股日线**、**全球指数**（美股指数 `.INX` 等经新浪；`HSI`/`SPX` 等简码经东财 `index_global_hist_em`）、**中国国债 K 线**、**美债收益率水平序列**（`bond_zh_us_rate`，默认 10Y 列；非持有期回报，看板有专门说明）、**C-REITs**（`reits_hist_em`）、**外汇**（`forex_hist_em`，如 `USDCNH`）。
- **量化诊断看板**：在选定区间内计算并展示 **CAGR**、**最大回撤**、**年化夏普 / Sortino / Calmar**（支持自定义无风险利率，默认 1.6%）、**95% 日 VaR / CVaR**、**累计收益率**，以及可配置的 SMA / 布林带 / RSI(14) / MACD，并展示数据源、复权口径、缓存命中状态与数据质量诊断。
- **交互式图表**：自适应亮暗主题的 Plotly 多面板图表（K 线 / 折线、成交量、滚动波动率、水下回撤与均线/布林带）。
- **多资产精确组合与样本外回测（「进阶金融算法」）**：支持跨市场日历对齐（`common_trading_dates` / `weekly_friday` / `mtm_outer`）、`CNY`/`USD`/`LOCAL` 基准货币换算与显式 USDT 锚定开关、Ledoit-Wolf 协方差收缩、SciPy SLSQP 精确有效前沿（GMV 与切点最大夏普 + 权重上下界），以及 Train/Test 静态划分或滚动/扩展 Walk-Forward 无泄露样本外回测（含再平衡频次与双边交易费率 bps）。
- **数据导出**：可下载与看板一致的 **清洗后 OHLCV CSV**（UTF-8 BOM，列：`Date`, `Open`, `High`, `Low`, `Close`, `Volume`）。
- **说明页**：第二个标签页提供指标相关的数学定义说明（Markdown + LaTeX）。
- **共享持久化缓存**：`data_sources/cache.py` 提供 Web 与 CLI 共享的 Parquet + JSON 元信息磁盘缓存（默认 TTL 3600 秒，支持区间子集复用与一键强制刷新/`--refresh`），Web 端叠加 Streamlit `cache_data`。
- **结果保留**：诊断、归一化、GBM 和组合优化分别保存本次会话的最新结果；侧边栏代理配置跨资产切换持久化，下载 CSV 或运行其他分析后仍可查看。

### 命令行（`main.py`）

- 面向 **全 13 类资产** 的端到端流程：共享持久化缓存拉数 → 可选基准货币折算（`--base-currency`）→ `OHLCVFeatureEngineer` 特征工程 → `MplfinanceVisualizer` 多面板图 → 控制台输出新版诊断报告（含 Sortino/Calmar/VaR/CVaR），并支持 `--save-chart` 与 `--export-csv` 无头批处理。

### 核心库模块

| 模块 | 作用 |
|------|------|
| `data_fetcher.py` | 保留现有数据 API；全球市场路由与多 ETF 宽表 |
| `data_sources/` | 境内权益回退、基金净值、东财/外汇 HTTP 适配、超时预算、线程代理隔离与 Parquet 共享缓存 |
| `ohlcv.py` | 日期校验、OHLCV 清洗、缺失值策略与质量诊断契约 |
| `asset_resolver.py` | 日期校验、境内 `load_china_equity_ohlcv`、`format_diagnosis_user_message` |
| `china_equity_entry.py` | 自 `asset_resolver` 的兼容 re-export |
| `asset_loader.py` / `asset_catalog.py` | 跨资产并发加载、多市场日历对齐、组合输入与预设模板 |
| `currency.py` | 跨市场计价推断、FX/USDT 汇率对齐与基准货币折算 |
| `feature_engineering.py` | SMA、布林带、EMA、RSI、MACD、对数收益波动率与最大回撤 |
| `analysis_metrics.py` / `quant_models.py` | Sortino/Calmar/VaR/CVaR；SLSQP 精确有效前沿、Ledoit-Wolf 收缩与 Walk-Forward OOS 回测 |
| `visualizer.py` | 基于 mplfinance 的 CLI 用高级图表（含纯折线模式） |
| `charts.py` / `app.py` | 主题自适应 Plotly 图表；Streamlit 页面与持久化状态 |

---

## 环境要求

- **Python**：最低 **3.11**（推荐 **3.11 / 3.12+**）。
- **网络**：A 股、基金、贵金属等依赖 **AKShare** 等数据源；加密货币依赖交易所 API（默认 Binance），部分地区可能需要代理。
- **共享磁盘缓存**：默认写入 `~/.cache/quantitative_analysis/ohlcv`，可通过环境变量 `QUANT_CACHE_DIR`、`QUANT_CACHE_TTL`（秒，默认 `3600`）与 `QUANT_DISABLE_CACHE=1` 控制。

---

## 安装

```bash
cd Quantitative_Analysis
python -m venv .venv
```

**Windows（PowerShell）** 激活虚拟环境示例：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：`pandas`、`numpy`、`scipy`、`streamlit`、`plotly`、`akshare`、`ccxt`、`pyarrow`、`matplotlib`、`mplfinance` 等（详见 `requirements.txt`）。

Web 入口要求 Streamlit **1.50+**，避免旧版前端与新版 Plotly 数组格式不兼容造成错误图表。PyArrow 保留 `<26` 约束。

开发环境（同时安装项目、测试与静态检查工具）：

```bash
python -m pip install -e ".[dev]"
```

`pyproject.toml` 从 `requirements.txt` 读取运行依赖，避免维护两份依赖列表。

---

## 使用方法

### 1. 启动 Web 应用（推荐）

```bash
streamlit run app.py
```

或在 Windows 下使用仓库中的包装脚本（等价于 `python -m streamlit run app.py`）：

```bat
streamlit.bat run app.py
```

浏览器打开终端提示的本地地址后：

1. 在左侧 **「参数设置」** 中选择 **资产类型**、输入 **代码**、**开始/结束日期**。
2. 点击 **「开始诊断」**。
3. 在 **「量化诊断看板」** 查看指标与图表；需要时可 **下载 CSV**。
4. **「数学原理」** 标签可阅读指标定义说明。
5. **「进阶金融算法」** 可独立运行归一化、GBM 和多资产组合分析。组合输入与代码参考分开展示，权重以百分比显示；代码参考与相关系数明细可按需展开。

手机上可通过左上角箭头展开参数栏。开始日期晚于结束日期或单标的代码为空时，页面会提示并禁用相应计算按钮；美债收益率列留空仍使用默认 10Y。

**各类型代码示例（默认值可参考应用内侧边栏）**：

| 资产类型 | 代码示例 | 说明 |
|----------|----------|------|
| A 股 ETF | `510300` | 与「ETF」类别对应，勿与个股混用 |
| A 股个股 | `600519` | 股票代码 |
| 公募基金 | `009691` | 基金代码（净值序列，Close 为累计净值） |
| 贵金属 | `GC`、`AU9999`（映射到 GC）等 | 来自 AKShare 外盘期货历史接口 |
| 加密货币 | `BTC/USDT` | CCXT + Binance；**可在侧边栏填写代理**（仅加密模式生效），格式如 `http://127.0.0.1:33210` |
| 美股 | `AAPL` | `stock_us_daily` |
| 港股 | `00700` | `stock_hk_daily`（纯数字会自动补零到 5 位） |
| 全球指数 | `.INX`（新浪美股指数）；或 `HSI`、`SPX`、`DXY` 等简码（东财 `index_global_hist_em`） | 简码见 `data_fetcher.GLOBAL_INDEX_EM_ALIASES` |
| 中国国债 K 线 | `sh010107` | `bond_zh_hs_daily` |
| 美债收益率 | `10Y` 或留空=默认 10 年期列；亦可填 `bond_zh_us_rate` 表中的**完整列名** | 水平序列，非债基持有回报 |
| 外盘商品期货 | `CL`、`HG` 等 | `futures_foreign_hist`，与贵金属同源 |
| C-REITs | `508097` | `reits_hist_em` |
| 外汇 | `USDCNH`；亦支持 `USD/CNY` 等别名 | `forex_hist_em` |

若拉取失败，界面会给出可读错误信息；**数据源异常**时会提示代码或接口暂不可用。空数据时会提示检查代码与资产类型是否匹配。

### 2. 命令行：全资产分析（`main.py`）

```bash
python main.py --help
```

安装项目后也可使用 `quantitative-analysis --help`，参数与 `python main.py` 一致。

示例：

```bash
# 默认：ETF 510300，约最近 3 年到今天
python main.py

# 美股按前复权抓取并折算为 CNY，无头保存图表与 CSV
python main.py --asset-type us_stock --symbol AAPL --adjust qfq --base-currency CNY --save-chart outputs/aapl.png --export-csv outputs/aapl.csv --no-show

# 强制刷新（跳过共享持久缓存）
python main.py --asset-type etf --symbol 510300 --refresh --no-show
```

参数说明：

- `--asset-type`：支持全 13 类资产（`etf`, `ashare`, `mutual_fund`, `gold`, `crypto`, `us_stock`, `hk_stock`, `global_index`, `bond_cn`, `bond_us_yield`, `commodity`, `creit`, `fx`）。
- `--symbol`：可选；缺省使用对应资产类别默认示例。
- `--start` / `--end`：可选，`YYYY-MM-DD`；默认结束为今天，开始约为结束日前推 3 年。
- `--adjust` / `--proxy` / `--base-currency` / `--save-chart` / `--export-csv` / `--no-show` / `--refresh`：复权口径、网络代理、基准货币折算（`CNY`/`USD`）、无头图表与 CSV 导出、跳过缓存强制刷新。

### 3. 运行测试

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
ruff format --check .
python -m pip check
```

测试目录为 `tests/`（配置见 `pytest.ini`），使用固定样本和模拟数据源，不访问行情网络。覆盖清洗、缺失值与交易日历、外汇折算、并发与超时预算、SLSQP 精确前沿、Ledoit-Wolf 收缩、Walk-Forward OOS 回测、共享磁盘缓存隔离、CSV、CLI 和 Streamlit。GitHub Actions 对 Python 3.11 / 3.12 执行检查。验证说明见 `docs/CHINA_EQUITY_VERIFICATION.md`；重构说明与兼容边界见 `docs/ARCHITECTURE.md`；实际浏览器验收记录见 `docs/FRONTEND_VERIFICATION.md`。

---

## 网络与代理说明

- **加密货币**：在 Web 侧通过侧边栏 **「代理地址」** 传入，由 CCXT 使用。
- **Windows 与系统代理**：导入库不会修改代理。入口启动或首次构造数据客户端时才应用 Windows 系统代理策略。AKShare 默认尝试直连；若需保留系统/环境代理，请在启动前设置 **`QUANT_USE_SYSTEM_PROXY=1`**，并按需设置 **`HTTP_PROXY` / `HTTPS_PROXY`**。直连上下文仅影响当前线程的 requests 代理解析，不删除进程环境变量，也不影响其他会话和 CCXT 的显式代理。

---

## 项目结构（简要）

```
Quantitative_Analysis/
├── app.py                 # Streamlit 入口
├── main.py                # CLI 入口（支持全 13 类资产）
├── asset_catalog.py       # 资产键、默认值与预设模板
├── asset_loader.py        # 跨资产并发加载与日历对齐
├── asset_resolver.py      # 境内权益入口与错误文案
├── china_equity_entry.py  # 旧导入路径兼容
├── currency.py            # 汇率加载与多资产基准货币折算
├── data_fetcher.py        # 数据公共 API 与全球市场路由
├── data_sources/          # 数据源适配、预算控制、代理与 Parquet 缓存
├── ohlcv.py               # 共用清洗、缺失值策略与质量诊断
├── feature_engineering.py # 批量与单指标计算（SMA/BB/EMA/RSI/MACD）
├── analysis_metrics.py    # Sortino / Calmar / VaR / CVaR 与诊断指标
├── quant_models.py        # GBM、SLSQP 精确有效前沿、Ledoit-Wolf 与 OOS 回测
├── charts.py              # 主题自适应 Plotly 图表
├── visualizer.py          # mplfinance 图表
├── pyproject.toml         # 安装、CLI 命令与开发工具配置
├── requirements.txt
├── .streamlit/config.toml # Streamlit 本地默认配置
├── tests/
└── docs/
```

---

## 免责声明

本工具仅用于数据展示与技术学习，不构成任何投资建议。行情与净值数据来自第三方接口，其准确性、完整性与时效性以数据源为准；指标与「信号」为规则化输出，请勿作为实盘决策的唯一依据。
