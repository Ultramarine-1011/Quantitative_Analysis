# Quantitative Analysis（量化诊断）

基于 Python 的量化分析工具集：提供 **Streamlit Web 应用** 做多资产统一诊断，以及 **命令行入口** 面向境内 ETF / A 股做离线图表与文本报告。数据经统一 **OHLCV**（开高低收量）契约清洗后，复用同一套特征工程与风险收益指标。

---

## 功能概览

### Web 应用（`app.py`）

- **多资产类型**：A 股 ETF/个股、公募基金、贵金属与**外盘商品期货**、加密货币（Binance + CCXT）、**美股/港股日线**、**全球指数**（美股指数 `.INX` 等经新浪；`HSI`/`SPX` 等简码经东财 `index_global_hist_em`）、**中国国债 K 线**、**美债收益率水平序列**（`bond_zh_us_rate`，默认 10Y 列；非持有期回报，看板有专门说明）、**C-REITs**（`reits_hist_em`）、**外汇**（`forex_hist_em`，如 `USDCNH`）。
- **量化诊断看板**：在选定区间内计算并展示 **CAGR**、**最大回撤**、**年化夏普比率**（内部无风险利率取常数约 1.6% 便于横向对比）、**累计收益率**（收益率水平类资产会改为「水平相对变化」等标签），以及基于布林带的 **「最新信号」** 文案。
- **交互式图表**：Plotly **K 线 + 成交量**（无成交量或公募/收益率水平序列时用**单面板折线或单面板 K 线**）；叠加均线与布林带。
- **数据导出**：可下载与看板一致的 **清洗后 OHLCV CSV**（UTF-8 BOM，列：`Date`, `Open`, `High`, `Low`, `Close`, `Volume`）。
- **说明页**：第二个标签页提供指标相关的数学定义说明（Markdown + LaTeX）。
- **缓存**：拉取数据使用 Streamlit `cache_data`（约 1 小时 TTL），减轻重复请求压力。

### 命令行（`main.py`）

- 面向 **境内 ETF / A 股个股** 的端到端流程：拉数 → `OHLCVFeatureEngineer` 特征工程 → `MplfinanceVisualizer` 多面板图（K 线、成交量、波动率、回撤等）→ 控制台输出 **资产数学诊断报告**。
- 与 Web 侧共用 `asset_resolver` / `china_equity_entry`（兼容 re-export）的校验与境内权益路由。

### 核心库模块

| 模块 | 作用 |
|------|------|
| `data_fetcher.py` | 保留现有数据 API；全球市场路由与多 ETF 宽表 |
| `data_sources/` | 境内权益回退、基金净值、东财/外汇 HTTP 适配、重试及代理策略 |
| `ohlcv.py` | 日期校验、OHLCV 清洗与契约；数据、特征、图表共用 |
| `asset_resolver.py` | 日期校验、境内 `load_china_equity_ohlcv`、`format_diagnosis_user_message` |
| `china_equity_entry.py` | 自 `asset_resolver` 的兼容 re-export |
| `asset_loader.py` / `asset_catalog.py` | 跨资产加载、组合输入解析；资产键、默认值与说明 |
| `feature_engineering.py` | SMA、布林带、对数收益波动率、回撤与最大回撤等 |
| `analysis_metrics.py` / `quant_models.py` | Web/CLI 共用诊断指标；GBM、有效前沿、组合回测 |
| `visualizer.py` | 基于 mplfinance 的 CLI 用高级图表 |
| `charts.py` / `app.py` | Plotly 图表；Streamlit 页面与一小时缓存 |

---

## 环境要求

- **Python**：最低 **3.9**，推荐 **3.12+**。macOS 系统 Python 3.9 的 LibreSSL 会触发 urllib3 警告，建议使用带 OpenSSL 的独立 Python 环境。
- **网络**：A 股、基金、贵金属等依赖 **AKShare** 等数据源；加密货币依赖交易所 API（默认 Binance），部分地区可能需要代理。

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

主要依赖包括：`pandas`、`numpy`、`streamlit`、`plotly`、`akshare`、`ccxt`、`matplotlib`、`mplfinance` 等（详见 `requirements.txt`）。

开发环境（同时安装项目、测试与静态检查工具）：

```bash
python -m pip install -e ".[dev]"
```

`pyproject.toml` 从 `requirements.txt` 读取运行依赖，避免维护两份依赖列表；Python 3.9 保留 CCXT `<4.5` 的兼容限制。

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
4. **「系统数学原理解析」** 标签可阅读指标定义说明。

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

### 2. 命令行：境内 ETF / A 股

```bash
python main.py --help
```

安装项目后也可使用 `quantitative-analysis --help`，参数与 `python main.py` 一致。

示例：

```bash
# 默认：ETF 510300，约最近 3 年到今天
python main.py

# A 股个股，指定代码与区间
python main.py --asset-type ashare --symbol 000001 --start 2022-01-01 --end 2025-12-31
```

参数说明：

- `--asset-type`：`etf` 或 `ashare`。
- `--symbol`：可选；默认 ETF `510300`，A 股 `600519`。
- `--start` / `--end`：可选，`YYYY-MM-DD`；默认结束为今天，开始约为结束日前推 3 年。

### 3. 运行测试

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
python -m pip check
```

测试目录为 `tests/`（配置见 `pytest.ini`），使用固定样本和模拟数据源，不访问行情网络。覆盖清洗、计算公式、回退/复权、代理隔离、组合模型、CSV、CLI 和 Streamlit。GitHub Actions 对 Python 3.9/3.12 执行检查。验证说明见 `docs/CHINA_EQUITY_VERIFICATION.md`；重构说明与兼容边界见 `docs/ARCHITECTURE.md`。

---

## 网络与代理说明

- **加密货币**：在 Web 侧通过侧边栏 **「代理地址」** 传入，由 CCXT 使用。
- **Windows 与系统代理**：导入库不会修改代理。入口启动或首次构造数据客户端时才应用 Windows 系统代理策略。AKShare 默认尝试直连；若需保留系统/环境代理，请在启动前设置 **`QUANT_USE_SYSTEM_PROXY=1`**，并按需设置 **`HTTP_PROXY` / `HTTPS_PROXY`**。直连上下文仅影响当前线程的 requests 代理解析，不删除进程环境变量，也不影响其他会话和 CCXT 的显式代理。

---

## 项目结构（简要）

```
Quantitative_Analysis/
├── app.py                 # Streamlit 入口
├── main.py                # CLI 入口（境内 ETF / A 股）
├── asset_catalog.py       # 资产键、默认值与说明
├── asset_loader.py        # 跨资产加载与组合输入
├── asset_resolver.py      # 境内权益入口与错误文案
├── china_equity_entry.py  # 旧导入路径兼容
├── data_fetcher.py        # 数据公共 API 与全球市场路由
├── data_sources/          # 按数据源拆分的实现
├── ohlcv.py               # 共用清洗、校验、日期契约
├── feature_engineering.py # 批量与单指标计算
├── analysis_metrics.py    # Web/CLI 共用诊断指标
├── quant_models.py        # GBM、有效前沿、组合回测
├── charts.py              # Plotly 图表
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
