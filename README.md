# Quantitative Analysis（量化诊断）

基于 Python 的量化分析工具集：提供 **Streamlit Web 应用** 做多资产统一诊断，以及 **命令行入口** 面向境内 ETF / A 股做离线图表与文本报告。数据经统一 **OHLCV**（开高低收量）契约清洗后，复用同一套特征工程与风险收益指标。

---

## 功能概览

### Web 应用（`app.py`）

- **多资产类型**：A 股 ETF、A 股个股、公募基金、全球贵金属（期货行情）、加密货币（默认 Binance + CCXT）。
- **量化诊断看板**：在选定区间内计算并展示 **CAGR**、**最大回撤**、**年化夏普比率**（内部无风险利率取常数约 1.6% 便于横向对比）、**累计收益率**，以及基于布林带的 **「最新信号」** 文案（如突破上轨、触底下轨、均值回归中等）。
- **交互式图表**：Plotly **K 线 + 成交量**（公募基金为 **累计净值折线**，无 K 线与成交量子图）；叠加短期/长期均线与布林带上下轨。
- **数据导出**：可下载与看板一致的 **清洗后 OHLCV CSV**（UTF-8 BOM，列：`Date`, `Open`, `High`, `Low`, `Close`, `Volume`）。
- **说明页**：第二个标签页提供指标相关的数学定义说明（Markdown + LaTeX）。
- **缓存**：拉取数据使用 Streamlit `cache_data`（约 1 小时 TTL），减轻重复请求压力。

### 命令行（`main.py`）

- 面向 **境内 ETF / A 股个股** 的端到端流程：拉数 → `OHLCVFeatureEngineer` 特征工程 → `MplfinanceVisualizer` 多面板图（K 线、成交量、波动率、回撤等）→ 控制台输出 **资产数学诊断报告**。
- 与 Web 侧共用 `china_equity_entry` 的校验与 AKShare 路由逻辑。

### 核心库模块

| 模块 | 作用 |
|------|------|
| `data_fetcher.py` | 数据抓取、公募基金净值、网络与代理策略（含 Windows 下可选行为） |
| `china_equity_entry.py` | 境内 ETF / A 股参数与 `load_china_equity_ohlcv` 统一入口 |
| `feature_engineering.py` | SMA、布林带、对数收益波动率、回撤与最大回撤等 |
| `visualizer.py` | 基于 mplfinance 的 CLI 用高级图表 |
| `app.py` | Streamlit 页面、跨资产 `load_asset_data`、Plotly 图表与指标汇总 |

---

## 环境要求

- **Python**：建议 **3.9+**（`requirements.txt` 注明 3.8+ 可用，3.7 无法满足依赖）。
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

主要依赖包括：`pandas`、`numpy`、`streamlit`、`plotly`、`akshare`、`ccxt`、`matplotlib`、`mplfinance`、`pytest` 等（详见 `requirements.txt`）。

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

若拉取失败，界面会给出可读错误信息；空数据时会提示检查代码与资产类型是否匹配。

### 2. 命令行：境内 ETF / A 股

```bash
python main.py --help
```

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
pytest
```

测试目录为 `tests/`（配置见 `pytest.ini`）。更完整的境内权益验证说明可参考 `docs/CHINA_EQUITY_VERIFICATION.md`。

---

## 网络与代理说明

- **加密货币**：在 Web 侧通过侧边栏 **「代理地址」** 传入，由 CCXT 使用。
- **Windows 与系统代理**：`data_fetcher` 在导入时会调用 `apply_default_network_proxy_policy()`。若需让 AKShare 等走 Windows「Internet 设置」中的系统代理，可设置环境变量 **`QUANT_USE_SYSTEM_PROXY=1`**（具体行为以 `data_fetcher.py` 实现为准）。仍可通过常规 **`HTTP_PROXY` / `HTTPS_PROXY`** 等环境变量显式指定代理。

---

## 项目结构（简要）

```
Quantitative_Analysis/
├── app.py                 # Streamlit 入口
├── main.py                # CLI 入口（境内 ETF / A 股）
├── data_fetcher.py        # 数据层与网络策略
├── china_equity_entry.py  # 境内权益统一入口与错误文案
├── feature_engineering.py # 特征工程
├── visualizer.py          # mplfinance 可视化
├── requirements.txt
├── .streamlit/config.toml # Streamlit 本地默认配置
├── tests/
└── docs/
```

---

## 免责声明

本工具仅用于数据展示与技术学习，不构成任何投资建议。行情与净值数据来自第三方接口，其准确性、完整性与时效性以数据源为准；指标与「信号」为规则化输出，请勿作为实盘决策的唯一依据。
