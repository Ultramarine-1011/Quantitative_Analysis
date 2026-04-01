# 境内 ETF / A 股：最小验证方案与故障排查

本文说明如何用**稳定样例**验收 ETF 与个股链路，如何运行**不依赖外网**的自动化测试，以及常见失败原因与处理思路。

## 最小验收清单（手工）

| 步骤 | ETF（样例 `510300`） | A 股（样例 `600519`） |
|------|----------------------|------------------------|
| 1. 环境 | `pip install -r requirements.txt` | 同上 |
| 2. CLI | `python main.py --asset-type etf --symbol 510300 --start 2022-01-01 --end 2024-12-31` | `python main.py --asset-type ashare --symbol 600519 --start 2022-01-01 --end 2024-12-31` |
| 3. Web | 启动 Streamlit 后选择 A 股 ETF，代码填 `510300`，日期区间合理 | 选择 A 股个股，代码填 `600519` |

**通过标准（手工）**：能拉取数据、弹出/显示图表，控制台或页面无未捕获异常；空数据时应看到「未找到该代码的数据…」类提示而非崩溃。

## 自动化测试（不依赖行情外网）

清洗与入口校验可通过伪造 DataFrame 覆盖，避免验收完全依赖第三方接口。

```bash
cd c:\Python_projects\Quantitative_Analysis
python -m pip install -r requirements.txt
python -m pytest tests\ -v
```

覆盖范围概要：

- `AKShareFetcher._normalize_ohlcv`：中文列映射、`DatetimeIndex`、去重、空表与缺列错误分支。
- `fetch_china_equity`：非法 `route` 的 `ValueError`。
- `china_equity_entry`：资产类型校验、日期区间、`format_china_equity_user_message` 文案。

## 故障排查

### 1. `ImportError` / 模块找不到（如 `akshare`、`streamlit`）

- 确认已在本机**同一 Python 解释器**下执行 `pip install -r requirements.txt`。
- IDE 或 `streamlit.bat` 若指向别的 Python，会出现「终端能跑、双击不能跑」；请统一用 `python -m streamlit run app.py`。

### 2. 抓取失败、`DataFetchError`、`ChinaEquityPipelineError`

- **网络**：防火墙、公司网络对行情域名的限制；可换网络或稍后重试。
- **代理**：Windows 上项目默认可能**忽略 WinINet 系统代理**（见 `data_fetcher` 中 `QUANT_USE_SYSTEM_PROXY`）。若你必须**完全按系统/环境变量走代理**（例如只通过 Clash 等本机代理上网），请设置 `QUANT_USE_SYSTEM_PROXY=1` 后重启 Streamlit/CLI。
- **提示 `AKShare ETF historical data fetch failed after 3 attempts`**：多为 **各数据源均不可达** 或 **代理配置错误**。当前链路顺序为：**新浪（ETF）→ 腾讯日线** → AKShare 东财接口 → **多个东方财富 K 线节点直连**。请确认网络可访问 `finance.sina.com.cn`、`proxy.finance.qq.com` 及东财域名，或按需设置 `QUANT_USE_SYSTEM_PROXY`。
- **代码与资产类型不匹配**：例如把股票代码填在 ETF 选项中，可能无数据或空表，界面会提示检查代码与类别。

### 3. `EmptySymbolDataError` / 「未找到该代码的数据」

- 核对代码是否为对应市场常用 6 位代码；ETF 与个股不要混用入口。
- 日期区间过短且遇长假可能出现可交易日为 0 的边界情况，可略微放宽区间试一次。

### 4. AKShare 升级后行为变化

- 主路径走 AKShare，失败时可能回退东方财富直连；若两端字段或接口变更，错误信息可能变为 `ChinaEquityPipelineError` 包裹的多段说明。
- 可固定 `akshare`  minor 版本（见 `requirements.txt` 下限）并在升级后跑一遍 `pytest` 与上述手工清单。

### 5. 其他依赖版本

- 使用 `requirements.txt` 中注明的 Python 版本建议（3.9+）；过旧的 `numpy`/`pandas` 可能导致类型或索引行为不一致。

## 相关代码入口

- 共用校验与拉取：`china_equity_entry.py`（`load_china_equity_ohlcv` 等）。
- 数据清洗与路由：`data_fetcher.py`（`AKShareFetcher.fetch_china_equity`、`_normalize_ohlcv`）。
- CLI：`main.py`；Web：`app.py`。
