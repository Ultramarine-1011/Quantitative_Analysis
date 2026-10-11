# 工程结构与兼容约定

## 职责与依赖

`app.py` 负责 Streamlit 控件、缓存和展示，`main.py` 负责 CLI 参数、报告和流程。`asset_loader.py` 提供不依赖 Streamlit 的多资产入口；`data_fetcher.py` 保留原有数据 API，数据源实现放在 `data_sources/`。

数据流为：数据源 → `ohlcv.py` 清洗 → `feature_engineering.py` / `analysis_metrics.py` / `quant_models.py` → `charts.py` 或 `visualizer.py`。数据层不导入 UI、指标或图表；资产目录只保存键、标签、默认代码与说明。增加数据源时先实现标准 OHLCV 加载，再补充目录和 `asset_loader.load_asset_data` 的分支即可。

`data_sources/base.py` 保留已有自定义抓取器的基类 API；`china_equity.py` 负责境内权益回退，`eastmoney.py` 负责 HTTP/K 线协议，`funds.py` 负责累计净值，`forex.py` 负责外汇，`network.py` 负责代理隔离。没有引入注册器、插件框架或新的继承层级。

## 数据与计算口径

- OHLCV 列固定为 `Open`、`High`、`Low`、`Close`、`Volume`，均为 float64；`Date` 为无时区的 datetime64[ns] 升序唯一索引。重复日期稳定保留输入中的最后一条。
- 数值先转换再将无穷值视作缺失，并前向填充。数据源边界删除开头仍缺失的行；特征/绘图输入拒绝仍缺失的核心值，保留额外指标列及 attrs。不执行价格向后填充。
- CAGR 仍使用起止价格比和自然日跨度 / 365.25；诊断 Sharpe 仍使用日对数收益、日无风险对数利率、样本标准差和 sqrt(252)。常数序列的 Sharpe 为 NaN，显示为 N/A。
- SMA、布林带及波动率保留完整滚动窗口和 ddof=1；回撤深度为 `1 - Close / cummax(Close)`。批量 transform 只校验一次输入，重复执行可覆盖已有特征列。
- 有效前沿使用日简单收益，均值乘交易日数、样本标准差乘其平方根；Dirichlet 权重和固定随机种子行为保留。直接由均值与协方差计算组合矩，避免创建“历史天数 × 组合数”的临时矩阵。
- 组合回测保留样本首日净值为 1 的规则；宽表仍按不同市场日历 outer 对齐并前向填充。多币种数据不会自动转换币种，GBM 未来日期仍为工作日近似，不包含各市场节假日。
- 基金仍使用累计净值，OHLC 相同、Volume=0；美债仍是收益率水平序列，保留专门标签与解释。

## 修复与兼容边界

恢复缺失的 `china_equity_entry` 兼容导入；`main.py` 使用实际的 `asset_resolver`。`data_fetcher` 的公共函数、异常、抓取器及常量仍可按原路径导入；`app` 的加载、指标和图表函数仍可使用。Web/CLI 启动命令、控件、CSV 编码及列顺序不变。

以下差异是明确的 Bug 修复：复权 ETF 不再使用未复权新浪历史；显式空复权参数不会被替换成 hfq；所有境内权益路径都裁剪日期；基金累计净值缺口不再从未来向前补齐；非法日期在调用数据源前拒绝；无穷值、无效权重及非正 GBM 价格得到明确处理；短历史 CLI 图表跳过尚无有效值的滚动指标；指标重算不复用失效 attrs。

导入库不改写代理。Windows 系统代理策略在入口启动/首次构造客户端时生效；requests 的直连上下文按线程隔离，异常和嵌套调用后恢复解析器，不修改 `os.environ`。代理上下文的安装/恢复使用可重入锁，直接调用数据源的会话可能串行执行。

`YFinanceFetcher` 作为已有 API 保留，仍是未实现的占位类；现有美股入口继续使用 AKShare。生产 GBM 移除了 NumPy 与 SciPy 的重复 IQR 自检，项目不再直接依赖 SciPy。

## 安装与验证

运行依赖只维护在 `requirements.txt`，`pyproject.toml` 动态读取它；`pip install -e ".[dev]"` 安装项目与 pytest/ruff，并提供 `quantitative-analysis` 命令。建议 Python 3.12+；最低 Python 3.9，CCXT 的旧版约束仅用于 Python 3.9。

自动化测试全部使用确定性样本或模拟数据源，不把第三方网络可用性作为测试通过条件。除原有测试外，加入独立公式核对、固定种子模型回归、回退复权、净值缺口、线程代理隔离、图表/CSV、缓存共享、CLI 与 Streamlit AppTest。CI 对 Python 3.9/3.12 执行 pytest、ruff 和依赖检查。

真实行情接口、交易所可用性、限流和跨源复权数据精度仍取决于上游；离线测试不能证明任一远端接口实时可用。手工行情验收见 `CHINA_EQUITY_VERIFICATION.md`。

## 本次验证记录（2026-10-11）

- 修复前测试因缺失 `china_equity_entry` 无法收集；修复后原有 40 项测试通过，新增回归用例后共 126 项。
- 原 Python 3.9.6 环境：126 项通过；仅残留系统 LibreSSL 的 urllib3 警告。依赖检查通过。
- 全新 Python 3.12.13 环境：126 项通过，无测试警告；使用 Pandas 3.0.6、NumPy 2.5.4、Plotly 7.1.0、Streamlit 1.65.0、AKShare 1.19.1、CCXT 4.5.85，依赖检查通过。
- Ruff 检查、格式检查、编译、wheel 构建与项目目录外的 CLI 验证通过；20 个本地库/入口模块未发现循环导入。
- 10,000 行固定样本与原特征实现逐列核对一致；本机单次测量约 34ms → 6ms，性能数值仅作局部参考。
