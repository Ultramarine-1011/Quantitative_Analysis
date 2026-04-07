"""Streamlit 量化诊断应用入口。

统一数据适配层覆盖 A 股、公募、贵金属、加密、全球股/指、债券、商品、REITs、外汇等，输出一致 OHLCV 契约，
供指标层与图表层复用。
"""

from __future__ import annotations

import importlib
import io
import math
import re
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from asset_resolver import (
    CHINA_EQUITY_LABELS,
    DEFAULT_CHINA_EQUITY_SYMBOLS,
    EMPTY_SYMBOL_FETCH_HINT,
    format_diagnosis_user_message,
    load_china_equity_ohlcv,
)
from data_fetcher import (
    DataFetchError,
    REQUIRED_OHLCV_COLUMNS,
    US_TREASURY_YIELD_10Y_CN_COL,
    apply_default_network_proxy_policy,
    get_fund_data,
    get_multiple_assets_close,
    load_bond_data,
    load_commodity_data,
    load_futures_foreign_ohlcv,
    load_fx_data,
    load_global_market_data,
    load_reit_data,
)
from feature_engineering import OHLCVFeatureEngineer
from quant_models import (
    EfficientFrontierResult,
    MonteCarloGBMResult,
    generate_efficient_frontier,
    normalize_close_to_base_one,
    run_monte_carlo_gbm,
)

apply_default_network_proxy_policy()


SUPPORTED_ASSET_TYPES: tuple[str, ...] = (
    "etf",
    "ashare",
    "mutual_fund",
    "gold",
    "crypto",
    "us_stock",
    "hk_stock",
    "global_index",
    "bond_cn",
    "bond_us_yield",
    "commodity",
    "creit",
    "fx",
)
ASSET_LABELS: dict[str, str] = {
    **CHINA_EQUITY_LABELS,
    "mutual_fund": "公募基金 (Mutual Fund)",
    "gold": "全球贵金属",
    "crypto": "加密货币",
    "us_stock": "美股 (US Stock)",
    "hk_stock": "港股 (HK Stock)",
    "global_index": "全球指数（含美股指数/东财国际指数）",
    "bond_cn": "中国国债 K 线",
    "bond_us_yield": "美债收益率曲线（水平序列）",
    "commodity": "外盘大宗商品期货",
    "creit": "C-REITs（基础设施）",
    "fx": "外汇即期",
}
DEFAULT_SYMBOLS: dict[str, str] = {
    **DEFAULT_CHINA_EQUITY_SYMBOLS,
    "mutual_fund": "009691",
    "gold": "GC",
    "crypto": "BTC/USDT",
    "us_stock": "AAPL",
    "hk_stock": "00700",
    "global_index": ".INX",
    "bond_cn": "sh010107",
    "bond_us_yield": "10Y",
    "commodity": "CL",
    "creit": "508097",
    "fx": "USDCNH",
}
LOOKBACK_YEARS = 3
RISK_FREE_RATE = 0.016
TRADING_DAYS = 252
GOLD_SYMBOL_ALIASES: dict[str, str] = {
    "GC": "GC",
    "AU9999": "GC",
}

# 资产种类说明：类型界定、社会意义、在本应用中的数学处理、代表代码（供用户对照输入）。
ASSET_TYPE_GUIDES: dict[str, dict[str, str]] = {
    "etf": {
        "kind": "在交易所上市交易的开放式指数基金或行业/主题 ETF，份额可像股票一样买卖，净值与标的指数挂钩。",
        "social": "为散户提供分散化、低门槛的指数化投资工具，反映市场对行业、风格与宏观主题的预期，流动性集中体现机构与散户配置行为。",
        "math": "本应用使用**后复权**日线 OHLCV，与 A 股个股同一套收益率、波动率、布林带与回撤公式；成交量为真实成交。",
        "examples": "`510300`（沪深300ETF）、`510500`（中证500ETF）、`159915`（创业板ETF）",
    },
    "ashare": {
        "kind": "在上海或深圳证券交易所挂牌的人民币普通股（A 股），公司所有权凭证，价格由二级市场供需决定。",
        "social": "连接实体经济融资与居民财富管理，个股走势反映行业景气、政策预期与公司治理等信息，是境内权益风险定价的核心载体之一。",
        "math": "后复权日 K 线；指标基于收盘价序列的几何/统计性质（对数收益、滚动波动、极值带），**不**包含基本面因子。",
        "examples": "`600519`、`000001`、`688981`",
    },
    "mutual_fund": {
        "kind": "公募开放式基金，按净值申购赎回；本应用拉取的是**累计净值**序列，已折算为伪 OHLCV（开高低收相同）。",
        "social": "专业管理、分散投资，服务长期储蓄与养老等需求；净值曲线体现基金经理资产配置与申赎压力的综合结果。",
        "math": "无真实日内高低价，OHLC 均为净值；**成交量恒为 0**，图表以折线展示；夏普/CAGR 等仍按净值变化率计算，与 ETF 市价口径不同。",
        "examples": "`009691`、`161725`、`110011`",
    },
    "gold": {
        "kind": "通过外盘期货主力连续合约报价的贵金属（如 COMEX 黄金），美元计价，反映避险与真实利率预期。",
        "social": "传统避险与储备资产锚，与美元信用、地缘政治和通胀预期相关，常作为多元资产配置的分散项。",
        "math": "标准 OHLCV 日频；可存在换月导致的跳空，指标对跳变敏感；与 A 股不同币种与交易时段。",
        "examples": "`GC`（黄金）、`AU9999`（映射至 GC）",
    },
    "crypto": {
        "kind": "去中心化或交易所报价的加密资产，本应用默认通过 CCXT 拉取 **Binance** 现货日线（如 `BTC/USDT`）。",
        "social": "高风险高波动的另类资产，反映全球流动性、监管叙事与技术采用预期；7×24 交易，与股市日历不完全对齐。",
        "math": "UTC 日线合成 OHLCV；可设代理访问；夏普与回撤对极端日收益敏感，不宜与股票直接横比。",
        "examples": "`BTC/USDT`、`ETH/USDT`、`SOL/USDT`",
    },
    "us_stock": {
        "kind": "美国主要交易所上市的公司普通股，美元计价，本应用使用 AKShare 美股日线。",
        "social": "全球科技与消费龙头集中地，股价反映全球增长预期与美元流动性；与 A 股存在时差与制度差异。",
        "math": "日 OHLCV；与境内股相同的特征工程管线；注意财报季跳空对波动率估计的影响。",
        "examples": "`AAPL`、`MSFT`、`NVDA`",
    },
    "hk_stock": {
        "kind": "在香港联合交易所上市的股票，港币计价；代码常为 5 位数字，本应用会自动补零。",
        "social": "连接内地资产与全球资本的中枢之一，大量中资股与美元利率、南向资金流量密切相关。",
        "math": "日 OHLCV；收益率统计与 A 股同构；节假日安排与沪深不同，对齐指数时需注意交易日历。",
        "examples": "`00700`、`09988`、`01810`",
    },
    "global_index": {
        "kind": "表征一组证券整体表现的统计量。以 `.` 开头（如 `.INX`）走**新浪美股指数**；`HSI`/`SPX` 等简码走**东财全球指数**中文名映射。",
        "social": "宏观与风险情绪的晴雨表，用于观察市场宽度、区域轮动与系统性风险，是资产配置与业绩基准的参照。",
        "math": "点位序列视为“价格”；部分来源**无成交量**（Volume=0），图表可能为单面板 K 线；与成分股加总并非线性关系。",
        "examples": "`.INX`（标普500新浪）、`HSI`（恒生）、`SPX`（标普500东财）、`DXY`（美元指数）",
    },
    "bond_cn": {
        "kind": "在沪深挂牌交易的**国债或债券**日 K（新浪源），代码形如 `sh010107`。",
        "social": "无风险利率与债市情绪的表征，影响全市场贴现率；机构配置与货币政策预期会体现在价格波动中。",
        "math": "标准 OHLCV；部分行情源无成交量字段时已补 0；久期与凸性等债项指标**未**在本应用中计算。",
        "examples": "`sh010107`、`sh019547`（示例，以数据源实际可交易代码为准）",
    },
    "bond_us_yield": {
        "kind": "中债登口径的**美国国债收益率曲线**日度数据（`bond_zh_us_rate`），`Close` 为**收益率水平（%）**，非债券价格。",
        "social": "全球资产定价的锚之一，影响汇率、房贷与企业债利差；10 年期常被视为“无风险长期利率”的代理变量。",
        "math": "伪 OHLCV，水平序列；看板中 CAGR/夏普等解释为**水平变化统计**，**不是**持有美债的持有期回报；默认列可用 `10Y`。",
        "examples": "`10Y`（默认美债10年列）、或粘贴表头完整列名如 `美国国债收益率10年`",
    },
    "commodity": {
        "kind": "外盘**大宗商品期货**主力连续（如原油、铜），与贵金属同源 `futures_foreign_hist`，美元计价为主。",
        "social": "实体经济成本与周期波动的晴雨表，连接通胀、制造业与地缘供给；常作为股票组合的商品敞口代理。",
        "math": "连续合约有换月跳空；波动率与趋势指标对供给冲击敏感；与股票相关性随周期变化。",
        "examples": "`CL`（原油）、`HG`（铜）、`NG`（天然气，视数据源是否提供）",
    },
    "creit": {
        "kind": "中国**基础设施公募 REITs**，在沪深交易，份额代表对底层基础设施收益权的证券化权益。",
        "social": "盘活存量基建资产、拓宽投融资渠道；价格反映分红预期、利率环境与流动性溢价。",
        "math": "东财日 K 线 OHLCV；与股票共用技术指标；分红除权规则与 ETF/股票不同，复权口径以数据源为准。",
        "examples": "`508097`、`180101`、`508000`（示例，以交易所挂牌为准）",
    },
    "fx": {
        "kind": "**外汇即期**汇率对（如离岸人民币 `USDCNH`），本应用经东财多节点拉取日 K，表示一单位基准货币兑另一货币的比价。",
        "social": "国际贸易与资本流动的价格纽带，影响出口竞争力与跨境资金成本；央行预期与利差是主要驱动。",
        "math": "汇率水平为正值序列，宜用 K 线观察；对数收益近似刻画日相对变化；与股票指数联合分析时需注意共线性与因果方向。",
        "examples": "`USDCNH`、`EURUSD`、`USD/JPY`（别名映射为数据源代码时以 `EURUSD` 等形式为准）",
    },
}


def format_asset_guide_markdown(asset_type: str) -> str:
    """生成当前资产类型的 Markdown 说明（侧栏与主区复用）。"""
    key = str(asset_type).strip().lower()
    g = ASSET_TYPE_GUIDES.get(key)
    if g is None:
        return ""
    title = ASSET_LABELS.get(key, key)
    return (
        "#### %s\n\n"
        "**类型**　%s\n\n"
        "**社会意义**　%s\n\n"
        "**数学与诊断口径**　%s\n\n"
        "**代表代码示例**　%s\n"
    ) % (title, g["kind"], g["social"], g["math"], g["examples"])


def _identity_cache_data(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """在未安装 Streamlit 时提供无副作用的缓存装饰器占位。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


try:
    st = importlib.import_module("streamlit")
except ImportError:
    st = None


cache_data = st.cache_data if st is not None else _identity_cache_data


def _require_streamlit() -> Any:
    """在真正渲染页面前检查 Streamlit 依赖。"""
    if st is None:
        raise ImportError(
            "运行 `app.py` 需要安装 streamlit。请先执行 `pip install streamlit plotly akshare ccxt`。"
        )
    return st


def _load_optional_dependency(module_name: str, install_name: str | None = None) -> Any:
    """按需导入三方依赖，避免模块导入阶段就因缺包失败。"""
    package_name = install_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "缺少运行依赖 `%s`，请先安装：`pip install %s`。"
            % (module_name, package_name)
        ) from exc


def _coerce_date(value: object, field_name: str) -> pd.Timestamp:
    """将外部日期输入转换为不带时区的日级时间戳。"""
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("%s 不是合法日期: %r" % (field_name, value)) from exc

    if pd.isna(ts):
        raise ValueError("%s 不是合法日期: %r" % (field_name, value))
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _validate_date_range(start_date: object, end_date: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    """校验用户输入的开始和结束日期。"""
    start_ts = _coerce_date(start_date, "start_date")
    end_ts = _coerce_date(end_date, "end_date")
    if start_ts > end_ts:
        raise ValueError("start_date 必须早于或等于 end_date。")
    return start_ts, end_ts


def _normalize_proxy_url(proxy_url: str | None) -> str | None:
    """将用户输入的代理地址规整为带协议前缀的 URL。"""
    if proxy_url is None:
        return None

    normalized = str(proxy_url).strip()
    if not normalized:
        return None

    if "://" not in normalized:
        normalized = "http://%s" % normalized

    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("proxy_url 格式无效，请使用 `http://127.0.0.1:端口`。")
    return normalized


def _normalize_ohlcv(df: pd.DataFrame, column_mapping: dict[str, str]) -> pd.DataFrame:
    """将任意原始行情表清洗为统一 OHLCV 契约。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必须是 pandas DataFrame。")
    if df.empty:
        raise ValueError("原始数据为空。")

    normalized = df.rename(columns=column_mapping).copy()

    if "Date" in normalized.columns:
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"])
        if normalized.empty:
            raise ValueError("Date 列无法解析为有效日期。")
        normalized = normalized.set_index("Date")
    elif isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index, errors="coerce")
        normalized = normalized[normalized.index.notna()]
    else:
        raise ValueError("原始数据必须包含 Date 列，或索引本身已是 DatetimeIndex。")

    missing_columns = [column for column in REQUIRED_OHLCV_COLUMNS if column not in normalized.columns]
    if missing_columns:
        raise ValueError("标准化后缺少 OHLCV 列: %s" % ", ".join(missing_columns))

    normalized = normalized.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    for column in REQUIRED_OHLCV_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.replace([float("inf"), float("-inf")], pd.NA)
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.ffill()
    normalized = normalized.dropna(subset=list(REQUIRED_OHLCV_COLUMNS))
    if normalized.empty:
        raise ValueError("标准化后没有可用的 OHLCV 数据。")

    normalized = normalized.astype("float64")
    normalized.index = pd.DatetimeIndex(normalized.index)
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index.name = "Date"
    return normalized


def _filter_by_date_range(
    df: pd.DataFrame,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """按用户选择的日期区间过滤数据。"""
    start_ts, end_ts = _validate_date_range(start_date, end_date)
    filtered = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    if filtered.empty:
        raise ValueError(
            "指定日期区间内没有可用数据：%s 至 %s。"
            % (start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        )
    return filtered


def _resolve_gold_symbol(symbol: str) -> tuple[str, str]:
    """将展示代码映射为底层抓取所需的贵金属代码。"""
    display_symbol = str(symbol).strip().upper()
    if not display_symbol:
        raise ValueError("黄金代码不能为空。")
    vendor_symbol = GOLD_SYMBOL_ALIASES.get(display_symbol, display_symbol)
    return display_symbol, vendor_symbol


def _fetch_etf_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 AKShareFetcher 路由抓取 ETF 并返回统一 OHLCV。"""
    return load_china_equity_ohlcv("etf", symbol, start_date, end_date, adjust="hfq")


def _fetch_ashare_stock_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 AKShare ``stock_zh_a_hist`` 抓取 A 股个股并返回统一 OHLCV（Open/High/Low/Close/Volume）。"""
    return load_china_equity_ohlcv("ashare", symbol, start_date, end_date, adjust="hfq")


def _fetch_mutual_fund_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 AKShare ``fund_open_fund_info_em`` 拉取开放式基金净值（Close 为累计净值）。"""
    fund_code = str(symbol).strip()
    if not fund_code:
        raise ValueError("公募基金代码不能为空。")
    df = get_fund_data(fund_code, start_date, end_date)
    df.attrs["asset_type"] = "mutual_fund"
    df.attrs["symbol"] = fund_code
    return df


def _fetch_gold_data(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
    """通过 ``futures_foreign_hist`` 加载贵金属期货行情。"""
    display_symbol, vendor_symbol = _resolve_gold_symbol(symbol)
    filtered = load_futures_foreign_ohlcv(
        vendor_symbol,
        start_date,
        end_date,
        asset_type="gold",
        unit_caption="USD · 贵金属外盘期货（主力连续，细则以数据源为准）",
    )
    filtered.attrs["symbol"] = display_symbol
    filtered.attrs["vendor_symbol"] = vendor_symbol
    return filtered


def _build_ccxt_proxy_config(proxy_url: str | None) -> dict[str, str] | None:
    """将单个代理地址扩展为 CCXT 期望的代理配置。"""
    normalized_proxy = _normalize_proxy_url(proxy_url)
    if normalized_proxy is None:
        return None
    return {
        "http": normalized_proxy,
        "https": normalized_proxy,
    }


def _fetch_crypto_ohlcv_rows(
    exchange: Any,
    symbol: str,
    start_date: object,
    end_date: object,
    timeframe: str = "1d",
    batch_limit: int = 1000,
) -> list[list[Any]]:
    """按日线分页拉取加密货币 K 线。"""
    start_ts, end_ts = _validate_date_range(start_date, end_date)
    timeframe_ms = 24 * 60 * 60 * 1000
    end_ms = int((end_ts + pd.Timedelta(days=1)).timestamp() * 1000)
    since_ms = int(start_ts.timestamp() * 1000)
    rows: list[list[Any]] = []

    # CCXT 的日线接口通常需要通过 since + limit 分页，避免长区间直接被截断。
    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=batch_limit,
        )
        if not batch:
            break

        rows.extend(batch)
        last_ts = int(batch[-1][0])
        next_since = last_ts + timeframe_ms
        if next_since <= since_ms:
            break
        since_ms = next_since

        if last_ts >= end_ms:
            break

    return rows


def _normalize_crypto_data(
    rows: list[list[Any]],
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """将 CCXT 返回的 OHLCV 列表标准化为 DataFrame。"""
    if not rows:
        raise ValueError("交易所未返回任何 OHLCV 数据。")

    raw_df = pd.DataFrame(
        rows,
        columns=["Date", "Open", "High", "Low", "Close", "Volume"],
    )
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], unit="ms", utc=True).dt.tz_localize(None)
    normalized = _normalize_ohlcv(
        raw_df,
        {
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        },
    )
    return _filter_by_date_range(normalized, start_date, end_date)


def _fetch_crypto_data(
    symbol: str,
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """通过 CCXT（默认 Binance）抓取加密货币 OHLCV。"""
    symbol_text = str(symbol).strip().upper()
    if not symbol_text:
        raise ValueError("加密货币交易对不能为空，例如 `BTC/USDT`。")

    ccxt = _load_optional_dependency("ccxt")
    exchange_config: dict[str, Any] = {
        "timeout": 30000,
        "enableRateLimit": True,
    }

    proxies = _build_ccxt_proxy_config(proxy_url)
    if proxies is not None:
        exchange_config["proxies"] = proxies

    exchange = ccxt.binance(exchange_config)
    rows = _fetch_crypto_ohlcv_rows(
        exchange=exchange,
        symbol=symbol_text,
        start_date=start_date,
        end_date=end_date,
    )
    normalized = _normalize_crypto_data(rows, start_date=start_date, end_date=end_date)
    normalized.attrs["asset_type"] = "crypto"
    normalized.attrs["symbol"] = symbol_text
    if proxies is not None:
        normalized.attrs["proxy_url"] = proxies["https"]
    return normalized


@cache_data(ttl=3600, show_spinner=False)
def load_asset_data(
    asset_type: str,
    symbol: str,
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """统一加载多类资产 OHLCV（契约一致）。"""
    asset_key = str(asset_type).strip().lower()
    if asset_key not in SUPPORTED_ASSET_TYPES:
        raise ValueError("不支持的资产类型: %s" % asset_type)

    if asset_key == "etf":
        return _fetch_etf_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "ashare":
        return _fetch_ashare_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "mutual_fund":
        return _fetch_mutual_fund_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "gold":
        return _fetch_gold_data(symbol=symbol, start_date=start_date, end_date=end_date)
    if asset_key == "crypto":
        return _fetch_crypto_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            proxy_url=proxy_url,
        )
    if asset_key == "us_stock":
        return load_global_market_data("us_stock", symbol, start_date, end_date)
    if asset_key == "hk_stock":
        return load_global_market_data("hk_stock", symbol, start_date, end_date)
    if asset_key == "global_index":
        return load_global_market_data("global_index", symbol, start_date, end_date)
    if asset_key == "bond_cn":
        return load_bond_data("cn_hist", symbol, start_date, end_date)
    if asset_key == "bond_us_yield":
        sym = str(symbol).strip()
        col: str | None = None
        if sym and sym.upper() not in ("10Y", "默认", "DEFAULT"):
            col = sym
        else:
            col = US_TREASURY_YIELD_10Y_CN_COL
        return load_bond_data(
            "us_yield",
            "",
            start_date,
            end_date,
            us_yield_column=col,
        )
    if asset_key == "commodity":
        return load_commodity_data(symbol, start_date, end_date)
    if asset_key == "creit":
        return load_reit_data(symbol, start_date, end_date)
    return load_fx_data(symbol, start_date, end_date)


@cache_data(ttl=3600, show_spinner=False)
def load_multiple_etfs_close(
    symbols_tuple: tuple[str, ...],
    start_date: object,
    end_date: object,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """多标的 A 股 ETF 收盘价宽表；参数为元组以便 Streamlit ``cache_data`` 键稳定。"""
    if not symbols_tuple:
        raise ValueError("symbols_tuple 不能为空。")
    return get_multiple_assets_close(list(symbols_tuple), start_date, end_date, adjust=adjust)


def _compute_cumulative_return(close: pd.Series) -> float:
    """计算区间累计收益率。"""
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def _compute_cagr(close: pd.Series) -> float:
    """基于起止价格和自然日跨度计算 CAGR。"""
    elapsed_days = float((close.index[-1] - close.index[0]).days)
    if elapsed_days <= 0:
        return float("nan")

    years = elapsed_days / 365.25
    if years <= 0:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0)


def _compute_sharpe_ratio(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> float:
    """使用日对数收益率计算年化 Sharpe。"""
    log_returns = engineer.compute_returns(df, price_col="Close", periods=1, log=True).dropna()
    if log_returns.empty:
        return float("nan")

    daily_rf_log = np.log1p(risk_free_rate) / float(trading_days)
    excess_log_returns = log_returns - daily_rf_log
    volatility = float(excess_log_returns.std())
    if not np.isfinite(volatility) or volatility == 0.0:
        return float("nan")

    sharpe = math.sqrt(float(trading_days)) * float(excess_log_returns.mean()) / volatility
    return float(sharpe)


def _describe_bollinger_position(df: pd.DataFrame) -> str:
    """描述最新收盘价位于布林带的哪个区间。"""
    feature_summary = df.attrs.get("feature_summary", {})
    bollinger_window = int(feature_summary.get("bollinger_window", 20))
    upper_col = "bollinger_upper_%d" % bollinger_window
    lower_col = "bollinger_lower_%d" % bollinger_window

    if upper_col not in df.columns or lower_col not in df.columns:
        return "布林带数据缺失"

    latest = df.iloc[-1]
    latest_close = float(latest["Close"])
    upper = latest.get(upper_col)
    lower = latest.get(lower_col)
    if pd.isna(upper) or pd.isna(lower):
        return "布林带数据不足"
    if latest_close >= float(upper):
        return "突破上轨"
    if latest_close <= float(lower):
        return "触底下轨"
    return "均值回归中"


def get_quant_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> dict[str, Any]:
    """将 CLI 数学诊断逻辑迁移为 Web 指标字典输出。"""
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    engineer = OHLCVFeatureEngineer()
    featured_df = engineer.transform(df)
    featured_df.attrs.update(df.attrs)
    featured_df.attrs["series_kind"] = series_kind

    close = pd.to_numeric(featured_df["Close"], errors="coerce").astype("float64")
    if close.isna().any():
        raise ValueError("Close 列存在 NaN，无法计算指标。")

    metrics: dict[str, Any] = {
        "cumulative_return": _compute_cumulative_return(close),
        "cagr": _compute_cagr(close),
        "max_drawdown": float(featured_df.attrs.get("max_drawdown", engineer.compute_max_drawdown(df))),
        "sharpe_ratio": _compute_sharpe_ratio(
            engineer,
            featured_df,
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
        ),
        "latest_signal": _describe_bollinger_position(featured_df),
        "featured_df": featured_df,
        "series_kind": series_kind,
    }
    if series_kind == "yield_level":
        metrics["metric_disclaimer"] = (
            "以下为收益率**水平**序列的统计（末/初比与波动），不代表债券持有期回报；"
            "夏普等指标基于水平序列的对数变化，仅作相对比较参考。"
        )
        metrics["cumulative_return_label"] = "水平相对变化（Close末/初-1）"
        metrics["cagr_label"] = "水平年化缩放（非票息回报）"
    else:
        metrics["metric_disclaimer"] = ""
        metrics["cumulative_return_label"] = "累计收益率"
        metrics["cagr_label"] = "CAGR"
    return metrics


def _format_percent_metric(value: float, signed: bool = False) -> str:
    """格式化百分比指标，支持 NaN 安全显示。"""
    if value is None or pd.isna(value):
        return "N/A"
    fmt = "%+.2f%%" if signed else "%.2f%%"
    return fmt % (100.0 * float(value))


def _format_ratio_metric(value: float) -> str:
    """格式化比率指标。"""
    if value is None or pd.isna(value):
        return "N/A"
    return "%.3f" % float(value)


def _load_plotly_modules() -> tuple[Any, Any]:
    """按需导入 Plotly，避免模块导入阶段缺包即失败。"""
    go = _load_optional_dependency("plotly.graph_objects", install_name="plotly")
    subplots = _load_optional_dependency("plotly.subplots", install_name="plotly")
    return go, subplots


def build_price_volume_figure(df: pd.DataFrame) -> Any:
    """绘制 K 线 + 成交量；净值/收益率水平/零成交量序列用单面板折线。"""
    go, subplots = _load_plotly_modules()
    make_subplots = subplots.make_subplots
    asset_key = str(df.attrs.get("asset_type", "")).strip().lower()
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    vol_max = float(pd.to_numeric(df["Volume"], errors="coerce").fillna(0).abs().max())
    use_line = asset_key == "mutual_fund" or series_kind == "yield_level"
    show_volume_panel = not use_line and vol_max > 0.0

    if use_line:
        line_name = "累计净值" if asset_key == "mutual_fund" else "收盘序列"
        if series_kind == "yield_level":
            line_name = "收益率水平"
        figure = make_subplots(rows=1, cols=1)
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=line_name,
                line={"width": 2.0, "color": "#1f77b4"},
            ),
            row=1,
            col=1,
        )
    elif show_volume_panel:
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
        )
        figure.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K线",
            ),
            row=1,
            col=1,
        )
    else:
        figure = make_subplots(rows=1, cols=1)
        figure.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K线",
            ),
            row=1,
            col=1,
        )

    feature_summary = df.attrs.get("feature_summary", {})
    short_window = int(feature_summary.get("sma_short_window", 20))
    long_window = int(feature_summary.get("sma_long_window", 60))
    bollinger_window = int(feature_summary.get("bollinger_window", 20))

    overlay_columns = [
        ("sma_%d" % short_window, "SMA %d" % short_window, "#1f77b4"),
        ("sma_%d" % long_window, "SMA %d" % long_window, "#ff7f0e"),
        ("bollinger_upper_%d" % bollinger_window, "布林上轨", "#9467bd"),
        ("bollinger_lower_%d" % bollinger_window, "布林下轨", "#9467bd"),
    ]
    for column, trace_name, color in overlay_columns:
        if column not in df.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode="lines",
                name=trace_name,
                line={"width": 1.6, "color": color},
                opacity=0.9 if "bollinger" not in column else 0.65,
            ),
            row=1,
            col=1,
        )

    if show_volume_panel:
        volume_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
        figure.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="成交量",
                marker={"color": volume_colors},
            ),
            row=2,
            col=1,
        )

    figure.update_layout(
        height=760,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    y1 = "净值" if asset_key == "mutual_fund" else "价格"
    if series_kind == "yield_level":
        y1 = "收益率水平"
    if use_line or not show_volume_panel:
        figure.update_yaxes(title_text=y1, row=1, col=1)
    if show_volume_panel:
        figure.update_yaxes(title_text=y1, row=1, col=1)
        figure.update_yaxes(title_text="成交量", row=2, col=1)
    return figure


def build_tab3_normalize_figure(dates: pd.Index, normalized: pd.Series) -> Any:
    """侧栏单一标的：收盘价归一化为期初=1 的净值曲线（暗色主题）。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=normalized,
            mode="lines",
            name="Nₜ = Pₜ / P₀",
            line={"width": 2.2, "color": "#7fdbff"},
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="归一化净值",
        xaxis_title="日期",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return figure


def build_tab3_gbm_figure(result: MonteCarloGBMResult) -> Any:
    """GBM 蒙特卡洛：历史收盘价、未来分位带、中位数与若干示例路径。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.historical_dates,
            y=result.historical_close,
            mode="lines",
            name="历史收盘价",
            line={"width": 2.0, "color": "#00d4ff"},
        )
    )
    fut_x = result.future_dates
    figure.add_trace(
        go.Scatter(
            x=fut_x,
            y=result.quantiles_p95,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=fut_x,
            y=result.quantiles_p05,
            mode="lines",
            name="5%–95% 分位带",
            fill="tonexty",
            fillcolor="rgba(0, 191, 255, 0.22)",
            line={"width": 0},
        )
    )
    n_paths = int(result.sample_paths.shape[0])
    for i in range(n_paths):
        figure.add_trace(
            go.Scatter(
                x=fut_x,
                y=result.sample_paths[i],
                mode="lines",
                name="示例路径" if i == 0 else None,
                legendgroup="samples",
                showlegend=(i == 0),
                line={"width": 0.9, "color": "rgba(180, 180, 200, 0.45)"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=fut_x,
            y=result.quantiles_p50,
            mode="lines",
            name="中位数路径",
            line={"width": 2.2, "color": "#ffcc00"},
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="价格",
        xaxis_title="日期",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0.0},
    )
    return figure


def build_tab3_efficient_frontier_figure(ef: EfficientFrontierResult) -> Any:
    """随机权重组合散点（颜色=夏普），最优组合红星。"""
    go, _ = _load_plotly_modules()
    vol = ef.volatility
    ret = ef.returns
    sharpe = ef.sharpe
    idx = int(ef.best_sharpe_idx)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=vol,
            y=ret,
            mode="markers",
            name="随机权重组合",
            marker={
                "size": 6,
                "color": sharpe,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "夏普"},
                "opacity": 0.85,
            },
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[float(vol[idx])],
            y=[float(ret[idx])],
            mode="markers",
            name="最大夏普组合",
            marker={
                "size": 20,
                "symbol": "star",
                "color": "#ff2a2a",
                "line": {"width": 1.2, "color": "#ffffff"},
            },
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        template="plotly_dark",
        hovermode="closest",
        xaxis_title="年化波动率",
        yaxis_title="年化收益",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return figure


def _sync_sidebar_state(streamlit: Any, asset_type: str) -> None:
    """让侧边栏默认值在资产类型切换时保持联动。"""
    session_state = streamlit.session_state
    last_asset_type = session_state.get("_last_asset_type")
    if "symbol_input" not in session_state:
        session_state["symbol_input"] = DEFAULT_SYMBOLS[asset_type]
    if "proxy_url_input" not in session_state:
        session_state["proxy_url_input"] = ""

    if last_asset_type != asset_type:
        session_state["symbol_input"] = DEFAULT_SYMBOLS[asset_type]
        if asset_type != "crypto":
            session_state["proxy_url_input"] = ""

    session_state["_last_asset_type"] = asset_type


_TAB2_SYSTEM_MATH_MARKDOWN = r"""
### 1. 收益与风险度量

设 $P_t$ 为第 $t$ 个交易日收盘价（或复权价），简单收益率与对数收益率常分别写为：

$$R_t=\frac{P_t-P_{t-1}}{P_{t-1}}, \qquad r_t=\ln\frac{P_t}{P_{t-1}}$$

在样本区间 $[0,T]$ 内，若共 $n$ 个交易日，年化复合收益率（CAGR）可表示为：

$$\mathrm{CAGR}=\left(\frac{P_T}{P_0}\right)^{252/n}-1$$

（上式将一年折算为 $252$ 个交易日，与本工具链常用的年化尺度一致。）

最大回撤（Maximum Drawdown）衡量自历史峰值以来的最深回落比例。设净值曲线为 $W_t$（例如从 1 起算的累计净值），则：

$$\mathrm{MDD}=\max_{\tau\in[0,T]}\left(\frac{\max_{s\le\tau}W_s-W_\tau}{\max_{s\le\tau}W_s}\right)$$

在诊断面板中，「最大回撤」以正数角标展示深度，可理解为 $\mathrm{MDD}$ 的绝对值。

夏普比率（Sharpe Ratio）在经典定义下使用超额收益相对波动率的比值，并按 $\sqrt{252}$ 做年化缩放：

$$S=\frac{E[R-r_f]}{\sigma(R)}\sqrt{252}$$

其中 $r_f$ 为年化无风险利率（本应用内部采用常数近似以便横截面可比），$R$ 为日收益率序列，$\sigma$ 为样本标准差。

---

### 2. 累计收益与信号展示

累计收益率可将各期收益链式复合，例如简单收益口径下：

$$1+R_{\mathrm{cum}}=\prod_{t=1}^{n}(1+R_t)$$

看板中的「最新信号」由特征工程与规则化输出共同决定，其底层是对价格、均线与波动包络等构造量的序贯更新；在阅读图表时，可将均线视为对 $P_t$ 的线性平滑，将布林带可视作对条件方差结构的启发式刻画。

---

### 3. 可视化与数据契约

K–线（OHLC）与成交量在同一时间轴上联合展示，横轴为交易日期。清洗后的数据表遵循统一列名契约（`Open` / `High` / `Low` / `Close` / `Volume`），以便跨境、跨资产类别复用同一套指标与绘图管线。

> **说明**：上述记号与年化习惯用于帮助理解面板指标；实际实现细节（如对数/简单收益选用、缺失值处理、复权口径等）以代码与拉取数据源为准。
"""

_WIN_FILENAME_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _sanitize_symbol_for_filename(display_symbol: str) -> str:
    """净化代码片段，避免 Windows 非法字符并将路径分隔符替换为下划线。"""
    text = str(display_symbol).strip() or "symbol"
    text = text.replace("/", "_")
    text = _WIN_FILENAME_FORBIDDEN.sub("_", text)
    text = text.strip(" .")
    return text or "symbol"


def _cleaned_ohlcv_to_csv_bytes_and_filename(
    df: pd.DataFrame,
    asset_type: str,
    display_symbol: str,
    start_date: object,
    end_date: object,
) -> tuple[bytes, str]:
    """将 load_asset_data 返回的清洗后 OHLCV 导出为 UTF-8-SIG CSV 字节与安全文件名。"""
    subset = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    subset = subset.sort_index()
    if not isinstance(subset.index, pd.DatetimeIndex):
        subset.index = pd.to_datetime(subset.index, errors="coerce")
    subset.index.name = "Date"
    text_buffer = io.StringIO()
    subset.to_csv(text_buffer, index=True, date_format="%Y-%m-%d")
    csv_bytes = text_buffer.getvalue().encode("utf-8-sig")
    at_key = str(asset_type).strip().lower()
    sym_part = _sanitize_symbol_for_filename(display_symbol)
    at_part = _sanitize_symbol_for_filename(at_key)
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    filename = "%s_%s_%s_%s_ohlcv.csv" % (at_part, sym_part, start_s, end_s)
    return csv_bytes, filename


def render_app() -> None:
    """渲染 Streamlit 量化诊断应用。"""
    streamlit = _require_streamlit()
    streamlit.set_page_config(page_title="量化诊断应用", layout="wide")
    streamlit.title("量化诊断应用")
    streamlit.caption(
        "支持 A 股、公募、贵金属、加密、美股/港股/全球指数、国债与美债收益率、外盘商品、REITs、外汇等；"
        "统一 OHLCV 契约下的诊断与图表（收益率水平类资产见看板说明）。"
    )

    today = pd.Timestamp.today().normalize().date()
    default_start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=LOOKBACK_YEARS)).date()
    streamlit.sidebar.header("参数设置")
    asset_type = streamlit.sidebar.selectbox(
        "资产类型",
        options=list(SUPPORTED_ASSET_TYPES),
        format_func=lambda key: ASSET_LABELS[key],
        key="asset_type",
    )
    _sync_sidebar_state(streamlit, asset_type)
    code_label = "代码"
    if asset_type == "bond_us_yield":
        code_label = "收益率列（默认 10Y=美债10年，可填表中完整列名）"
    symbol = streamlit.sidebar.text_input(code_label, key="symbol_input").strip()
    start_date = streamlit.sidebar.date_input(
        "开始日期",
        value=default_start,
    )
    end_date = streamlit.sidebar.date_input(
        "结束日期",
        value=today,
    )

    proxy_url: str | None = None
    if asset_type == "crypto":
        proxy_url = streamlit.sidebar.text_input(
            "代理地址",
            key="proxy_url_input",
            placeholder="http://127.0.0.1:33210",
        )
        streamlit.sidebar.caption("代理仅在加密货币模式下生效。")

    with streamlit.sidebar.expander("资产类型说明", expanded=False):
        streamlit.markdown(format_asset_guide_markdown(asset_type))

    clicked = streamlit.sidebar.button("开始诊断", use_container_width=True)

    tab1, tab2, tab3 = streamlit.tabs(
        ["📊 量化诊断看板", "📚 系统数学原理解析", "🧮 进阶金融算法"]
    )

    with tab2:
        streamlit.markdown(_TAB2_SYSTEM_MATH_MARKDOWN)

    with tab3:
        streamlit.markdown(
            "本页使用**左侧**资产类型、代码与起止日期；各区块**独立按钮**拉数与计算，无需先点击「开始诊断」。"
        )
        streamlit.caption(
            "有效前沿（模块 B）仅支持 **A 股 ETF 代码** 逗号分隔列表；GBM 与归一化沿用当前侧栏单一标的。"
        )

        streamlit.subheader("0 · 基准归一化")
        streamlit.markdown(
            r"将收盘价序列归一化为以样本首日价格为 1 的净值：$N_t = P_t / P_{t_0}$（$P$ 为 `Close`）。"
        )
        if streamlit.button("显示归一化净值曲线", key="tab3_btn_normalize", use_container_width=True):
            try:
                with streamlit.spinner("正在加载行情…"):
                    df_norm = load_asset_data(
                        asset_type=asset_type,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        proxy_url=proxy_url,
                    )
                if df_norm.empty:
                    streamlit.error(EMPTY_SYMBOL_FETCH_HINT)
                else:
                    series_n = normalize_close_to_base_one(df_norm)
                    fig_n = build_tab3_normalize_figure(df_norm.index, series_n)
                    streamlit.plotly_chart(fig_n, use_container_width=True)
            except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                streamlit.error(format_diagnosis_user_message(exc))

        streamlit.subheader("A · GBM 蒙特卡洛")
        streamlit.markdown(
            "基于历史日对数收益估计波动与漂移，按几何布朗运动离散递推未来 **252** 个交易日路径；"
            "图中为分位数带与若干示例路径。"
        )
        if streamlit.button("运行 GBM 模拟", key="tab3_btn_gbm", use_container_width=True):
            try:
                with streamlit.spinner("正在加载数据并模拟路径…"):
                    df_gbm = load_asset_data(
                        asset_type=asset_type,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        proxy_url=proxy_url,
                    )
                if df_gbm.empty:
                    streamlit.error(EMPTY_SYMBOL_FETCH_HINT)
                else:
                    mc = run_monte_carlo_gbm(
                        df_gbm,
                        days=TRADING_DAYS,
                        num_simulations=500,
                        random_state=42,
                        num_sample_paths=5,
                    )
                    streamlit.caption(
                        "估计参数：日漂移 μ≈%.6f，日波动 σ≈%.6f，起点 S₀=%.4f。"
                        % (mc.mu_daily, mc.sigma_daily, mc.s0)
                    )
                    fig_g = build_tab3_gbm_figure(mc)
                    streamlit.plotly_chart(fig_g, use_container_width=True)
            except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                streamlit.error(format_diagnosis_user_message(exc))

        streamlit.subheader("B · 有效前沿（随机权重）")
        ef_default = "%s,%s" % (
            DEFAULT_SYMBOLS.get("etf", "510300"),
            "510500",
        )
        ef_symbols_text = streamlit.text_input(
            "ETF 代码（英文逗号分隔，至少 2 个）",
            value=ef_default,
            key="tab3_ef_symbol_input",
        )
        if streamlit.button("运行协方差矩阵优化", key="tab3_btn_ef", use_container_width=True):
            raw_parts = [p.strip() for p in str(ef_symbols_text).split(",")]
            codes = [p for p in raw_parts if p]
            if len(codes) < 2:
                streamlit.error("请至少输入两个有效 ETF 代码。")
            else:
                try:
                    sym_tuple = tuple(sorted(codes))
                    with streamlit.spinner("正在拉取多标的收盘价并模拟组合…"):
                        wide_close = load_multiple_etfs_close(
                            sym_tuple,
                            start_date,
                            end_date,
                            adjust="hfq",
                        )
                    returns_wide = wide_close.pct_change().dropna()
                    if returns_wide.shape[1] < 2:
                        streamlit.error("有效资产列不足，请检查代码是否正确。")
                    else:
                        ef_res: EfficientFrontierResult = generate_efficient_frontier(
                            returns_wide,
                            num_portfolios=5000,
                            risk_free_rate=RISK_FREE_RATE,
                            trading_days=TRADING_DAYS,
                            random_state=42,
                        )
                        fig_ef = build_tab3_efficient_frontier_figure(ef_res)
                        streamlit.plotly_chart(fig_ef, use_container_width=True)
                        w_best = ef_res.weights[ef_res.best_sharpe_idx]
                        weights_df = pd.DataFrame(
                            [w_best],
                            columns=list(returns_wide.columns),
                        )
                        streamlit.markdown("**最大夏普比率组合的权重**")
                        streamlit.dataframe(weights_df, use_container_width=True)
                except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                    streamlit.error(format_diagnosis_user_message(exc))

    with tab1:
        if not clicked:
            streamlit.info("在左侧选择资产和时间区间后点击“开始诊断”，即可查看 Web 版量化分析结果。")
            with streamlit.expander("资产类型说明（与左侧「资产类型」联动）", expanded=False):
                streamlit.markdown(format_asset_guide_markdown(asset_type))
        else:
            try:
                with streamlit.spinner("正在加载数据并计算量化指标..."):
                    df = load_asset_data(
                        asset_type=asset_type,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        proxy_url=proxy_url,
                    )
                    if df.empty:
                        streamlit.error(EMPTY_SYMBOL_FETCH_HINT)
                        return
                    metrics = get_quant_metrics(df)
                    figure = build_price_volume_figure(metrics["featured_df"])
            except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                streamlit.error(format_diagnosis_user_message(exc))
                return

            display_symbol = str(df.attrs.get("symbol", symbol)).strip() or symbol
            streamlit.subheader("%s · %s" % (ASSET_LABELS[asset_type], display_symbol))
            unit_cap = str(df.attrs.get("unit_caption", "") or "").strip()
            if unit_cap:
                streamlit.caption("单位与口径：%s" % unit_cap)
            streamlit.caption(
                "样本区间：%s 至 %s，共 %d 条记录。"
                % (
                    df.index[0].strftime("%Y-%m-%d"),
                    df.index[-1].strftime("%Y-%m-%d"),
                    len(df),
                )
            )
            if metrics.get("metric_disclaimer"):
                streamlit.warning(metrics["metric_disclaimer"])
            csv_bytes, csv_filename = _cleaned_ohlcv_to_csv_bytes_and_filename(
                df,
                asset_type=asset_type,
                display_symbol=display_symbol,
                start_date=start_date,
                end_date=end_date,
            )
            streamlit.download_button(
                label="下载清洗后 OHLCV（CSV）",
                data=csv_bytes,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=False,
                key="download_cleaned_ohlcv_csv",
            )
            streamlit.caption(
                "CSV 含 Date 与 Open、High、Low、Close、Volume 五列，与看板清洗后数据一致（不含均线、布林带等特征列）。"
            )

            metric_col_1, metric_col_2, metric_col_3 = streamlit.columns(3)
            metric_col_1.metric(metrics["cagr_label"], _format_percent_metric(metrics["cagr"]))
            metric_col_2.metric("最大回撤", _format_percent_metric(-abs(metrics["max_drawdown"]), signed=True))
            metric_col_3.metric("Sharpe", _format_ratio_metric(metrics["sharpe_ratio"]))

            streamlit.info(
                "最新信号：%s | %s：%s"
                % (
                    metrics["latest_signal"],
                    metrics["cumulative_return_label"],
                    _format_percent_metric(metrics["cumulative_return"], signed=True),
                )
            )
            streamlit.plotly_chart(figure, use_container_width=True)

            with streamlit.expander("查看清洗后与特征后数据", expanded=False):
                streamlit.markdown("**清洗后 OHLCV 数据尾部**")
                streamlit.dataframe(df.tail(20), use_container_width=True)
                streamlit.markdown("**特征工程结果尾部**")
                streamlit.dataframe(metrics["featured_df"].tail(20), use_container_width=True)
                streamlit.caption("标准列: %s" % ", ".join(list(REQUIRED_OHLCV_COLUMNS)))


if __name__ == "__main__":
    render_app()
