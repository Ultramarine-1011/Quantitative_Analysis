"""Streamlit 量化诊断应用入口。

统一数据适配层覆盖 A 股 ETF、A 股个股、公募基金、贵金属与加密货币，输出一致 OHLCV 契约，
供指标层与图表层复用。
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from china_equity_entry import (
    CHINA_EQUITY_LABELS,
    DEFAULT_CHINA_EQUITY_SYMBOLS,
    EMPTY_SYMBOL_FETCH_HINT,
    format_china_equity_user_message,
    load_china_equity_ohlcv,
)
from data_fetcher import (
    DataFetchError,
    REQUIRED_OHLCV_COLUMNS,
    apply_default_network_proxy_policy,
    get_fund_data,
)
from feature_engineering import OHLCVFeatureEngineer

apply_default_network_proxy_policy()


SUPPORTED_ASSET_TYPES: tuple[str, ...] = ("etf", "ashare", "mutual_fund", "gold", "crypto")
ASSET_LABELS: dict[str, str] = {
    **CHINA_EQUITY_LABELS,
    "mutual_fund": "公募基金 (Mutual Fund)",
    "gold": "全球贵金属",
    "crypto": "加密货币",
}
DEFAULT_SYMBOLS: dict[str, str] = {
    **DEFAULT_CHINA_EQUITY_SYMBOLS,
    "mutual_fund": "009691",
    "gold": "GC",
    "crypto": "BTC/USDT",
}
LOOKBACK_YEARS = 3
RISK_FREE_RATE = 0.016
TRADING_DAYS = 252
GOLD_SYMBOL_ALIASES: dict[str, str] = {
    "GC": "GC",
    "AU9999": "GC",
}
GOLD_COLUMN_MAPPING: dict[str, str] = {
    "date": "Date",
    "Date": "Date",
    "open": "Open",
    "Open": "Open",
    "high": "High",
    "High": "High",
    "low": "Low",
    "Low": "Low",
    "close": "Close",
    "Close": "Close",
    "volume": "Volume",
    "Volume": "Volume",
}


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
    """通过 AKShare `futures_foreign_hist` 加载贵金属期货行情。"""
    display_symbol, vendor_symbol = _resolve_gold_symbol(symbol)
    ak = _load_optional_dependency("akshare")

    raw_df = ak.futures_foreign_hist(symbol=vendor_symbol)
    normalized = _normalize_ohlcv(raw_df, dict(GOLD_COLUMN_MAPPING))
    filtered = _filter_by_date_range(normalized, start_date, end_date)
    filtered.attrs["asset_type"] = "gold"
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
    """统一加载 ETF、A 股个股、公募基金、黄金、加密货币等资产数据。"""
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
    return _fetch_crypto_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        proxy_url=proxy_url,
    )


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
    engineer = OHLCVFeatureEngineer()
    featured_df = engineer.transform(df)
    featured_df.attrs.update(df.attrs)

    close = pd.to_numeric(featured_df["Close"], errors="coerce").astype("float64")
    if close.isna().any():
        raise ValueError("Close 列存在 NaN，无法计算指标。")

    metrics = {
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
    }
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
    """绘制 K 线 + 成交量双层图；公募基金为累计净值折线单层图（无 K 线、无成交量）。"""
    go, subplots = _load_plotly_modules()
    make_subplots = subplots.make_subplots
    asset_key = str(df.attrs.get("asset_type", "")).strip().lower()
    is_mutual_fund = asset_key == "mutual_fund"

    if is_mutual_fund:
        figure = make_subplots(rows=1, cols=1)
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="累计净值",
                line={"width": 2.0, "color": "#1f77b4"},
            ),
            row=1,
            col=1,
        )
    else:
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

    if not is_mutual_fund:
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
    if is_mutual_fund:
        figure.update_yaxes(title_text="净值", row=1, col=1)
    else:
        figure.update_yaxes(title_text="价格", row=1, col=1)
        figure.update_yaxes(title_text="成交量", row=2, col=1)
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


def render_app() -> None:
    """渲染 Streamlit 量化诊断应用。"""
    streamlit = _require_streamlit()
    streamlit.set_page_config(page_title="量化诊断应用", layout="wide")
    streamlit.title("量化诊断应用")
    streamlit.caption(
        "支持 A 股 ETF / A 股个股 / 公募基金 / 贵金属 / 加密货币 的统一诊断、K 线量价图和核心量化指标展示。"
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
    symbol = streamlit.sidebar.text_input("代码", key="symbol_input").strip()
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

    if streamlit.sidebar.button("开始诊断", use_container_width=True):
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
            streamlit.error(format_china_equity_user_message(exc))
            return

        display_symbol = str(df.attrs.get("symbol", symbol)).strip() or symbol
        streamlit.subheader("%s · %s" % (ASSET_LABELS[asset_type], display_symbol))
        streamlit.caption(
            "样本区间：%s 至 %s，共 %d 条记录。"
            % (
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"),
                len(df),
            )
        )

        metric_col_1, metric_col_2, metric_col_3 = streamlit.columns(3)
        metric_col_1.metric("CAGR", _format_percent_metric(metrics["cagr"]))
        metric_col_2.metric("最大回撤", _format_percent_metric(-abs(metrics["max_drawdown"]), signed=True))
        metric_col_3.metric("Sharpe", _format_ratio_metric(metrics["sharpe_ratio"]))

        streamlit.info(
            "最新信号：%s | 累计收益率：%s"
            % (
                metrics["latest_signal"],
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
    else:
        streamlit.info("在左侧选择资产和时间区间后点击“开始诊断”，即可查看 Web 版量化分析结果。")


if __name__ == "__main__":
    render_app()
