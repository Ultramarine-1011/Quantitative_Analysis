"""境内 ETF / A 股个股：Web 与 CLI 共用的参数校验、抓取路由与用户可见错误文案。"""

from __future__ import annotations

import pandas as pd

from data_fetcher import AKShareFetcher, EmptySymbolDataError

CHINA_EQUITY_ASSET_TYPES: tuple[str, ...] = ("etf", "ashare")
CHINA_EQUITY_LABELS: dict[str, str] = {
    "etf": "A股 ETF",
    "ashare": "A股个股 (A-Share Stock)",
}
DEFAULT_CHINA_EQUITY_SYMBOLS: dict[str, str] = {
    "etf": "510300",
    "ashare": "600519",
}
EMPTY_SYMBOL_FETCH_HINT = (
    "未找到该代码的数据，请检查代码是否输入正确，或该资产类别是否匹配"
    "（例如不要把股票代码输入到ETF选项中）。"
)
GENERIC_DIAGNOSIS_FAILURE_PREFIX = "诊断失败"


def coerce_analysis_date(value: object, field_name: str) -> pd.Timestamp:
    """将外部日期输入转换为不带时区的日级时间戳（与 Web 侧逻辑一致）。"""
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("%s 不是合法日期: %r" % (field_name, value)) from exc

    if pd.isna(ts):
        raise ValueError("%s 不是合法日期: %r" % (field_name, value))
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def validate_analysis_date_range(
    start_date: object,
    end_date: object,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """校验分析区间的起止日期。"""
    start_ts = coerce_analysis_date(start_date, "start_date")
    end_ts = coerce_analysis_date(end_date, "end_date")
    if start_ts > end_ts:
        raise ValueError("start_date 必须早于或等于 end_date。")
    return start_ts, end_ts


def resolve_analysis_date_window(
    start_date: object | None,
    end_date: object | None,
    *,
    lookback_years: int = 3,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """解析 CLI/Web 共用的默认区间：缺省结束日为今天，缺省开始日为向前 ``lookback_years`` 年。"""
    if end_date is None:
        end_ts = pd.Timestamp.today().normalize()
    else:
        end_ts = coerce_analysis_date(end_date, "end_date")
    if start_date is None:
        start_ts = end_ts - pd.DateOffset(years=lookback_years)
    else:
        start_ts = coerce_analysis_date(start_date, "start_date")
    return validate_analysis_date_range(start_ts, end_ts)


def normalize_china_equity_asset_type(value: object) -> str:
    """解析并校验资产类型关键字，返回小写 ``etf`` 或 ``ashare``。"""
    key = str(value).strip().lower()
    if key not in CHINA_EQUITY_ASSET_TYPES:
        raise ValueError(
            "不支持的境内权益类型: %r，可选: %s"
            % (value, ", ".join(CHINA_EQUITY_ASSET_TYPES))
        )
    return key


def validate_china_equity_symbol(asset_type: str, symbol: object) -> str:
    """校验非空代码；文案与 Streamlit 侧一致。"""
    sym = str(symbol).strip()
    if not sym:
        if asset_type == "etf":
            raise ValueError("ETF 代码不能为空。")
        raise ValueError("A 股代码不能为空。")
    return sym


def fetch_route_for_china_equity(asset_type: str) -> str:
    """将 UI 资产类型映射到 ``AKShareFetcher.fetch_china_equity`` 的路由。"""
    if asset_type == "etf":
        return "etf"
    return "stock"


def load_china_equity_ohlcv(
    asset_type: str | object,
    symbol: str | object,
    start_date: object,
    end_date: object,
    *,
    adjust: str | None = "hfq",
    fetcher: AKShareFetcher | None = None,
) -> pd.DataFrame:
    """拉取境内 ETF 或 A 股 OHLCV，并写入 ``attrs``（``asset_type`` / ``symbol``）。"""
    asset_key = normalize_china_equity_asset_type(asset_type)
    sym = validate_china_equity_symbol(asset_key, symbol)
    start_ts, end_ts = validate_analysis_date_range(start_date, end_date)
    route = fetch_route_for_china_equity(asset_key)
    client = fetcher or AKShareFetcher(max_retries=3, retry_delay=1.0)
    df = client.fetch_china_equity(route, sym, start=start_ts, end=end_ts, adjust=adjust)
    df.attrs["asset_type"] = asset_key
    df.attrs["symbol"] = sym
    return df


def format_china_equity_user_message(exc: BaseException) -> str:
    """将异常转换为用户可见文案（与 ``app.render_app`` 中 ETF/个股分支一致）。"""
    if isinstance(exc, EmptySymbolDataError):
        return EMPTY_SYMBOL_FETCH_HINT
    return "%s: %s" % (GENERIC_DIAGNOSIS_FAILURE_PREFIX, exc)
