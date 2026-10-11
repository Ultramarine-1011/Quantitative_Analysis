"""Public multi-asset data API. Vendor implementation lives in data_sources."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ohlcv import (
    A_SHARE_COLUMN_MAPPING,
    DateLike,
    REQUIRED_OHLCV_COLUMNS,
    coerce_analysis_date as _coerce_analysis_timestamp,
    filter_by_date_range as _filter_ohlcv_by_user_date_range,
    normalize_ohlcv,
    validate_analysis_date_range,
)
from data_sources.base import BaseDataFetcher, YFinanceFetcher
from data_sources.china_equity import AKShareChinaPrimarySource, AKShareFetcher
from data_sources.eastmoney import EastMoneyDailyKlineClient
from data_sources.errors import (
    ChinaEquityPipelineError,
    DataFetchError,
    EmptySymbolDataError,
    NetworkFetchError,
    VendorUpstreamError,
)
from data_sources.forex import _fetch_forex_hist_em_robust, _forex_kline_json_to_dataframe
from data_sources.funds import (
    get_fund_data,
    _fund_build_pseudo_ohlcv_raw,
    _fund_raw_to_normalized_ohlcv,
)
from data_sources.network import apply_default_network_proxy_policy, _call_akshare, _import_akshare

FOREIGN_FUTURES_COLUMN_MAPPING: dict[str, str] = {
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

# 东方财富 K 线类接口常见列（指数 / 外汇 / REITs 历史）
_EM_KLINE_OHLCV_MAPPING: dict[str, str] = {
    "日期": "Date",
    "今开": "Open",
    "最高": "High",
    "最低": "Low",
    "最新价": "Close",
    "成交量": "Volume",
}

# 全球指数（东财）中文名称；用户可输入简码或中文
GLOBAL_INDEX_EM_ALIASES: dict[str, str] = {
    "HSI": "恒生指数",
    "HSCEI": "国企指数",
    "SPX": "标普500",
    "INX": "标普500",
    "GSPC": "标普500",
    "NDX": "纳斯达克",
    "IXIC": "纳斯达克",
    "DJI": "道琼斯",
    "DJIA": "道琼斯",
    "UDI": "美元指数",
    "DXY": "美元指数",
    "N225": "日经225",
    "FTSE": "英国富时100",
    "GDAXI": "德国DAX30",
    "FCHI": "法国CAC40",
    "SX5E": "欧洲斯托克50",
}

# 美债收益率表默认列（bond_zh_us_rate）
US_TREASURY_YIELD_10Y_CN_COL = "美国国债收益率10年"

FX_PAIR_ALIASES: dict[str, str] = {
    "USD/CNY": "USDCNH",
    "USD/CNH": "USDCNH",
    "USDCNY": "USDCNH",
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
}


def _em_kline_frame_to_ohlcv_raw(df: pd.DataFrame) -> pd.DataFrame:
    """为缺少成交量列的东财 K 线表补 ``成交量``，再交给 ``A_SHARE_COLUMN_MAPPING`` 清洗。"""
    work = df.copy()
    if "成交量" not in work.columns:
        # 指数、收益率等序列常没有成交量；补 0 是为了复用统一 OHLCV 契约。
        work["成交量"] = 0.0
    return work


def _normalize_hk_stock_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if not text:
        raise ValueError("港股代码不能为空。")
    if text.isdigit():
        return text.zfill(5)
    return text


def _resolve_global_index_em_name(symbol: str) -> str:
    raw = str(symbol).strip()
    if not raw:
        raise ValueError("指数代码或名称不能为空。")
    if raw in GLOBAL_INDEX_EM_ALIASES:
        return GLOBAL_INDEX_EM_ALIASES[raw]
    upper = raw.upper()
    if upper in GLOBAL_INDEX_EM_ALIASES:
        return GLOBAL_INDEX_EM_ALIASES[upper]
    return raw


def load_futures_foreign_ohlcv(
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    asset_type: str,
    unit_caption: str = "",
    series_kind: str = "price",
) -> pd.DataFrame:
    """外盘期货日线（新浪全球期货），输出统一 OHLCV。"""
    display = str(symbol).strip().upper()
    if not display:
        raise ValueError("期货品种代码不能为空。")
    validate_analysis_date_range(start_date, end_date)
    ak = _import_akshare()
    raw_df = _call_akshare(
        "futures_foreign_hist(%r)" % display,
        ak.futures_foreign_hist,
        symbol=display,
    )
    normalized = normalize_ohlcv(raw_df, FOREIGN_FUTURES_COLUMN_MAPPING)
    out = _filter_ohlcv_by_user_date_range(normalized, start_date, end_date)
    out.attrs["asset_type"] = str(asset_type)
    out.attrs["symbol"] = display
    out.attrs["series_kind"] = str(series_kind)
    if unit_caption:
        out.attrs["unit_caption"] = str(unit_caption)
    return out


def load_global_market_data(
    route: str,
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
) -> pd.DataFrame:
    """全球权益/指数：``route`` 为 ``us_stock`` | ``hk_stock`` | ``global_index``。"""
    route_key = str(route).strip().lower()
    sym = str(symbol).strip()
    resolved_symbol = sym
    if not sym:
        raise ValueError("代码不能为空。")
    if route_key not in {"us_stock", "hk_stock", "global_index"}:
        raise ValueError("load_global_market_data: 未知 route %r。" % route)
    validate_analysis_date_range(start_date, end_date)
    ak = _import_akshare()

    if route_key == "us_stock":
        # 美股和港股接口返回英文 OHLCV 列，可直接使用外盘映射清洗。
        raw = _call_akshare(
            "stock_us_daily(%r)" % sym,
            ak.stock_us_daily,
            symbol=sym,
            adjust="",
        )
        out = normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        atype = "us_stock"
        caption = "USD 计价 · 美股日线"
    elif route_key == "hk_stock":
        hk_sym = _normalize_hk_stock_symbol(sym)
        raw = _call_akshare(
            "stock_hk_daily(%r)" % hk_sym,
            ak.stock_hk_daily,
            symbol=hk_sym,
            adjust="",
        )
        out = normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        resolved_symbol = hk_sym
        atype = "hk_stock"
        caption = "HKD 计价 · 港股日线"
    elif route_key == "global_index":
        display_sym = sym.strip()
        if display_sym.startswith("."):
            # 以点开头的美股指数代码走新浪接口，例如 .INX。
            raw = _call_akshare(
                "index_us_stock_sina(%r)" % display_sym,
                ak.index_us_stock_sina,
                symbol=display_sym,
            )
            out = normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
            atype = "global_index"
            caption = "美股指数（新浪）· 点"
        else:
            # 常见简码先映射到东财接口需要的中文/英文指数名称。
            em_name = _resolve_global_index_em_name(display_sym)
            raw = _call_akshare(
                "index_global_hist_em(%r)" % em_name,
                ak.index_global_hist_em,
                symbol=em_name,
            )
            raw2 = _em_kline_frame_to_ohlcv_raw(raw)
            out = normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
            resolved_symbol = em_name
            atype = "global_index"
            caption = "全球指数（东财）· 点"
    else:
        raise ValueError("load_global_market_data: 未知 route %r。" % route)

    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = atype
    out.attrs["symbol"] = resolved_symbol
    out.attrs["series_kind"] = "price"
    out.attrs["unit_caption"] = caption
    return out


def load_bond_data(
    kind: str,
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    us_yield_column: str | None = None,
) -> pd.DataFrame:
    """债券：``cn_hist`` 中国国债 K 线；``us_yield`` 美债收益率时间序列（伪 OHLCV）。"""
    kind_key = str(kind).strip().lower()
    if kind_key not in {"cn_hist", "us_yield"}:
        raise ValueError("load_bond_data: 未知 kind %r。" % kind)
    start_ts, end_ts = validate_analysis_date_range(start_date, end_date)
    ak = _import_akshare()

    if kind_key == "cn_hist":
        code = str(symbol).strip()
        if not code:
            raise ValueError("国债代码不能为空，例如 sh010107。")
        raw = _call_akshare("bond_zh_hs_daily(%r)" % code, ak.bond_zh_hs_daily, symbol=code)
        if "volume" not in raw.columns:
            raw = raw.copy()
            raw["volume"] = 0.0
        out = normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
        out.attrs["asset_type"] = "bond_cn"
        out.attrs["symbol"] = code
        out.attrs["series_kind"] = "price"
        out.attrs["unit_caption"] = "人民币计价 · 国债现货/债券 K 线（列含义以数据源为准）"
        return out

    if kind_key == "us_yield":
        start_arg = start_ts.strftime("%Y%m%d")
        raw = _call_akshare("bond_zh_us_rate", ak.bond_zh_us_rate, start_date=start_arg)
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise EmptySymbolDataError("bond_zh_us_rate returned empty.")

        date_col = raw.columns[0]
        col_name = us_yield_column or US_TREASURY_YIELD_10Y_CN_COL
        if col_name not in raw.columns:
            candidates = [c for c in raw.columns if c != date_col]
            raise ValueError(
                "未找到收益率列 %r，可选列示例: %s"
                % (col_name, ", ".join(str(c) for c in candidates[:8]))
            )

        work = pd.DataFrame(
            {
                "日期": pd.to_datetime(raw[date_col], errors="coerce"),
                "收盘": pd.to_numeric(raw[col_name], errors="coerce"),
            }
        ).dropna(subset=["日期", "收盘"])
        if work.empty:
            raise EmptySymbolDataError("美债收益率解析后无有效行。")
        # 美债收益率是“水平序列”，不是可交易价格；伪 OHLCV 只是为了复用图表和指标管线。
        pseudo = _fund_build_pseudo_ohlcv_raw(work.set_index("日期")["收盘"])
        out = normalize_ohlcv(pseudo, A_SHARE_COLUMN_MAPPING)
        out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
        out.attrs["asset_type"] = "bond_us_yield"
        out.attrs["symbol"] = col_name
        out.attrs["series_kind"] = "yield_level"
        out.attrs["unit_caption"] = (
            "收益率水平 %% · %s（非持有期回报，指标为水平序列统计）" % col_name
        )
        out.attrs["yield_column"] = col_name
        return out

    raise ValueError("load_bond_data: 未知 kind %r。" % kind)


def load_commodity_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """工业商品等外盘期货（与贵金属共用 ``futures_foreign_hist``）。"""
    return load_futures_foreign_ohlcv(
        symbol,
        start_date,
        end_date,
        asset_type="commodity",
        unit_caption="USD 等 · 外盘期货主力连续（合约细则以数据源为准）",
    )


def load_reit_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """沪深 C-REITs 日线（东财 K 线）。"""
    code = str(symbol).strip()
    if not code:
        raise ValueError("REITs 基金代码不能为空，例如 508097。")
    validate_analysis_date_range(start_date, end_date)
    ak = _import_akshare()
    raw = _call_akshare("reits_hist_em(%r)" % code, ak.reits_hist_em, symbol=code)
    raw2 = _em_kline_frame_to_ohlcv_raw(raw)
    out = normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = "creit"
    out.attrs["symbol"] = code
    out.attrs["series_kind"] = "price"
    out.attrs["unit_caption"] = "CNY 计价 · 基础设施 REITs"
    return out


def load_fx_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """外汇即期日线（东财 ``forex_hist_em``）。"""
    raw_key = str(symbol).strip().upper().replace(" ", "")
    if not raw_key:
        raise ValueError("外汇代码不能为空，例如 USDCNH。")
    pair = FX_PAIR_ALIASES.get(raw_key, raw_key.replace("/", ""))
    validate_analysis_date_range(start_date, end_date)
    raw = _fetch_forex_hist_em_robust(pair)
    raw2 = _em_kline_frame_to_ohlcv_raw(raw)
    out = normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = "fx"
    out.attrs["symbol"] = pair
    out.attrs["series_kind"] = "price"
    out.attrs["unit_caption"] = "汇率 · %s（间接标价含义以数据源为准）" % pair
    return out


def get_multiple_assets_close(
    symbols_list: list[str] | tuple[str, ...],
    start_date: DateLike,
    end_date: DateLike,
    *,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """拉取多只 A 股 ETF 的收盘价宽表（按日期 outer 对齐后前向填充）。

    假设 ``symbols_list`` 中均为沪深 ETF 代码。通过本模块内 ``AKShareFetcher`` 逐只拉取并裁剪区间，
    逻辑上等价于单标的 ETF 拉取的组合；**不**依赖 ``asset_resolver``，以避免循环导入。

    Parameters
    ----------
    symbols_list
        ETF 代码列表；元素会先 ``strip``、去重（保序），且不得包含去空白后的空串。
    start_date, end_date
        分析区间；与 ``_filter_ohlcv_by_user_date_range`` 使用相同的日级时间戳与空区间语义。
    adjust
        复权方式，``"qfq"`` 或 ``"hfq"``，默认后复权。

    Returns
    -------
    pd.DataFrame
        索引为 ``DatetimeIndex``（升序），每列名为对应 ETF 代码，值为 ``Close``（``float64``）。

    Raises
    ------
    ValueError
        代码列表非法、复权参数非法、日期区间非法，或合并后某代码列全为缺失（与用户区间无重叠或无有效数据）。
    """
    if not isinstance(symbols_list, (list, tuple)):
        raise TypeError("symbols_list must be a list or tuple of ETF codes.")

    stripped = [str(s).strip() for s in symbols_list]
    if any(not s for s in stripped):
        raise ValueError("symbols_list must not contain empty codes after stripping whitespace.")

    symbols_ordered = list(dict.fromkeys(stripped))

    if not symbols_ordered:
        raise ValueError("symbols_list must be a non-empty list of ETF codes after normalization.")

    adjust_key = str(adjust).strip().lower()
    if adjust_key not in {"qfq", "hfq"}:
        raise ValueError("adjust must be 'qfq' or 'hfq', got %r." % (adjust,))

    start_ts = _coerce_analysis_timestamp(start_date, "start_date")
    end_ts = _coerce_analysis_timestamp(end_date, "end_date")
    if start_ts > end_ts:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    fetcher = AKShareFetcher(max_retries=3, retry_delay=1.0)
    close_frames: list[pd.Series] = []
    for sym in symbols_ordered:
        ohlcv = fetcher.fetch_china_equity(
            "etf",
            sym,
            start=start_ts,
            end=end_ts,
            adjust=adjust_key,
        )
        clipped = _filter_ohlcv_by_user_date_range(ohlcv, start_date, end_date)
        close_frames.append(clipped["Close"].rename(sym))

    wide = pd.concat(close_frames, axis=1, join="outer")
    wide = wide.sort_index().ffill()

    for col in wide.columns:
        if wide[col].isna().all():
            raise ValueError(
                "标的 %r 在指定区间内无有效收盘价，或与所选标的无可用重叠数据。" % str(col)
            )

    wide.index.name = "Date"
    return wide.astype("float64")


__all__ = [
    "DateLike",
    "REQUIRED_OHLCV_COLUMNS",
    "A_SHARE_COLUMN_MAPPING",
    "DataFetchError",
    "EmptySymbolDataError",
    "NetworkFetchError",
    "VendorUpstreamError",
    "ChinaEquityPipelineError",
    "apply_default_network_proxy_policy",
    "BaseDataFetcher",
    "AKShareFetcher",
    "AKShareChinaPrimarySource",
    "EastMoneyDailyKlineClient",
    "YFinanceFetcher",
    "get_fund_data",
    "FOREIGN_FUTURES_COLUMN_MAPPING",
    "GLOBAL_INDEX_EM_ALIASES",
    "US_TREASURY_YIELD_10Y_CN_COL",
    "FX_PAIR_ALIASES",
    "load_futures_foreign_ohlcv",
    "load_global_market_data",
    "load_bond_data",
    "load_commodity_data",
    "load_reit_data",
    "load_fx_data",
    "get_multiple_assets_close",
]
