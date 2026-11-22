"""Public multi-asset data API. Vendor implementation lives in data_sources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from asset_catalog import (
    DOLLAR_INDEX_ALIASES,
    FX_PAIR_ALIASES,
    US_TREASURY_YIELD_10Y_CN_COL,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
)
from data_sources.base import BaseDataFetcher, YFinanceFetcher
from data_sources.china_equity import AKShareChinaPrimarySource, AKShareFetcher
from data_sources.eastmoney import EastMoneyDailyKlineClient, eastmoney_sec_market_id
from data_sources.errors import (
    ChinaEquityPipelineError,
    DataFetchError,
    EmptySymbolDataError,
    NetworkFetchError,
    VendorUpstreamError,
)
from data_sources.forex import (
    fetch_forex_hist_em_robust,
    fetch_sina_kline_daily,
    forex_kline_json_to_dataframe,
)
from data_sources.funds import (
    fund_build_pseudo_ohlcv_raw,
    fund_raw_to_normalized_ohlcv,
    get_fund_data,
)
from data_sources.network import (
    apply_default_network_proxy_policy,
    call_akshare,
    import_akshare,
    resolve_akshare_callable,
)
from data_sources.quality import annotate_quote_quality

# 兼容已有单测与外部调用对私有符号的 monkeypatch（新代码请使用公开符号）。
_fetch_forex_hist_em_robust = fetch_forex_hist_em_robust
_forex_kline_json_to_dataframe = forex_kline_json_to_dataframe
_fund_build_pseudo_ohlcv_raw = fund_build_pseudo_ohlcv_raw
_fund_raw_to_normalized_ohlcv = fund_raw_to_normalized_ohlcv
_call_akshare = call_akshare
_import_akshare = import_akshare
from ohlcv import (
    A_SHARE_COLUMN_MAPPING,
    REQUIRED_OHLCV_COLUMNS,
    DateLike,
    coerce_analysis_date as _coerce_analysis_timestamp,
    filter_by_date_range as _filter_ohlcv_by_user_date_range,
    normalize_ohlcv,
    validate_analysis_date_range,
)

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


_GLOBAL_INDEX_US_SINA_ALIASES: dict[str, str] = {
    "SPX": ".INX",
    "INX": ".INX",
    "GSPC": ".INX",
    "标普500": ".INX",
    "NDX": ".NDX",
    "IXIC": ".IXIC",
    "纳斯达克": ".IXIC",
    "DJI": ".DJI",
    "DJIA": ".DJI",
    "道琼斯": ".DJI",
}
_GLOBAL_INDEX_HK_SINA_ALIASES: dict[str, str] = {
    "HSI": "HSI",
    "恒生指数": "HSI",
    "HSCEI": "HSCEI",
    "国企指数": "HSCEI",
}
_GLOBAL_INDEX_GLOBAL_SINA_ALIASES: dict[str, str] = {
    "FTSE": "英国富时100指数",
    "英国富时100": "英国富时100指数",
    "GDAXI": "德国DAX 30种股价指数",
    "德国DAX30": "德国DAX 30种股价指数",
    "FCHI": "法CAC40指数",
    "法国CAC40": "法CAC40指数",
    "SX5E": "欧洲Stoxx50指数",
    "欧洲斯托克50": "欧洲Stoxx50指数",
    "N225": "日经225指数",
    "日经225": "日经225指数",
}


def _em_kline_frame_to_ohlcv_raw(df: pd.DataFrame) -> pd.DataFrame:
    """为缺少成交量列的东财 K 线表补 ``成交量``，再交给 ``A_SHARE_COLUMN_MAPPING`` 清洗。"""
    work = df.copy()
    if "成交量" not in work.columns:
        # 指数、收益率等序列常没有成交量；补 0 是为了复用统一 OHLCV 契约。
        work["成交量"] = 0.0
    return work


def _sina_ohlcv_with_optional_volume(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize Sina OHLCV frames that may omit ``volume``."""
    work = raw.copy()
    if "volume" not in work.columns and "Volume" not in work.columns:
        work["volume"] = 0.0
    return normalize_ohlcv(work, FOREIGN_FUTURES_COLUMN_MAPPING)


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


def _resolve_global_sina_index_name(ak: Any, display_sym: str, em_name: str) -> str | None:
    upper = display_sym.upper()
    for candidate in (display_sym, upper, em_name):
        if candidate in _GLOBAL_INDEX_GLOBAL_SINA_ALIASES:
            return _GLOBAL_INDEX_GLOBAL_SINA_ALIASES[candidate]
    table_fn = getattr(ak, "index_global_name_table", None)
    if callable(table_fn):
        try:
            tbl = table_fn()
            if isinstance(tbl, pd.DataFrame) and {"指数名称", "代码"}.issubset(tbl.columns):
                for _, row in tbl.iterrows():
                    name_val = str(row["指数名称"]).strip()
                    code_val = str(row["代码"]).strip().upper()
                    if upper == code_val or display_sym == name_val or em_name == name_val:
                        return name_val
        except Exception:  # noqa: BLE001
            pass
    return None


_US_INDEX_SINA_CANDIDATES: tuple[str, ...] = ("index_us_daily_sina", "index_us_stock_sina")


def _try_us_index_sina_candidates(
    ak: Any,
    idx_sym: str,
    warnings_list: list[str],
    *,
    max_retries: int = 1,
) -> tuple[pd.DataFrame | None, str, bool, bool, BaseException | None]:
    """Try compatible US index Sina AKShare APIs until valid OHLCV data is obtained."""
    any_callable = False
    saw_empty = False
    last_error: BaseException | None = None
    for fn_name in _US_INDEX_SINA_CANDIDATES:
        us_fn = getattr(ak, fn_name, None)
        if not callable(us_fn):
            continue
        any_callable = True
        try:
            raw = _call_akshare(
                f"{fn_name}({idx_sym!r})", us_fn, symbol=idx_sym, max_retries=max_retries
            )
            if not isinstance(raw, pd.DataFrame):
                raise TypeError(
                    f"{fn_name}({idx_sym}) returned non-DataFrame payload: {type(raw).__name__}"
                )
            if raw.empty:
                saw_empty = True
                warnings_list.append(f"akshare.{fn_name}({idx_sym}) returned empty data")
                continue
            out = _sina_ohlcv_with_optional_volume(raw)
            return out, f"akshare.{fn_name}", any_callable, saw_empty, None
        except EmptySymbolDataError as exc:
            saw_empty = True
            last_error = exc
            warnings_list.append(f"akshare.{fn_name}({idx_sym}) returned empty data: {exc!r}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            warnings_list.append(f"akshare.{fn_name}({idx_sym}) failed: {exc!r}")
    return None, "", any_callable, saw_empty, last_error


def _load_non_dot_global_index(
    ak: Any, display_sym: str
) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Resolve non-dot global index across Sina AKShare -> EastMoney -> Sina DXY fallback."""
    em_name = _resolve_global_index_em_name(display_sym)
    upper = display_sym.upper()
    warnings_list: list[str] = []

    us_sina_sym = _GLOBAL_INDEX_US_SINA_ALIASES.get(upper) or _GLOBAL_INDEX_US_SINA_ALIASES.get(
        em_name
    )
    if us_sina_sym:
        out, data_source, any_us_callable, _, _ = _try_us_index_sina_candidates(
            ak, us_sina_sym, warnings_list, max_retries=1
        )
        if out is not None:
            return out, us_sina_sym, data_source, warnings_list
        if not any_us_callable:
            warnings_list.append("akshare missing index_us_daily_sina/index_us_stock_sina")

    hk_sina_sym = _GLOBAL_INDEX_HK_SINA_ALIASES.get(upper) or _GLOBAL_INDEX_HK_SINA_ALIASES.get(
        em_name
    )
    if hk_sina_sym:
        hk_fn = getattr(ak, "index_hk_daily_sina", None)
        if callable(hk_fn):
            try:
                raw = _call_akshare(
                    f"index_hk_daily_sina({hk_sina_sym!r})",
                    hk_fn,
                    symbol=hk_sina_sym,
                    max_retries=1,
                )
                out = _sina_ohlcv_with_optional_volume(raw)
                return out, hk_sina_sym, "akshare.index_hk_daily_sina", warnings_list
            except Exception as exc:  # noqa: BLE001
                warnings_list.append(f"akshare.index_hk_daily_sina({hk_sina_sym}) failed: {exc!r}")

    global_sina_name = _resolve_global_sina_index_name(ak, display_sym, em_name)
    if global_sina_name:
        g_fn = getattr(ak, "index_global_hist_sina", None)
        if callable(g_fn):
            try:
                raw = _call_akshare(
                    f"index_global_hist_sina({global_sina_name!r})",
                    g_fn,
                    symbol=global_sina_name,
                    max_retries=1,
                )
                out = _sina_ohlcv_with_optional_volume(raw)
                return out, global_sina_name, "akshare.index_global_hist_sina", warnings_list
            except Exception as exc:  # noqa: BLE001
                warnings_list.append(
                    f"akshare.index_global_hist_sina({global_sina_name}) failed: {exc!r}"
                )

    em_fn = getattr(ak, "index_global_hist_em", None)
    if callable(em_fn):
        try:
            raw = _call_akshare(
                f"index_global_hist_em({em_name!r})", em_fn, symbol=em_name, max_retries=1
            )
            raw2 = _em_kline_frame_to_ohlcv_raw(raw)
            out = normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
            return out, em_name, "akshare.index_global_hist_em", warnings_list
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"akshare.index_global_hist_em({em_name}) failed: {exc!r}")
    else:
        warnings_list.append("akshare missing index_global_hist_em")

    if upper in DOLLAR_INDEX_ALIASES or em_name in DOLLAR_INDEX_ALIASES:
        try:
            raw_dxy = fetch_sina_kline_daily("DINIW", display_symbol=display_sym)
            raw2 = _em_kline_frame_to_ohlcv_raw(raw_dxy)
            out = normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
            resolved = "DXY" if upper in {"DXY", "UDI", "DINIW"} else em_name
            return out, resolved, "sina_forex_direct", warnings_list
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"sina_forex_direct(DINIW) failed: {exc!r}")

    detail = "; ".join(warnings_list) if warnings_list else f"未知全球指数 {display_sym!r}"
    raise DataFetchError(f"全球指数 {display_sym!r} 行情抓取失败: {detail}")


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
    fn_name, fn = resolve_akshare_callable(ak, "futures_foreign_hist_sina", "futures_foreign_hist")
    raw_df = _call_akshare(
        f"{fn_name}({display!r})",
        fn,
        symbol=display,
    )
    normalized = normalize_ohlcv(raw_df, FOREIGN_FUTURES_COLUMN_MAPPING)
    out = _filter_ohlcv_by_user_date_range(normalized, start_date, end_date)
    out.attrs["asset_type"] = str(asset_type)
    out.attrs["symbol"] = display
    out.attrs["series_kind"] = str(series_kind)
    out.attrs["return_basis"] = resolve_return_basis(str(asset_type), attrs=out.attrs)
    out.attrs["data_source"] = f"akshare.{fn_name}"
    if unit_caption:
        out.attrs["unit_caption"] = str(unit_caption)
    return out


def load_global_market_data(
    route: str,
    symbol: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    adjust: str | None = None,
) -> pd.DataFrame:
    """全球权益/指数：``route`` 为 ``us_stock`` | ``hk_stock`` | ``global_index``。

    ``us_stock`` 默认前复权（``'qfq'``，接口仅支持 ``'qfq'`` / ``''``）；
    ``hk_stock`` 默认后复权（``'hfq'``，接口支持 ``'hfq'`` / ``'qfq'`` / ``''``）。
    """
    route_key = str(route).strip().lower()
    sym = str(symbol).strip()
    resolved_symbol = sym
    if not sym:
        raise ValueError("代码不能为空。")
    if route_key not in {"us_stock", "hk_stock", "global_index"}:
        raise ValueError("load_global_market_data: 未知 route %r。" % route)
    validate_analysis_date_range(start_date, end_date)
    selected_adjust = (
        resolve_adjustment(route_key, adjust) if route_key in {"us_stock", "hk_stock"} else ""
    )
    ak = _import_akshare()
    fallback_warnings: list[str] = []

    if route_key == "us_stock":
        us_sym = sym.upper()
        fn_name, fn = resolve_akshare_callable(ak, "stock_us_daily_sina", "stock_us_daily")
        # 美股（新浪 stock_us_daily_sina）仅支持前复权 ('qfq') 与不复权 ('')。
        raw = _call_akshare(
            f"{fn_name}({us_sym!r}, adjust={selected_adjust!r})",
            fn,
            symbol=us_sym,
            adjust=selected_adjust,
        )
        out = normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        resolved_symbol = us_sym
        atype = "us_stock"
        data_source = f"akshare.{fn_name}"
        adj_label = "前复权" if selected_adjust == "qfq" else "不复权"
        caption = f"USD 计价 · 美股日线（{adj_label}）"
    elif route_key == "hk_stock":
        hk_sym = _normalize_hk_stock_symbol(sym)
        fn_name, fn = resolve_akshare_callable(ak, "stock_hk_daily_sina", "stock_hk_daily")
        raw = _call_akshare(
            f"{fn_name}({hk_sym!r}, adjust={selected_adjust!r})",
            fn,
            symbol=hk_sym,
            adjust=selected_adjust,
        )
        out = normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        resolved_symbol = hk_sym
        atype = "hk_stock"
        data_source = f"akshare.{fn_name}"
        adj_label = {"hfq": "后复权", "qfq": "前复权", "": "不复权"}.get(
            selected_adjust, selected_adjust
        )
        caption = f"HKD 计价 · 港股日线（{adj_label}）"
    elif route_key == "global_index":
        display_sym = sym.strip()
        if display_sym.startswith("."):
            idx_sym = display_sym.upper()
            out_us, data_source, any_us_callable, saw_empty, last_error = (
                _try_us_index_sina_candidates(ak, idx_sym, fallback_warnings, max_retries=1)
            )
            if out_us is None:
                if not any_us_callable:
                    tried = ", ".join(_US_INDEX_SINA_CANDIDATES)
                    raise VendorUpstreamError(
                        f"当前 AKShare 版本缺少可用接口（已尝试: {tried}），"
                        "请检查 AKShare 版本或数据路由配置。"
                    )
                detail = "; ".join(fallback_warnings)
                if saw_empty and not any("failed:" in w for w in fallback_warnings):
                    raise EmptySymbolDataError(
                        f"美股指数 ({idx_sym}) 未返回有效 K 线数据: {detail}"
                    )
                raise DataFetchError(f"美股指数 ({idx_sym}) 行情抓取失败: {detail}") from last_error
            out = out_us
            resolved_symbol = idx_sym
            atype = "global_index"
            caption = "美股指数（新浪）· 点"
        else:
            out, resolved_symbol, data_source, fallback_warnings = _load_non_dot_global_index(
                ak, display_sym
            )
            atype = "global_index"
            caption = "全球指数 · 点"
    else:
        raise ValueError("load_global_market_data: 未知 route %r。" % route)

    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = atype
    out.attrs["symbol"] = resolved_symbol
    out.attrs["series_kind"] = "price"
    out.attrs["unit_caption"] = caption
    out.attrs["data_source"] = data_source
    if fallback_warnings:
        out.attrs["fallback_warnings"] = fallback_warnings
    if route_key in {"us_stock", "hk_stock"}:
        out.attrs["adjust"] = selected_adjust
        out.attrs["return_basis"] = (
            "unadjusted_price" if selected_adjust == "" else "adjusted_price"
        )
    else:
        out.attrs["return_basis"] = resolve_return_basis(atype, attrs=out.attrs)
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
        fn_name, fn = resolve_akshare_callable(ak, "bond_zh_hs_daily_sina", "bond_zh_hs_daily")
        raw = _call_akshare(f"{fn_name}({code!r})", fn, symbol=code)
        if "volume" not in raw.columns:
            raw = raw.copy()
            raw["volume"] = 0.0
        out = normalize_ohlcv(raw, FOREIGN_FUTURES_COLUMN_MAPPING)
        out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
        out.attrs["asset_type"] = "bond_cn"
        out.attrs["symbol"] = code
        out.attrs["series_kind"] = "price"
        out.attrs["return_basis"] = "exchange_bond_price"
        out.attrs["unit_caption"] = "人民币计价 · 国债现货/债券 K 线（列含义以数据源为准）"
        out.attrs["data_source"] = f"akshare.{fn_name}"
        return out

    if kind_key == "us_yield":
        start_arg = start_ts.strftime("%Y%m%d")
        fn_name, fn = resolve_akshare_callable(ak, "bond_zh_us_yield_em", "bond_zh_us_rate")
        raw = _call_akshare(fn_name, fn, start_date=start_arg)
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise EmptySymbolDataError(f"{fn_name} returned empty.")

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
        out.attrs["return_basis"] = "yield_level"
        out.attrs["unit_caption"] = (
            "收益率水平 %% · %s（非持有期回报，指标为水平序列统计）" % col_name
        )
        out.attrs["yield_column"] = col_name
        out.attrs["data_source"] = f"akshare.{fn_name}"
        return out

    raise ValueError("load_bond_data: 未知 kind %r。" % kind)


def load_commodity_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """工业商品等外盘期货（与贵金属共用 ``futures_foreign_hist_sina``）。"""
    return load_futures_foreign_ohlcv(
        symbol,
        start_date,
        end_date,
        asset_type="commodity",
        unit_caption="USD 等 · 外盘期货主力连续（合约细则以数据源为准）",
    )


def _parse_tencent_reit_kline(
    raw_tx: Any,
    qq: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, bool, str | None]:
    """Validate Tencent kline frame without ever mapping turnover ``amount`` to ``Volume``.

    Returns ``(valid_real_vol_df, missing_vol_fallback_df, saw_empty, warning)``.
    """
    if not isinstance(raw_tx, pd.DataFrame):
        raise TypeError(
            f"stock_zh_a_hist_tx({qq}) returned non-DataFrame payload: {type(raw_tx).__name__}"
        )
    if raw_tx.empty:
        return None, None, True, f"akshare.stock_zh_a_hist_tx({qq}) returned empty data"

    col_lookup: dict[str, str] = {}
    for target, candidates in (
        ("Date", ("date", "Date", "日期")),
        ("Open", ("open", "Open", "开盘")),
        ("High", ("high", "High", "最高")),
        ("Low", ("low", "Low", "最低")),
        ("Close", ("close", "Close", "收盘")),
    ):
        matched = next((c for c in candidates if c in raw_tx.columns), None)
        if matched is None:
            raise ValueError(f"Tencent kline DataFrame missing required {target} column.")
        col_lookup[target] = matched

    vol_col = next((c for c in ("volume", "Volume", "成交量") if c in raw_tx.columns), None)
    base_frame = pd.DataFrame(
        {target: raw_tx[src] for target, src in col_lookup.items()},
        index=raw_tx.index,
    )

    if vol_col is not None:
        vol_series = pd.to_numeric(raw_tx[vol_col], errors="coerce")
        if vol_series.notna().any():
            base_frame["Volume"] = vol_series
            return normalize_ohlcv(base_frame, {}), None, False, None
        warn = f"akshare.stock_zh_a_hist_tx({qq}) returned no valid 'volume' values"
    else:
        # 腾讯接口的 amount 为成交金额（元），禁止在缺少 volume 时将其映射为 Volume。
        has_amount = "amount" in raw_tx.columns or "成交额" in raw_tx.columns
        detail = (
            "missing 'volume' column (found 'amount', refusing to map turnover amount to Volume)"
            if has_amount
            else "missing 'volume' column"
        )
        warn = f"akshare.stock_zh_a_hist_tx({qq}) {detail}"

    # C-REITs 允许缺失成交量：先规范化请求窗口内的价格序列作为显式缺失兜底，
    # 但仍返回 out=None，以便调用方优先尝试其他具备真实成交量的数据源。
    dates = pd.to_datetime(base_frame["Date"], errors="coerce")
    lo = start_ts.normalize()
    hi = end_ts.normalize()
    in_range = base_frame.loc[
        dates.notna() & (dates.dt.normalize() >= lo) & (dates.dt.normalize() <= hi)
    ].copy()
    if in_range.empty:
        return None, None, True, warn

    in_range["Volume"] = float("nan")
    try:
        missing_vol_norm = normalize_ohlcv(in_range, {})
    except EmptySymbolDataError:
        return None, None, True, warn
    return None, missing_vol_norm, False, warn


def load_reit_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """沪深 C-REITs 日线（腾讯前复权 -> AKShare 东财 -> 东财直连 -> 新浪不复权）。"""
    code = normalize_symbol_alias("creit", symbol)
    if not code:
        raise ValueError("REITs 基金代码不能为空，例如 508097。")
    start_ts, end_ts = validate_analysis_date_range(start_date, end_date)
    ak = _import_akshare()
    fallback_warnings: list[str] = []
    saw_empty = False
    tx_prefix: str | None = None
    if code.isdigit():
        tx_prefix = "sh" if eastmoney_sec_market_id(code) == 1 else "sz"

    out: pd.DataFrame | None = None
    tx_missing_vol_fallback: pd.DataFrame | None = None
    data_source = ""

    # 1) 腾讯日线（前复权 qfq 与东财 fqt=1 口径一致，且不依赖 push2）
    tx_fn = getattr(ak, "stock_zh_a_hist_tx", None)
    if callable(tx_fn) and tx_prefix is not None:
        qq = f"{tx_prefix}{code}"
        try:
            raw_tx = _call_akshare(
                f"stock_zh_a_hist_tx({qq!r})",
                tx_fn,
                symbol=qq,
                start_date=start_ts.strftime("%Y%m%d"),
                end_date=end_ts.strftime("%Y%m%d"),
                adjust="qfq",
                max_retries=1,
            )
            out, tx_missing_vol_fallback, tx_empty, tx_warn = _parse_tencent_reit_kline(
                raw_tx, qq, start_ts, end_ts
            )
            if out is not None:
                data_source = "akshare.stock_zh_a_hist_tx"
            if tx_empty:
                saw_empty = True
            if tx_warn:
                fallback_warnings.append(tx_warn)
        except Exception as exc:  # noqa: BLE001
            fallback_warnings.append(f"akshare.stock_zh_a_hist_tx({qq}) failed: {exc!r}")

    # 2) AKShare 东财 REITs 接口（1.20.46 为 fund_reits_hist_em，兼容旧版 reits_hist_em）
    if out is None:
        any_reit_callable = False
        for reit_name in ("fund_reits_hist_em", "reits_hist_em"):
            reit_fn = getattr(ak, reit_name, None)
            if not callable(reit_fn):
                continue
            any_reit_callable = True
            try:
                raw_em = _call_akshare(
                    f"{reit_name}({code!r})", reit_fn, symbol=code, max_retries=1
                )
                if not isinstance(raw_em, pd.DataFrame):
                    raise TypeError(
                        f"{reit_name}({code}) returned non-DataFrame payload: "
                        f"{type(raw_em).__name__}"
                    )
                if raw_em.empty:
                    saw_empty = True
                    fallback_warnings.append(f"akshare.{reit_name}({code}) returned empty data")
                    continue
                raw2 = _em_kline_frame_to_ohlcv_raw(raw_em)
                out = normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
                data_source = f"akshare.{reit_name}"
                break
            except EmptySymbolDataError as exc:
                saw_empty = True
                fallback_warnings.append(
                    f"akshare.{reit_name}({code}) returned empty data: {exc!r}"
                )
            except Exception as exc:  # noqa: BLE001
                fallback_warnings.append(f"akshare.{reit_name}({code}) failed: {exc!r}")
        if not any_reit_callable:
            fallback_warnings.append("akshare missing fund_reits_hist_em/reits_hist_em")

    # 3) 东方财富直连 K 线兜底
    if out is None and code.isdigit():
        try:
            direct_df = EastMoneyDailyKlineClient().fetch_daily_klines(
                code, start_ts, end_ts, "qfq", trust_env=False
            )
            if not isinstance(direct_df, pd.DataFrame):
                raise TypeError(
                    f"eastmoney_direct({code}) returned non-DataFrame payload: "
                    f"{type(direct_df).__name__}"
                )
            if not direct_df.empty:
                out = normalize_ohlcv(direct_df, A_SHARE_COLUMN_MAPPING)
                data_source = "eastmoney_direct"
            else:
                saw_empty = True
                fallback_warnings.append(f"eastmoney_direct({code}) returned empty data")
        except Exception as exc:  # noqa: BLE001
            fallback_warnings.append(f"eastmoney_direct({code}) failed: {exc!r}")

    # 4) 新浪交易所基金接口兜底（注：不复权）
    if out is None and tx_prefix is not None:
        sina_fn = getattr(ak, "fund_etf_hist_sina", None)
        if callable(sina_fn):
            qq = f"{tx_prefix}{code}"
            try:
                raw_s = _call_akshare(
                    f"fund_etf_hist_sina({qq!r})", sina_fn, symbol=qq, max_retries=1
                )
                if not isinstance(raw_s, pd.DataFrame):
                    raise TypeError(
                        f"fund_etf_hist_sina({qq}) returned non-DataFrame payload: "
                        f"{type(raw_s).__name__}"
                    )
                if not raw_s.empty:
                    out = normalize_ohlcv(raw_s, FOREIGN_FUTURES_COLUMN_MAPPING)
                    data_source = "akshare.fund_etf_hist_sina"
                    fallback_warnings.append(
                        "已降级为新浪交易所行情 akshare.fund_etf_hist_sina（不复权口径，分红除权日可能存在跳空）。"
                    )
                else:
                    saw_empty = True
                    fallback_warnings.append(
                        f"akshare.fund_etf_hist_sina({qq}) returned empty data"
                    )
            except Exception as exc:  # noqa: BLE001
                fallback_warnings.append(f"akshare.fund_etf_hist_sina({qq}) failed: {exc!r}")

    if out is None and tx_missing_vol_fallback is not None:
        out = tx_missing_vol_fallback
        data_source = "akshare.stock_zh_a_hist_tx"
        fallback_warnings.append(
            "未能从其他数据源获取真实成交量，已沿用 akshare.stock_zh_a_hist_tx "
            "价格序列并按 OHLCV 契约显式处理缺失成交量。"
        )

    if out is None:
        detail = "; ".join(fallback_warnings)
        if saw_empty and not any("failed:" in w for w in fallback_warnings):
            raise EmptySymbolDataError(f"C-REITs ({code}) 未返回有效 K 线数据: {detail}")
        raise DataFetchError(f"C-REITs ({code}) 历史行情抓取失败: {detail}")

    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = "creit"
    out.attrs["symbol"] = code
    out.attrs["series_kind"] = "price"
    out.attrs["return_basis"] = resolve_return_basis("creit", attrs=out.attrs)
    out.attrs["unit_caption"] = "CNY 计价 · 基础设施 REITs"
    out.attrs["data_source"] = data_source
    if fallback_warnings:
        out.attrs["fallback_warnings"] = fallback_warnings
    annotate_quote_quality(out, "creit", None)
    return out


def load_fx_data(symbol: str, start_date: DateLike, end_date: DateLike) -> pd.DataFrame:
    """外汇即期日线（东财直连 -> AKShare -> 新浪外汇）。"""
    raw_key = str(symbol).strip().upper().replace(" ", "")
    if not raw_key:
        raise ValueError("外汇代码不能为空，例如 USDCNH。")
    pair = FX_PAIR_ALIASES.get(raw_key, raw_key.replace("/", ""))
    validate_analysis_date_range(start_date, end_date)
    raw = _fetch_forex_hist_em_robust(pair)
    fx_source = raw.attrs.get("data_source", "eastmoney_forex_direct")
    fallback_warnings = list(raw.attrs.get("fallback_warnings") or ())
    raw2 = _em_kline_frame_to_ohlcv_raw(raw)
    out = normalize_ohlcv(raw2, {**A_SHARE_COLUMN_MAPPING, **_EM_KLINE_OHLCV_MAPPING})
    out = _filter_ohlcv_by_user_date_range(out, start_date, end_date)
    out.attrs["asset_type"] = "fx"
    out.attrs["symbol"] = pair
    out.attrs["series_kind"] = "price"
    out.attrs["return_basis"] = "fx_rate"
    out.attrs["unit_caption"] = "汇率 · %s（间接标价含义以数据源为准）" % pair
    out.attrs["data_source"] = fx_source
    if fallback_warnings:
        out.attrs["fallback_warnings"] = fallback_warnings
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

    def _fetch_one_etf(sym: str) -> pd.Series:
        fetcher = AKShareFetcher(max_retries=3, retry_delay=1.0)
        ohlcv = fetcher.fetch_china_equity(
            "etf",
            sym,
            start=start_ts,
            end=end_ts,
            adjust=adjust_key,
        )
        clipped = _filter_ohlcv_by_user_date_range(ohlcv, start_date, end_date)
        return clipped["Close"].rename(sym)

    if len(symbols_ordered) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(symbols_ordered))) as pool:
            futures = [pool.submit(_fetch_one_etf, sym) for sym in symbols_ordered]
            close_frames = [fut.result() for fut in futures]
    else:
        close_frames = [_fetch_one_etf(symbols_ordered[0])]

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
