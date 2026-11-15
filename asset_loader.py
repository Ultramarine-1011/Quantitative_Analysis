"""UI-independent asset routing, crypto loading and multi-asset alignment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

from asset_catalog import (
    ASSET_RETURN_BASIS,
    ASSET_RETURN_BASIS_NOTES,
    GOLD_SYMBOL_ALIASES,
    SUPPORTED_ADJUSTMENTS,
    SUPPORTED_ASSET_TYPES,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
)
from asset_resolver import load_china_equity_ohlcv
from currency import convert_prices_to_base_currency, infer_asset_currency, normalize_currency_code
from data_fetcher import (
    US_TREASURY_YIELD_10Y_CN_COL,
    get_fund_data,
    get_multiple_assets_close,
    load_bond_data,
    load_commodity_data,
    load_futures_foreign_ohlcv,
    load_fx_data,
    load_global_market_data,
    load_reit_data,
)
from data_sources.cache import load_cached_ohlcv, save_cached_ohlcv
from data_sources.crypto import (
    build_ccxt_proxy_config,
    fetch_crypto_data,
    fetch_crypto_ohlcv_rows,
    normalize_crypto_data,
    normalize_proxy_url,
    redact_proxy_url,
)
from ohlcv import (
    validate_analysis_date_range as _validate_date_range,
)

# 兼容已有单测与调用方直接访问原私有函数名。
_build_ccxt_proxy_config = build_ccxt_proxy_config
_fetch_crypto_data = fetch_crypto_data
_fetch_crypto_ohlcv_rows = fetch_crypto_ohlcv_rows
_normalize_crypto_data = normalize_crypto_data
_normalize_proxy_url = normalize_proxy_url
_redact_proxy_url = redact_proxy_url


def _resolve_gold_symbol(symbol: str) -> tuple[str, str]:
    """将展示代码映射为底层抓取所需的贵金属代码。"""
    display_symbol = str(symbol).strip().upper()
    if not display_symbol:
        raise ValueError("黄金代码不能为空。")
    vendor_symbol = GOLD_SYMBOL_ALIASES.get(display_symbol, display_symbol)
    return display_symbol, vendor_symbol


def _fetch_etf_data(
    symbol: str,
    start_date: object,
    end_date: object,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """通过 AKShareFetcher 路由抓取 ETF 并返回统一 OHLCV。"""
    return load_china_equity_ohlcv("etf", symbol, start_date, end_date, adjust=adjust)


def _fetch_ashare_stock_data(
    symbol: str,
    start_date: object,
    end_date: object,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """通过 AKShare ``stock_zh_a_hist`` 抓取 A 股个股并返回统一 OHLCV（Open/High/Low/Close/Volume）。"""
    return load_china_equity_ohlcv("ashare", symbol, start_date, end_date, adjust=adjust)


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


def load_asset_data(
    asset_type: str,
    symbol: str,
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
    *,
    adjust: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """统一加载多类资产 OHLCV（契约一致），支持共享持久化缓存与 ``adjust`` 复权口径。"""
    start_date, end_date = _validate_date_range(start_date, end_date)
    asset_key = str(asset_type).strip().lower()
    if asset_key not in SUPPORTED_ASSET_TYPES:
        raise ValueError("不支持的资产类型: %s" % asset_type)
    if adjust is not None:
        if asset_key in SUPPORTED_ADJUSTMENTS:
            resolve_adjustment(asset_key, adjust)
        elif adjust != "":
            resolve_adjustment(asset_key, adjust)
    if asset_key == "crypto" and proxy_url is not None:
        _normalize_proxy_url(proxy_url)

    cached = load_cached_ohlcv(
        asset_key,
        symbol,
        start_date,
        end_date,
        adjust=adjust,
        force_refresh=force_refresh,
    )
    if cached is not None:
        resolved_sym = str(cached.attrs.get("symbol", symbol) or symbol).strip()
        cached.attrs.setdefault("asset_type", asset_key)
        if asset_key in SUPPORTED_ADJUSTMENTS and "adjust" not in cached.attrs:
            cached.attrs["adjust"] = resolve_adjustment(asset_key, adjust)
        cached.attrs["return_basis"] = resolve_return_basis(
            asset_key, adjust=adjust, attrs=cached.attrs
        )
        cached.attrs.setdefault(
            "canonical_symbol", normalize_symbol_alias(asset_key, resolved_sym or symbol)
        )
        cached.attrs.setdefault(
            "currency", infer_asset_currency(asset_key, resolved_sym, cached.attrs)
        )
        return cached

    # 这里是 Web/CLI 的数据总路由：不同资产来源不同，但返回值都遵循统一 OHLCV 契约。
    if asset_key == "etf":
        china_sym = normalize_symbol_alias(asset_key, symbol)
        df = (
            _fetch_etf_data(symbol=china_sym, start_date=start_date, end_date=end_date)
            if adjust is None
            else _fetch_etf_data(
                symbol=china_sym, start_date=start_date, end_date=end_date, adjust=adjust
            )
        )
    elif asset_key == "ashare":
        china_sym = normalize_symbol_alias(asset_key, symbol)
        df = (
            _fetch_ashare_stock_data(symbol=china_sym, start_date=start_date, end_date=end_date)
            if adjust is None
            else _fetch_ashare_stock_data(
                symbol=china_sym, start_date=start_date, end_date=end_date, adjust=adjust
            )
        )
    elif asset_key == "mutual_fund":
        df = _fetch_mutual_fund_data(symbol=symbol, start_date=start_date, end_date=end_date)
    elif asset_key == "gold":
        df = _fetch_gold_data(symbol=symbol, start_date=start_date, end_date=end_date)
    elif asset_key == "crypto":
        df = _fetch_crypto_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            proxy_url=proxy_url,
        )
    elif asset_key == "us_stock":
        df = (
            load_global_market_data("us_stock", symbol, start_date, end_date)
            if adjust is None
            else load_global_market_data("us_stock", symbol, start_date, end_date, adjust=adjust)
        )
    elif asset_key == "hk_stock":
        df = (
            load_global_market_data("hk_stock", symbol, start_date, end_date)
            if adjust is None
            else load_global_market_data("hk_stock", symbol, start_date, end_date, adjust=adjust)
        )
    elif asset_key == "global_index":
        df = load_global_market_data("global_index", symbol, start_date, end_date)
    elif asset_key == "bond_cn":
        df = load_bond_data("cn_hist", symbol, start_date, end_date)
    elif asset_key == "bond_us_yield":
        sym = str(symbol).strip()
        col: str | None = None
        if sym and sym.upper() not in ("10Y", "默认", "DEFAULT"):
            col = sym
        else:
            col = US_TREASURY_YIELD_10Y_CN_COL
        # 美债收益率用列名选择期限；默认 10Y，也允许用户填完整列名。
        df = load_bond_data(
            "us_yield",
            "",
            start_date,
            end_date,
            us_yield_column=col,
        )
    elif asset_key == "commodity":
        df = load_commodity_data(symbol, start_date, end_date)
    elif asset_key == "creit":
        df = load_reit_data(symbol, start_date, end_date)
    else:
        df = load_fx_data(symbol, start_date, end_date)

    resolved_sym = str(df.attrs.get("symbol", symbol) or symbol).strip()
    df.attrs.setdefault("asset_type", asset_key)
    if asset_key in SUPPORTED_ADJUSTMENTS and "adjust" not in df.attrs:
        df.attrs["adjust"] = resolve_adjustment(asset_key, adjust)
    df.attrs["return_basis"] = resolve_return_basis(asset_key, adjust=adjust, attrs=df.attrs)
    df.attrs.setdefault(
        "canonical_symbol", normalize_symbol_alias(asset_key, resolved_sym or symbol)
    )
    df.attrs.setdefault("currency", infer_asset_currency(asset_key, resolved_sym, df.attrs))
    df.attrs.setdefault("fetched_at", datetime.now(UTC).isoformat())
    df.attrs.setdefault("cache_hit", False)
    if not df.empty:
        saved_path = save_cached_ohlcv(
            df,
            asset_key,
            symbol,
            start_date,
            end_date,
            adjust=adjust,
        )
        if saved_path is not None:
            df.attrs["cache_path"] = str(saved_path)
    return df


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


def multi_asset_leg_column_name(asset_type_key: str, symbol: str) -> str:
    """有效前沿宽表列名：`{资产类型}:{代码}`，避免跨类型代码冲突。"""
    return "%s:%s" % (str(asset_type_key).strip().lower(), str(symbol).strip())


def _split_asset_leg_segment(segment: str) -> tuple[str, str]:
    """解析单行 leg：`类型,代码` 或 `类型|代码`（仅拆第一处分隔符）。"""
    s = str(segment).strip()
    if not s:
        raise ValueError("空的标的行。")
    if "," in s:
        left, right = s.split(",", 1)
    elif "|" in s:
        left, right = s.split("|", 1)
    else:
        raise ValueError(
            "每行需为 `资产类型,代码` 或 `资产类型|代码`，例如 `etf,510300`；无效行：%r" % segment
        )
    asset_raw, symbol_raw = left.strip(), right.strip()
    if not asset_raw:
        raise ValueError("资产类型不能为空：%r" % segment)
    if not symbol_raw:
        raise ValueError("代码不能为空：%r" % segment)
    return asset_raw, symbol_raw


def parse_multi_asset_legs(text: str) -> tuple[tuple[str, str], ...]:
    """从多行文本解析 Tab3 有效前沿标的列表；校验类型、去重（保序）、至少 2 条。

    支持：换行分段；同一行内可用英文分号 `;` 分隔多条。空行忽略。
    返回的元组中资产类型已规范为小写键（``SUPPORTED_ASSET_TYPES`` 成员）。
    """
    segments: list[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if part:
                segments.append(part)

    legs_ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for seg in segments:
        asset_raw, symbol_raw = _split_asset_leg_segment(seg)
        asset_key = asset_raw.strip().lower()
        if asset_key not in SUPPORTED_ASSET_TYPES:
            raise ValueError(
                "不支持的资产类型 %r；可选：%s" % (asset_raw, ", ".join(SUPPORTED_ASSET_TYPES))
            )
        key = (asset_key, symbol_raw)
        if key in seen:
            continue
        seen.add(key)
        legs_ordered.append((asset_key, symbol_raw))

    if len(legs_ordered) < 2:
        raise ValueError("有效前沿至少需要 2 个不重复标的（`类型,代码` 每行一条，或用 `;` 分隔）。")

    return tuple(legs_ordered)


def _fetch_legs_frames(
    legs_tuple: tuple[tuple[str, str], ...],
    start_date: object,
    end_date: object,
    *,
    proxy_url: str | None = None,
    loader: Callable[..., pd.DataFrame] | None = None,
    concurrent: bool = True,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load multiple heterogeneous legs, concurrently for library calls and sequentially for custom loaders."""
    ordered_cols: list[tuple[str, str, str]] = []
    seen_cols: set[str] = set()
    for asset_key, symbol in legs_tuple:
        col = multi_asset_leg_column_name(asset_key, symbol)
        if col in seen_cols:
            raise ValueError(f"多标的组合不能包含重复标的或列名冲突: {col}")
        seen_cols.add(col)
        ordered_cols.append((col, asset_key, symbol))

    load_leg = loader if loader is not None else load_asset_data

    def _one(asset_key: str, symbol: str) -> pd.DataFrame:
        kwargs: dict[str, Any] = {
            "asset_type": asset_key,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "proxy_url": proxy_url,
        }
        if force_refresh:
            kwargs["force_refresh"] = True
        return load_leg(**kwargs)

    # Custom loaders (e.g. Streamlit @st.cache_data wrappers) may require main-thread ScriptRunContext.
    if concurrent and loader is None and len(ordered_cols) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(ordered_cols))) as pool:
            futs = [pool.submit(_one, a, s) for _, a, s in ordered_cols]
            return {col: fut.result() for (col, _, _), fut in zip(ordered_cols, futs)}
    return {col: _one(a, s) for col, a, s in ordered_cols}


def load_multi_leg_close_wide(
    legs_tuple: tuple[tuple[str, str], ...],
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
    *,
    loader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """多类型标的收盘价宽表：逐 leg ``load_asset_data``，按日期 outer 合并后前向填充。"""
    if not legs_tuple:
        raise ValueError("legs_tuple 不能为空。")

    leg_frames = _fetch_legs_frames(
        legs_tuple,
        start_date,
        end_date,
        proxy_url=proxy_url,
        loader=loader,
    )
    close_frames: list[pd.Series] = []
    for col, df_leg in leg_frames.items():
        if df_leg.empty or "Close" not in df_leg.columns:
            raise ValueError("标的 %r 在指定区间内无可用数据或缺少 Close 列。" % col)
        close = pd.to_numeric(df_leg["Close"], errors="coerce")
        if close.isna().all():
            raise ValueError("标的 %r 在指定区间内 Close 全为缺失。" % col)
        close_frames.append(close.rename(col))

    if len({frame.name for frame in close_frames}) != len(close_frames):
        raise ValueError("多标的宽表不能包含重复标的列。")
    wide = pd.concat(close_frames, axis=1, join="outer")
    # 不同市场交易日不同，先 outer 对齐再前向填充，以保留尽可能多的联合样本。
    wide = wide.sort_index().ffill()

    for col in wide.columns:
        if wide[col].isna().all():
            raise ValueError(
                "标的列 %r 在指定区间内无有效收盘价，或与所选标的无可用重叠数据。" % str(col)
            )

    wide.index.name = "Date"
    return wide.astype("float64")


CalendarMode = Literal["common_trading_dates", "weekly_friday", "mtm_outer"]


@dataclass
class MultiAssetAlignmentResult:
    """Unified valuation calendar and base-currency aligned multi-asset bundle."""

    base_prices: pd.DataFrame
    local_prices: pd.DataFrame
    fx_rates: pd.DataFrame
    returns_base: pd.DataFrame
    returns_local: pd.DataFrame
    correlation_returns_base: pd.DataFrame
    correlation_returns_local: pd.DataFrame
    tradable_mask: pd.DataFrame
    base_date: pd.Timestamp
    calendar_mode: CalendarMode
    base_currency: str
    asset_currencies: dict[str, str]
    effective_periods_per_year: float
    backtest_periods_per_year: float
    overlap_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _clean_close_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"标的 {col_name!r} 无可用数据或缺少 Close 列。")
    s = pd.to_numeric(df["Close"], errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="raise"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.as_unit("ns")
    s = pd.Series(s.to_numpy(dtype=float), index=idx, name=col_name)
    s = s.sort_index(kind="stable")
    s = s.loc[~s.index.duplicated(keep="last")].dropna()
    if s.empty:
        raise ValueError(f"标的 {col_name!r} 收盘价全为缺失。")
    if (s <= 0.0).any():
        raise ValueError(f"标的 {col_name!r} 包含非正收盘价，无法计算对数/几何组合收益。")
    return s


def _compute_periods_per_year(
    index: pd.DatetimeIndex, n_returns: int, default: float = 252.0
) -> float:
    if len(index) < 2 or n_returns < 1:
        return default
    span_days = float((index[-1] - index[0]).days)
    if span_days <= 0.0:
        return default
    return float(n_returns) / (span_days / 365.25)


def align_multi_asset_frames(
    asset_frames: Mapping[str, pd.DataFrame],
    *,
    calendar_mode: CalendarMode = "common_trading_dates",
    base_currency: str = "CNY",
    asset_currencies: Mapping[str, str] | None = None,
    fx_loader: Callable[..., pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[..., pd.DataFrame] | None = None,
    proxy_url: str | None = None,
    allow_usdt_usd_peg: bool = False,
    max_fx_staleness_days: int = 4,
) -> MultiAssetAlignmentResult:
    """Align multi-asset OHLCV frames onto a unified valuation timeline and base currency.

    - ``common_trading_dates`` (default): aligns price levels on the exact date intersection
      where all assets have valid observed quotes, then computes multi-period compound returns
      between consecutive valuation dates (capturing e.g. crypto weekend moves without inserting
      fake 0.0 returns on equity closures).
    - ``weekly_friday``: resamples each asset to weekly Friday closing prices and intersects weeks.
    - ``mtm_outer``: outer mark-to-market price grid from the latest asset start date onward;
      ``tradable_mask`` tracks real trading dates so backtests never rebalance on closed days,
      while ``correlation_returns_*`` still uses concurrent observed intervals to avoid zero-return bias.
    """
    if len(asset_frames) < 2:
        raise ValueError("多资产对齐至少需要 2 个标的。")
    if calendar_mode not in ("common_trading_dates", "weekly_friday", "mtm_outer"):
        raise ValueError(f"不支持的日历对齐模式: {calendar_mode!r}")

    series_map: dict[str, pd.Series] = {}
    inferred_ccy: dict[str, str] = {}
    warnings: list[str] = []
    seen_basis_notes: set[str] = set()
    raw_counts: dict[str, int] = {}
    raw_ranges: dict[str, tuple[str, str]] = {}
    data_sources: dict[str, str] = {}
    adjustments: dict[str, str] = {}

    for col, frame in asset_frames.items():
        attrs = getattr(frame, "attrs", {}) or {}
        if ":" in col:
            parsed_type, parsed_sym = col.split(":", 1)
        else:
            parsed_type, parsed_sym = (
                str(attrs.get("asset_type", "")),
                str(attrs.get("symbol", col)),
            )
        atype = str(attrs.get("asset_type") or parsed_type).strip().lower()
        sym = str(attrs.get("symbol") or parsed_sym).strip()
        series_kind = str(attrs.get("series_kind", "price") or "price").strip().lower()
        basis = str(attrs.get("return_basis") or ASSET_RETURN_BASIS.get(atype, "")).strip()

        if series_kind == "yield_level" or basis == "yield_level" or atype == "bond_us_yield":
            raise ValueError(
                f"标的 {col!r} 为收益率水平序列（yield_level），不是资产价格或持有期回报，不能直接纳入组合优化。"
            )
        if (
            series_kind == "fx_rate" or basis == "fx_rate" or atype == "fx"
        ) and normalize_currency_code(base_currency) != "LOCAL":
            target_ccy = normalize_currency_code(base_currency)
            raise ValueError(
                f"标的 {col!r} 为汇率比值序列（fx_rate），不支持按单币种资产折算至 {target_ccy}；"
                "请在基准货币中选择 LOCAL 或移出组合。"
            )
        if (
            basis in ("futures_continuous", "price_index", "fx_rate")
            and basis not in seen_basis_notes
        ):
            seen_basis_notes.add(basis)
            note = ASSET_RETURN_BASIS_NOTES.get(basis)
            if note:
                warnings.append(f"[{atype}] {note}")

        clean_s = _clean_close_series(frame, col)
        series_map[col] = clean_s
        raw_counts[col] = int(len(clean_s))
        raw_ranges[col] = (
            clean_s.index[0].strftime("%Y-%m-%d"),
            clean_s.index[-1].strftime("%Y-%m-%d"),
        )
        if attrs.get("data_source"):
            data_sources[col] = str(attrs["data_source"])
        if "adjust" in attrs:
            adjustments[col] = str(attrs["adjust"])

        if asset_currencies and col in asset_currencies:
            inferred_ccy[col] = str(asset_currencies[col]).strip().upper()
        else:
            inferred_ccy[col] = infer_asset_currency(atype, sym, attrs)

    cols = list(series_map.keys())
    series_list = [series_map[c] for c in cols]

    if calendar_mode == "common_trading_dates":
        common_idx = series_list[0].index
        for s in series_list[1:]:
            common_idx = common_idx.intersection(s.index)
        common_idx = common_idx.sort_values()
        if len(common_idx) < 3:
            raise ValueError(
                f"多标的共同交易日重叠样本过少（仅 {len(common_idx)} 条），请检查区间或各资产上市日期。"
            )
        local_prices = pd.concat([s.reindex(common_idx) for s in series_list], axis=1)
        local_prices.columns = cols
        tradable_mask = pd.DataFrame(True, index=common_idx, columns=cols)
        corr_local_prices = local_prices
    elif calendar_mode == "weekly_friday":
        weekly_list = [s.resample("W-FRI").last().dropna() for s in series_list]
        common_idx = weekly_list[0].index
        for w in weekly_list[1:]:
            common_idx = common_idx.intersection(w.index)
        common_idx = common_idx.sort_values()
        if len(common_idx) < 3:
            raise ValueError(f"周频对齐后共同样本过少（仅 {len(common_idx)} 周），请扩大日期区间。")
        local_prices = pd.concat([w.reindex(common_idx) for w in weekly_list], axis=1)
        local_prices.columns = cols
        tradable_mask = pd.DataFrame(True, index=common_idx, columns=cols)
        corr_local_prices = local_prices
    else:  # mtm_outer
        effective_start = max(s.index.min() for s in series_list)
        trimmed = [s.loc[s.index >= effective_start] for s in series_list]
        raw_outer = pd.concat(trimmed, axis=1, join="outer").sort_index()
        raw_outer.columns = cols
        tradable_mask = raw_outer.notna()
        local_prices = raw_outer.ffill().dropna()
        if len(local_prices) < 3:
            raise ValueError("逐日盯市对齐后有效样本不足 3 条，请检查资产区间重叠。")
        tradable_mask = tradable_mask.loc[local_prices.index]
        corr_local_prices = raw_outer.dropna()
        if len(corr_local_prices) < 3:
            raise ValueError("共同真实报价日期不足 3 条，无法无偏估计跨资产相关性。")
        warnings.append(
            "逐日盯市模式（mtm_outer）仅用于连续净值展示；非交易日不触发调仓，相关性与协方差仍基于真实共同报价区间计算。"
        )

    dropped_by_asset = {c: raw_counts[c] - int(len(local_prices)) for c in cols}
    max_raw = max(raw_counts.values())
    if len(local_prices) < 0.7 * max_raw and calendar_mode == "common_trading_dates":
        warnings.append(
            f"共同交易日对齐后保留 {len(local_prices)} 个估值日（最长单资产有 {max_raw} 条），跨市场节假日或上市时间差异导致部分区间被裁剪。"
        )

    conv = convert_prices_to_base_currency(
        local_prices,
        inferred_ccy,
        base_currency=base_currency,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
        proxy_url=proxy_url,
        allow_usdt_usd_peg=allow_usdt_usd_peg,
        max_fx_staleness_days=max_fx_staleness_days,
    )
    warnings.extend(conv.warnings)
    base_prices = conv.base_prices
    local_prices = conv.local_prices
    fx_rates = conv.fx_rates
    tradable_mask = tradable_mask.loc[base_prices.index]

    if len(base_prices) < 3:
        raise ValueError("汇率对齐后剩余有效估值日不足 3 条，无法计算组合收益。")

    if calendar_mode == "mtm_outer":
        corr_conv = convert_prices_to_base_currency(
            corr_local_prices,
            inferred_ccy,
            base_currency=base_currency,
            fx_loader=fx_loader,
            usdt_usd_loader=usdt_usd_loader,
            proxy_url=proxy_url,
            allow_usdt_usd_peg=allow_usdt_usd_peg,
            max_fx_staleness_days=max_fx_staleness_days,
        )
        corr_base_prices = corr_conv.base_prices
        corr_local_prices = corr_conv.local_prices
    else:
        corr_base_prices = base_prices
        corr_local_prices = local_prices

    returns_base = base_prices.pct_change(fill_method=None).iloc[1:].astype("float64")
    returns_local = local_prices.pct_change(fill_method=None).iloc[1:].astype("float64")
    corr_returns_base = corr_base_prices.pct_change(fill_method=None).iloc[1:].astype("float64")
    corr_returns_local = corr_local_prices.pct_change(fill_method=None).iloc[1:].astype("float64")

    default_ppy = 52.0 if calendar_mode == "weekly_friday" else 252.0
    eff_ppy = _compute_periods_per_year(
        corr_base_prices.index, len(corr_returns_base), default=default_ppy
    )
    bt_ppy = _compute_periods_per_year(base_prices.index, len(returns_base), default=default_ppy)

    base_prices.index.name = "Date"
    local_prices.index.name = "Date"
    fx_rates.index.name = "Date"
    returns_base.index.name = "Date"
    returns_local.index.name = "Date"

    overlap_summary = {
        "effective_start": base_prices.index[0].strftime("%Y-%m-%d"),
        "effective_end": base_prices.index[-1].strftime("%Y-%m-%d"),
        "valuation_points": int(len(base_prices)),
        "return_periods": int(len(returns_base)),
        "raw_counts": raw_counts,
        "raw_ranges": raw_ranges,
        "dropped_non_overlap_days_by_asset": dropped_by_asset,
        "currency_metadata": conv.metadata,
        "data_sources": data_sources,
        "adjustments": adjustments,
    }

    return MultiAssetAlignmentResult(
        base_prices=base_prices,
        local_prices=local_prices,
        fx_rates=fx_rates,
        returns_base=returns_base,
        returns_local=returns_local,
        correlation_returns_base=corr_returns_base,
        correlation_returns_local=corr_returns_local,
        tradable_mask=tradable_mask,
        base_date=pd.Timestamp(base_prices.index[0]),
        calendar_mode=calendar_mode,
        base_currency=conv.base_currency,
        asset_currencies=conv.asset_currencies,
        effective_periods_per_year=eff_ppy,
        backtest_periods_per_year=bt_ppy,
        overlap_summary=overlap_summary,
        warnings=warnings,
    )


def load_and_align_multi_asset_portfolio(
    legs_tuple: tuple[tuple[str, str], ...],
    start_date: object,
    end_date: object,
    *,
    proxy_url: str | None = None,
    calendar_mode: CalendarMode = "common_trading_dates",
    base_currency: str = "CNY",
    allow_usdt_usd_peg: bool = False,
    max_fx_staleness_days: int = 4,
    loader: Callable[..., pd.DataFrame] | None = None,
    fx_loader: Callable[..., pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[..., pd.DataFrame] | None = None,
    concurrent: bool = True,
    force_refresh: bool = False,
) -> MultiAssetAlignmentResult:
    """Fetch multiple heterogeneous asset legs and align onto unified valuation & currency."""
    if not legs_tuple:
        raise ValueError("legs_tuple 不能为空。")
    frames = _fetch_legs_frames(
        legs_tuple,
        start_date,
        end_date,
        proxy_url=proxy_url,
        loader=loader,
        concurrent=concurrent,
        force_refresh=force_refresh,
    )

    return align_multi_asset_frames(
        frames,
        calendar_mode=calendar_mode,
        base_currency=base_currency,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
        proxy_url=proxy_url,
        allow_usdt_usd_peg=allow_usdt_usd_peg,
        max_fx_staleness_days=max_fx_staleness_days,
    )
