"""UI-independent asset routing, crypto loading and multi-asset alignment."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd

from asset_catalog import ASSET_RETURN_BASIS, ASSET_RETURN_BASIS_NOTES, SUPPORTED_ASSET_TYPES
from asset_resolver import load_china_equity_ohlcv
from currency import convert_prices_to_base_currency, infer_asset_currency
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
from ohlcv import (
    validate_analysis_date_range as _validate_date_range,
    normalize_ohlcv as _normalize_ohlcv,
    filter_by_date_range as _filter_by_date_range,
)

GOLD_SYMBOL_ALIASES = {"GC": "GC", "AU9999": "GC"}


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
    if timeframe != "1d":
        raise ValueError("Only the daily 1d timeframe is supported.")
    if isinstance(batch_limit, bool) or not isinstance(batch_limit, int) or batch_limit < 1:
        raise ValueError("batch_limit must be a positive integer.")
    timeframe_ms = 24 * 60 * 60 * 1000
    end_ms = int((end_ts + pd.Timedelta(days=1)).tz_localize("UTC").timestamp() * 1000)
    since_ms = int(start_ts.tz_localize("UTC").timestamp() * 1000)
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
        # 下一页从最后一根 K 线之后开始，避免重复抓取同一天。
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

    try:
        ccxt = importlib.import_module("ccxt")
    except ImportError as exc:
        raise ImportError("缺少运行依赖 `ccxt`，请先安装：`pip install ccxt`。") from exc
    exchange_config: dict[str, Any] = {
        "timeout": 30000,
        "enableRateLimit": True,
    }

    proxies = _build_ccxt_proxy_config(proxy_url)
    if proxies is not None:
        # 代理只注入 CCXT 交易所客户端，不影响其他资产类型的数据源。
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


def load_asset_data(
    asset_type: str,
    symbol: str,
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """统一加载多类资产 OHLCV（契约一致）。"""
    start_date, end_date = _validate_date_range(start_date, end_date)
    asset_key = str(asset_type).strip().lower()
    if asset_key not in SUPPORTED_ASSET_TYPES:
        raise ValueError("不支持的资产类型: %s" % asset_type)

    # 这里是 Web 应用的数据总路由：不同资产来源不同，但返回值都必须是 OHLCV。
    if asset_key == "etf":
        df = _fetch_etf_data(symbol=symbol, start_date=start_date, end_date=end_date)
    elif asset_key == "ashare":
        df = _fetch_ashare_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
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
        df = load_global_market_data("us_stock", symbol, start_date, end_date)
    elif asset_key == "hk_stock":
        df = load_global_market_data("hk_stock", symbol, start_date, end_date)
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
    df.attrs.setdefault("return_basis", ASSET_RETURN_BASIS.get(asset_key, "adjusted_or_price"))
    df.attrs.setdefault("currency", infer_asset_currency(asset_key, resolved_sym, df.attrs))
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

    close_frames: list[pd.Series] = []
    load_leg = loader if loader is not None else load_asset_data
    for asset_key, symbol in legs_tuple:
        # 每条 leg 独立拉取，再统一成 Close 宽表，列名包含类型以避免代码冲突。
        col = multi_asset_leg_column_name(asset_key, symbol)
        df_leg = load_leg(
            asset_type=asset_key,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            proxy_url=proxy_url,
        )
        if df_leg.empty or "Close" not in df_leg.columns:
            raise ValueError(
                "标的 %r（列名 %r）在指定区间内无可用数据或缺少 Close 列。"
                % ("%s:%s" % (asset_key, symbol), col)
            )
        close = pd.to_numeric(df_leg["Close"], errors="coerce")
        if close.isna().all():
            raise ValueError(
                "标的 %r（列名 %r）在指定区间内 Close 全为缺失。"
                % ("%s:%s" % (asset_key, symbol), col)
            )
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

        if series_kind == "yield_level" or atype == "bond_us_yield":
            raise ValueError(
                f"标的 {col!r} 为收益率水平序列（yield_level），不是资产价格或持有期回报，不能直接纳入组合优化。"
            )

        basis = str(attrs.get("return_basis") or ASSET_RETURN_BASIS.get(atype, "")).strip()
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
) -> MultiAssetAlignmentResult:
    """Fetch multiple heterogeneous asset legs and align onto unified valuation & currency."""
    if not legs_tuple:
        raise ValueError("legs_tuple 不能为空。")
    load_leg = loader if loader is not None else load_asset_data
    frames: dict[str, pd.DataFrame] = {}
    for asset_key, symbol in legs_tuple:
        col = multi_asset_leg_column_name(asset_key, symbol)
        if col in frames:
            raise ValueError(f"多标的组合不能包含重复标的: {col}")
        df_leg = load_leg(
            asset_type=asset_key,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            proxy_url=proxy_url,
        )
        frames[col] = df_leg

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
