"""Cross-market currency inference, FX alignment, and base-currency price conversion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from asset_catalog import ASSET_DEFAULT_CURRENCY
from data_fetcher import load_fx_data
from ohlcv import DateLike, compute_ohlcv_quality

SUPPORTED_BASE_CURRENCIES: tuple[str, ...] = ("CNY", "USD", "LOCAL")

# Known global index quote currencies; unknown indices require explicit currency or LOCAL mode.
GLOBAL_INDEX_CURRENCIES: dict[str, str] = {
    ".INX": "USD",
    ".DJI": "USD",
    ".IXIC": "USD",
    "SPX": "USD",
    "INX": "USD",
    "GSPC": "USD",
    "NDX": "USD",
    "IXIC": "USD",
    "DJI": "USD",
    "DJIA": "USD",
    "标普500": "USD",
    "纳斯达克": "USD",
    "道琼斯": "USD",
    "HSI": "HKD",
    "HSCEI": "HKD",
    "恒生指数": "HKD",
    "国企指数": "HKD",
    "N225": "JPY",
    "日经225": "JPY",
    "FTSE": "GBP",
    "英国富时100": "GBP",
    "GDAXI": "EUR",
    "FCHI": "EUR",
    "SX5E": "EUR",
    "德国DAX30": "EUR",
    "法国CAC40": "EUR",
    "欧洲斯托克50": "EUR",
}

_DIRECT_FX_PAIRS: dict[tuple[str, str], str] = {
    ("USD", "CNY"): "USDCNH",
    ("CNY", "USD"): "CNHUSD",
    ("HKD", "CNY"): "HKDCNH",
    ("CNY", "HKD"): "CNHHKD",
    ("EUR", "CNY"): "EURCNH",
    ("CNY", "EUR"): "CNHEUR",
    ("GBP", "CNY"): "GBPCNH",
    ("CNY", "GBP"): "CNHGBP",
    ("JPY", "CNY"): "JPYCNH",
    ("CNY", "JPY"): "CNHJPY",
    ("HKD", "USD"): "HKDUSD",
    ("USD", "HKD"): "USDHKD",
    ("EUR", "USD"): "EURUSD",
    ("USD", "EUR"): "USDEUR",
    ("GBP", "USD"): "GBPUSD",
    ("USD", "GBP"): "USDGBP",
    ("JPY", "USD"): "JPYUSD",
    ("USD", "JPY"): "USDJPY",
}

_KNOWN_CRYPTO_QUOTES: tuple[str, ...] = ("USDT", "USDC", "BUSD", "USD", "CNY", "EUR", "BTC", "ETH")


def normalize_currency_code(currency: str) -> str:
    """Normalize currency aliases (`CNH` -> `CNY`) and uppercase."""
    code = str(currency).strip().upper()
    if code == "CNH":
        return "CNY"
    return code


def parse_fx_pair_currencies(symbol: str) -> tuple[str, str] | None:
    """Extract `(base, quote)` from an FX code such as `USDCNH` or `USD/JPY`."""
    cleaned = str(symbol).strip().upper().replace(" ", "")
    if "/" in cleaned:
        parts = cleaned.split("/", 1)
        if len(parts[0]) == 3 and len(parts[1]) == 3:
            return normalize_currency_code(parts[0]), normalize_currency_code(parts[1])
    if len(cleaned) == 6 and cleaned.isalpha():
        return normalize_currency_code(cleaned[:3]), normalize_currency_code(cleaned[3:])
    return None


def infer_asset_currency(
    asset_type: str,
    symbol: str,
    attrs: Mapping[str, Any] | None = None,
) -> str:
    """Infer an asset's native denomination currency from metadata, asset type, and symbol."""
    if attrs and attrs.get("currency"):
        return normalize_currency_code(str(attrs["currency"]))

    atype = str(asset_type).strip().lower()
    sym = str(symbol).strip()
    default_entry = ASSET_DEFAULT_CURRENCY.get(atype, "UNKNOWN")
    if default_entry not in ("QUOTE", "INDEX", "FX", "UNKNOWN"):
        return normalize_currency_code(default_entry)

    if atype == "crypto":
        upper = sym.upper()
        if "/" in upper:
            _, quote = upper.split("/", 1)
            return normalize_currency_code(quote)
        for suffix in _KNOWN_CRYPTO_QUOTES:
            if upper.endswith(suffix) and len(upper) > len(suffix):
                return normalize_currency_code(suffix)
        return "UNKNOWN"

    if atype == "global_index":
        if sym in GLOBAL_INDEX_CURRENCIES:
            return GLOBAL_INDEX_CURRENCIES[sym]
        upper = sym.upper()
        if upper in GLOBAL_INDEX_CURRENCIES:
            return GLOBAL_INDEX_CURRENCIES[upper]
        return "UNKNOWN"

    if atype == "fx":
        parsed = parse_fx_pair_currencies(sym)
        if parsed is not None:
            # An FX rate quotes units of quote currency per 1 unit of base currency.
            return parsed[1]
        return "UNKNOWN"

    return "UNKNOWN"


def align_fx_series_to_dates(
    fx_series: pd.Series,
    target_dates: pd.DatetimeIndex,
    *,
    max_staleness_days: int = 4,
) -> tuple[pd.Series, dict[str, int]]:
    """Align FX rates onto target valuation timestamps with bounded staleness.

    Uses the most recent FX quote on or before each valuation timestamp, only when the
    calendar age does not exceed ``max_staleness_days`` (handles weekends/short holidays
    without masking prolonged FX data outages). Never backfills from future FX rates.
    """
    if max_staleness_days < 0:
        raise ValueError("max_staleness_days must be non-negative.")

    clean_fx = pd.to_numeric(fx_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    clean_fx = clean_fx.loc[clean_fx > 0.0]
    if clean_fx.empty:
        raise ValueError("FX series contains no valid positive quotes.")

    fx_idx = pd.DatetimeIndex(pd.to_datetime(clean_fx.index, errors="raise"))
    if fx_idx.tz is not None:
        fx_idx = fx_idx.tz_localize(None)
    fx_idx = fx_idx.as_unit("ns")
    clean_fx = pd.Series(clean_fx.to_numpy(dtype=float), index=fx_idx).sort_index(kind="stable")
    clean_fx = clean_fx.loc[~clean_fx.index.duplicated(keep="last")]

    tgt = pd.DatetimeIndex(pd.to_datetime(target_dates, errors="raise"))
    if tgt.tz is not None:
        tgt = tgt.tz_localize(None)
    tgt = tgt.as_unit("ns").sort_values().unique()

    left = pd.DataFrame({"_target_dt": tgt})
    right = pd.DataFrame({"_fx_dt": clean_fx.index, "_fx_val": clean_fx.to_numpy(dtype=float)})
    merged = pd.merge_asof(
        left,
        right,
        left_on="_target_dt",
        right_on="_fx_dt",
        direction="backward",
    )

    age_days = (merged["_target_dt"] - merged["_fx_dt"]).dt.days
    valid = merged["_fx_val"].notna() & (age_days <= int(max_staleness_days))
    stale_exact = merged["_fx_val"].notna() & (age_days > 0) & (age_days <= int(max_staleness_days))
    expired = merged["_fx_val"].notna() & (age_days > int(max_staleness_days))
    missing_leading = merged["_fx_dt"].isna()

    aligned = pd.Series(
        np.where(valid, merged["_fx_val"], np.nan),
        index=tgt,
        dtype="float64",
    )
    stats = {
        "exact_match_days": int((valid & (age_days == 0)).sum()),
        "stale_filled_days": int(stale_exact.sum()),
        "expired_gap_days": int(expired.sum()),
        "leading_missing_days": int(missing_leading.sum()),
    }
    return aligned, stats


def _default_usdt_usd_loader(
    start_date: DateLike, end_date: DateLike, proxy_url: str | None
) -> pd.DataFrame:
    """Attempt to fetch live USDT/USD spot daily OHLCV from a USD-quoted exchange via CCXT."""
    from data_sources.crypto import fetch_usdt_usd_ohlcv

    return fetch_usdt_usd_ohlcv(start_date, end_date, proxy_url=proxy_url)


@dataclass
class CurrencyConversionResult:
    """Base-currency converted prices alongside local prices, FX rates, and diagnostics."""

    base_prices: pd.DataFrame
    local_prices: pd.DataFrame
    fx_rates: pd.DataFrame
    asset_currencies: dict[str, str]
    base_currency: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_fx_rate_series(
    from_currency: str,
    to_currency: str,
    target_dates: pd.DatetimeIndex,
    start_date: DateLike,
    end_date: DateLike,
    *,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
    proxy_url: str | None = None,
    allow_usdt_usd_peg: bool = False,
    max_fx_staleness_days: int = 4,
) -> tuple[pd.Series, list[str], dict[str, Any]]:
    """Resolve spot FX series `X_{from->to}(t)` aligned to `target_dates`.

    Never silently treats `USDT` as `USD` unless ``allow_usdt_usd_peg=True``.
    """
    src = normalize_currency_code(from_currency)
    dst = normalize_currency_code(to_currency)
    tgt = pd.DatetimeIndex(pd.to_datetime(target_dates)).as_unit("ns").sort_values().unique()

    if src == dst:
        return (
            pd.Series(1.0, index=tgt, dtype="float64"),
            [],
            {"source": "identity", "stale_filled_days": 0, "usdt_peg_assumed": False},
        )
    if src == "UNKNOWN" or dst == "UNKNOWN":
        raise ValueError(
            f"无法确定币种换算路径（{src} -> {dst}）。请为未知币种资产指定币种或使用 LOCAL 原币模式。"
        )

    load_fx = fx_loader if fx_loader is not None else load_fx_data
    load_usdt = usdt_usd_loader if usdt_usd_loader is not None else _default_usdt_usd_loader
    warnings: list[str] = []
    meta: dict[str, Any] = {"usdt_peg_assumed": False, "stale_filled_days": 0, "pairs_used": []}

    # Handle USDT leg separately: never treat USDT as unconditionally equal to USD.
    if src == "USDT" or dst == "USDT":
        try:
            usdt_df = load_usdt(start_date, end_date, proxy_url)
            usdt_raw = pd.to_numeric(usdt_df["Close"], errors="coerce")
            usdt_aligned, usdt_stats = align_fx_series_to_dates(
                usdt_raw, tgt, max_staleness_days=max_fx_staleness_days
            )
            meta["pairs_used"].append("USDT/USD")
            meta["stale_filled_days"] += usdt_stats["stale_filled_days"]
            if usdt_stats["stale_filled_days"] > 0:
                warnings.append(
                    f"USDT/USD 在 {usdt_stats['stale_filled_days']} 个估值日使用最近有效汇率（<= {max_fx_staleness_days} 天）。"
                )
        except Exception as exc:  # noqa: BLE001
            if not allow_usdt_usd_peg:
                raise ValueError(
                    "无法获取真实的 USDT/USD 市场汇率；USDT 不得无条件视作 USD。"
                    "请切换为 LOCAL 原币模式，或在参数中显式开启「允许 USDT≈1 USD 近似」。"
                    f"（详情：{exc}）"
                ) from exc
            usdt_aligned = pd.Series(1.0, index=tgt, dtype="float64")
            meta["usdt_peg_assumed"] = True
            warnings.append(
                "未获取到真实 USDT/USD 市场汇率，已按显式授权使用 1:1 近似（存在稳定币脱锚偏差风险）。"
            )

        if src == "USDT" and dst == "USD":
            return usdt_aligned, warnings, meta
        if src == "USD" and dst == "USDT":
            return 1.0 / usdt_aligned, warnings, meta
        if src == "USDT":
            usd_to_dst, sub_warn, sub_meta = resolve_fx_rate_series(
                "USD",
                dst,
                tgt,
                start_date,
                end_date,
                fx_loader=load_fx,
                usdt_usd_loader=load_usdt,
                proxy_url=proxy_url,
                allow_usdt_usd_peg=allow_usdt_usd_peg,
                max_fx_staleness_days=max_fx_staleness_days,
            )
            warnings.extend(sub_warn)
            meta["pairs_used"].extend(sub_meta.get("pairs_used", []))
            meta["stale_filled_days"] += int(sub_meta.get("stale_filled_days", 0))
            return usdt_aligned * usd_to_dst, warnings, meta

        src_to_usd, sub_warn, sub_meta = resolve_fx_rate_series(
            src,
            "USD",
            tgt,
            start_date,
            end_date,
            fx_loader=load_fx,
            usdt_usd_loader=load_usdt,
            proxy_url=proxy_url,
            allow_usdt_usd_peg=allow_usdt_usd_peg,
            max_fx_staleness_days=max_fx_staleness_days,
        )
        warnings.extend(sub_warn)
        meta["pairs_used"].extend(sub_meta.get("pairs_used", []))
        meta["stale_filled_days"] += int(sub_meta.get("stale_filled_days", 0))
        return src_to_usd / usdt_aligned, warnings, meta

    def _fetch_aligned_pair(pair_code: str) -> tuple[pd.Series, dict[str, int]]:
        # Pad start_date back by max_fx_staleness_days so leading valuation timestamps can match prior FX quotes.
        padded_start = pd.Timestamp(start_date) - pd.Timedelta(
            days=max(7, max_fx_staleness_days + 3)
        )
        fx_df = load_fx(pair_code, padded_start, end_date)
        if fx_df.empty or "Close" not in fx_df.columns:
            raise ValueError(f"汇率品种 {pair_code} 未返回有效数据。")
        return align_fx_series_to_dates(
            fx_df["Close"], tgt, max_staleness_days=max_fx_staleness_days
        )

    if (src, dst) in _DIRECT_FX_PAIRS:
        pair = _DIRECT_FX_PAIRS[(src, dst)]
        try:
            series, stats = _fetch_aligned_pair(pair)
            meta["pairs_used"].append(pair)
            meta["stale_filled_days"] += stats["stale_filled_days"]
            if stats["stale_filled_days"] > 0:
                warnings.append(
                    f"汇率 {pair} 在 {stats['stale_filled_days']} 个非外汇报价日沿用了 <= {max_fx_staleness_days} 天的最近即期汇率。"
                )
            return series, warnings, meta
        except Exception as direct_exc:
            if (dst, src) in _DIRECT_FX_PAIRS:
                inv_pair = _DIRECT_FX_PAIRS[(dst, src)]
                series_inv, stats = _fetch_aligned_pair(inv_pair)
                meta["pairs_used"].append(f"1/{inv_pair}")
                meta["stale_filled_days"] += stats["stale_filled_days"]
                return 1.0 / series_inv, warnings, meta
            if src != "USD" and dst != "USD":
                # Fallback to USD triangulation below.
                pass
            else:
                raise ValueError(
                    f"无法获取汇率 {src}->{dst}（{pair}）: {direct_exc}"
                ) from direct_exc

    if (dst, src) in _DIRECT_FX_PAIRS:
        inv_pair = _DIRECT_FX_PAIRS[(dst, src)]
        series_inv, stats = _fetch_aligned_pair(inv_pair)
        meta["pairs_used"].append(f"1/{inv_pair}")
        meta["stale_filled_days"] += stats["stale_filled_days"]
        return 1.0 / series_inv, warnings, meta

    if src != "USD" and dst != "USD":
        leg1, w1, m1 = resolve_fx_rate_series(
            src,
            "USD",
            tgt,
            start_date,
            end_date,
            fx_loader=load_fx,
            usdt_usd_loader=load_usdt,
            proxy_url=proxy_url,
            allow_usdt_usd_peg=allow_usdt_usd_peg,
            max_fx_staleness_days=max_fx_staleness_days,
        )
        leg2, w2, m2 = resolve_fx_rate_series(
            "USD",
            dst,
            tgt,
            start_date,
            end_date,
            fx_loader=load_fx,
            usdt_usd_loader=load_usdt,
            proxy_url=proxy_url,
            allow_usdt_usd_peg=allow_usdt_usd_peg,
            max_fx_staleness_days=max_fx_staleness_days,
        )
        warnings.extend(w1 + w2)
        meta["pairs_used"].extend(m1.get("pairs_used", []) + m2.get("pairs_used", []))
        meta["stale_filled_days"] += int(m1.get("stale_filled_days", 0)) + int(
            m2.get("stale_filled_days", 0)
        )
        return leg1 * leg2, warnings, meta

    raise ValueError(f"暂不支持从 {src} 到 {dst} 的汇率换算。")


def convert_prices_to_base_currency(
    local_prices: pd.DataFrame,
    asset_currencies: Mapping[str, str],
    base_currency: str = "CNY",
    *,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
    proxy_url: str | None = None,
    allow_usdt_usd_peg: bool = False,
    max_fx_staleness_days: int = 4,
) -> CurrencyConversionResult:
    """Convert a wide local-currency price DataFrame into a unified investor `base_currency`."""
    if local_prices.empty:
        raise ValueError("local_prices cannot be empty.")

    base_ccy = normalize_currency_code(base_currency)
    if base_ccy == "LOCAL":
        ones = pd.DataFrame(1.0, index=local_prices.index, columns=local_prices.columns)
        return CurrencyConversionResult(
            base_prices=local_prices.astype("float64").copy(),
            local_prices=local_prices.astype("float64").copy(),
            fx_rates=ones,
            asset_currencies={
                col: normalize_currency_code(asset_currencies.get(col, "UNKNOWN"))
                for col in local_prices.columns
            },
            base_currency="LOCAL",
            warnings=["当前采用原币种（LOCAL）对比口径，未做跨币种汇率折算。"],
            metadata={"fx_pairs": {}, "usdt_peg_assumed": False},
        )

    if base_ccy not in ("CNY", "USD"):
        raise ValueError(
            f"Unsupported base_currency {base_currency!r}; choose 'CNY', 'USD', or 'LOCAL'."
        )

    tgt = pd.DatetimeIndex(local_prices.index)
    start_dt = tgt.min()
    end_dt = tgt.max()

    fx_cache: dict[str, tuple[pd.Series, list[str], dict[str, Any]]] = {}
    fx_columns: dict[str, pd.Series] = {}
    resolved_currencies: dict[str, str] = {}
    all_warnings: list[str] = []
    fx_meta: dict[str, Any] = {}
    usdt_peg_used = False

    for col in local_prices.columns:
        src_ccy = normalize_currency_code(asset_currencies.get(col, "UNKNOWN"))
        resolved_currencies[col] = src_ccy
        if src_ccy not in fx_cache:
            fx_cache[src_ccy] = resolve_fx_rate_series(
                src_ccy,
                base_ccy,
                tgt,
                start_dt,
                end_dt,
                fx_loader=fx_loader,
                usdt_usd_loader=usdt_usd_loader,
                proxy_url=proxy_url,
                allow_usdt_usd_peg=allow_usdt_usd_peg,
                max_fx_staleness_days=max_fx_staleness_days,
            )
        series, warns, meta = fx_cache[src_ccy]
        fx_columns[col] = series
        for w in warns:
            if w not in all_warnings:
                all_warnings.append(w)
        fx_meta[col] = meta
        if meta.get("usdt_peg_assumed"):
            usdt_peg_used = True

    fx_df = pd.DataFrame(fx_columns, index=tgt).astype("float64")
    # Drop timestamps where any required FX quote exceeded staleness limits.
    valid_fx_rows = fx_df.notna().all(axis=1)
    dropped_fx_days = int((~valid_fx_rows).sum())
    if not valid_fx_rows.any():
        raise ValueError("汇率对齐后无有效共同估值日期，请扩大时间区间或检查汇率数据源。")
    if dropped_fx_days > 0:
        all_warnings.append(
            f"因部分交易日缺少有效汇率（超过 {max_fx_staleness_days} 天窗口或早于汇率起点），已剔除 {dropped_fx_days} 个共同估值日。"
        )

    aligned_local = local_prices.loc[valid_fx_rows].astype("float64").copy()
    aligned_fx = fx_df.loc[valid_fx_rows].astype("float64").copy()
    aligned_base = (aligned_local * aligned_fx).astype("float64")

    return CurrencyConversionResult(
        base_prices=aligned_base,
        local_prices=aligned_local,
        fx_rates=aligned_fx,
        asset_currencies=resolved_currencies,
        base_currency=base_ccy,
        warnings=all_warnings,
        metadata={
            "fx_pairs": fx_meta,
            "dropped_fx_days": dropped_fx_days,
            "usdt_peg_assumed": usdt_peg_used,
        },
    )


def convert_ohlcv_to_base_currency(
    df: pd.DataFrame,
    asset_key: str,
    symbol: str,
    target_currency: str,
    *,
    proxy_url: str | None = None,
    allow_usdt_peg: bool = False,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """将单标的 OHLCV 价格列按对齐汇率同比例折算为目标基准货币（拒绝 yield_level 与 fx_rate）。"""
    target_norm = normalize_currency_code(target_currency)
    asset_norm = str(asset_key or "").strip().lower()
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    return_basis = str(df.attrs.get("return_basis", "") or "").strip().lower()
    if target_norm == "LOCAL":
        native_ccy = infer_asset_currency(asset_norm, symbol, df.attrs)
        out = df.copy()
        out.attrs = dict(df.attrs)
        out.attrs["native_currency"] = native_ccy
        out.attrs["currency"] = native_ccy
        out.attrs["fx_converted"] = False
        out.attrs["fx_pair"] = None
        return out

    if (
        series_kind == "yield_level"
        or return_basis == "yield_level"
        or asset_norm == "bond_us_yield"
    ):
        raise ValueError("收益率水平序列（yield_level）不是货币计价资产价格，不支持基准货币换算。")
    if series_kind == "fx_rate" or return_basis == "fx_rate" or asset_norm == "fx":
        raise ValueError("即期汇率序列（fx_rate）已是货币相对比价，不支持再折算为基准货币。")

    native_ccy = infer_asset_currency(asset_norm, symbol, df.attrs)
    if target_norm == native_ccy:
        out = df.copy()
        out.attrs = dict(df.attrs)
        out.attrs["native_currency"] = native_ccy
        out.attrs["currency"] = native_ccy
        out.attrs["fx_converted"] = False
        out.attrs["fx_pair"] = None
        return out

    col_key = f"{asset_norm}:{symbol}"
    conv = convert_prices_to_base_currency(
        pd.DataFrame({col_key: df["Close"]}),
        {col_key: native_ccy},
        base_currency=target_norm,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
        proxy_url=proxy_url,
        allow_usdt_usd_peg=allow_usdt_peg,
    )
    aligned_idx = conv.base_prices.index
    local_close = conv.local_prices[col_key]
    base_close = conv.base_prices[col_key]
    fx_ratio = base_close / local_close

    out = df.reindex(aligned_idx).copy()
    for price_col in ("Open", "High", "Low", "Close"):
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce") * fx_ratio

    out.attrs = dict(df.attrs)
    out.attrs["native_currency"] = native_ccy
    out.attrs["currency"] = conv.base_currency
    out.attrs["fx_converted"] = True

    pair_meta = (conv.metadata.get("fx_pairs") or {}).get(col_key) or {}
    pairs_used = pair_meta.get("pairs_used") or ()
    out.attrs["fx_pair"] = (
        " * ".join(str(p) for p in pairs_used)
        if pairs_used
        else f"{native_ccy}/{conv.base_currency}"
    )

    merged_warnings: list[str] = []
    for w in (*(out.attrs.get("fx_warnings") or ()), *conv.warnings):
        text = str(w).strip()
        if text and text not in merged_warnings:
            merged_warnings.append(text)
    if merged_warnings:
        out.attrs["fx_warnings"] = merged_warnings

    prev_quality = out.attrs.get("data_quality")
    if len(out) != len(df):
        prev_map = prev_quality if isinstance(prev_quality, Mapping) else {}
        out.attrs["data_quality"] = compute_ohlcv_quality(
            out,
            missing_policy=str(prev_map.get("missing_policy", "drop") or "drop"),
            dropped_invalid_rows=int(prev_map.get("dropped_invalid_rows", 0) or 0)
            + max(0, len(df) - len(out)),
            filled_price_rows=int(prev_map.get("filled_price_rows", 0) or 0),
            missing_volume_rows=int(prev_map.get("missing_volume_rows", 0) or 0),
        )
    return out


__all__ = [
    "GLOBAL_INDEX_CURRENCIES",
    "SUPPORTED_BASE_CURRENCIES",
    "CurrencyConversionResult",
    "align_fx_series_to_dates",
    "convert_ohlcv_to_base_currency",
    "convert_prices_to_base_currency",
    "infer_asset_currency",
    "normalize_currency_code",
    "parse_fx_pair_currencies",
    "resolve_fx_rate_series",
]
