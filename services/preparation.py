"""Shared window, loader and metadata preparation for research services."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from asset_catalog import (
    SUPPORTED_ADJUSTMENTS,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
    resolve_semantic_series_kind,
)
from asset_loader import (
    MultiAssetAlignmentResult,
    load_asset_data,
)
from contracts import (
    AssetSpec,
    QualityReport,
    SeriesMetadata,
    build_quality_report,
    build_series_metadata,
    sync_metadata_to_frame,
    validate_frame_metadata_consistency,
)
from currency import infer_asset_currency
from data_sources.errors import EmptySymbolDataError
from data_sources.quality import annotate_quote_quality
from ohlcv import compute_ohlcv_quality

AssetLoaderFn = Callable[..., pd.DataFrame]


def prepare_quote_metadata(
    frame: pd.DataFrame, asset_type: str, symbol: str, adjust: str | None
) -> str:
    """Bind identity and actual economic conventions to an owned, date-sliced frame."""
    annotate_quote_quality(frame, asset_type, adjust)
    frame.attrs["asset_type"] = asset_type
    requested_canonical = normalize_symbol_alias(asset_type, symbol)
    raw_attr_symbol = str(frame.attrs.get("symbol") or "").strip()
    if raw_attr_symbol and (
        normalize_symbol_alias(asset_type, raw_attr_symbol) == requested_canonical
    ):
        resolved_sym = raw_attr_symbol
    else:
        resolved_sym = symbol
    frame.attrs["symbol"] = resolved_sym
    has_attr_adjust = "adjust" in frame.attrs and frame.attrs.get("adjust") is not None
    raw_attr_adjust = str(frame.attrs.get("adjust") or "").strip() if has_attr_adjust else None
    if raw_attr_adjust is not None:
        if asset_type not in SUPPORTED_ADJUSTMENTS and raw_attr_adjust == "none":
            effective_adjust = ""
        elif asset_type in SUPPORTED_ADJUSTMENTS:
            effective_adjust = resolve_adjustment(asset_type, raw_attr_adjust)
        else:
            effective_adjust = raw_attr_adjust
    else:
        effective_adjust = adjust
    frame.attrs["adjust"] = effective_adjust
    frame.attrs["return_basis"] = resolve_return_basis(
        asset_type,
        adjust=effective_adjust if asset_type in SUPPORTED_ADJUSTMENTS else None,
        attrs=frame.attrs,
    )
    frame.attrs["series_kind"] = resolve_semantic_series_kind(
        asset_type,
        return_basis=frame.attrs["return_basis"],
        attrs=frame.attrs,
    )
    frame.attrs["canonical_symbol"] = normalize_symbol_alias(asset_type, resolved_sym)
    if not frame.attrs.get("currency"):
        frame.attrs["currency"] = infer_asset_currency(asset_type, resolved_sym, frame.attrs)
    return resolved_sym


def slice_to_requested_window(
    df: pd.DataFrame,
    *,
    asset_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Clip a loaded OHLCV frame to ``[start_date, end_date]`` and update slice quality if narrowed."""
    idx = pd.to_datetime(df.index, errors="coerce")
    valid_mask = ~pd.isna(idx)
    if not valid_mask.any():
        raise EmptySymbolDataError(
            f"未找到标的 {asset_type}:{symbol} 在 {start_date}~{end_date} 的有效数据。"
        )
    idx_valid = pd.DatetimeIndex(idx[valid_mask])
    if idx_valid.tz is not None:
        idx_cmp = idx_valid.tz_convert("UTC").tz_localize(None).normalize()
    else:
        idx_cmp = idx_valid.normalize()

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    range_mask = (idx_cmp >= start_ts) & (idx_cmp <= end_ts)

    full_mask = np.zeros(len(df), dtype=bool)
    full_mask[np.asarray(valid_mask, dtype=bool)] = np.asarray(range_mask, dtype=bool)
    if not full_mask.any():
        raise EmptySymbolDataError(
            f"未找到标的 {asset_type}:{symbol} 在 {start_date}~{end_date} 的有效数据。"
        )
    if full_mask.all():
        return df

    sliced = df.loc[full_mask].copy()
    sliced.attrs = dict(df.attrs)
    raw_quality = sliced.attrs.get("data_quality")
    existing = dict(raw_quality) if isinstance(raw_quality, Mapping) else {}
    if existing and "source_data_quality" not in sliced.attrs:
        sliced.attrs["source_data_quality"] = dict(existing)
    policy = str(existing.get("missing_policy", "drop") or "drop")
    sliced.attrs["data_quality"] = compute_ohlcv_quality(
        sliced,
        missing_policy=policy,
        dropped_invalid_rows=0,
        filled_price_rows=0,
        missing_volume_rows=0,
    )
    sliced.attrs["data_quality"]["quality_scope"] = "slice"
    return sliced


def dedupe_messages(*groups: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
    return tuple(merged)


def _invoke_asset_loader(
    fetch_fn: Callable[..., pd.DataFrame],
    spec: AssetSpec,
    start_date: str,
    end_date: str,
    *,
    proxy_url: str | None,
    force_refresh: bool,
) -> pd.DataFrame:
    """Invoke ``load_asset_data`` or a custom test/mock loader without breaking narrow signatures."""
    kwargs: dict[str, Any] = {
        "asset_type": spec.asset_type,
        "symbol": spec.symbol,
        "start_date": start_date,
        "end_date": end_date,
        "proxy_url": proxy_url,
    }
    try:
        sig = inspect.signature(fetch_fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if (has_var_kw or "adjust" in params) and spec.asset_type in SUPPORTED_ADJUSTMENTS:
            kwargs["adjust"] = spec.adjust
        if (has_var_kw or "force_refresh" in params) and force_refresh:
            kwargs["force_refresh"] = True
    except TypeError, ValueError:
        pass
    return fetch_fn(**kwargs)


def _load_and_slice_spec(
    spec: AssetSpec,
    start_date: str,
    end_date: str,
    *,
    fetch_fn: Callable[..., pd.DataFrame],
    proxy_url: str | None,
    force_refresh: bool,
) -> pd.DataFrame:
    loaded = _invoke_asset_loader(
        fetch_fn,
        spec,
        start_date,
        end_date,
        proxy_url=proxy_url,
        force_refresh=force_refresh,
    )
    if loaded is None or loaded.empty:
        raise EmptySymbolDataError(
            f"未找到标的 {spec.leg_name} 在 {start_date}~{end_date} 的有效数据。"
        )
    sliced = slice_to_requested_window(
        loaded,
        asset_type=spec.asset_type,
        symbol=spec.symbol,
        start_date=start_date,
        end_date=end_date,
    )
    if sliced is loaded:
        sliced = loaded.copy()
        sliced.attrs = dict(loaded.attrs)

    prepare_quote_metadata(sliced, spec.asset_type, spec.symbol, spec.adjust)
    return sliced


def fetch_multi_asset_frames(
    spec_map: Mapping[str, AssetSpec],
    start_date: str,
    end_date: str,
    *,
    loader: AssetLoaderFn | None,
    proxy_url: str | None,
    force_refresh: bool,
) -> dict[str, pd.DataFrame]:
    fetch_fn = loader or load_asset_data
    items = list(spec_map.items())

    if loader is None and len(items) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            futs = [
                pool.submit(
                    _load_and_slice_spec,
                    spec,
                    start_date,
                    end_date,
                    fetch_fn=fetch_fn,
                    proxy_url=proxy_url,
                    force_refresh=force_refresh,
                )
                for _, spec in items
            ]
            return {col_key: fut.result() for (col_key, _), fut in zip(items, futs)}

    return {
        col_key: _load_and_slice_spec(
            spec,
            start_date,
            end_date,
            fetch_fn=fetch_fn,
            proxy_url=proxy_url,
            force_refresh=force_refresh,
        )
        for col_key, spec in items
    }


def collect_metadata_and_quality(
    specs_by_key: Mapping[str, AssetSpec],
    raw_frames: Mapping[str, pd.DataFrame],
    aligned: MultiAssetAlignmentResult,
    *,
    trading_days: int | None,
) -> tuple[dict[str, SeriesMetadata], dict[str, QualityReport], bool, list[str]]:
    """Build per-asset metadata & quality reports and surface FX / calendar / local-currency caveats."""
    ccy_meta_root = dict(aligned.overlap_summary.get("currency_metadata", {}))
    fx_pairs_meta = dict(ccy_meta_root.get("fx_pairs", {}))
    assets_meta: dict[str, SeriesMetadata] = {}
    quality_map: dict[str, QualityReport] = {}
    extra_warnings: list[str] = []
    degraded = False

    for leg, spec in specs_by_key.items():
        frame = raw_frames[leg]
        q = build_quality_report(frame)
        base_meta = build_series_metadata(
            frame,
            asset_type=spec.asset_type,
            symbol=spec.symbol,
            adjust=str(frame.attrs.get("adjust", spec.adjust) or ""),
            return_basis=str(frame.attrs.get("return_basis") or ""),
            series_kind=str(frame.attrs.get("series_kind") or ""),
            output_currency=frame.attrs.get("currency"),
            trading_days=trading_days,
            strict=True,
        )
        validate_frame_metadata_consistency(frame, base_meta)
        sync_metadata_to_frame(frame, base_meta, q)
        leg_ccy_meta = dict(fx_pairs_meta.get(leg, {}))
        native_ccy = (
            str(
                leg_ccy_meta.get("native_currency")
                or base_meta.native_currency
                or aligned.asset_currencies.get(leg, "")
            ).strip()
            or None
        )
        output_ccy = (
            str(
                leg_ccy_meta.get("output_currency")
                or (native_ccy if aligned.base_currency == "LOCAL" else aligned.base_currency)
            ).strip()
            or None
        )
        pairs_used = leg_ccy_meta.get("pairs_used") or ()
        is_identity = leg_ccy_meta.get("source") == "identity" or (
            native_ccy is not None and output_ccy is not None and native_ccy == output_ccy
        )
        inferred_converted = bool(
            aligned.base_currency != "LOCAL"
            and not is_identity
            and (pairs_used or leg_ccy_meta.get("usdt_peg_assumed") or (native_ccy and output_ccy))
        )
        fx_converted = bool(
            leg_ccy_meta.get("fx_converted", base_meta.fx_converted or inferred_converted)
        )
        inferred_pair = (
            " * ".join(str(p) for p in pairs_used)
            if pairs_used
            else (
                f"{native_ccy}/{output_ccy}" if fx_converted and native_ccy and output_ccy else ""
            )
        )
        fx_pair = (
            str(leg_ccy_meta.get("fx_pair") or base_meta.fx_pair or inferred_pair).strip() or None
        )

        assets_meta[leg] = SeriesMetadata(
            asset_type=base_meta.asset_type,
            symbol=base_meta.symbol,
            canonical_symbol=base_meta.canonical_symbol,
            adjust=base_meta.adjust,
            return_basis=base_meta.return_basis,
            series_kind=base_meta.series_kind,
            native_currency=native_ccy,
            output_currency=output_ccy,
            fx_converted=fx_converted,
            fx_pair=fx_pair,
            trading_days=base_meta.trading_days,
            data_source=base_meta.data_source,
            fetched_at=base_meta.fetched_at,
            cache_hit=base_meta.cache_hit,
            warnings=base_meta.warnings,
            volume_unit=base_meta.volume_unit,
        )
        quality_map[leg] = q
        extra_warnings.extend(f"{leg}: {code}" for code in frame.attrs.get("quality_warnings", ()))
        if q.has_anomalies or base_meta.warnings:
            degraded = True

    distinct_local_ccys = {
        str(ccy).strip().upper() for ccy in aligned.asset_currencies.values() if str(ccy).strip()
    }
    mixed_local = aligned.base_currency == "LOCAL" and len(distinct_local_ccys) > 1
    if mixed_local:
        degraded = True
        extra_warnings.append(
            f"当前处于 base_currency='LOCAL' 且标的跨多种计价货币（{sorted(distinct_local_ccys)}）；"
            "跨币种名义收益汇总或相关对比未包含汇率折算（等效于假设零成本完全对冲），请改用 CNY/USD 统一计价口径。"
        )
    if any("fallback" in w.lower() or "peg" in w.lower() or "裁剪" in w for w in aligned.warnings):
        degraded = True

    return assets_meta, quality_map, (degraded or mixed_local), extra_warnings
