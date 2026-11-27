"""Bridge between UI components/workspaces and backend services, reporting, and caches."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

import asset_loader as _asset_loader_mod
import currency as _currency_mod
import data_fetcher as _data_fetcher_mod
from asset_inputs import parse_asset_list, parse_weights
from contracts import AnalysisResult, AssetSpec, MultiAssetResult, QualityReport
from reporting import build_html_report, build_markdown_report, build_plotly_figures
from ui.theme import apply_ui_plotly_theme


def date_str(value: object) -> str:
    """Format any date-like input as ``YYYY-MM-DD``."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def make_service_loaders(
    *,
    force_refresh: bool,
    cached_loader: Callable[..., pd.DataFrame],
) -> tuple[Callable[..., pd.DataFrame], Callable[[str, object, object], pd.DataFrame]]:
    """Build service-compatible ``(asset_loader, fx_loader)`` honoring cache & monkeypatches."""

    def _service_asset_loader(
        asset_type: str,
        symbol: str,
        start_date: object,
        end_date: object,
        proxy_url: str | None = None,
        adjust: str | None = None,
        force_refresh: bool = force_refresh,
    ) -> pd.DataFrame:
        s_str = date_str(start_date)
        e_str = date_str(end_date)
        if force_refresh:
            return _asset_loader_mod.load_asset_data(
                asset_type=asset_type,
                symbol=symbol,
                start_date=s_str,
                end_date=e_str,
                proxy_url=proxy_url,
                adjust=adjust,
                force_refresh=True,
            )
        return cached_loader(
            asset_type=asset_type,
            symbol=symbol,
            start_date=s_str,
            end_date=e_str,
            proxy_url=proxy_url,
            adjust=adjust,
        )

    def _service_fx_loader(symbol: str, start_date: object, end_date: object) -> pd.DataFrame:
        # Honor direct currency monkeypatches in CLI/hybrid tests when asset_loader wasn't patched.
        if (
            _currency_mod.load_fx_data is not _data_fetcher_mod.load_fx_data
            and _asset_loader_mod.load_fx_data is _data_fetcher_mod.load_fx_data
        ):
            return _currency_mod.load_fx_data(symbol, start_date, end_date)
        s_str = date_str(start_date)
        e_str = date_str(end_date)
        if force_refresh:
            return _asset_loader_mod.load_asset_data(
                asset_type="fx",
                symbol=symbol,
                start_date=s_str,
                end_date=e_str,
                force_refresh=True,
            )
        return cached_loader(
            asset_type="fx",
            symbol=symbol,
            start_date=s_str,
            end_date=e_str,
        )

    return _service_asset_loader, _service_fx_loader


def extract_quality_counts(
    obj: AnalysisResult | QualityReport | pd.DataFrame | Mapping[str, Any] | None,
    legacy_metrics: Mapping[str, Any] | None = None,
) -> tuple[int, int, int, int, int, str]:
    """Normalize quality counters from QualityReport, AnalysisResult, DataFrame, or legacy dict."""
    quality_obj: Any = None
    if isinstance(obj, AnalysisResult):
        quality_obj = obj.quality
    elif isinstance(obj, QualityReport):
        quality_obj = obj
    elif isinstance(obj, pd.DataFrame):
        quality_obj = obj.attrs.get("data_quality")
    elif isinstance(obj, Mapping):
        quality_obj = obj.get("data_quality", obj)

    if quality_obj is None and isinstance(legacy_metrics, Mapping):
        quality_obj = legacy_metrics.get("data_quality")

    if isinstance(quality_obj, QualityReport):
        return (
            int(quality_obj.dropped_invalid_rows),
            int(quality_obj.filled_price_rows),
            int(quality_obj.missing_volume_rows),
            int(quality_obj.non_positive_price_rows),
            int(quality_obj.negative_volume_rows),
            str(quality_obj.missing_policy or ""),
        )
    if isinstance(quality_obj, Mapping):
        dropped = int(
            quality_obj.get("dropped_invalid_rows", quality_obj.get("dropped_rows", 0)) or 0
        )
        filled = int(quality_obj.get("filled_price_rows", quality_obj.get("ffilled_cells", 0)) or 0)
        missing_vol = int(quality_obj.get("missing_volume_rows", 0) or 0)
        non_pos = int(quality_obj.get("non_positive_price_rows", 0) or 0)
        neg_vol = int(quality_obj.get("negative_volume_rows", 0) or 0)
        policy = str(quality_obj.get("missing_policy", "") or "")
        return dropped, filled, missing_vol, non_pos, neg_vol, policy
    return 0, 0, 0, 0, 0, ""


def format_data_quality_warning(
    obj: AnalysisResult | QualityReport | pd.DataFrame | Mapping[str, Any] | None,
    legacy_metrics: Mapping[str, Any] | None = None,
) -> str:
    """Build user-facing Chinese data quality warning when rows were dropped, filled, or anomalous."""
    dropped, filled, missing_vol, non_pos, neg_vol, policy = extract_quality_counts(
        obj, legacy_metrics
    )
    parts: list[str] = []
    if dropped > 0:
        parts.append(f"清理异常/缺失 {dropped} 行")
    if filled > 0:
        parts.append(f"前向填补价格 {filled} 行")
    if missing_vol > 0:
        parts.append(f"缺失成交量补零 {missing_vol} 行")
    if non_pos > 0:
        parts.append(f"存在非正价格 {non_pos} 行")
    if neg_vol > 0:
        parts.append(f"存在负成交量 {neg_vol} 行")
    if not parts:
        return ""
    policy_note = f"（策略：{policy}）" if policy else ""
    return f"数据质量提示{policy_note}：" + "；".join(parts) + "。"


def cleaned_ohlcv_to_csv_bytes_and_filename(
    df: pd.DataFrame,
    asset_type: str,
    symbol: str,
    start_date: object,
    end_date: object,
) -> tuple[bytes, str]:
    """Serialize OHLCV DataFrame with UTF-8-SIG BOM and filesystem-safe filename."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and not out.index.name:
        out.index.name = "Date"
    write_idx = not isinstance(out.index, pd.RangeIndex)
    csv_bytes = out.to_csv(index=write_idx, encoding="utf-8-sig").encode("utf-8-sig")
    safe_sym = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", str(symbol or "asset").strip()).strip("_") or "asset"
    )
    s_str = date_str(start_date)
    e_str = date_str(end_date)
    filename = f"{asset_type}_{safe_sym}_{s_str}_{e_str}_ohlcv.csv"
    return csv_bytes, filename


def get_or_build_figures(
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    chart_types: Sequence[str] | None = None,
    on_error: Callable[[ImportError], None] | None = None,
) -> tuple[dict[str, tuple[str, str, Any]] | None, dict[str, str]]:
    """Build and memoize Claude Dark Plotly figures keyed by ``(result_version, chart_types)``."""
    requested_key = tuple(chart_types) if chart_types is not None else None
    cache_key = (int(state.get("result_version", 0)), requested_key)

    cached_figs = state.get("figures")
    if isinstance(cached_figs, dict) and state.get("figures_cache_key") == cache_key:
        return cached_figs, dict(state.get("skipped") or {})

    try:
        raw_figs, skipped = build_plotly_figures(result, chart_types=chart_types)
    except ImportError as exc:
        state["figures"] = None
        state["figures_cache_key"] = None
        if on_error is not None:
            on_error(exc)
        return None, {}

    themed_figs: dict[str, tuple[str, str, Any]] = {
        ct: (title, subtitle, apply_ui_plotly_theme(fig))
        for ct, (title, subtitle, fig) in raw_figs.items()
    }
    state["figures"] = themed_figs
    state["skipped"] = skipped
    state["figures_requested"] = requested_key
    state["figures_cache_key"] = cache_key
    return themed_figs, skipped


def get_or_build_report_exports(
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    skipped_charts: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Memoize Markdown, HTML, JSON, and CSV exports keyed by ``(result_version, skipped_charts)``."""
    skipped_key = tuple(sorted((skipped_charts or {}).items()))
    cache_key = (int(state.get("result_version", 0)), skipped_key)
    cached = state.get("exports")
    if isinstance(cached, dict) and cached and state.get("exports_cache_key") == cache_key:
        return cached

    md_text = build_markdown_report(result, skipped_charts=skipped_charts)
    html_text = build_html_report(result, skipped_charts=skipped_charts)
    json_text = json.dumps(
        result.to_dict(include_data=False),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    if isinstance(result, AnalysisResult):
        primary_csv, _ = cleaned_ohlcv_to_csv_bytes_and_filename(
            result.data_frame,
            result.asset.asset_type,
            result.asset.canonical_symbol,
            result.quality.start_date,
            result.quality.end_date,
        )
    else:
        primary_df = result.data_frame.copy()
        if isinstance(primary_df.index, pd.DatetimeIndex) and not primary_df.index.name:
            primary_df.index.name = "Date"
        write_idx = not isinstance(primary_df.index, pd.RangeIndex)
        primary_csv = primary_df.to_csv(index=write_idx, encoding="utf-8-sig").encode("utf-8-sig")

    exports: dict[str, Any] = {
        "md": md_text,
        "html": html_text,
        "json": json_text,
        "csv": primary_csv,
    }
    if isinstance(result, MultiAssetResult):
        extra_csvs: dict[str, bytes] = {}
        for frame_key in ("weights", "backtest", "returns", "correlation", "covariance"):
            frame = result.frames.get(frame_key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                use_idx = not isinstance(frame.index, pd.RangeIndex)
                extra_csvs[frame_key] = frame.to_csv(index=use_idx, encoding="utf-8-sig").encode(
                    "utf-8-sig"
                )
        exports["extra_csvs"] = extra_csvs

    state["exports"] = exports
    state["exports_cache_key"] = cache_key
    return exports


def preview_multi_asset_specs(text: str) -> tuple[tuple[AssetSpec, ...], str | None]:
    """Parse UI asset list for live non-blocking chip preview."""
    if not str(text or "").strip():
        return (), "请输入至少 2 个资产条目，例如 `etf:510300`。"
    try:
        specs = parse_asset_list([text], syntax="web")
        return specs, None
    except ValueError as exc:
        return (), str(exc)


def preview_custom_weights(
    raw: str,
    *,
    expected_count: int | None = None,
) -> tuple[tuple[float, ...] | None, str | None]:
    """Validate UI custom weight text for non-blocking inline feedback."""
    cleaned = str(raw or "").strip()
    if not cleaned:
        return None, "请输入与资产数量一致的正权重（逗号分隔，例如 `0.4, 0.3, 0.3`）。"
    try:
        weights = parse_weights(cleaned, web=True)
    except ValueError as exc:
        return None, str(exc)
    if expected_count is not None and len(weights) != expected_count:
        return (
            weights,
            f"当前输入 {len(weights)} 个权重，与已解析的 {expected_count} 个资产不一致。",
        )
    if sum(weights) <= 0:
        return weights, "权重总和必须大于 0。"
    return weights, None
