"""Streamlit Web UI for Quantitative Analysis (M5 / V1.0).

Serves as the thin interactive presentation layer over unified M1-M4 services:
- Workspace 1: 单资产探索 (`services.analysis_service.analyze_single_asset`)
- Workspace 2: 多资产比较 (`services.multi_asset_service.compare_assets`)
- Workspace 3: 投资组合研究 (`services.multi_asset_service.analyze_portfolio`)
- Unified charts & reports (`reporting.build_plotly_figures`, `build_markdown_report`, `build_html_report`)
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # Allow headless library/CLI imports without Streamlit installed.
    st = None  # type: ignore[assignment]

import asset_loader as _asset_loader_mod
import currency as _currency_mod
import data_fetcher as _data_fetcher_mod
from analysis_metrics import compute_cagr as _compute_cagr, get_quant_metrics
from asset_catalog import (
    _TAB2_SYSTEM_MATH_MARKDOWN,
    ASSET_LABELS,
    ASSET_RETURN_BASIS_NOTES,
    DEFAULT_SYMBOLS,
    PRESET_PORTFOLIOS,
    SUPPORTED_ADJUSTMENTS,
    SUPPORTED_ASSET_TYPES,
    build_ef_multi_asset_codes_reference_df,
    format_asset_guide_markdown,
    get_asset_trading_days,
    normalize_symbol_alias,
    resolve_adjustment,
)
from asset_loader import multi_asset_leg_column_name, parse_multi_asset_legs
from asset_resolver import format_diagnosis_user_message, resolve_analysis_date_window
from charts import (
    build_tab3_correlation_heatmap,
    build_tab3_gbm_figure,
    build_tab3_normalize_figure,
)
from contracts import (
    SOFTWARE_VERSION,
    SUPPORTED_CALENDAR_MODES,
    SUPPORTED_COV_METHODS,
    SUPPORTED_PORTFOLIO_STRATEGIES,
    SUPPORTED_REBALANCE_FREQS,
    SUPPORTED_RETURN_CONVENTIONS,
    SUPPORTED_VALIDATION_MODES,
    AnalysisRequest,
    AnalysisResult,
    AssetSpec,
    CompareRequest,
    MultiAssetResult,
    PortfolioRequest,
    QualityReport,
)
from data_sources.errors import DataFetchError, EmptySymbolDataError
from data_sources.network import apply_default_network_proxy_policy
from quant_models import normalize_close_to_base_one, run_monte_carlo_gbm
from reporting import (
    build_html_report,
    build_markdown_report,
    build_plotly_figures,
    format_drawdown_percent,
    format_integer,
    format_metric_value,
    format_p_value,
    format_percent,
    format_price_with_unit,
    format_ratio,
    format_reason_label,
    is_valid_number,
)
from services.analysis_service import analyze_single_asset, fetch_single_asset
from services.multi_asset_service import analyze_portfolio, compare_assets

__all__ = [
    "DataFetchError",
    "EmptySymbolDataError",
    "SUPPORTED_ASSET_TYPES",
    "_VAL_MODE_ROLLING",
    "_compute_cagr",
    "_format_data_quality_warning",
    "build_tab3_correlation_heatmap",
    "get_quant_metrics",
    "load_asset_data",
    "load_multi_leg_close_wide",
    "load_multiple_etfs_close",
    "main",
    "multi_asset_leg_column_name",
    "parse_multi_asset_legs",
]

_VAL_MODE_ROLLING = "walk_forward_rolling"

_ADJUST_LABELS: dict[str, str] = {
    "hfq": "后复权 (hfq)",
    "qfq": "前复权 (qfq)",
    "": "不复权 (raw)",
}

_BASE_CURRENCY_LABELS: dict[str, str] = {
    "LOCAL": "原生计价货币 (LOCAL)",
    "CNY": "人民币统一折算 (CNY)",
    "USD": "美元统一折算 (USD)",
}

_RETURN_CONVENTION_LABELS: dict[str, str] = {
    "log": "对数收益率 (log)",
    "simple": "简单算术收益率 (simple)",
}

_CALENDAR_MODE_LABELS: dict[str, str] = {
    "common_trading_dates": "共同交易日交集 (common_trading_dates)",
    "weekly_friday": "周频周五重采样 (weekly_friday)",
    "mtm_outer": "跨市场外连接按市值重估 (mtm_outer)",
}

_COV_METHOD_LABELS: dict[str, str] = {
    "ledoit_wolf": "Ledoit-Wolf 收缩估计 (ledoit_wolf)",
    "sample": "样本协方差 (sample)",
    "diagonal": "对角方差阵 (diagonal)",
}

_STRATEGY_LABELS: dict[str, str] = {
    "max_sharpe": "最大夏普切点组合 (max_sharpe)",
    "min_variance": "全局最小方差组合 GMV (min_variance)",
    "risk_parity": "风险平价 ERC (risk_parity)",
    "inverse_volatility": "倒数波动率加权 (inverse_volatility)",
    "equal_weight": "等权重基准 1/N (equal_weight)",
    "custom": "自定义权重组合 (custom)",
}

_VALIDATION_MODE_LABELS: dict[str, str] = {
    "train_test": "静态样本外划分 OOS (train_test)",
    "walk_forward_rolling": "滚动窗口 Walk-Forward OOS (walk_forward_rolling)",
    "walk_forward_expanding": "扩展窗口 Walk-Forward OOS (walk_forward_expanding)",
    "in_sample": "全样本拟合 In-Sample (含前视偏差警告)",
}

_REBALANCE_FREQ_LABELS: dict[str, str] = {
    "monthly": "月度再平衡 (monthly)",
    "quarterly": "季度再平衡 (quarterly)",
    "daily": "日度再平衡 (daily)",
    "buy_and_hold": "买入持有不调仓 (buy_and_hold)",
}

_COMPARE_CHART_LABELS: dict[str, str] = {
    "normalized": "归一化净值曲线 (Base=1)",
    "drawdown": "水下回撤对比",
    "rolling": "滚动风险与相关性",
    "correlation": "收益率相关性热力图",
}

_PORTFOLIO_CHART_LABELS: dict[str, str] = {
    "backtest": "策略回测净值与回撤",
    "weights": "权重分布与 Euler 风险贡献",
    "frontier": "SLSQP 精确有效前沿",
    "correlation": "标的同期相关性热力图",
    "normalized": "组合标的归一化净值",
    "drawdown": "组合标的水下回撤",
    "weight_history": "Walk-Forward / 调仓权重演化",
}

_SINGLE_CHART_LABELS: dict[str, str] = {
    "price": "价格 / 均线 / 布林带与成交量",
    "rolling": "滚动波动率、VaR 与技术动量",
    "drawdown": "历史水下回撤曲线",
    "normalized": "区间归一化净值 (Base=1)",
}

_WORKSPACE_SINGLE_KEY = "_workspace_single"
_WORKSPACE_COMPARE_KEY = "_workspace_compare"
_WORKSPACE_PORTFOLIO_KEY = "_workspace_portfolio"
_LEGACY_RESULTS_KEY = "_analysis_results"
_EMPTY_LEG_MSG = "每行需为 `资产类型,代码`、`资产类型|代码` 或 `资产类型:代码`。"


# ---------------------------------------------------------------------------
# Caching & Unified Loader Bridges
# ---------------------------------------------------------------------------


def _cache_data(ttl: int = 3600) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Use ``st.cache_data`` inside Streamlit, or provide a no-op ``.clear()`` fallback."""
    if st is None:

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func.clear = lambda: None  # type: ignore[attr-defined]
            return func

        return _decorator
    return st.cache_data(ttl=ttl, show_spinner=False)


# Bind cache directly onto asset_loader functions so __module__ matches in both `import app` and AppTest.
load_asset_data = _cache_data(ttl=3600)(_asset_loader_mod.load_asset_data)
load_multiple_etfs_close = _cache_data(ttl=3600)(_asset_loader_mod.load_multiple_etfs_close)


@_cache_data(ttl=3600)
def load_multi_leg_close_wide(
    legs_tuple: tuple[tuple[str, str], ...],
    start_date: str,
    end_date: str,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible wide close loader delegating each leg to cached ``load_asset_data``."""
    return _asset_loader_mod.load_multi_leg_close_wide(
        legs_tuple=legs_tuple,
        start_date=start_date,
        end_date=end_date,
        proxy_url=proxy_url,
        loader=load_asset_data,
    )


def _date_str(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _make_service_loaders(
    *, force_refresh: bool
) -> tuple[Callable[..., pd.DataFrame], Callable[[str, object, object], pd.DataFrame]]:
    """Build service-compatible ``(asset_loader, fx_loader)`` respecting cache & monkeypatches."""

    def _service_asset_loader(
        asset_type: str,
        symbol: str,
        start_date: object,
        end_date: object,
        proxy_url: str | None = None,
        adjust: str | None = None,
        force_refresh: bool = force_refresh,
    ) -> pd.DataFrame:
        s_str = _date_str(start_date)
        e_str = _date_str(end_date)
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
        return load_asset_data(
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
        s_str = _date_str(start_date)
        e_str = _date_str(end_date)
        if force_refresh:
            return _asset_loader_mod.load_asset_data(
                asset_type="fx",
                symbol=symbol,
                start_date=s_str,
                end_date=e_str,
                force_refresh=True,
            )
        return load_asset_data(
            asset_type="fx",
            symbol=symbol,
            start_date=s_str,
            end_date=e_str,
        )

    return _service_asset_loader, _service_fx_loader


# ---------------------------------------------------------------------------
# Pure Parsing, Validation, Quality Formatting & Table Helpers
# ---------------------------------------------------------------------------


def _split_ui_leg_token(segment: str) -> tuple[str, str]:
    """Parse ``type,symbol``, ``type|symbol``, or ``type:symbol`` (preserving slashes)."""
    cleaned = str(segment).strip()
    if not cleaned:
        raise ValueError("空的标的行。")
    for sep in (",", "|", ":"):
        if sep in cleaned:
            raw_type, raw_sym = cleaned.split(sep, 1)
            asset_key = raw_type.strip().lower()
            sym = raw_sym.strip()
            if not asset_key:
                raise ValueError(f"资产类型不能为空：{segment!r}")
            if not sym:
                raise ValueError(f"代码不能为空：{segment!r}")
            if asset_key not in SUPPORTED_ASSET_TYPES:
                supported = ", ".join(SUPPORTED_ASSET_TYPES)
                raise ValueError(f"不支持的资产类型 {raw_type!r}；可选：{supported}")
            return asset_key, normalize_symbol_alias(asset_key, sym)
    raise ValueError(f"{_EMPTY_LEG_MSG} 无效输入：{segment!r}")


def _parse_ui_multi_asset_specs(text: str) -> tuple[AssetSpec, ...]:
    """Parse textarea asset list into deduplicated ``AssetSpec`` tuple (min 2 assets)."""
    segments: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for part in line.split(";"):
            tok = part.strip()
            if tok:
                segments.append(tok)

    ordered: list[AssetSpec] = []
    seen: set[tuple[str, str]] = set()
    for seg in segments:
        asset_key, sym = _split_ui_leg_token(seg)
        dedup_key = (asset_key, sym)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        ordered.append(AssetSpec(asset_type=asset_key, symbol=sym))

    if len(ordered) < 2:
        raise ValueError("至少需要 2 个不重复标的（支持 `类型,代码` 或 `类型:代码`）。")
    return tuple(ordered)


def _parse_ui_benchmark_spec(raw: str) -> AssetSpec | None:
    """Parse optional benchmark string (`etf:510300` or `etf,510300`), or None when blank."""
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    asset_key, sym = _split_ui_leg_token(cleaned)
    return AssetSpec(asset_type=asset_key, symbol=sym)


def _parse_ui_custom_weights(raw: str) -> tuple[float, ...] | None:
    """Parse comma/semicolon/space-separated non-negative custom weights matching CLI semantics."""
    cleaned = str(raw).strip()
    if not cleaned:
        raise ValueError(
            "自定义权重策略 (custom) 需要输入与标的数一致的权重，例如 `0.4, 0.3, 0.3`。"
        )
    parts = [p for p in cleaned.replace(";", ",").replace(" ", ",").split(",") if p != ""]
    if not parts:
        raise ValueError("请输入有效的数字权重，例如 `0.5, 0.5`。")
    weights: list[float] = []
    for part in parts:
        try:
            val = float(part)
        except ValueError as exc:
            raise ValueError(f"权重包含无效数字：{part!r}") from exc
        if not math.isfinite(val):
            raise ValueError(f"权重不能为 NaN 或 Inf：{part!r}")
        weights.append(val)
    return tuple(weights)


def _extract_quality_counts(
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


def _format_data_quality_warning(
    obj: AnalysisResult | QualityReport | pd.DataFrame | Mapping[str, Any] | None,
    legacy_metrics: Mapping[str, Any] | None = None,
) -> str:
    """Build user-facing Chinese data quality warning when rows were dropped, filled, or anomalous."""
    dropped, filled, missing_vol, non_pos, neg_vol, policy = _extract_quality_counts(
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


def _inject_ui_theme_css() -> None:
    """Apply lightweight Light/Dark-safe responsive styling."""
    if st is None:
        return
    st.markdown(
        """
        <style>
        .quant-hero {
            padding: 0.9rem 1.15rem;
            border-radius: 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.24);
            background: rgba(128, 128, 128, 0.06);
            margin-bottom: 0.75rem;
        }
        .quant-hero-title {
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0 0 0.25rem 0;
            line-height: 1.25;
        }
        .quant-hero-sub {
            font-size: 0.92rem;
            opacity: 0.86;
            margin: 0;
        }
        .quant-pill {
            display: inline-block;
            padding: 0.16rem 0.55rem;
            margin: 0.35rem 0.35rem 0 0;
            border-radius: 999px;
            font-size: 0.78rem;
            border: 1px solid rgba(128, 128, 128, 0.28);
            background: rgba(128, 128, 128, 0.10);
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 0.65rem;
            padding: 0.65rem 0.85rem;
            background: rgba(128, 128, 128, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _workspace_state(key: str) -> dict[str, Any]:
    """Retrieve or initialize an isolated workspace dict in ``st.session_state``."""
    assert st is not None
    state = st.session_state.get(key)
    if not isinstance(state, dict):
        state = {
            "signature": None,
            "fetch_signature": None,
            "result": None,
            "figures": None,
            "skipped": None,
            "exports": None,
            "error": None,
        }
        st.session_state[key] = state
    return state


def _reset_workspace_outputs(state: dict[str, Any], *, keep_error: bool = False) -> None:
    state["result"] = None
    state["fetch_signature"] = None
    state["figures"] = None
    state["skipped"] = None
    state["exports"] = None
    if not keep_error:
        state["error"] = None


def _get_or_build_figures(
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    chart_types: Sequence[str] | None = None,
) -> tuple[dict[str, tuple[str, str, Any]] | None, dict[str, str]]:
    """Build and memoize Plotly figures on workspace state so reruns avoid re-plotting."""
    assert st is not None
    requested_key = tuple(chart_types) if chart_types is not None else None
    cached_figs = state.get("figures")
    cached_req = state.get("figures_requested")
    if isinstance(cached_figs, dict) and cached_req == requested_key:
        return cached_figs, dict(state.get("skipped") or {})

    try:
        figs, skipped = build_plotly_figures(result, chart_types=chart_types)
    except ImportError as exc:
        st.error(str(exc))
        return None, {}
    state["figures"] = figs
    state["skipped"] = skipped
    state["figures_requested"] = requested_key
    return figs, skipped


def _cleaned_ohlcv_to_csv_bytes_and_filename(
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
    s_str = _date_str(start_date)
    e_str = _date_str(end_date)
    filename = f"{asset_type}_{safe_sym}_{s_str}_{e_str}_ohlcv.csv"
    return csv_bytes, filename


def _get_or_build_report_exports(
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    skipped_charts: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Memoize Markdown, HTML, JSON, and UTF-8-SIG CSV exports on workspace state."""
    cached = state.get("exports")
    if isinstance(cached, dict) and cached:
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
        primary_csv, _ = _cleaned_ohlcv_to_csv_bytes_and_filename(
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
    return exports


def _render_report_and_download_section(
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    file_prefix: str,
    skipped_charts: Mapping[str, str] | None,
    key_prefix: str,
) -> None:
    """Render unified Markdown/HTML/JSON/CSV download buttons and inline report preview."""
    assert st is not None
    exports = _get_or_build_report_exports(state, result, skipped_charts=skipped_charts)
    st.subheader("📑 统一研究报告与数据导出")

    cols = st.columns(4)
    cols[0].download_button(
        label="📄 Markdown 报告 (.md)",
        data=exports["md"].encode("utf-8"),
        file_name=f"{file_prefix}_report.md",
        mime="text/markdown",
        key=f"{key_prefix}_dl_md",
        width="stretch",
    )
    cols[1].download_button(
        label="🌐 HTML 报告 (.html)",
        data=exports["html"].encode("utf-8"),
        file_name=f"{file_prefix}_report.html",
        mime="text/html",
        key=f"{key_prefix}_dl_html",
        width="stretch",
    )
    cols[2].download_button(
        label="🧾 结构化 JSON (.json)",
        data=exports["json"].encode("utf-8"),
        file_name=f"{file_prefix}_result.json",
        mime="application/json",
        key=f"{key_prefix}_dl_json",
        width="stretch",
    )
    csv_label = (
        "📊 OHLCV 指标明细 (.csv)" if isinstance(result, AnalysisResult) else "📊 对齐价格 (.csv)"
    )
    cols[3].download_button(
        label=csv_label,
        data=exports["csv"],
        file_name=f"{file_prefix}_data.csv",
        mime="text/csv",
        key=f"{key_prefix}_dl_csv",
        width="stretch",
    )

    if isinstance(result, MultiAssetResult):
        extra_csvs = exports.get("extra_csvs") or {}
        if extra_csvs:
            dl_cols = st.columns(len(extra_csvs))
            label_map = {
                "weights": "⚖️ 权重与风险贡献 (.csv)",
                "backtest": "📈 回测净值序列 (.csv)",
                "returns": "📉 对齐收益率 (.csv)",
                "correlation": "🔗 相关系数矩阵 (.csv)",
                "covariance": "📐 年化协方差矩阵 (.csv)",
            }
            for col, (frame_key, payload) in zip(dl_cols, extra_csvs.items(), strict=False):
                col.download_button(
                    label=label_map.get(frame_key, f"{frame_key}.csv"),
                    data=payload,
                    file_name=f"{file_prefix}_{frame_key}.csv",
                    mime="text/csv",
                    key=f"{key_prefix}_dl_extra_{frame_key}",
                    width="stretch",
                )

    with st.expander("👁️ 在线预览研究报告 (Markdown)", expanded=False):
        st.markdown(exports["md"])


def _render_chart_bundle(
    figs: Mapping[str, tuple[str, str, Any]],
    skipped: Mapping[str, str],
    order: Sequence[str],
    labels: Mapping[str, str],
    *,
    key_prefix: str,
) -> None:
    """Render ordered Plotly charts and surface any skipped-chart reasons cleanly."""
    assert st is not None
    available = [ct for ct in order if ct in figs]
    for idx, ct in enumerate(available):
        title, subtitle, fig = figs[ct]
        heading = labels.get(ct, title)
        st.markdown(f"**{heading}**")
        if subtitle:
            st.caption(subtitle)
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_plotly_{ct}_{idx}")

    if skipped:
        reasons = [
            f"`{labels.get(ct, ct)}` 已跳过（{format_reason_label(reason)}）"
            for ct, reason in skipped.items()
        ]
        st.info("图表口径限制提示：" + "；".join(reasons))


def _render_quality_and_warnings(
    warnings_list: Sequence[str],
    *,
    single_result: AnalysisResult | None = None,
) -> None:
    """Display formatted data-quality banner and deduplicated service warnings."""
    assert st is not None
    shown_messages: set[str] = set()
    if single_result is not None:
        formatted_q = _format_data_quality_warning(single_result)
        if formatted_q:
            st.warning(formatted_q)
            shown_messages.add(formatted_q)

    for raw_warn in warnings_list:
        msg = str(raw_warn).strip()
        if not msg or msg in shown_messages:
            continue
        # Suppress raw English summary duplicate when clean Chinese quality warning already shown.
        if (
            single_result is not None
            and shown_messages
            and msg.startswith("数据质量提示：dropped=")
        ):
            continue
        shown_messages.add(msg)
        st.warning(msg)


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------


def _sync_symbol_with_asset_type() -> None:
    """Update symbol & adaptive trading_days smoothly when user switches ``asset_type``."""
    assert st is not None
    new_type = str(st.session_state.get("asset_type", "etf"))
    prev_type = str(st.session_state.get("_prev_asset_type", "etf"))
    current_sym = str(st.session_state.get("symbol_input", "")).strip()
    prev_default = DEFAULT_SYMBOLS.get(prev_type, "")
    if not current_sym or current_sym == prev_default:
        st.session_state["symbol_input"] = DEFAULT_SYMBOLS.get(new_type, "")

    prev_td = get_asset_trading_days(prev_type)
    cur_td = st.session_state.get("single_trading_days_input")
    if cur_td is None or int(cur_td) == prev_td:
        st.session_state["single_trading_days_input"] = get_asset_trading_days(new_type)

    st.session_state["_prev_asset_type"] = new_type


def _apply_compare_preset() -> None:
    assert st is not None
    preset = str(st.session_state.get("compare_preset_template", ""))
    rows = PRESET_PORTFOLIOS.get(preset)
    if rows:
        st.session_state["compare_assets_input"] = "\n".join(rows)


def _apply_portfolio_preset() -> None:
    assert st is not None
    preset = str(st.session_state.get("tab3_preset_template", ""))
    rows = PRESET_PORTFOLIOS.get(preset)
    if rows:
        st.session_state["tab3_ef_legs_input"] = "\n".join(rows)


def _render_sidebar() -> dict[str, Any]:
    """Render shared runtime + Single-Asset primary controls cleanly in the sidebar."""
    assert st is not None
    default_start, default_end = resolve_analysis_date_window(None, None)

    st.sidebar.header("⚙️ 全局与单资产控制")
    start_val = st.sidebar.date_input(
        "开始日期",
        value=default_start.date(),
        key="start_date_input",
    )
    end_val = st.sidebar.date_input(
        "结束日期",
        value=default_end.date(),
        key="end_date_input",
    )

    date_error: str | None = None
    start_str: str | None = None
    end_str: str | None = None
    if (
        start_val is None
        or end_val is None
        or isinstance(start_val, tuple)
        or isinstance(end_val, tuple)
    ):
        date_error = "请选择开始和结束日期。"
    elif isinstance(start_val, dt.date) and isinstance(end_val, dt.date):
        if start_val > end_val:
            date_error = "开始日期不能晚于结束日期，请调整后再运行。"
        else:
            start_str = start_val.strftime("%Y-%m-%d")
            end_str = end_val.strftime("%Y-%m-%d")
    else:
        date_error = "请选择开始和结束日期。"

    if date_error:
        st.sidebar.error(date_error)
        st.error(date_error)

    rf_pct = float(
        st.sidebar.number_input(
            "无风险年化利率 Rf (%)",
            min_value=0.0,
            max_value=20.0,
            value=1.6,
            step=0.1,
            key="rf_pct_input",
            help="用于夏普、Sortino、特雷诺及切线组合计算；默认 1.6% 与 CLI 保持一致。",
        )
    )

    with st.sidebar.expander("🌐 网络代理、缓存与稳定币口径", expanded=False):
        proxy_raw = st.text_input(
            "HTTP/SOCKS 代理地址 (可选)",
            value="",
            placeholder="http://127.0.0.1:7890",
            key="proxy_url_input",
        )
        force_refresh = st.checkbox(
            "跳过本地缓存（强制在线刷新）",
            value=False,
            key="force_refresh_input",
        )
        allow_usdt_peg = st.checkbox(
            "允许加密货币按 USDT=USD (1:1) 近似折算",
            value=False,
            key="allow_usdt_peg_input",
            help="默认严格抓取实际 USDT/USD 汇率；若不可得时可显式勾选允许 1:1 近似。",
        )

    st.sidebar.divider()
    st.sidebar.subheader("📈 单资产探索参数")

    if "asset_type" not in st.session_state:
        st.session_state["asset_type"] = "etf"
    if "_prev_asset_type" not in st.session_state:
        st.session_state["_prev_asset_type"] = st.session_state["asset_type"]
    if "symbol_input" not in st.session_state:
        st.session_state["symbol_input"] = DEFAULT_SYMBOLS["etf"]
    if "single_trading_days_input" not in st.session_state:
        st.session_state["single_trading_days_input"] = get_asset_trading_days("etf")

    asset_type = str(
        st.sidebar.selectbox(
            "资产类别",
            options=SUPPORTED_ASSET_TYPES,
            format_func=lambda key: ASSET_LABELS.get(key, key),
            key="asset_type",
            on_change=_sync_symbol_with_asset_type,
        )
    )
    symbol = str(
        st.sidebar.text_input(
            "资产代码",
            key="symbol_input",
            help="支持基金/ETF/A股/美股/港股/全球指数/加密货币/汇率/债券/REITs。",
        )
    ).strip()

    with st.sidebar.expander("💡 常见代码格式与品种提示", expanded=False):
        st.markdown(format_asset_guide_markdown(asset_type))

    supported_adjusts = SUPPORTED_ADJUSTMENTS.get(asset_type)
    if supported_adjusts:
        cur_adj = st.session_state.get("adjust_input")
        if cur_adj not in supported_adjusts:
            st.session_state["adjust_input"] = supported_adjusts[0]
        adjust: str | None = str(
            st.sidebar.selectbox(
                "复权口径",
                options=supported_adjusts,
                format_func=lambda code: _ADJUST_LABELS.get(code, code),
                key="adjust_input",
            )
        )
    else:
        st.sidebar.selectbox(
            "复权口径",
            options=("该资产不适用复权",),
            index=0,
            disabled=True,
            key="adjust_input_disabled",
        )
        adjust = None

    single_base_currency = str(
        st.sidebar.selectbox(
            "估值基准货币",
            options=("LOCAL", "CNY", "USD"),
            index=0,
            format_func=lambda c: _BASE_CURRENCY_LABELS.get(c, c),
            key="single_base_currency_input",
            help="默认保持资产原生货币 LOCAL（与 `quant analyze` 默认口径一致）。",
        )
    )

    with st.sidebar.expander("🔧 单资产高级技术与统计参数", expanded=False):
        single_return_convention = str(
            st.selectbox(
                "收益率计算口径",
                options=SUPPORTED_RETURN_CONVENTIONS,
                index=0,
                format_func=lambda m: _RETURN_CONVENTION_LABELS.get(m, m),
                key="single_return_convention_input",
                help="单资产默认使用时间可加的对数收益率 (log)，可切换为简单收益率。",
            )
        )
        sma_short = int(
            st.number_input("短期均线窗口 (SMA Short)", 2, 120, 20, step=1, key="sma_short_input")
        )
        sma_long = int(
            st.number_input("长期均线窗口 (SMA Long)", 5, 250, 60, step=5, key="sma_long_input")
        )
        bb_window = int(
            st.number_input("布林带滚动窗口 (BB Window)", 2, 120, 20, step=1, key="bb_window_input")
        )
        single_trading_days = int(
            st.number_input(
                "年化基准天数 (Trading Days)",
                min_value=1,
                max_value=366,
                step=1,
                key="single_trading_days_input",
                help="股票/ETF 默认 252，外汇默认 260，加密货币默认 365。",
            )
        )
        include_momentum = bool(
            st.checkbox(
                "计算并绘制动量副图 (RSI-14 / MACD)", value=False, key="include_momentum_input"
            )
        )

    symbol_missing = (not symbol) and (asset_type != "bond_us_yield")
    if symbol_missing:
        st.sidebar.warning("请输入有效的资产代码（美债利率支持留空默认为 10Y）。")

    single_clicked = st.sidebar.button(
        "🔍 运行单资产分析",
        key="single_run_btn",
        type="primary",
        width="stretch",
        disabled=bool(date_error or symbol_missing),
    )

    return {
        "dates_valid": date_error is None,
        "date_error": date_error,
        "start_str": start_str,
        "end_str": end_str,
        "rf_rate": rf_pct / 100.0,
        "proxy_url": proxy_raw.strip() or None,
        "force_refresh": force_refresh,
        "allow_usdt_peg": allow_usdt_peg,
        "asset_type": asset_type,
        "symbol": symbol,
        "adjust": adjust,
        "single_base_currency": single_base_currency,
        "single_return_convention": single_return_convention,
        "sma_short": sma_short,
        "sma_long": sma_long,
        "bb_window": bb_window,
        "single_trading_days": single_trading_days,
        "include_momentum": include_momentum,
        "single_clicked": single_clicked,
    }


# ---------------------------------------------------------------------------
# Workspace 1: Single-Asset Exploration (`analyze_single_asset`)
# ---------------------------------------------------------------------------


def _render_single_asset_kpis(result: AnalysisResult) -> None:
    """Render unified headline KPIs and full metric details matching M1-M4 contracts."""
    assert st is not None
    metrics = result.metrics
    reasons = result.metric_reasons
    series_kind = result.asset.series_kind
    output_ccy = result.asset.output_currency or result.asset.native_currency

    if series_kind == "yield_level":
        st.info(
            "利率水平序列提示：当前标的为利率/收益率水平 (%)，并非总收益资产价格，"
            "按金融严谨性禁止计算复利回报、CAGR、回撤百分比或夏普比率。"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "最新利率水平",
            format_metric_value(
                "latest_level",
                metrics.get("latest_level"),
                series_kind=series_kind,
                output_currency=output_ccy,
            ),
        )
        c2.metric(
            "区间绝对变动",
            format_metric_value(
                "level_change",
                metrics.get("level_change"),
                series_kind=series_kind,
                output_currency=output_ccy,
            ),
        )
        lo = format_metric_value(
            "level_min",
            metrics.get("level_min"),
            series_kind=series_kind,
            output_currency=output_ccy,
        )
        hi = format_metric_value(
            "level_max",
            metrics.get("level_max"),
            series_kind=series_kind,
            output_currency=output_ccy,
        )
        c3.metric("区间高低范围", f"{lo} ~ {hi}")

        rows = [
            {
                "指标": label,
                "数值": "N/A（口径禁用）",
                "禁用原因": format_reason_label(reasons.get(key, "yield_level_not_return")),
            }
            for key, label in (
                ("cumulative_return", "累计收益率"),
                ("cagr", "年化复合收益 (CAGR)"),
                ("max_drawdown", "最大回撤"),
                ("annualized_volatility", "年化波动率"),
                ("sharpe_ratio", "夏普比率"),
                ("sortino_ratio", "Sortino 比率"),
                ("calmar_ratio", "Calmar 比率"),
            )
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        return

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "CAGR",
        format_percent(metrics.get("cagr"))
        if is_valid_number(metrics.get("cagr"))
        else format_reason_label(reasons.get("cagr")),
    )
    c2.metric(
        "最大回撤",
        format_drawdown_percent(metrics.get("max_drawdown"))
        if is_valid_number(metrics.get("max_drawdown"))
        else format_reason_label(reasons.get("max_drawdown")),
    )
    c3.metric(
        "夏普比率",
        format_ratio(metrics.get("sharpe_ratio"), decimals=3)
        if is_valid_number(metrics.get("sharpe_ratio"))
        else format_reason_label(reasons.get("sharpe_ratio")),
    )

    detail_rows = [
        {
            "累计收益率": format_percent(metrics.get("cumulative_return")),
            "年化波动率": format_percent(metrics.get("annualized_volatility")),
            "Sortino 比率": format_ratio(metrics.get("sortino_ratio"), decimals=3),
            "Calmar 比率": format_ratio(metrics.get("calmar_ratio"), decimals=3),
            "95% 日 VaR": format_percent(metrics.get("var_95")),
            "95% 日 CVaR": format_percent(metrics.get("cvar_95")),
            "最新收盘价": format_price_with_unit(
                metrics.get("latest_close"),
                series_kind=series_kind,
                output_currency=output_ccy,
            ),
            "均线/均值状态": str(metrics.get("latest_signal", "N/A")),
            "年化天数 / 收益口径": f"{result.asset.trading_days} / {metrics.get('return_convention', 'log')}",
        }
    ]
    st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)


def _render_single_advanced_explorer(cfg: dict[str, Any], state: dict[str, Any]) -> None:
    """Collapsible Single-Asset Normalize + Asset-Adaptive GBM Monte Carlo simulator."""
    assert st is not None
    with st.expander("🎲 高级探索：单资产基准归一化与蒙特卡洛（GBM）路径模拟", expanded=False):
        st.caption(
            "参数前瞻模拟工具（非历史已实现绩效）：基于对数收益率的漂移项与波动率估算几何布朗运动路径，"
            f"年化基准自动适配当前资产类别（{cfg['asset_type']} → {get_asset_trading_days(cfg['asset_type'])} 期/年）。"
        )
        col_l, col_r = st.columns([1, 2])
        with col_l:
            norm_clicked = st.button(
                "📈 生成归一化净值曲线 (Base=1)",
                key="single_btn_normalize",
                width="stretch",
                disabled=bool(
                    not cfg["dates_valid"]
                    or (not cfg["symbol"] and cfg["asset_type"] != "bond_us_yield")
                ),
            )
            gbm_days = int(
                st.number_input(
                    "模拟未来期数 (Days/Steps)",
                    min_value=10,
                    max_value=756,
                    value=int(get_asset_trading_days(cfg["asset_type"])),
                    step=21,
                    key="single_gbm_days",
                )
            )
            gbm_sims = int(
                st.number_input(
                    "蒙特卡洛模拟路径数",
                    min_value=20,
                    max_value=2000,
                    value=200,
                    step=20,
                    key="single_gbm_sims",
                )
            )
            gbm_clicked = st.button(
                "🎲 运行 GBM 蒙特卡洛模拟",
                key="single_btn_gbm",
                width="stretch",
                disabled=bool(
                    not cfg["dates_valid"]
                    or (not cfg["symbol"] and cfg["asset_type"] != "bond_us_yield")
                ),
            )

        sub_sig = (
            cfg["asset_type"],
            cfg["symbol"],
            cfg["adjust"],
            cfg["single_base_currency"],
            cfg["start_str"],
            cfg["end_str"],
            cfg["allow_usdt_peg"],
        )
        if state.get("advanced_sig") != sub_sig:
            state["advanced_sig"] = sub_sig
            state["norm_fig"] = None
            state["norm_error"] = None
            state["gbm_result"] = None
            state["gbm_fig"] = None
            state["gbm_error"] = None
        if state.get("gbm_params") != (gbm_days, gbm_sims):
            state["gbm_params"] = (gbm_days, gbm_sims)
            state["gbm_result"] = None
            state["gbm_fig"] = None

        def _obtain_price_frame() -> pd.DataFrame:
            existing: AnalysisResult | None = state.get("result")
            fetch_sig = (
                cfg["asset_type"],
                cfg["symbol"] or "10Y",
                cfg["adjust"],
                cfg["single_base_currency"],
                str(cfg["start_str"]),
                str(cfg["end_str"]),
                cfg["allow_usdt_peg"],
            )
            if (
                not cfg["force_refresh"]
                and existing is not None
                and existing.data_frame is not None
                and state.get("fetch_signature") == fetch_sig
            ):
                if existing.asset.series_kind == "yield_level":
                    raise ValueError("利率水平序列不支持净值归一化或 GBM 价格路径模拟。")
                return existing.data_frame

            loader_fn, fx_fn = _make_service_loaders(force_refresh=cfg["force_refresh"])
            fetched = fetch_single_asset(
                AnalysisRequest(
                    asset_type=cfg["asset_type"],
                    symbol=cfg["symbol"] or "10Y",
                    start_date=str(cfg["start_str"]),
                    end_date=str(cfg["end_str"]),
                    adjust=cfg["adjust"],
                    base_currency=cfg["single_base_currency"],
                    allow_usdt_peg=cfg["allow_usdt_peg"],
                    force_refresh=cfg["force_refresh"],
                    proxy_url=cfg["proxy_url"],
                ),
                loader=loader_fn,
                fx_loader=fx_fn,
            )
            if fetched.asset.series_kind == "yield_level":
                raise ValueError("利率水平序列不支持净值归一化或 GBM 价格路径模拟。")
            return fetched.data_frame

        if norm_clicked:
            with st.spinner("正在计算归一化净值曲线..."):
                try:
                    df_norm = _obtain_price_frame()
                    norm_s = normalize_close_to_base_one(df_norm)
                    state["norm_fig"] = build_tab3_normalize_figure(df_norm.index, norm_s)
                    state["norm_error"] = None
                except Exception as exc:
                    state["norm_fig"] = None
                    state["norm_error"] = format_diagnosis_user_message(exc)

        if gbm_clicked:
            with st.spinner("正在模拟几何布朗运动（GBM）路径..."):
                try:
                    df_gbm = _obtain_price_frame()
                    td = get_asset_trading_days(cfg["asset_type"])
                    gbm_res = run_monte_carlo_gbm(
                        df_gbm,
                        days=gbm_days,
                        num_simulations=gbm_sims,
                        random_state=42,
                        trading_days=td,
                    )
                    state["gbm_result"] = gbm_res
                    state["gbm_fig"] = build_tab3_gbm_figure(gbm_res)
                    state["gbm_error"] = None
                except Exception as exc:
                    state["gbm_result"] = None
                    state["gbm_fig"] = None
                    state["gbm_error"] = format_diagnosis_user_message(exc)

        with col_r:
            if state.get("norm_error"):
                st.error(state["norm_error"])
            if state.get("norm_fig") is not None:
                st.plotly_chart(state["norm_fig"], width="stretch", key="single_norm_chart")

            if state.get("gbm_error"):
                st.error(state["gbm_error"])
            gbm_out = state.get("gbm_result")
            if gbm_out is not None and state.get("gbm_fig") is not None:
                st.plotly_chart(state["gbm_fig"], width="stretch", key="single_gbm_chart")
                q_row = {
                    "当前初值 S0": f"{gbm_out.s0:.2f}",
                    "P5 (悲观)": f"{float(gbm_out.quantiles_p05[-1]):.2f}",
                    "P50 (中位数)": f"{float(gbm_out.quantiles_p50[-1]):.2f}",
                    "P95 (乐观)": f"{float(gbm_out.quantiles_p95[-1]):.2f}",
                    "日连续漂移 μ / 波动 σ": f"{gbm_out.mu_daily:.5f} / {gbm_out.sigma_daily:.5f}",
                }
                st.dataframe(pd.DataFrame([q_row]), width="stretch", hide_index=True)


def _render_single_asset_workspace(cfg: dict[str, Any]) -> None:
    """Render Workspace 1: Single-Asset Exploration backed by ``analyze_single_asset``."""
    assert st is not None
    state = _workspace_state(_WORKSPACE_SINGLE_KEY)

    cur_sig = (
        cfg["asset_type"],
        cfg["symbol"],
        cfg["adjust"],
        cfg["single_base_currency"],
        cfg["single_return_convention"],
        cfg["sma_short"],
        cfg["sma_long"],
        cfg["bb_window"],
        cfg["single_trading_days"],
        cfg["include_momentum"],
        round(cfg["rf_rate"], 8),
        cfg["start_str"],
        cfg["end_str"],
        cfg["allow_usdt_peg"],
    )
    if (state.get("result") is not None or state.get("error") is not None) and state.get(
        "signature"
    ) != cur_sig:
        _reset_workspace_outputs(state, keep_error=False)

    if cfg["single_clicked"]:
        if not cfg["dates_valid"]:
            _reset_workspace_outputs(state, keep_error=True)
            state["error"] = cfg["date_error"]
        elif cfg["sma_short"] >= cfg["sma_long"]:
            _reset_workspace_outputs(state, keep_error=True)
            state["error"] = (
                "参数错误：短期均线窗口 (SMA Short) 必须严格小于长期均线窗口 (SMA Long)。"
            )
        else:
            apply_default_network_proxy_policy()
            loader_fn, fx_fn = _make_service_loaders(force_refresh=cfg["force_refresh"])
            req = AnalysisRequest(
                asset_type=cfg["asset_type"],
                symbol=cfg["symbol"] or "10Y",
                start_date=str(cfg["start_str"]),
                end_date=str(cfg["end_str"]),
                adjust=cfg["adjust"],
                base_currency=cfg["single_base_currency"],
                allow_usdt_peg=cfg["allow_usdt_peg"],
                risk_free_rate=cfg["rf_rate"],
                trading_days=cfg["single_trading_days"],
                return_convention=cfg["single_return_convention"],
                sma_short=cfg["sma_short"],
                sma_long=cfg["sma_long"],
                bb_window=cfg["bb_window"],
                include_momentum=cfg["include_momentum"],
                force_refresh=cfg["force_refresh"],
                proxy_url=cfg["proxy_url"],
            )
            with st.spinner("正在执行单资产定量分析..."):
                try:
                    res = analyze_single_asset(req, loader=loader_fn, fx_loader=fx_fn)
                except Exception as exc:
                    _reset_workspace_outputs(state, keep_error=False)
                    state["signature"] = cur_sig
                    state["error"] = format_diagnosis_user_message(exc)
                else:
                    _reset_workspace_outputs(state, keep_error=False)
                    state["signature"] = cur_sig
                    state["fetch_signature"] = (
                        cfg["asset_type"],
                        cfg["symbol"] or "10Y",
                        cfg["adjust"],
                        cfg["single_base_currency"],
                        str(cfg["start_str"]),
                        str(cfg["end_str"]),
                        cfg["allow_usdt_peg"],
                    )
                    state["result"] = res

    if state.get("error"):
        st.error(state["error"])

    result: AnalysisResult | None = state.get("result")
    if result is None:
        st.info(
            "请在左侧边栏配置资产代码与分析日期后，点击 **「🔍 运行单资产分析」** 查看诊断面板与报告。"
        )
        _render_single_advanced_explorer(cfg, state)
        return

    cache_label = (
        f"本地缓存命中（{result.asset.fetched_at or 'disk cache'}）"
        if result.asset.cache_hit
        else f"在线实时拉取（{result.asset.fetched_at or 'fresh'}）"
    )
    ccy_note = (
        f"{result.asset.native_currency} → {result.asset.output_currency}（汇率 {result.asset.fx_pair}）"
        if result.asset.fx_converted
        else f"{result.asset.output_currency or result.asset.native_currency or 'N/A'}"
    )
    st.caption(
        f"**{ASSET_LABELS.get(result.asset.asset_type, result.asset.asset_type)} · "
        f"`{result.asset.canonical_symbol}`** | 样本：{result.quality.start_date} ~ {result.quality.end_date} "
        f"({result.quality.total_rows} 条) | 数据源：`{result.asset.data_source or 'unknown'}` | "
        f"货币：{ccy_note} | {cache_label}"
    )

    _render_quality_and_warnings(result.warnings, single_result=result)
    _render_single_asset_kpis(result)

    figs, skipped = _get_or_build_figures(state, result)
    if figs is not None:
        _render_chart_bundle(
            figs,
            skipped,
            order=("price", "rolling", "drawdown", "normalized"),
            labels=_SINGLE_CHART_LABELS,
            key_prefix="single",
        )

    with st.expander("🔍 数据质量审计、收益率口径与风险声明", expanded=False):
        basis_note = ASSET_RETURN_BASIS_NOTES.get(result.asset.return_basis, "")
        if basis_note:
            st.markdown(f"- **价格/收益口径 (`{result.asset.return_basis}`)**: {basis_note}")
        q_dict = {
            "样本总量": result.quality.total_rows,
            "清洗缺失策略": result.quality.missing_policy,
            "清理异常/缺失行数": result.quality.dropped_invalid_rows,
            "前向填补价格行数": result.quality.filled_price_rows,
            "缺失成交量补零行数": result.quality.missing_volume_rows,
            "零成交量行数": result.quality.zero_volume_rows,
            "OHLC 关系异常行数": result.quality.ohlc_anomaly_rows,
            "最大日历间隔 (天)": result.quality.max_calendar_gap_days,
        }
        st.dataframe(pd.DataFrame([q_dict]), width="stretch", hide_index=True)
        for disc in result.disclaimers:
            st.caption(f"• {disc}")

    _render_single_advanced_explorer(cfg, state)
    _render_report_and_download_section(
        state,
        result,
        file_prefix=f"{result.asset.asset_type}_{result.asset.canonical_symbol}".replace("/", "_"),
        skipped_charts=skipped,
        key_prefix="single",
    )


# ---------------------------------------------------------------------------
# Workspace 2: Multi-Asset Comparison (`compare_assets`)
# ---------------------------------------------------------------------------


def _render_compare_workspace(cfg: dict[str, Any]) -> None:
    """Render Workspace 2: Multi-Asset Comparison backed by ``compare_assets``."""
    assert st is not None
    state = _workspace_state(_WORKSPACE_COMPARE_KEY)

    if "compare_assets_input" not in st.session_state:
        first_preset = next(iter(PRESET_PORTFOLIOS.values()))
        st.session_state["compare_assets_input"] = "\n".join(first_preset)

    st.subheader("⚖️ 跨市场多资产历史对比与统计检验")
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.selectbox(
            "预设跨市场资产模板（可选）",
            options=tuple(PRESET_PORTFOLIOS.keys()),
            key="compare_preset_template",
            on_change=_apply_compare_preset,
        )
        assets_text = st.text_area(
            "对比资产列表（每行或分号分隔 `类型,代码` 或 `类型:代码`）",
            key="compare_assets_input",
            height=115,
            help="示例：etf,510300 / us_stock:AAPL / crypto,BTC/USDT。注：不支持将利率水平纳入净值对比。",
        )
        benchmark_text = st.text_input(
            "对比基准（可选，例如 `etf:510300` 或 `us_stock:SPY`）",
            value="",
            key="compare_benchmark_input",
            help="留空时不计算相对基准 Alpha/Beta、Tracking Error 与 Memmel Sharpe 差异检验。",
        )
    with col_right:
        calendar_mode = str(
            st.selectbox(
                "跨市场日历对齐方式",
                options=SUPPORTED_CALENDAR_MODES,
                index=0,
                format_func=lambda m: _CALENDAR_MODE_LABELS.get(m, m),
                key="compare_calendar_mode",
            )
        )
        base_currency = str(
            st.selectbox(
                "统一估值货币",
                options=("CNY", "USD", "LOCAL"),
                index=0,
                format_func=lambda c: _BASE_CURRENCY_LABELS.get(c, c),
                key="compare_base_currency",
                help="跨币种比较默认统一折算为 CNY（与 `quant compare` 一致）。",
            )
        )
        return_convention = str(
            st.selectbox(
                "比较收益率口径",
                options=("simple", "log"),
                index=0,
                format_func=lambda m: _RETURN_CONVENTION_LABELS.get(m, m),
                key="compare_return_convention",
            )
        )

    with st.expander("⚙️ 高级统计与滚动检验参数", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        cov_method = str(
            adv1.selectbox(
                "协方差估计方法",
                options=SUPPORTED_COV_METHODS,
                index=list(SUPPORTED_COV_METHODS).index("ledoit_wolf"),
                format_func=lambda m: _COV_METHOD_LABELS.get(m, m),
                key="compare_cov_method",
            )
        )
        rolling_window = int(
            adv2.number_input("滚动计算窗口期", 5, 252, 20, step=5, key="compare_rolling_window")
        )
        trading_days_override = int(
            adv3.number_input(
                "年化期数覆盖（0 = 自动）",
                min_value=0,
                max_value=366,
                value=0,
                step=1,
                key="compare_trading_days_override",
            )
        )

    run_clicked = st.button(
        "⚖️ 运行多资产比较",
        key="compare_run_btn",
        type="primary",
        width="stretch",
        disabled=not cfg["dates_valid"],
    )

    cur_sig = (
        assets_text,
        benchmark_text,
        calendar_mode,
        base_currency,
        return_convention,
        cov_method,
        rolling_window,
        trading_days_override,
        round(cfg["rf_rate"], 8),
        cfg["start_str"],
        cfg["end_str"],
        cfg["allow_usdt_peg"],
    )
    if (state.get("result") is not None or state.get("error") is not None) and state.get(
        "signature"
    ) != cur_sig:
        _reset_workspace_outputs(state, keep_error=False)

    if run_clicked:
        if not cfg["dates_valid"]:
            _reset_workspace_outputs(state, keep_error=True)
            state["error"] = cfg["date_error"]
        else:
            apply_default_network_proxy_policy()
            loader_fn, fx_fn = _make_service_loaders(force_refresh=cfg["force_refresh"])
            with st.spinner("正在对齐多市场日历、折算汇率并运行比较统计..."):
                try:
                    specs = _parse_ui_multi_asset_specs(assets_text)
                    bench_spec = _parse_ui_benchmark_spec(benchmark_text)
                    req = CompareRequest(
                        assets=specs,
                        start_date=str(cfg["start_str"]),
                        end_date=str(cfg["end_str"]),
                        benchmark=bench_spec,
                        calendar_mode=calendar_mode,
                        base_currency=base_currency,
                        allow_usdt_peg=cfg["allow_usdt_peg"],
                        risk_free_rate=cfg["rf_rate"],
                        trading_days=(
                            None if trading_days_override <= 0 else trading_days_override
                        ),
                        return_convention=return_convention,
                        rolling_window=rolling_window,
                        cov_method=cov_method,
                        force_refresh=cfg["force_refresh"],
                        proxy_url=cfg["proxy_url"],
                    )
                    res = compare_assets(req, loader=loader_fn, fx_loader=fx_fn)
                except Exception as exc:
                    _reset_workspace_outputs(state, keep_error=False)
                    state["signature"] = cur_sig
                    state["error"] = format_diagnosis_user_message(exc)
                else:
                    _reset_workspace_outputs(state, keep_error=False)
                    state["signature"] = cur_sig
                    state["result"] = res

    if state.get("error"):
        st.error(state["error"])

    result: MultiAssetResult | None = state.get("result")
    if result is None:
        st.info(
            "配置 2 个及以上对比资产并点击 **「⚖️ 运行多资产比较」**，即可生成归一化净值、分布检验与相关性面板。"
        )
        return

    summary = result.data_summary
    methodology = result.methodology
    _render_quality_and_warnings(result.warnings)

    m1, m2, m3 = st.columns(3)
    m1.metric("对齐估值样本数", format_integer(summary.get("valuation_points")))
    m2.metric("同期收益观测期数", format_integer(summary.get("concurrent_return_periods")))
    m3.metric(
        "有效年化期数 / 年",
        format_ratio(methodology.get("effective_periods_per_year"), decimals=1),
    )

    eff_start = summary.get("effective_start") or summary.get("requested_start") or ""
    eff_end = summary.get("effective_end") or summary.get("requested_end") or ""
    st.caption(
        f"估值货币：`{methodology.get('base_currency', base_currency)}` | "
        f"区间：{eff_start} ~ {eff_end} | "
        f"日历口径：`{methodology.get('calendar_mode', calendar_mode)}` | "
        f"协方差估计：`{methodology.get('cov_method', cov_method)}`"
    )

    st.markdown("#### 📊 历史实现绩效对照 (Historical Realized Performance)")
    per_asset = result.metrics.get("per_asset") or {}
    perf_rows = []
    for leg in result.asset_order:
        row_m = per_asset.get(leg, {})
        perf_rows.append(
            {
                "资产": leg,
                "累计收益": format_percent(row_m.get("cumulative_return")),
                "CAGR": format_percent(row_m.get("cagr")),
                "年化波动率": format_percent(row_m.get("annualized_volatility")),
                "最大回撤": format_drawdown_percent(row_m.get("max_drawdown")),
                "夏普比率": format_ratio(row_m.get("sharpe_ratio"), decimals=3),
                "Sortino": format_ratio(row_m.get("sortino_ratio"), decimals=3),
                "Calmar": format_ratio(row_m.get("calmar_ratio"), decimals=3),
                "95% 日 VaR": format_percent(row_m.get("var_95")),
                "95% 日 CVaR": format_percent(row_m.get("cvar_95")),
            }
        )
    st.dataframe(pd.DataFrame(perf_rows), width="stretch", hide_index=True)

    dist_map = result.metrics.get("distribution") or {}
    if dist_map:
        st.markdown("#### 🔬 收益分布统计推断 (Statistical Inference)")
        dist_rows = []
        for leg in result.asset_order:
            d = dist_map.get(leg, {})
            is_norm = d.get("is_normal_at_5pct")
            norm_label = (
                "未拒绝正态假设 (5%)"
                if is_norm is True
                else ("拒绝正态假设 (5%)" if is_norm is False else "N/A")
            )
            dist_rows.append(
                {
                    "资产": leg,
                    "期均值": format_percent(d.get("mean"), decimals=3),
                    "偏度 (Skew)": format_ratio(d.get("skewness"), decimals=3),
                    "超额峰度 (Kurt)": format_ratio(d.get("excess_kurtosis"), decimals=3),
                    "Jarque-Bera": format_ratio(d.get("jarque_bera_stat"), decimals=2),
                    "JB p-value": format_p_value(d.get("jarque_bera_p_value")),
                    "正态性检验": norm_label,
                    "胜率": format_percent(d.get("win_rate"), decimals=1),
                    "均值检验 p-value": format_p_value(d.get("mean_p_value")),
                }
            )
        st.dataframe(pd.DataFrame(dist_rows), width="stretch", hide_index=True)

    bench_comp = result.metrics.get("benchmark_comparison") or {}
    if result.benchmark and bench_comp:
        bench_name = result.benchmark.get("leg_name", "Benchmark")
        st.markdown(f"#### 🧭 相对基准归因与 Memmel Sharpe 检验 (基准：`{bench_name}`)")
        rel_rows = []
        for leg, rel in bench_comp.items():
            memmel = rel.get("memmel_sharpe_test") or {}
            rel_rows.append(
                {
                    "资产": leg,
                    "超额年化收益": format_percent(rel.get("excess_annualized_return")),
                    "跟踪误差 TE": format_percent(rel.get("tracking_error")),
                    "信息比率 IR": format_ratio(rel.get("information_ratio"), decimals=3),
                    "CAPM Beta": format_ratio(rel.get("beta"), decimals=3),
                    "Jensen Alpha": format_percent(rel.get("jensen_alpha")),
                    "Memmel Z / p-value": (
                        f"{format_ratio(memmel.get('z_stat'), decimals=2)} "
                        f"(p={format_p_value(memmel.get('p_value'))})"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rel_rows), width="stretch", hide_index=True)

    figs, skipped = _get_or_build_figures(state, result)
    if figs is not None:
        _render_chart_bundle(
            figs,
            skipped,
            order=("normalized", "drawdown", "rolling", "correlation"),
            labels=_COMPARE_CHART_LABELS,
            key_prefix="compare",
        )

    with st.expander("📐 相关性 / 协方差矩阵与方法学审计", expanded=False):
        c1, c2 = st.columns(2)
        corr_df = result.frames.get("correlation")
        cov_df = result.frames.get("covariance")
        if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
            c1.markdown("**同期收益率相关性矩阵**")
            c1.dataframe(corr_df.round(4), width="stretch")
        if isinstance(cov_df, pd.DataFrame) and not cov_df.empty:
            c2.markdown("**年化协方差矩阵**")
            c2.dataframe(cov_df.round(6), width="stretch")
        st.json(result.methodology)

    _render_report_and_download_section(
        state,
        result,
        file_prefix="compare_assets",
        skipped_charts=skipped,
        key_prefix="compare",
    )


# ---------------------------------------------------------------------------
# Workspace 3: Portfolio Research (`analyze_portfolio`)
# ---------------------------------------------------------------------------


def _render_portfolio_workspace(cfg: dict[str, Any]) -> None:
    """Render Workspace 3: Portfolio Optimization, Frontier & Backtesting via ``analyze_portfolio``."""
    assert st is not None
    state = _workspace_state(_WORKSPACE_PORTFOLIO_KEY)

    if "tab3_ef_legs_input" not in st.session_state:
        first_preset = next(iter(PRESET_PORTFOLIOS.values()))
        st.session_state["tab3_ef_legs_input"] = "\n".join(first_preset)

    st.subheader("🎯 投资组合优化、有效前沿与样本外回测研究")
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.selectbox(
            "预设组合模板（可选）",
            options=tuple(PRESET_PORTFOLIOS.keys()),
            key="tab3_preset_template",
            on_change=_apply_portfolio_preset,
        )
        legs_text = st.text_area(
            "组合标的列表（至少 2 个，`类型,代码` 或 `类型:代码`）",
            key="tab3_ef_legs_input",
            height=115,
        )
        benchmark_text = st.text_input(
            "组合对比基准（可选，例如 `etf:510300`）",
            value="",
            key="portfolio_benchmark_input",
        )
    with col_right:
        strategy = str(
            st.selectbox(
                "资产配置 / 优化策略",
                options=SUPPORTED_PORTFOLIO_STRATEGIES,
                index=list(SUPPORTED_PORTFOLIO_STRATEGIES).index("max_sharpe"),
                format_func=lambda s: _STRATEGY_LABELS.get(s, s),
                key="portfolio_strategy_input",
            )
        )
        custom_weights_raw = ""
        if strategy == "custom":
            custom_weights_raw = str(
                st.text_input(
                    "自定义资产权重（逗号分隔，按上方资产顺序）",
                    value="",
                    placeholder="0.4, 0.3, 0.3",
                    key="portfolio_custom_weights_input",
                )
            )
        validation_mode = str(
            st.selectbox(
                "回测与样本划分协议",
                options=SUPPORTED_VALIDATION_MODES,
                index=list(SUPPORTED_VALIDATION_MODES).index("train_test"),
                format_func=lambda m: _VALIDATION_MODE_LABELS.get(m, m),
                key="tab3_val_mode",
            )
        )
        c_cal, c_ccy = st.columns(2)
        calendar_mode = str(
            c_cal.selectbox(
                "日历对齐",
                options=SUPPORTED_CALENDAR_MODES,
                index=0,
                format_func=lambda m: _CALENDAR_MODE_LABELS.get(m, m),
                key="portfolio_calendar_mode",
            )
        )
        base_currency = str(
            c_ccy.selectbox(
                "组合计价币种",
                options=("CNY", "USD", "LOCAL"),
                index=0,
                format_func=lambda c: _BASE_CURRENCY_LABELS.get(c, c),
                key="portfolio_base_currency",
            )
        )

    with st.expander("⚙️ 回测约束、有效前沿、蒙特卡洛散点与敏感性参数", expanded=False):
        p1, p2, p3 = st.columns(3)
        cov_method = str(
            p1.selectbox(
                "协方差矩阵估计",
                options=SUPPORTED_COV_METHODS,
                index=list(SUPPORTED_COV_METHODS).index("ledoit_wolf"),
                format_func=lambda m: _COV_METHOD_LABELS.get(m, m),
                key="portfolio_cov_method",
            )
        )
        rebalance_freq = str(
            p2.selectbox(
                "再平衡调仓频率",
                options=SUPPORTED_REBALANCE_FREQS,
                index=list(SUPPORTED_REBALANCE_FREQS).index("monthly"),
                format_func=lambda f: _REBALANCE_FREQ_LABELS.get(f, f),
                key="portfolio_rebalance_freq",
            )
        )
        tx_cost_bps = float(
            p3.number_input(
                "单边换手交易成本 (bps)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=5.0,
                key="portfolio_tx_bps",
            )
        )

        b1, b2, b3, b4 = st.columns(4)
        min_weight = float(
            b1.number_input("单资产权重下界", 0.0, 0.99, 0.0, step=0.05, key="portfolio_min_weight")
        )
        max_weight = float(
            b2.number_input("单资产权重上界", 0.01, 1.0, 1.0, step=0.05, key="portfolio_max_weight")
        )
        train_ratio = float(
            b3.number_input(
                "Train 样本占比 (train_test)",
                0.2,
                0.9,
                0.7,
                step=0.05,
                key="portfolio_train_ratio",
            )
        )
        train_window = int(
            b4.number_input(
                "Walk-Forward 训练窗口",
                min_value=15,
                max_value=504,
                value=60,
                step=10,
                key="portfolio_train_window",
            )
        )

        e1, e2, e3, e4 = st.columns(4)
        split_date_raw = str(
            e1.text_input(
                "指定划分日期 (可选 YYYY-MM-DD)",
                value="",
                key="portfolio_split_date",
            )
        ).strip()
        frontier_points = int(
            e2.number_input("精确前沿点数", 0, 100, 25, step=5, key="portfolio_frontier_points")
        )
        num_random_portfolios = int(
            e3.number_input(
                "随机散点模拟数 (0 = 仅精确前沿)",
                min_value=0,
                max_value=3000,
                value=0,
                step=100,
                key="portfolio_num_random",
                help="默认 0（仅计算 SLSQP 数学精确前沿，与 CLI 一致）；可设为 >0 添加约束 Dirichlet 可行域云图。",
            )
        )
        trading_days_override = int(
            e4.number_input(
                "年化期数覆盖 (0 = 自动)",
                min_value=0,
                max_value=366,
                value=0,
                step=1,
                key="portfolio_trading_days_override",
            )
        )

        c_opt1, c_opt2 = st.columns(2)
        include_sensitivity = bool(
            c_opt1.checkbox(
                "计算参数稳健性敏感性矩阵（协方差 / 交易成本 / 调仓频率）",
                value=True,
                key="portfolio_include_sensitivity",
            )
        )
        show_extra_charts = bool(
            c_opt2.checkbox(
                "显示附加图表（标的归一化、水下回撤、动态权重演化）",
                value=False,
                key="tab3_show_extra_charts",
            )
        )

    run_clicked = st.button(
        "🎯 运行组合优化与回测",
        key="tab3_btn_ef",
        type="primary",
        width="stretch",
        disabled=not cfg["dates_valid"],
    )

    cur_sig = (
        legs_text,
        benchmark_text,
        strategy,
        custom_weights_raw,
        validation_mode,
        calendar_mode,
        base_currency,
        cov_method,
        rebalance_freq,
        round(tx_cost_bps, 6),
        round(min_weight, 6),
        round(max_weight, 6),
        round(train_ratio, 6),
        train_window,
        split_date_raw,
        frontier_points,
        num_random_portfolios,
        trading_days_override,
        include_sensitivity,
        round(cfg["rf_rate"], 8),
        cfg["start_str"],
        cfg["end_str"],
        cfg["allow_usdt_peg"],
    )
    if (state.get("result") is not None or state.get("error") is not None) and state.get(
        "signature"
    ) != cur_sig:
        _reset_workspace_outputs(state, keep_error=False)

    if run_clicked:
        if not cfg["dates_valid"]:
            _reset_workspace_outputs(state, keep_error=True)
            state["error"] = cfg["date_error"]
        else:
            apply_default_network_proxy_policy()
            loader_fn, fx_fn = _make_service_loaders(force_refresh=cfg["force_refresh"])
            with st.spinner("正在执行组合协方差估计、SLSQP 优化与回测验证..."):
                try:
                    specs = _parse_ui_multi_asset_specs(legs_text)
                    bench_spec = _parse_ui_benchmark_spec(benchmark_text)
                    custom_w = (
                        _parse_ui_custom_weights(custom_weights_raw)
                        if strategy == "custom"
                        else None
                    )
                    req = PortfolioRequest(
                        assets=specs,
                        start_date=str(cfg["start_str"]),
                        end_date=str(cfg["end_str"]),
                        strategy=strategy,
                        validation_mode=validation_mode,
                        benchmark=bench_spec,
                        calendar_mode=calendar_mode,
                        base_currency=base_currency,
                        allow_usdt_peg=cfg["allow_usdt_peg"],
                        risk_free_rate=cfg["rf_rate"],
                        trading_days=(
                            None if trading_days_override <= 0 else trading_days_override
                        ),
                        cov_method=cov_method,
                        rebalance_freq=rebalance_freq,
                        transaction_cost_bps=tx_cost_bps,
                        min_weight=min_weight,
                        max_weight=max_weight,
                        train_ratio=train_ratio,
                        split_date=split_date_raw or None,
                        train_window=train_window,
                        custom_weights=custom_w,
                        frontier_points=frontier_points,
                        num_random_portfolios=num_random_portfolios,
                        random_seed=42,
                        include_sensitivity=include_sensitivity,
                        force_refresh=cfg["force_refresh"],
                        proxy_url=cfg["proxy_url"],
                    )
                    res = analyze_portfolio(req, loader=loader_fn, fx_loader=fx_fn)
                except Exception as exc:
                    _reset_workspace_outputs(state, keep_error=False)
                    state["signature"] = cur_sig
                    state["error"] = format_diagnosis_user_message(exc)
                else:
                    _reset_workspace_outputs(state, keep_error=False)
                    state["signature"] = cur_sig
                    state["result"] = res

    if state.get("error"):
        st.error(state["error"])

    result: MultiAssetResult | None = state.get("result")
    if result is None:
        st.info(
            "配置组合标的与验证协议后点击 **「🎯 运行组合优化与回测」**，即可生成风险贡献、有效前沿与回测绩效。"
        )
        return

    methodology = result.methodology
    validation_meta = result.metrics.get("validation") or {}
    backtest_metrics = result.metrics.get("backtest") or {}
    res_val_mode = str(methodology.get("validation_mode") or validation_mode)
    is_oos = res_val_mode != "in_sample" and bool(methodology.get("is_out_of_sample", True))
    has_lookahead = bool(methodology.get("has_lookahead_bias", False))

    if not is_oos:
        if has_lookahead:
            st.warning(
                "⚠️ 当前为 **全样本拟合 (In-Sample Fit)**：权重估计与评估使用全区间事后信息，"
                "存在前视偏差（Lookahead Bias），不代表真实样本外可执行收益。"
            )
        else:
            st.warning(
                "⚠️ 当前为 **全样本固定权重回测 (In-Sample Non-OOS Backtest)**："
                "采用外生自定义权重在完整历史区间直接回测，未经过时间切分或滚动样本外隔离，"
                "非样本外评估（OOS）；若权重基于事后观察手工选定，仍可能存在选择偏差。"
            )
    elif res_val_mode == "train_test":
        st.info(
            f"样本外独立测试协议 (`train_test`)：训练集 {validation_meta.get('train_start')} ~ "
            f"{validation_meta.get('train_end')} ({validation_meta.get('train_rows')} 期) | "
            f"严格样本外测试集 {validation_meta.get('test_start')} ~ "
            f"{validation_meta.get('test_end')} ({validation_meta.get('test_rows')} 期)"
        )
    else:
        st.info(
            f"因果 Walk-Forward 样本外协议 (`{res_val_mode}`)："
            f"OOS 区间 {validation_meta.get('oos_start')} ~ {validation_meta.get('oos_end')} "
            f"({validation_meta.get('oos_periods')} 期，共执行 {validation_meta.get('rebalance_count')} 次滚动优化调仓)"
        )

    _render_quality_and_warnings(result.warnings)

    primary_perf = (
        (backtest_metrics.get("Portfolio") or backtest_metrics.get("OOS_Portfolio") or {})
        if not is_oos
        else (backtest_metrics.get("OOS_Portfolio") or backtest_metrics.get("Portfolio") or {})
    )
    ew_perf = backtest_metrics.get("Equal_Weight_Portfolio") or {}
    if is_oos:
        primary_title = "样本外策略"
    elif has_lookahead:
        primary_title = "样本内拟合"
    else:
        primary_title = "样本内全样本"

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{primary_title} CAGR", format_percent(primary_perf.get("cagr")))
    m2.metric(f"{primary_title}最大回撤", format_drawdown_percent(primary_perf.get("max_drawdown")))
    m3.metric(
        f"{primary_title}夏普比率", format_ratio(primary_perf.get("sharpe_ratio"), decimals=3)
    )

    # Compare strategy vs equal weight vs benchmark / hindsight reference in clean table
    perf_comp_rows = [
        {
            "组合/基准": f"{primary_title} ({strategy})",
            "累计收益": format_percent(primary_perf.get("cumulative_return")),
            "CAGR": format_percent(primary_perf.get("cagr")),
            "年化波动": format_percent(primary_perf.get("annualized_volatility")),
            "最大回撤": format_drawdown_percent(primary_perf.get("max_drawdown")),
            "夏普": format_ratio(primary_perf.get("sharpe_ratio"), decimals=3),
            "Sortino": format_ratio(primary_perf.get("sortino_ratio"), decimals=3),
            "总换手": format_ratio(primary_perf.get("total_turnover"), decimals=2),
            "估算交易成本": format_percent(primary_perf.get("total_cost_ratio")),
        },
        {
            "组合/基准": "等权重基准 (Equal Weight 1/N)",
            "累计收益": format_percent(ew_perf.get("cumulative_return")),
            "CAGR": format_percent(ew_perf.get("cagr")),
            "年化波动": format_percent(ew_perf.get("annualized_volatility")),
            "最大回撤": format_drawdown_percent(ew_perf.get("max_drawdown")),
            "夏普": format_ratio(ew_perf.get("sharpe_ratio"), decimals=3),
            "Sortino": format_ratio(ew_perf.get("sortino_ratio"), decimals=3),
            "总换手": format_ratio(ew_perf.get("total_turnover"), decimals=2),
            "估算交易成本": format_percent(ew_perf.get("total_cost_ratio")),
        },
    ]
    if "In_Sample_Portfolio" in backtest_metrics:
        is_perf = backtest_metrics["In_Sample_Portfolio"]
        perf_comp_rows.append(
            {
                "组合/基准": "样本内事后参考 (含前视偏差，仅对比)",
                "累计收益": format_percent(is_perf.get("cumulative_return")),
                "CAGR": format_percent(is_perf.get("cagr")),
                "年化波动": format_percent(is_perf.get("annualized_volatility")),
                "最大回撤": format_drawdown_percent(is_perf.get("max_drawdown")),
                "夏普": format_ratio(is_perf.get("sharpe_ratio"), decimals=3),
                "Sortino": format_ratio(is_perf.get("sortino_ratio"), decimals=3),
                "总换手": format_ratio(is_perf.get("total_turnover"), decimals=2),
                "估算交易成本": format_percent(is_perf.get("total_cost_ratio")),
            }
        )
    if result.benchmark and isinstance(result.benchmark.get("performance"), Mapping):
        bp = result.benchmark["performance"]
        perf_comp_rows.append(
            {
                "组合/基准": f"用户基准 ({result.benchmark.get('leg_name', 'Benchmark')})",
                "累计收益": format_percent(bp.get("cumulative_return")),
                "CAGR": format_percent(bp.get("cagr")),
                "年化波动": format_percent(bp.get("annualized_volatility")),
                "最大回撤": format_drawdown_percent(bp.get("max_drawdown")),
                "夏普": format_ratio(bp.get("sharpe_ratio"), decimals=3),
                "Sortino": format_ratio(bp.get("sortino_ratio"), decimals=3),
                "总换手": "0.00",
                "估算交易成本": "0.00%",
            }
        )
    st.markdown("#### 📈 组合回测与基准绩效对比")
    st.dataframe(pd.DataFrame(perf_comp_rows), width="stretch", hide_index=True)

    # Weights & Euler risk decomposition
    port_meta = result.metrics.get("portfolio") or {}
    rc_meta = result.metrics.get("risk_contributions") or {}
    cov_meta = result.metrics.get("covariance") or {}
    st.caption(
        f"求解状态：`{port_meta.get('solver_message', 'ok')}` | "
        f"分散化比率 (Diversification Ratio)：{format_ratio(rc_meta.get('diversification_ratio'), decimals=3)} | "
        f"有效资产数 ENB (权重 / 风险)：{format_ratio(rc_meta.get('effective_n_weights'), decimals=2)} / "
        f"{format_ratio(rc_meta.get('effective_n_risk'), decimals=2)} | "
        f"协方差条件数：{format_ratio(cov_meta.get('condition_number'), decimals=2)}"
    )

    weights_df = result.frames.get("weights")
    if isinstance(weights_df, pd.DataFrame) and not weights_df.empty:
        st.markdown("#### ⚖️ 资产权重与 Euler 风险贡献拆解")
        disp_w = pd.DataFrame(
            {
                "资产": weights_df.index.astype(str),
                "策略权重": [format_percent(v) for v in weights_df["weight"]],
                "风险贡献占比": [format_percent(v) for v in weights_df["pct_risk_contribution"]],
                "年化预期收益(估计)": [
                    format_percent(v) for v in weights_df["expected_return_ann"]
                ],
                "年化波动率(估计)": [format_percent(v) for v in weights_df["volatility_ann"]],
                "等权 1/N": [format_percent(v) for v in weights_df["equal_weight"]],
                "GMV 权重": [format_percent(v) for v in weights_df["min_variance_weight"]],
                "Tangency 权重": [format_percent(v) for v in weights_df["max_sharpe_weight"]],
            }
        )
        st.dataframe(disp_w, width="stretch", hide_index=True)

    chart_selection = ["backtest", "weights", "frontier", "correlation"]
    if show_extra_charts:
        chart_selection.extend(["normalized", "drawdown"])
        if "weight_history" in result.frames:
            chart_selection.append("weight_history")

    figs, skipped = _get_or_build_figures(state, result, chart_types=tuple(chart_selection))
    if figs is not None:
        _render_chart_bundle(
            figs,
            skipped,
            order=chart_selection,
            labels=_PORTFOLIO_CHART_LABELS,
            key_prefix="portfolio",
        )

    sens = result.metrics.get("sensitivity") or {}
    if sens:
        with st.expander("🔬 参数与交易成本敏感性分析", expanded=False):
            for sens_key, sens_title in (
                ("covariance_method", "协方差估计方法敏感性"),
                ("transaction_cost_bps", "交易成本敏感性 (bps)"),
                ("rebalance_freq", "再平衡频率敏感性"),
            ):
                rows = sens.get(sens_key)
                if isinstance(rows, list) and rows:
                    st.markdown(f"**{sens_title}**")
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("📐 协方差矩阵、基准归因检验与方法学审计", expanded=False):
        if result.benchmark and isinstance(result.benchmark.get("relative_to_strategy"), Mapping):
            rel = result.benchmark["relative_to_strategy"]
            mem = rel.get("memmel_sharpe_test") or {}
            st.markdown("**策略相对基准统计检验**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "超额年化收益": format_percent(rel.get("excess_annualized_return")),
                            "Tracking Error": format_percent(rel.get("tracking_error")),
                            "Information Ratio": format_ratio(
                                rel.get("information_ratio"), decimals=3
                            ),
                            "Beta": format_ratio(rel.get("beta"), decimals=3),
                            "Jensen Alpha": format_percent(rel.get("jensen_alpha")),
                            "Memmel Z": format_ratio(mem.get("z_stat"), decimals=3),
                            "Memmel p-value": format_p_value(mem.get("p_value")),
                        }
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        cov_df = result.frames.get("covariance")
        if isinstance(cov_df, pd.DataFrame) and not cov_df.empty:
            st.markdown("**年化协方差矩阵**")
            st.dataframe(cov_df.round(6), width="stretch")
        st.json(methodology)

    _render_report_and_download_section(
        state,
        result,
        file_prefix=f"portfolio_{strategy}_{validation_mode}",
        skipped_charts=skipped,
        key_prefix="portfolio",
    )


# ---------------------------------------------------------------------------
# App Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the unified M5 / V1.0 Streamlit application."""
    if st is None:
        raise ImportError("缺少运行依赖 `streamlit`，请先安装：`pip install streamlit`。")

    st.set_page_config(
        page_title="Quant Insight · 量化金融分析工作台",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_ui_theme_css()

    st.markdown(
        f"""
        <div class="quant-hero">
          <div class="quant-hero-title">📈 Quant Insight · 多资产量化分析与组合研究工作台</div>
          <p class="quant-hero-sub">
            Web 与 <code>quant</code> CLI 共用 M1–M4 统一金融内核、严格语义防线、中性化日历与无前视回测引擎。
          </p>
          <span class="quant-pill">版本 V{SOFTWARE_VERSION}</span>
          <span class="quant-pill">13 类跨市场资产</span>
          <span class="quant-pill">SLSQP 精确前沿 &amp; Walk-Forward OOS</span>
          <span class="quant-pill">Markdown / HTML / JSON / CSV 报告直出</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📐 计算口径、数学原理与多市场代码参考（点击展开）", expanded=False):
        st.markdown(
            "- **单资产与多资产默认口径差异**：单资产探索默认保留原生货币 (`LOCAL`) 与对数收益率 (`log`)；"
            "多资产比较与组合研究默认折算为统一货币 (`CNY`) 与离散简单收益率 (`simple`，保证组合加权线性性)。\n"
            "- **统计严格区分**：所有界面与报告均明确区分 **历史已实现绩效**、**正态性/Memmel 统计推断**、"
            "**样本内拟合（含前视偏差警告）** 与 **严格因果样本外回测（OOS）**。"
        )
        st.markdown(_TAB2_SYSTEM_MATH_MARKDOWN)
        st.dataframe(
            build_ef_multi_asset_codes_reference_df(),
            width="stretch",
            hide_index=True,
        )

    cfg = _render_sidebar()

    tab_single, tab_compare, tab_portfolio = st.tabs(
        ["📈 单资产探索", "⚖️ 多资产比较", "🎯 投资组合研究"]
    )
    with tab_single:
        _render_single_asset_workspace(cfg)
    with tab_compare:
        _render_compare_workspace(cfg)
    with tab_portfolio:
        _render_portfolio_workspace(cfg)

    _sync_legacy_analysis_results()


def _sync_legacy_analysis_results() -> None:
    """Keep ``st.session_state['_analysis_results']`` synchronized across M5 workspaces."""
    assert st is not None
    summary: dict[str, Any] = {}
    s_state = st.session_state.get(_WORKSPACE_SINGLE_KEY) or {}
    if s_state.get("result") is not None:
        summary["single"] = s_state["result"]
    if s_state.get("norm_fig") is not None:
        summary["normalize"] = s_state["norm_fig"]
    if s_state.get("gbm_result") is not None:
        summary["gbm"] = s_state["gbm_result"]
    c_state = st.session_state.get(_WORKSPACE_COMPARE_KEY) or {}
    if c_state.get("result") is not None:
        summary["compare"] = c_state["result"]
    p_state = st.session_state.get(_WORKSPACE_PORTFOLIO_KEY) or {}
    if p_state.get("result") is not None:
        summary["frontier"] = p_state["result"]
    st.session_state[_LEGACY_RESULTS_KEY] = summary


def _in_streamlit_runtime() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            return get_script_run_ctx() is not None
        except Exception:
            return False


if __name__ == "__main__" or _in_streamlit_runtime():
    main()
