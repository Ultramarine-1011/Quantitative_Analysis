"""Streamlit Web UI entrypoint for Quant Insight.

Thin interactive presentation shell delegating all analytics to ``services/`` and
``reporting/``, with task-driven navigation (Home, Single Asset, Compare, Portfolio),
cross-workspace context continuity, neutral desktop styling, and cache invalidation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:  # Allow headless library/CLI imports without Streamlit installed.
    st = None  # type: ignore[assignment]

import asset_loader as _asset_loader_mod
from analysis_metrics import compute_cagr as _compute_cagr, get_quant_metrics
from asset_catalog import SUPPORTED_ASSET_TYPES
from asset_inputs import parse_asset_list, parse_asset_token, parse_weights
from asset_loader import multi_asset_leg_column_name, parse_multi_asset_legs
from charts import build_tab3_correlation_heatmap
from contracts import AssetSpec
from data_sources.errors import DataFetchError, EmptySymbolDataError
from ui.components.sidebar import render_sidebar
from ui.services_bridge import (
    cleaned_ohlcv_to_csv_bytes_and_filename as _cleaned_ohlcv_to_csv_bytes_and_filename,
    format_data_quality_warning as _format_data_quality_warning,
)
from ui.state import apply_pending_transitions, sync_legacy_analysis_results
from ui.theme import inject_ui_theme_css, normalize_theme_mode
from ui.workspaces.compare import (
    invalidate_compare_from_persisted,
    render_compare_workspace,
)
from ui.workspaces.home import render_home_workspace
from ui.workspaces.portfolio import (
    invalidate_portfolio_from_persisted,
    render_portfolio_workspace,
)
from ui.workspaces.single_asset import (
    invalidate_single_from_persisted,
    render_single_asset_workspace,
)

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


def _cache_data(ttl: int = 3600) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Use ``st.cache_data`` inside Streamlit, or provide a no-op ``.clear()`` fallback."""
    if st is None:

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func.clear = lambda: None  # type: ignore[attr-defined]
            return func

        return _decorator
    return st.cache_data(ttl=ttl, show_spinner=False)


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


def _split_ui_leg_token(segment: str) -> tuple[str, str]:
    spec = parse_asset_token(segment, syntax="web")
    return spec.asset_type, spec.symbol


def _parse_ui_multi_asset_specs(text: str) -> tuple[AssetSpec, ...]:
    return parse_asset_list([text], syntax="web")


def _parse_ui_benchmark_spec(raw: str) -> AssetSpec | None:
    return parse_asset_token(raw, syntax="web") if raw.strip() else None


def _parse_ui_custom_weights(raw: str) -> tuple[float, ...] | None:
    return parse_weights(raw, web=True)


def main() -> None:
    """Render the task-driven Quant Insight Streamlit workbench."""
    if st is None:
        raise ImportError("缺少运行依赖 `streamlit`，请先安装：`pip install streamlit`。")

    st.set_page_config(
        page_title="Quant Insight · 量化研究工作台",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_pending_transitions(st)

    active_theme = normalize_theme_mode(st.session_state.get("ui_theme_mode", "auto"))
    inject_ui_theme_css(st, active_theme)

    cfg = render_sidebar(st)
    if cfg["theme_mode"] != active_theme:
        inject_ui_theme_css(st, cfg["theme_mode"])

    active_view = cfg["active_view"]
    invalidated_any = False
    invalidated_any |= invalidate_single_from_persisted(st, cfg)
    invalidated_any |= invalidate_compare_from_persisted(st, cfg)
    invalidated_any |= invalidate_portfolio_from_persisted(st, cfg)
    if invalidated_any:
        st.rerun()

    if cfg.get("date_error"):
        st.error(str(cfg["date_error"]))

    if active_view == "home":
        render_home_workspace(st, cfg)
    elif active_view == "single":
        render_single_asset_workspace(st, cfg, cached_loader=load_asset_data)
    elif active_view == "compare":
        render_compare_workspace(st, cfg, cached_loader=load_asset_data)
    else:
        render_portfolio_workspace(st, cfg, cached_loader=load_asset_data)

    sync_legacy_analysis_results(st)


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
