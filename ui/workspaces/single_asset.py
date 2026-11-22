"""Workspace 1: Single-Asset Exploration backed by ``analyze_single_asset``."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import pandas as pd

from asset_catalog import ASSET_LABELS, get_asset_trading_days
from charts import build_tab3_gbm_figure, build_tab3_normalize_figure
from contracts import AnalysisRequest, AnalysisResult
from data_sources.network import apply_default_network_proxy_policy
from quant_models import normalize_close_to_base_one, run_monte_carlo_gbm
from services.analysis_service import analyze_single_asset, fetch_single_asset
from ui.components.alerts import (
    render_empty_state,
    render_error_banner,
    render_quality_and_warnings,
)
from ui.components.charts import render_chart_explorer
from ui.components.exports import render_report_and_download_section
from ui.components.metrics import render_single_asset_kpis
from ui.components.tables import render_single_asset_audit_and_preview
from ui.errors import classify_exception, format_ui_error_markdown, make_validation_error
from ui.labels import ADJUST_LABELS, SINGLE_CHART_LABELS, SINGLE_CHART_ORDER
from ui.services_bridge import get_or_build_figures, make_service_loaders
from ui.state import (
    WORKSPACE_SINGLE_KEY,
    invalidate_if_signature_changed,
    store_workspace_error,
    store_workspace_result,
    workspace_state,
)
from ui.theme import apply_ui_plotly_theme, normalize_theme_mode


def _render_single_advanced_explorer(
    st_module: Any,
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    cached_loader: Callable[..., pd.DataFrame],
) -> None:
    """Render collapsible Single-Asset Normalize + Asset-Adaptive GBM Monte Carlo simulator."""
    with st_module.expander(
        "🎲 高级探索：单资产基准归一化与蒙特卡洛（GBM）路径模拟", expanded=False
    ):
        st_module.caption(
            "参数前瞻模拟工具（非历史已实现绩效）：基于对数收益率的漂移项与波动率估算几何布朗运动路径，"
            f"年化基准自动适配当前资产类别（{cfg['asset_type']} → {get_asset_trading_days(cfg['asset_type'])} 期/年）。"
        )
        col_l, col_r = st_module.columns([1, 2])
        disabled_btn = bool(
            not cfg["dates_valid"] or (not cfg["symbol"] and cfg["asset_type"] != "bond_us_yield")
        )
        with col_l:
            norm_clicked = st_module.button(
                "📈 生成归一化净值曲线 (Base=1)",
                key="single_btn_normalize",
                width="stretch",
                disabled=disabled_btn,
            )
            gbm_days = int(
                st_module.number_input(
                    "模拟未来期数 (Days/Steps)",
                    min_value=10,
                    max_value=756,
                    value=int(get_asset_trading_days(cfg["asset_type"])),
                    step=21,
                    key="single_gbm_days",
                )
            )
            gbm_sims = int(
                st_module.number_input(
                    "蒙特卡洛模拟路径数",
                    min_value=20,
                    max_value=2000,
                    value=200,
                    step=20,
                    key="single_gbm_sims",
                )
            )
            gbm_clicked = st_module.button(
                "🎲 运行 GBM 蒙特卡洛模拟",
                key="single_btn_gbm",
                width="stretch",
                disabled=disabled_btn,
            )

        sub_sig = (
            cfg["asset_type"],
            cfg["symbol"],
            cfg["adjust"],
            cfg["single_base_currency"],
            cfg["start_str"],
            cfg["end_str"],
            cfg["allow_usdt_peg"],
            cfg["proxy_url"],
        )
        if state.get("advanced_sig") != sub_sig:
            state["advanced_sig"] = sub_sig
            state["norm_fig_raw"] = None
            state["norm_fig"] = None
            state["norm_fig_theme"] = None
            state["norm_error"] = None
            state["gbm_result"] = None
            state["gbm_fig_raw"] = None
            state["gbm_fig"] = None
            state["gbm_fig_theme"] = None
            state["gbm_error"] = None
        if state.get("gbm_params") != (gbm_days, gbm_sims):
            state["gbm_params"] = (gbm_days, gbm_sims)
            state["gbm_result"] = None
            state["gbm_fig_raw"] = None
            state["gbm_fig"] = None
            state["gbm_fig_theme"] = None

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
                cfg["proxy_url"],
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

            loader_fn, fx_fn = make_service_loaders(
                force_refresh=cfg["force_refresh"],
                cached_loader=cached_loader,
            )
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

        theme_mode = normalize_theme_mode(
            cfg.get("resolved_theme_mode") or cfg.get("theme_mode", "auto")
        )

        if norm_clicked:
            with st_module.spinner("正在计算归一化净值曲线..."):
                try:
                    df_norm = _obtain_price_frame()
                    norm_s = normalize_close_to_base_one(df_norm)
                    raw_fig = build_tab3_normalize_figure(df_norm.index, norm_s)
                    state["norm_fig_raw"] = raw_fig
                    state["norm_fig"] = apply_ui_plotly_theme(raw_fig, theme_mode)
                    state["norm_fig_theme"] = theme_mode
                    state["norm_error"] = None
                except Exception as exc:
                    state["norm_fig_raw"] = None
                    state["norm_fig"] = None
                    state["norm_fig_theme"] = None
                    state["norm_error"] = format_ui_error_markdown(classify_exception(exc))

        if gbm_clicked:
            with st_module.spinner("正在模拟几何布朗运动（GBM）路径..."):
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
                    raw_gbm = build_tab3_gbm_figure(gbm_res)
                    state["gbm_result"] = gbm_res
                    state["gbm_fig_raw"] = raw_gbm
                    state["gbm_fig"] = apply_ui_plotly_theme(raw_gbm, theme_mode)
                    state["gbm_fig_theme"] = theme_mode
                    state["gbm_error"] = None
                except Exception as exc:
                    state["gbm_result"] = None
                    state["gbm_fig_raw"] = None
                    state["gbm_fig"] = None
                    state["gbm_fig_theme"] = None
                    state["gbm_error"] = format_ui_error_markdown(classify_exception(exc))

        # Re-theme memoized advanced figures transparently when theme switches.
        if state.get("norm_fig_raw") is not None and state.get("norm_fig_theme") != theme_mode:
            state["norm_fig"] = apply_ui_plotly_theme(state["norm_fig_raw"], theme_mode)
            state["norm_fig_theme"] = theme_mode
        if state.get("gbm_fig_raw") is not None and state.get("gbm_fig_theme") != theme_mode:
            state["gbm_fig"] = apply_ui_plotly_theme(state["gbm_fig_raw"], theme_mode)
            state["gbm_fig_theme"] = theme_mode

        with col_r:
            if state.get("norm_error"):
                st_module.error(state["norm_error"])
            if state.get("norm_fig") is not None:
                st_module.plotly_chart(
                    state["norm_fig"],
                    width="stretch",
                    key="single_norm_chart",
                    config={"displaylogo": False, "responsive": True},
                )

            if state.get("gbm_error"):
                st_module.error(state["gbm_error"])
            gbm_out = state.get("gbm_result")
            if gbm_out is not None and state.get("gbm_fig") is not None:
                st_module.plotly_chart(
                    state["gbm_fig"],
                    width="stretch",
                    key="single_gbm_chart",
                    config={"displaylogo": False, "responsive": True},
                )
                q_row = {
                    "当前初值 S0": f"{gbm_out.s0:.2f}",
                    "P5 (悲观)": f"{float(gbm_out.quantiles_p05[-1]):.2f}",
                    "P50 (中位数)": f"{float(gbm_out.quantiles_p50[-1]):.2f}",
                    "P95 (乐观)": f"{float(gbm_out.quantiles_p95[-1]):.2f}",
                    "日连续漂移 μ / 波动 σ": f"{gbm_out.mu_daily:.5f} / {gbm_out.sigma_daily:.5f}",
                }
                st_module.dataframe(pd.DataFrame([q_row]), width="stretch", hide_index=True)


def render_single_asset_workspace(
    st_module: Any,
    cfg: dict[str, Any],
    *,
    cached_loader: Callable[..., pd.DataFrame],
) -> None:
    """Render Workspace 1: Single-Asset Exploration backed by ``analyze_single_asset``."""
    if st_module is None:
        return
    state = workspace_state(st_module, WORKSPACE_SINGLE_KEY)

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
        cfg["proxy_url"],
    )
    invalidate_if_signature_changed(state, cur_sig)

    if cfg["single_clicked"]:
        if not cfg["dates_valid"]:
            store_workspace_error(
                state,
                signature=cur_sig,
                error_info=make_validation_error(str(cfg["date_error"] or "请选择有效日期区间。")),
            )
        elif cfg["sma_short"] >= cfg["sma_long"]:
            store_workspace_error(
                state,
                signature=cur_sig,
                error_info=make_validation_error(
                    "参数错误：短期均线窗口 (SMA Short) 必须严格小于长期均线窗口 (SMA Long)。",
                    hint="请在左侧边栏“单资产高级技术与统计参数”中调小 SMA Short 或调大 SMA Long。",
                ),
            )
        else:
            apply_default_network_proxy_policy()
            loader_fn, fx_fn = make_service_loaders(
                force_refresh=cfg["force_refresh"],
                cached_loader=cached_loader,
            )
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
            with st_module.spinner("正在拉取行情、清洗 OHLCV 并执行单资产定量分析..."):
                try:
                    res = analyze_single_asset(req, loader=loader_fn, fx_loader=fx_fn)
                except Exception as exc:
                    store_workspace_error(
                        state,
                        signature=cur_sig,
                        error_info=classify_exception(exc),
                    )
                else:
                    fetch_sig = (
                        cfg["asset_type"],
                        cfg["symbol"] or "10Y",
                        cfg["adjust"],
                        cfg["single_base_currency"],
                        str(cfg["start_str"]),
                        str(cfg["end_str"]),
                        cfg["allow_usdt_peg"],
                        cfg["proxy_url"],
                    )
                    store_workspace_result(
                        state,
                        signature=cur_sig,
                        result=res,
                        fetch_signature=fetch_sig,
                    )

    if state.get("error_info") or state.get("error"):
        render_error_banner(st_module, state.get("error_info") or state.get("error"))

    result: AnalysisResult | None = state.get("result")
    if result is None:
        adj_label = (
            ADJUST_LABELS.get(cfg["adjust"], str(cfg["adjust"]))
            if cfg["adjust"] is not None
            else "无复权"
        )
        render_empty_state(
            st_module,
            title="📈 单资产历史诊断与风险面板",
            subtitle="在左侧边栏确认资产类别、标的代码与分析区间后，点击「🔍 运行单资产分析」生成核心绩效指标、交互式 K 线/回撤图与多格式研究报告。",
            chips=[
                ASSET_LABELS.get(cfg["asset_type"], cfg["asset_type"]),
                f"代码: {cfg['symbol'] or '10Y'}",
                f"复权: {adj_label}",
                f"货币: {cfg['single_base_currency']}",
                f"口径: {cfg['single_return_convention']}",
            ],
        )
        _render_single_advanced_explorer(st_module, cfg, state, cached_loader=cached_loader)
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
    asset_lbl = ASSET_LABELS.get(result.asset.asset_type, result.asset.asset_type)
    st_module.markdown(
        f"""
        <div class="qa-context-strip">
          <div class="qa-context-title">
            <span>{html.escape(asset_lbl)}</span>
            <span class="qa-code-tag">{html.escape(result.asset.canonical_symbol)}</span>
          </div>
          <div class="qa-masthead-badges">
            <span class="qa-pill">样本 {html.escape(result.quality.start_date)} ~ {html.escape(result.quality.end_date)} ({result.quality.total_rows} 条)</span>
            <span class="qa-pill">源 {html.escape(result.asset.data_source or "unknown")}</span>
            <span class="qa-pill qa-pill--accent">{html.escape(ccy_note)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st_module.caption(f"数据与缓存溯源：{cache_label} · 估值货币：{ccy_note}")

    render_quality_and_warnings(st_module, result.warnings, single_result=result)
    render_single_asset_kpis(st_module, result)

    figs, skipped = get_or_build_figures(
        state,
        result,
        theme_mode=cfg.get("resolved_theme_mode") or cfg.get("theme_mode", "auto"),
        on_error=lambda exc: render_error_banner(st_module, classify_exception(exc)),
    )
    if figs is not None:
        render_chart_explorer(
            st_module,
            figs,
            skipped,
            order=SINGLE_CHART_ORDER,
            labels=SINGLE_CHART_LABELS,
            key_prefix="single",
        )

    render_single_asset_audit_and_preview(st_module, result)
    _render_single_advanced_explorer(st_module, cfg, state, cached_loader=cached_loader)
    render_report_and_download_section(
        st_module,
        state,
        result,
        file_prefix=f"{result.asset.asset_type}_{result.asset.canonical_symbol}".replace("/", "_"),
        skipped_charts=skipped,
        key_prefix="single",
    )
