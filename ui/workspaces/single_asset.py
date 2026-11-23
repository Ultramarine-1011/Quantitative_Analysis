"""Workspace 1: Single-Asset Analysis with results-first disclosure and cross-workspace handoff."""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from asset_catalog import (
    ASSET_LABELS,
    DEFAULT_SYMBOLS,
    SUPPORTED_ADJUSTMENTS,
    SUPPORTED_ASSET_TYPES,
    format_asset_guide_markdown,
    get_asset_trading_days,
)
from charts import build_tab3_gbm_figure, build_tab3_normalize_figure
from contracts import AnalysisRequest, AnalysisResult
from data_sources.network import apply_default_network_proxy_policy
from quant_models import normalize_close_to_base_one, run_monte_carlo_gbm
from reporting import format_drawdown_percent, format_percent, format_ratio
from services.analysis_service import analyze_single_asset, fetch_single_asset
from ui.asset_directory import (
    POPULAR_ASSETS,
    AssetEntry,
    format_asset_display,
    format_asset_picker_option,
    parse_legs_loose,
    serialize_legs,
)
from ui.components.alerts import (
    render_change_notice,
    render_delta_comparison,
    render_empty_state,
    render_error_banner,
    render_quality_and_warnings,
    render_takeaway_card,
)
from ui.components.charts import render_chart_explorer
from ui.components.exports import render_export_popover, render_report_preview_drawer
from ui.components.header import render_workspace_topbar
from ui.components.metrics import render_single_asset_kpis
from ui.components.tables import render_single_asset_audit_and_preview
from ui.errors import classify_exception, format_ui_error_markdown, make_validation_error
from ui.labels import (
    ADJUST_LABELS,
    BASE_CURRENCY_LABELS,
    RETURN_CONVENTION_LABELS,
    SINGLE_CHART_LABELS,
    SINGLE_CHART_ORDER,
)
from ui.services_bridge import get_or_build_figures, make_service_loaders
from ui.state import (
    WORKSPACE_SINGLE_KEY,
    consume_auto_run,
    hydrate_widget_inputs,
    invalidate_if_signature_changed,
    queue_transition,
    read_persisted_inputs,
    record_basket_assets,
    save_widget_inputs,
    set_mirrored_input,
    store_workspace_error,
    store_workspace_result,
    workspace_state,
    workspace_status,
)
from ui.theme import apply_ui_plotly_theme, normalize_theme_mode

SINGLE_INPUT_DEFAULTS: dict[str, Any] = {
    "asset_type": "etf",
    "_prev_asset_type": "etf",
    "symbol_input": DEFAULT_SYMBOLS["etf"],
    "adjust_input": "hfq",
    "single_base_currency_input": "LOCAL",
    "single_return_convention_input": "log",
    "sma_short_input": 20,
    "sma_long_input": 60,
    "bb_window_input": 20,
    "single_trading_days_input": get_asset_trading_days("etf"),
    "include_momentum_input": False,
}

SINGLE_WIDGET_KEYS: tuple[str, ...] = tuple(SINGLE_INPUT_DEFAULTS.keys())


def _resolve_effective_adjust(asset_type: str, raw_adjust: Any) -> str | None:
    supported = SUPPORTED_ADJUSTMENTS.get(asset_type)
    if not supported:
        return None
    if raw_adjust in supported:
        return str(raw_adjust)
    return supported[0]


def build_single_sig_and_meta(
    inputs: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build deterministic cache signature and human-readable meta from single-asset inputs."""
    asset_type = str(inputs.get("asset_type", "etf")).strip().lower()
    symbol = str(inputs.get("symbol_input", "") or "").strip()
    adjust = _resolve_effective_adjust(asset_type, inputs.get("adjust_input"))
    base_ccy = str(inputs.get("single_base_currency_input", "LOCAL"))
    ret_conv = str(inputs.get("single_return_convention_input", "log"))
    sma_short = int(inputs.get("sma_short_input", 20))
    sma_long = int(inputs.get("sma_long_input", 60))
    bb_window = int(inputs.get("bb_window_input", 20))
    trading_days = int(inputs.get("single_trading_days_input", get_asset_trading_days(asset_type)))
    include_momentum = bool(inputs.get("include_momentum_input", False))

    sig = (
        asset_type,
        symbol,
        adjust,
        base_ccy,
        ret_conv,
        sma_short,
        sma_long,
        bb_window,
        trading_days,
        include_momentum,
        round(float(cfg.get("rf_rate", 0.016)), 8),
        cfg.get("start_str"),
        cfg.get("end_str"),
        bool(cfg.get("allow_usdt_peg", False)),
        cfg.get("proxy_url"),
    )
    eff_symbol = symbol or ("10Y" if asset_type == "bond_us_yield" else "")
    adj_label = ADJUST_LABELS.get(adjust, str(adjust)) if adjust is not None else "不适用"
    date_window = (
        f"{cfg.get('start_str')} ~ {cfg.get('end_str')}"
        if cfg.get("dates_valid") and cfg.get("start_str") and cfg.get("end_str")
        else "区间待修正"
    )
    meta = {
        "scope_key": (asset_type, eff_symbol.upper()),
        "date_window": date_window,
        "rf_pct": f"{float(cfg.get('rf_rate', 0.016)) * 100.0:.2f}%",
        "adjust": adj_label,
        "base_currency": base_ccy,
        "return_convention": ret_conv,
    }
    return sig, meta


def invalidate_single_from_persisted(st_module: Any, cfg: Mapping[str, Any]) -> bool:
    """Invalidate single-asset workspace if global settings changed while another view is active."""
    state = workspace_state(st_module, WORKSPACE_SINGLE_KEY)
    if (
        state.get("result") is None
        and state.get("error") is None
        and state.get("error_info") is None
    ):
        return False
    inputs = read_persisted_inputs(st_module, SINGLE_INPUT_DEFAULTS)
    sig, meta = build_single_sig_and_meta(inputs, cfg)
    return invalidate_if_signature_changed(state, sig, cur_meta=meta)


def sync_symbol_with_asset_type(st_module: Any) -> None:
    """Update default symbol, adjustment, and adaptive trading_days when ``asset_type`` changes."""
    if st_module is None:
        return
    new_type = str(st_module.session_state.get("asset_type", "etf"))
    prev_type = str(st_module.session_state.get("_prev_asset_type", "etf"))
    current_sym = str(st_module.session_state.get("symbol_input", "")).strip()
    prev_default = DEFAULT_SYMBOLS.get(prev_type, "")
    if not current_sym or current_sym == prev_default:
        set_mirrored_input(st_module, "symbol_input", DEFAULT_SYMBOLS.get(new_type, ""))

    prev_td = get_asset_trading_days(prev_type)
    cur_td = st_module.session_state.get("single_trading_days_input")
    if cur_td is None or int(cur_td) == prev_td:
        set_mirrored_input(st_module, "single_trading_days_input", get_asset_trading_days(new_type))

    supported = SUPPORTED_ADJUSTMENTS.get(new_type)
    if supported:
        cur_adj = st_module.session_state.get("adjust_input")
        if cur_adj not in supported:
            set_mirrored_input(st_module, "adjust_input", supported[0])

    set_mirrored_input(st_module, "_prev_asset_type", new_type)


def _apply_quick_pick(st_module: Any) -> None:
    picked = st_module.session_state.get("single_quick_pick")
    if not isinstance(picked, AssetEntry):
        return
    set_mirrored_input(st_module, "asset_type", picked.asset_type)
    set_mirrored_input(st_module, "_prev_asset_type", picked.asset_type)
    set_mirrored_input(st_module, "symbol_input", picked.symbol)
    set_mirrored_input(
        st_module, "single_trading_days_input", get_asset_trading_days(picked.asset_type)
    )
    supported = SUPPORTED_ADJUSTMENTS.get(picked.asset_type)
    if supported:
        set_mirrored_input(st_module, "adjust_input", supported[0])


def _build_single_summary(res: AnalysisResult, meta: Mapping[str, Any]) -> dict[str, Any]:
    if res.asset.series_kind == "yield_level":
        return {
            "scope_key": meta["scope_key"],
            "context_label": str(meta["date_window"]),
            "metrics": [
                {
                    "key": "latest_level",
                    "label": "最新利率",
                    "value": res.metrics.get("latest_level"),
                    "fmt": "ratio",
                },
                {
                    "key": "level_change",
                    "label": "区间变动",
                    "value": res.metrics.get("level_change"),
                    "fmt": "ratio",
                },
            ],
        }
    return {
        "scope_key": meta["scope_key"],
        "context_label": str(meta["date_window"]),
        "metrics": [
            {
                "key": "cagr",
                "label": "年化回报 CAGR",
                "value": res.metrics.get("cagr"),
                "fmt": "pct",
            },
            {
                "key": "max_drawdown",
                "label": "最大回撤",
                "value": res.metrics.get("max_drawdown"),
                "fmt": "dd",
            },
            {
                "key": "sharpe_ratio",
                "label": "夏普比率",
                "value": res.metrics.get("sharpe_ratio"),
                "fmt": "ratio",
            },
        ],
    }


def _render_single_takeaway(st_module: Any, res: AnalysisResult) -> None:
    display_name = format_asset_display(res.asset.asset_type, res.asset.canonical_symbol)
    span = f"{res.quality.start_date} ~ {res.quality.end_date}"
    if res.asset.series_kind == "yield_level":
        latest_lvl = format_ratio(res.metrics.get("latest_level"), decimals=3)
        chg = format_ratio(res.metrics.get("level_change"), decimals=3)
        render_takeaway_card(
            st_module,
            headline=f"{display_name} 在 {span} 期间最新利率水平为 {latest_lvl}%，区间变化 {chg} 个百分点。",
            detail="注：利率水平序列按口径规则仅展示利率变化，不计算股票式净值收益率或回撤。",
        )
        return

    cum_txt = format_percent(res.metrics.get("cumulative_return"), signed=True)
    cagr_txt = format_percent(res.metrics.get("cagr"), signed=True)
    mdd_txt = format_drawdown_percent(res.metrics.get("max_drawdown"))
    sh_txt = format_ratio(res.metrics.get("sharpe_ratio"), decimals=3)
    ann_vol = format_percent(res.metrics.get("annualized_volatility"))
    render_takeaway_card(
        st_module,
        headline=(
            f"{display_name} 在 {span} 累计收益 {cum_txt}（年化 {cagr_txt}），"
            f"期间最大回撤 {mdd_txt}，夏普比率 {sh_txt}。"
        ),
        detail=f"年化波动率 {ann_vol} · 共 {res.quality.total_rows} 个交易日样本 · 估值货币 {res.asset.output_currency or res.asset.native_currency}",
    )


def _handoff_to_compare(st_module: Any, asset_type: str, symbol: str) -> None:
    basket = record_basket_assets(st_module, [(asset_type, symbol)])
    existing_compare = parse_legs_loose(
        str(st_module.session_state.get("compare_assets_input", "") or "")
    )
    ordered: list[tuple[str, str]] = [(asset_type, symbol)]
    # Carry other assets explored in basket first (if user already touched others), otherwise keep just current.
    for pair in basket:
        if pair not in ordered:
            ordered.append(pair)
    if len(ordered) == 1 and len(existing_compare) >= 2:
        # Only inherit existing compare items if user explicitly built a multi-asset basket earlier;
        # starting fresh with 1 asset encourages explicit choice of the second asset.
        pass
    queue_transition(
        st_module,
        view="compare",
        updates={"compare_assets_input": serialize_legs(ordered)},
    )
    st_module.rerun()


def _handoff_to_portfolio(st_module: Any, asset_type: str, symbol: str) -> None:
    record_basket_assets(st_module, [(asset_type, symbol)])
    existing = parse_legs_loose(str(st_module.session_state.get("tab3_ef_legs_input", "") or ""))
    ordered = [(asset_type, symbol), *[p for p in existing if p != (asset_type, symbol)]]
    queue_transition(
        st_module,
        view="portfolio",
        updates={"tab3_ef_legs_input": serialize_legs(ordered)},
    )
    st_module.rerun()


def _render_single_advanced_explorer(
    st_module: Any,
    cfg: Mapping[str, Any],
    inputs: Mapping[str, Any],
    state: dict[str, Any],
    *,
    cached_loader: Callable[..., pd.DataFrame],
) -> None:
    """Render collapsible Single-Asset Normalize + Asset-Adaptive GBM Monte Carlo simulator."""
    asset_type = str(inputs["asset_type"])
    symbol = str(inputs["symbol_input"] or "").strip()
    adjust = _resolve_effective_adjust(asset_type, inputs.get("adjust_input"))
    base_currency = str(inputs["single_base_currency_input"])

    with st_module.expander("高级探索：归一化净值与蒙特卡洛（GBM）未来路径模拟", expanded=False):
        st_module.caption(
            "前瞻模拟工具（非历史已实现绩效）：基于历史波动率估算几何布朗运动路径，"
            f"年化基准自动适配当前资产（{asset_type} → {get_asset_trading_days(asset_type)} 期/年）。"
        )
        col_l, col_r = st_module.columns([1, 2])
        disabled_btn = bool(
            not cfg["dates_valid"] or (not symbol and asset_type != "bond_us_yield")
        )
        with col_l:
            norm_clicked = st_module.button(
                "生成归一化净值曲线 (Base=1)",
                key="single_btn_normalize",
                width="stretch",
                disabled=disabled_btn,
            )
            gbm_days = int(
                st_module.number_input(
                    "模拟未来期数 (Days/Steps)",
                    min_value=10,
                    max_value=756,
                    value=int(get_asset_trading_days(asset_type)),
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
                "运行 GBM 蒙特卡洛模拟",
                key="single_btn_gbm",
                width="stretch",
                disabled=disabled_btn,
            )

        sub_sig = (
            asset_type,
            symbol,
            adjust,
            base_currency,
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
                asset_type,
                symbol or "10Y",
                adjust,
                base_currency,
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
                    asset_type=asset_type,
                    symbol=symbol or "10Y",
                    start_date=str(cfg["start_str"]),
                    end_date=str(cfg["end_str"]),
                    adjust=adjust,
                    base_currency=base_currency,
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
                    td = get_asset_trading_days(asset_type)
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
    """Render Workspace 1: Single-Asset Analysis with results-first disclosure."""
    if st_module is None:
        return
    state = workspace_state(st_module, WORKSPACE_SINGLE_KEY)
    inputs = read_persisted_inputs(st_module, SINGLE_INPUT_DEFAULTS)

    cur_sig, cur_meta = build_single_sig_and_meta(inputs, cfg)
    invalidate_if_signature_changed(state, cur_sig, cur_meta=cur_meta)

    asset_type = str(inputs["asset_type"])
    symbol = str(inputs["symbol_input"] or "").strip()
    symbol_missing = (not symbol) and (asset_type != "bond_us_yield")
    display_name = (
        format_asset_display(asset_type, symbol or "10Y")
        if not symbol_missing
        else "未填写资产代码"
    )

    render_workspace_topbar(
        st_module,
        view_title="分析资产",
        context_label=display_name,
        status_kind=workspace_status(state, inputs_complete=not symbol_missing),
        cfg=cfg,
    )

    recompute_clicked = False
    with st_module.container():
        if state.get("change_notice"):
            recompute_clicked = render_change_notice(
                st_module,
                notice=state.get("change_notice"),
                button_label="重新计算",
                button_key="single_recompute_btn",
                disabled=bool(cfg.get("date_error") or symbol_missing),
            )

    existing_result: AnalysisResult | None = state.get("result")
    show_editor = (existing_result is None) or bool(state.get("editor_open"))

    run_clicked = False
    if show_editor:
        hydrate_widget_inputs(st_module, SINGLE_INPUT_DEFAULTS)
        adv_expanded = bool(state.pop("_keep_adv_open", False))
        with st_module.container(border=True):
            st_module.selectbox(
                "快速按名称搜索常用标的（可选）",
                options=(None, *POPULAR_ASSETS),
                index=0,
                format_func=lambda item: (
                    "输入中文名称或代码快速填入（可选）..."
                    if item is None
                    else format_asset_picker_option(item)
                ),
                key="single_quick_pick",
                on_change=lambda: _apply_quick_pick(st_module),
            )

            c_type, c_sym, c_btn = st_module.columns([2, 2, 1.4])
            asset_type = str(
                c_type.selectbox(
                    "资产类别",
                    options=SUPPORTED_ASSET_TYPES,
                    format_func=lambda key: ASSET_LABELS.get(key, key),
                    key="asset_type",
                    on_change=lambda: sync_symbol_with_asset_type(st_module),
                )
            )
            symbol = str(
                c_sym.text_input(
                    "资产代码",
                    key="symbol_input",
                    help="支持 ETF / A股 / 美股 / 港股 / 全球指数 / 黄金 / 加密货币 / 外汇 / 债券。",
                )
            ).strip()
            symbol_missing = (not symbol) and (asset_type != "bond_us_yield")
            with c_btn:
                st_module.markdown('<div class="qa-btn-spacer"></div>', unsafe_allow_html=True)
                run_clicked = bool(
                    st_module.button(
                        "运行单资产分析",
                        key="single_run_btn",
                        type="primary",
                        width="stretch",
                        disabled=bool(cfg.get("date_error") or symbol_missing),
                    )
                )

            if symbol_missing:
                st_module.warning(
                    "输入未完成：请输入有效的资产代码（如 `510300` 或 `AAPL`；美债利率可留空默认 10Y）。"
                )

            with st_module.expander(
                "高级参数：复权方式、估值币种、收益率口径与技术指标",
                expanded=adv_expanded,
            ):
                a1, a2, a3 = st_module.columns(3)
                supported_adjusts = SUPPORTED_ADJUSTMENTS.get(asset_type)
                if supported_adjusts:
                    cur_adj = st_module.session_state.get("adjust_input")
                    if cur_adj not in supported_adjusts:
                        st_module.session_state["adjust_input"] = supported_adjusts[0]
                    a1.selectbox(
                        "复权口径",
                        options=supported_adjusts,
                        format_func=lambda code: ADJUST_LABELS.get(code, code),
                        key="adjust_input",
                    )
                else:
                    a1.selectbox(
                        "复权口径",
                        options=("该资产不适用复权",),
                        index=0,
                        disabled=True,
                        key="adjust_input_disabled",
                    )

                a2.selectbox(
                    "估值基准货币",
                    options=("LOCAL", "CNY", "USD"),
                    format_func=lambda c: BASE_CURRENCY_LABELS.get(c, c),
                    key="single_base_currency_input",
                    help="默认保留原生货币 LOCAL，如需统一折算人民币/美元可切换。",
                )
                a3.selectbox(
                    "收益率计算口径",
                    options=("log", "simple"),
                    format_func=lambda m: RETURN_CONVENTION_LABELS.get(m, m),
                    key="single_return_convention_input",
                )

                t1, t2, t3, t4 = st_module.columns(4)
                sma_short_val = int(
                    t1.number_input("短期均线 (SMA Short)", 2, 120, step=1, key="sma_short_input")
                )
                sma_long_val = int(
                    t2.number_input("长期均线 (SMA Long)", 5, 250, step=5, key="sma_long_input")
                )
                t3.number_input("布林带窗口 (BB Window)", 2, 120, step=1, key="bb_window_input")
                t4.number_input(
                    "年化基准天数",
                    min_value=1,
                    max_value=366,
                    step=1,
                    key="single_trading_days_input",
                )
                if sma_short_val >= sma_long_val:
                    st_module.caption("短期均线窗口应严格小于长期均线窗口。")
                st_module.checkbox(
                    "计算并绘制动量副图 (RSI-14 / MACD)",
                    key="include_momentum_input",
                )
                guide_md = format_asset_guide_markdown(asset_type)
                if guide_md:
                    st_module.markdown(guide_md)

            if existing_result is not None:
                if st_module.button("收起参数，专注查看结果", key="single_collapse_editor_btn"):
                    state["editor_open"] = False
                    st_module.rerun()

        save_widget_inputs(st_module, SINGLE_WIDGET_KEYS)
        inputs = read_persisted_inputs(st_module, SINGLE_INPUT_DEFAULTS)
        cur_sig, cur_meta = build_single_sig_and_meta(inputs, cfg)
        if invalidate_if_signature_changed(state, cur_sig, cur_meta=cur_meta):
            state["_keep_adv_open"] = True
            st_module.rerun()
    else:
        # Compact 2-row results-first summary & action bar (no truncation on narrower panes)
        c_sum, c_edit = st_module.columns([3.2, 1.0])
        c_sum.markdown(
            f'<div class="qa-summary-inline">'
            f'<span class="qa-summary-strong">{html.escape(display_name)}</span>'
            f'<span class="qa-summary-meta">{html.escape(cur_meta["adjust"])} · {html.escape(cur_meta["base_currency"])} · {html.escape(cur_meta["return_convention"])}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if c_edit.button("调整参数", key="single_toggle_editor_btn", width="stretch"):
            state["editor_open"] = True
            st_module.rerun()
        c_cmp, c_port, c_exp = st_module.columns([1.3, 1.1, 1.3])
        if c_cmp.button(
            "与其它资产比较 →",
            key="single_to_compare_btn",
            type="primary",
            width="stretch",
        ):
            _handoff_to_compare(st_module, asset_type, symbol or "10Y")
        if c_port.button(
            "加入组合 →",
            key="single_to_portfolio_btn",
            width="stretch",
        ):
            _handoff_to_portfolio(st_module, asset_type, symbol or "10Y")

    auto_run = consume_auto_run(st_module, "single")
    if run_clicked or recompute_clicked or auto_run:
        eff_type = str(inputs["asset_type"])
        eff_sym = str(inputs["symbol_input"] or "").strip() or "10Y"
        eff_adj = _resolve_effective_adjust(eff_type, inputs.get("adjust_input"))
        eff_ccy = str(inputs["single_base_currency_input"])
        eff_conv = str(inputs["single_return_convention_input"])
        sma_s = int(inputs["sma_short_input"])
        sma_l = int(inputs["sma_long_input"])
        bb_w = int(inputs["bb_window_input"])
        td = int(inputs["single_trading_days_input"])
        inc_mom = bool(inputs["include_momentum_input"])

        if not cfg["dates_valid"]:
            store_workspace_error(
                state,
                signature=cur_sig,
                error_info=make_validation_error(str(cfg["date_error"] or "请选择有效日期区间。")),
            )
        elif sma_s >= sma_l:
            store_workspace_error(
                state,
                signature=cur_sig,
                error_info=make_validation_error(
                    "参数错误：短期均线窗口 (SMA Short) 必须严格小于长期均线窗口 (SMA Long)。",
                    hint="请在“高级参数”中调小短期均线或调大长期均线后重新运行。",
                ),
            )
        else:
            apply_default_network_proxy_policy()
            loader_fn, fx_fn = make_service_loaders(
                force_refresh=cfg["force_refresh"],
                cached_loader=cached_loader,
            )
            req = AnalysisRequest(
                asset_type=eff_type,
                symbol=eff_sym,
                start_date=str(cfg["start_str"]),
                end_date=str(cfg["end_str"]),
                adjust=eff_adj,
                base_currency=eff_ccy,
                allow_usdt_peg=cfg["allow_usdt_peg"],
                risk_free_rate=cfg["rf_rate"],
                trading_days=td,
                return_convention=eff_conv,
                sma_short=sma_s,
                sma_long=sma_l,
                bb_window=bb_w,
                include_momentum=inc_mom,
                force_refresh=cfg["force_refresh"],
                proxy_url=cfg["proxy_url"],
            )
            with st_module.spinner("正在拉取行情并计算单资产定量指标..."):
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
                        eff_type,
                        eff_sym,
                        eff_adj,
                        eff_ccy,
                        str(cfg["start_str"]),
                        str(cfg["end_str"]),
                        cfg["allow_usdt_peg"],
                        cfg["proxy_url"],
                    )
                    summary = _build_single_summary(res, cur_meta)
                    store_workspace_result(
                        state,
                        signature=cur_sig,
                        result=res,
                        fetch_signature=fetch_sig,
                        meta=cur_meta,
                        summary=summary,
                    )
                    record_basket_assets(
                        st_module, [(res.asset.asset_type, res.asset.canonical_symbol)]
                    )
                    if not show_editor or auto_run or recompute_clicked:
                        st_module.rerun()
                    else:
                        # Collapse editor into results-first view immediately on fresh run.
                        st_module.rerun()

    if state.get("error_info") or state.get("error"):
        render_error_banner(st_module, state.get("error_info") or state.get("error"))

    result: AnalysisResult | None = state.get("result")
    if result is None:
        render_empty_state(
            st_module,
            title="等待运行单资产分析",
            subtitle=(
                "确认上方资产与分析区间后点击「运行单资产分析」，即可查看累计收益、最大回撤、夏普比率与交互式价格图表。"
                if not state.get("change_notice")
                else "分析参数或区间已变更，点击上方「重新计算」即可更新全部图表与前后对比。"
            ),
            chips=[
                ASSET_LABELS.get(asset_type, asset_type),
                f"代码: {symbol or '10Y'}",
                f"复权: {cur_meta['adjust']}",
                f"货币: {cur_meta['base_currency']}",
            ],
        )
        _render_single_advanced_explorer(st_module, cfg, inputs, state, cached_loader=cached_loader)
        return

    figs, skipped = get_or_build_figures(
        state,
        result,
        theme_mode=cfg.get("resolved_theme_mode") or cfg.get("theme_mode", "auto"),
        on_error=lambda exc: render_error_banner(st_module, classify_exception(exc)),
    )

    # Mount export popover cleanly in top bar when editor is collapsed, or right above results when editor is open.
    file_prefix = f"{result.asset.asset_type}_{result.asset.canonical_symbol}".replace("/", "_")
    if not show_editor:
        with c_exp:
            render_export_popover(
                st_module,
                state,
                result,
                file_prefix=file_prefix,
                skipped_charts=skipped,
                key_prefix="single",
            )
    else:
        c_act1, c_act2, c_act3 = st_module.columns([1.4, 1.2, 1.4])
        if c_act1.button("与其它资产比较 →", key="single_to_compare_btn", width="stretch"):
            _handoff_to_compare(st_module, asset_type, symbol or "10Y")
        if c_act2.button("加入组合 →", key="single_to_portfolio_btn", width="stretch"):
            _handoff_to_portfolio(st_module, asset_type, symbol or "10Y")
        with c_act3:
            render_export_popover(
                st_module,
                state,
                result,
                file_prefix=file_prefix,
                skipped_charts=skipped,
                key_prefix="single",
            )

    render_quality_and_warnings(st_module, result.warnings, single_result=result)
    _render_single_takeaway(st_module, result)
    render_delta_comparison(
        st_module,
        prev_summary=state.get("prev_summary"),
        cur_summary=state.get("last_summary"),
    )
    render_single_asset_kpis(st_module, result)

    if figs is not None:
        render_chart_explorer(
            st_module,
            figs,
            skipped,
            order=SINGLE_CHART_ORDER,
            labels=SINGLE_CHART_LABELS,
            key_prefix="single",
        )

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
    st_module.caption(
        f"数据与缓存溯源：{cache_label} · 估值货币：{ccy_note} · 数据源：{result.asset.data_source or 'unknown'}"
    )

    render_single_asset_audit_and_preview(st_module, result)
    _render_single_advanced_explorer(st_module, cfg, inputs, state, cached_loader=cached_loader)
    render_report_preview_drawer(st_module, state, result, skipped_charts=skipped)
