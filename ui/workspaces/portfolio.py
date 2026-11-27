"""Workspace 3: Portfolio Construction, Optimization & OOS Backtesting via ``analyze_portfolio``."""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from asset_catalog import PRESET_PORTFOLIOS
from asset_inputs import parse_asset_list, parse_asset_token, parse_weights
from contracts import (
    SUPPORTED_CALENDAR_MODES,
    SUPPORTED_COV_METHODS,
    SUPPORTED_PORTFOLIO_STRATEGIES,
    SUPPORTED_REBALANCE_FREQS,
    SUPPORTED_VALIDATION_MODES,
    MultiAssetResult,
    PortfolioRequest,
)
from data_sources.network import apply_default_network_proxy_policy
from reporting import format_drawdown_percent, format_percent, format_ratio, is_valid_number
from services.multi_asset_service import analyze_portfolio
from ui.asset_directory import format_asset_display, parse_legs_loose, serialize_legs
from ui.components.alerts import (
    render_change_notice,
    render_delta_comparison,
    render_empty_state,
    render_error_banner,
    render_quality_and_warnings,
    render_takeaway_card,
    render_validation_protocol_banner,
)
from ui.components.asset_picker import render_visual_asset_picker
from ui.components.charts import render_chart_explorer
from ui.components.exports import render_export_popover, render_report_preview_drawer
from ui.components.header import render_workspace_topbar
from ui.components.metrics import (
    KPIItem,
    render_metric_cards,
    render_secondary_kpi_strip,
    tone_for_signed_number,
)
from ui.components.tables import (
    render_portfolio_backtest_table,
    render_portfolio_methodology_audit,
    render_portfolio_sensitivity_drawer,
    render_portfolio_weights_table,
)
from ui.errors import classify_exception, make_validation_error
from ui.labels import (
    BASE_CURRENCY_LABELS,
    CALENDAR_MODE_LABELS,
    COV_METHOD_LABELS,
    COV_SHORT_LABELS,
    KPI_HELP_TEXTS,
    PORTFOLIO_CHART_LABELS,
    PORTFOLIO_CORE_CHARTS,
    PORTFOLIO_EXTRA_CHARTS,
    REBALANCE_FREQ_LABELS,
    REBALANCE_SHORT_LABELS,
    STRATEGY_HINTS,
    STRATEGY_LABELS,
    STRATEGY_SHORT_LABELS,
    VALIDATION_MODE_HINTS,
    VALIDATION_MODE_LABELS,
    VALIDATION_SHORT_LABELS,
)
from ui.services_bridge import (
    get_or_build_figures,
    make_service_loaders,
    preview_custom_weights,
)
from ui.state import (
    WORKSPACE_PORTFOLIO_KEY,
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

PORTFOLIO_INPUT_DEFAULTS: dict[str, Any] = {
    "tab3_preset_template": next(iter(PRESET_PORTFOLIOS.keys())),
    "tab3_ef_legs_input": "\n".join(next(iter(PRESET_PORTFOLIOS.values()))),
    "portfolio_benchmark_input": "",
    "portfolio_strategy_input": "max_sharpe",
    "portfolio_custom_weights_input": "",
    "tab3_val_mode": "train_test",
    "portfolio_calendar_mode": "common_trading_dates",
    "portfolio_base_currency": "CNY",
    "portfolio_cov_method": "ledoit_wolf",
    "portfolio_rebalance_freq": "monthly",
    "portfolio_tx_bps": 10.0,
    "portfolio_min_weight": 0.0,
    "portfolio_max_weight": 1.0,
    "portfolio_train_ratio": 0.7,
    "portfolio_train_window": 60,
    "portfolio_split_date": "",
    "portfolio_frontier_points": 25,
    "portfolio_num_random": 0,
    "portfolio_trading_days_override": 0,
    "portfolio_include_sensitivity": True,
    "tab3_show_extra_charts": False,
}

PORTFOLIO_WIDGET_KEYS: tuple[str, ...] = tuple(PORTFOLIO_INPUT_DEFAULTS.keys())


def _sync_rf_from_portfolio(st_module: Any) -> None:
    val = float(st_module.session_state.get("portfolio_rf_pct_input", 1.6))
    set_mirrored_input(st_module, "rf_pct_input", val)
    set_mirrored_input(st_module, "compare_rf_pct_input", val)


def build_portfolio_sig_and_meta(
    inputs: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build deterministic cache signature and human-readable meta for Portfolio workspace."""
    legs_text = str(inputs.get("tab3_ef_legs_input", ""))
    benchmark_text = str(inputs.get("portfolio_benchmark_input", ""))
    strategy = str(inputs.get("portfolio_strategy_input", "max_sharpe"))
    custom_weights_raw = (
        str(inputs.get("portfolio_custom_weights_input", "")) if strategy == "custom" else ""
    )
    validation_mode = str(inputs.get("tab3_val_mode", "train_test"))
    calendar_mode = str(inputs.get("portfolio_calendar_mode", "common_trading_dates"))
    base_currency = str(inputs.get("portfolio_base_currency", "CNY"))
    cov_method = str(inputs.get("portfolio_cov_method", "ledoit_wolf"))
    rebalance_freq = str(inputs.get("portfolio_rebalance_freq", "monthly"))
    tx_cost_bps = float(inputs.get("portfolio_tx_bps", 10.0))
    min_weight = float(inputs.get("portfolio_min_weight", 0.0))
    max_weight = float(inputs.get("portfolio_max_weight", 1.0))
    train_ratio = float(inputs.get("portfolio_train_ratio", 0.7))
    train_window = int(inputs.get("portfolio_train_window", 60))
    split_date_raw = str(inputs.get("portfolio_split_date", "") or "").strip()
    frontier_points = int(inputs.get("portfolio_frontier_points", 25))
    num_random_portfolios = int(inputs.get("portfolio_num_random", 0))
    trading_days_override = int(inputs.get("portfolio_trading_days_override", 0))
    include_sensitivity = bool(inputs.get("portfolio_include_sensitivity", True))

    sig = (
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
        round(float(cfg.get("rf_rate", 0.016)), 8),
        cfg.get("start_str"),
        cfg.get("end_str"),
        bool(cfg.get("allow_usdt_peg", False)),
        cfg.get("proxy_url"),
    )
    parsed = tuple(parse_legs_loose(legs_text))
    date_window = (
        f"{cfg.get('start_str')} ~ {cfg.get('end_str')}"
        if cfg.get("dates_valid") and cfg.get("start_str") and cfg.get("end_str")
        else "区间待修正"
    )
    meta = {
        "scope_key": parsed,
        "date_window": date_window,
        "rf_pct": f"{float(cfg.get('rf_rate', 0.016)) * 100.0:.2f}%",
        "strategy": strategy,
        "validation_mode": validation_mode,
        "cov_method": cov_method,
        "rebalance_freq": rebalance_freq,
        "tx_cost_bps": f"{tx_cost_bps:.1f} bps",
        "calendar_mode": calendar_mode,
        "base_currency": base_currency,
        "train_ratio": f"{train_ratio:.2f}",
        "train_window": train_window,
        "min_weight": f"{min_weight:.2f}",
        "max_weight": f"{max_weight:.2f}",
        "benchmark": benchmark_text.strip() or "无",
    }
    return sig, meta


def invalidate_portfolio_from_persisted(st_module: Any, cfg: Mapping[str, Any]) -> bool:
    """Invalidate Portfolio workspace if global settings changed while another view is active."""
    state = workspace_state(st_module, WORKSPACE_PORTFOLIO_KEY)
    if (
        state.get("result") is None
        and state.get("error") is None
        and state.get("error_info") is None
    ):
        return False
    inputs = read_persisted_inputs(st_module, PORTFOLIO_INPUT_DEFAULTS)
    sig, meta = build_portfolio_sig_and_meta(inputs, cfg)
    return invalidate_if_signature_changed(state, sig, cur_meta=meta)


def _extract_primary_perf(result: MultiAssetResult, *, is_oos: bool) -> Mapping[str, Any]:
    backtest_metrics = result.metrics.get("backtest") or {}
    if not is_oos:
        return backtest_metrics.get("Portfolio") or backtest_metrics.get("OOS_Portfolio") or {}
    return backtest_metrics.get("OOS_Portfolio") or backtest_metrics.get("Portfolio") or {}


def _build_portfolio_summary(
    result: MultiAssetResult, meta: Mapping[str, Any], *, is_oos: bool
) -> dict[str, Any]:
    perf = _extract_primary_perf(result, is_oos=is_oos)
    return {
        "scope_key": meta["scope_key"],
        "context_label": str(meta["date_window"]),
        "metrics": [
            {"key": "cagr", "label": "年化回报 CAGR", "value": perf.get("cagr"), "fmt": "pct"},
            {
                "key": "max_drawdown",
                "label": "最大回撤",
                "value": perf.get("max_drawdown"),
                "fmt": "dd",
            },
            {
                "key": "sharpe_ratio",
                "label": "夏普比率",
                "value": perf.get("sharpe_ratio"),
                "fmt": "ratio",
            },
            {
                "key": "total_turnover",
                "label": "总换手",
                "value": perf.get("total_turnover"),
                "fmt": "neutral_ratio",
            },
        ],
    }


def _render_portfolio_takeaway(
    st_module: Any,
    result: MultiAssetResult,
    *,
    primary_title: str,
    primary_perf: Mapping[str, Any],
    ew_perf: Mapping[str, Any],
) -> None:
    cagr_txt = format_percent(primary_perf.get("cagr"), signed=True)
    mdd_txt = format_drawdown_percent(primary_perf.get("max_drawdown"))
    sh_txt = format_ratio(primary_perf.get("sharpe_ratio"), decimals=3)

    sh_new = primary_perf.get("sharpe_ratio")
    sh_ew = ew_perf.get("sharpe_ratio")
    vs_ew = ""
    if is_valid_number(sh_new) and is_valid_number(sh_ew):
        diff = float(sh_new) - float(sh_ew)
        ew_txt = format_ratio(sh_ew, decimals=3)
        if diff > 0.02:
            vs_ew = f"，风险调整收益优于等权基准 1/N（基准夏普 {ew_txt}，高出 {diff:+.3f}）"
        elif diff < -0.02:
            vs_ew = f"，风险调整收益略逊于等权基准 1/N（基准夏普 {ew_txt}，差值 {diff:+.3f}）"
        else:
            vs_ew = f"，与等权基准 1/N（夏普 {ew_txt}）风险收益比接近"

    weights_map = (result.metrics.get("portfolio") or {}).get("weights") or {}
    top_weight_note = ""
    if weights_map:
        top_leg, top_w = max(weights_map.items(), key=lambda x: float(x[1]))
        top_weight_note = f"最大配比资产：{top_leg}（{format_percent(top_w)}） · "

    render_takeaway_card(
        st_module,
        headline=f"{primary_title}组合年化回报 {cagr_txt}，最大回撤 {mdd_txt}，夏普比率 {sh_txt}{vs_ew}。",
        detail=f"{top_weight_note}组合共 {len(result.asset_order)} 项资产 · 计价币种 {result.methodology.get('base_currency', 'CNY')}",
    )


def _extract_result_pairs(result: MultiAssetResult) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for leg in result.asset_order:
        meta = result.assets.get(leg)
        if meta is None:
            continue
        sym = meta.symbol if meta.asset_type == "global_index" else meta.canonical_symbol
        pairs.append((meta.asset_type, sym))
    return pairs


def render_portfolio_workspace(
    st_module: Any,
    cfg: dict[str, Any],
    *,
    cached_loader: Callable[..., pd.DataFrame],
) -> None:
    """Render Workspace 3: Portfolio Optimization, Frontier & OOS Backtesting."""
    if st_module is None:
        return
    state = workspace_state(st_module, WORKSPACE_PORTFOLIO_KEY)
    inputs = read_persisted_inputs(st_module, PORTFOLIO_INPUT_DEFAULTS)

    cur_sig, cur_meta = build_portfolio_sig_and_meta(inputs, cfg)
    invalidate_if_signature_changed(state, cur_sig, cur_meta=cur_meta)

    parsed_items = parse_legs_loose(str(inputs.get("tab3_ef_legs_input", "")))
    ctx_labels = [format_asset_display(a, s) for a, s in parsed_items[:3]]
    ctx_text = " + ".join(ctx_labels) + (" ..." if len(parsed_items) > 3 else "")

    render_workspace_topbar(
        st_module,
        view_title="构建组合",
        context_label=ctx_text or "请选择至少 2 项资产",
        status_kind=workspace_status(state, inputs_complete=len(parsed_items) >= 2),
        cfg=cfg,
    )

    recompute_clicked = False
    with st_module.container():
        if state.get("change_notice"):
            recompute_clicked = render_change_notice(
                st_module,
                notice=state.get("change_notice"),
                button_label="重新计算",
                button_key="portfolio_recompute_btn",
                disabled=bool(not cfg["dates_valid"] or len(parsed_items) < 2),
            )

    existing_result: MultiAssetResult | None = state.get("result")
    show_editor = (existing_result is None) or bool(state.get("editor_open"))

    run_clicked = False
    parse_error: str | None = None
    if show_editor:
        hydrate_widget_inputs(st_module, PORTFOLIO_INPUT_DEFAULTS)
        st_module.session_state["portfolio_rf_pct_input"] = float(cfg.get("rf_pct", 1.6))
        adv_expanded = bool(state.pop("_keep_adv_open", False))
        with st_module.container(border=True, key="portfolio_editor_panel"):
            parsed_items, parse_error = render_visual_asset_picker(
                st_module,
                text_key="tab3_ef_legs_input",
                preset_key="tab3_preset_template",
                key_prefix="portfolio",
                min_assets=2,
            )

            if len(parsed_items) < 2 and not parse_error:
                st_module.info(
                    f"输入未完成：当前组合仅有 {len(parsed_items)} 项资产，请再添加至少 {2 - len(parsed_items)} 项资产后即可开始优化与回测。"
                )

            c_strat, c_val = st_module.columns(2)
            with c_strat:
                strategy = str(
                    st_module.selectbox(
                        "优化目标",
                        options=SUPPORTED_PORTFOLIO_STRATEGIES,
                        format_func=lambda s: STRATEGY_LABELS.get(s, s),
                        key="portfolio_strategy_input",
                    )
                )
                if strategy in STRATEGY_HINTS:
                    st_module.caption(STRATEGY_HINTS[strategy])

                if strategy == "custom":
                    custom_weights_raw = str(
                        st_module.text_input(
                            "自定义资产权重（逗号分隔，按上方资产顺序）",
                            placeholder="0.4, 0.3, 0.3",
                            key="portfolio_custom_weights_input",
                        )
                    )
                    parsed_w, w_err = preview_custom_weights(
                        custom_weights_raw,
                        expected_count=len(parsed_items) if parsed_items else None,
                    )
                    if w_err:
                        st_module.caption(f"权重校验：{w_err}")
                    elif parsed_w is not None:
                        st_module.caption(
                            f"已识别 {len(parsed_w)} 项权重（合计 {sum(parsed_w):.4f}，求解时自动归一化）。"
                        )

            with c_val:
                validation_mode = str(
                    st_module.selectbox(
                        "回测验证方式",
                        options=SUPPORTED_VALIDATION_MODES,
                        format_func=lambda m: VALIDATION_MODE_LABELS.get(m, m),
                        key="tab3_val_mode",
                    )
                )
                if validation_mode in VALIDATION_MODE_HINTS:
                    st_module.caption(VALIDATION_MODE_HINTS[validation_mode])

            with st_module.expander(
                "专业参数：协方差估计、无风险利率、调仓频率与回测约束",
                expanded=adv_expanded,
            ):
                p1, p2, p3 = st_module.columns(3)
                p1.selectbox(
                    "协方差矩阵估计",
                    options=SUPPORTED_COV_METHODS,
                    format_func=lambda m: COV_METHOD_LABELS.get(m, m),
                    key="portfolio_cov_method",
                )
                p2.number_input(
                    "无风险年化利率 Rf (%)",
                    min_value=0.0,
                    max_value=20.0,
                    step=0.1,
                    key="portfolio_rf_pct_input",
                    on_change=lambda: _sync_rf_from_portfolio(st_module),
                    help="与全局左侧面板同步；用于计算夏普比率与切线组合。",
                )
                p3.selectbox(
                    "再平衡调仓频率",
                    options=SUPPORTED_REBALANCE_FREQS,
                    format_func=lambda f: REBALANCE_FREQ_LABELS.get(f, f),
                    key="portfolio_rebalance_freq",
                )

                m1, m2, m3, m4 = st_module.columns(4)
                m1.number_input(
                    "单边换手交易成本 (bps)",
                    min_value=0.0,
                    max_value=500.0,
                    step=5.0,
                    key="portfolio_tx_bps",
                )
                m2.selectbox(
                    "日历对齐",
                    options=SUPPORTED_CALENDAR_MODES,
                    format_func=lambda m: CALENDAR_MODE_LABELS.get(m, m),
                    key="portfolio_calendar_mode",
                )
                m3.selectbox(
                    "组合计价币种",
                    options=("CNY", "USD", "LOCAL"),
                    format_func=lambda c: BASE_CURRENCY_LABELS.get(c, c),
                    key="portfolio_base_currency",
                )
                m4.text_input(
                    "组合对比基准（可选）",
                    placeholder="etf:510300",
                    key="portfolio_benchmark_input",
                )

                b1, b2, b3, b4 = st_module.columns(4)
                b1.number_input("单资产权重下界", 0.0, 0.99, step=0.05, key="portfolio_min_weight")
                b2.number_input("单资产权重上界", 0.01, 1.0, step=0.05, key="portfolio_max_weight")
                b3.number_input(
                    "Train 样本占比 (train_test)",
                    0.2,
                    0.9,
                    step=0.05,
                    key="portfolio_train_ratio",
                )
                b4.number_input(
                    "Walk-Forward 训练窗口",
                    min_value=15,
                    max_value=504,
                    step=10,
                    key="portfolio_train_window",
                )

                e1, e2, e3, e4 = st_module.columns(4)
                e1.text_input(
                    "指定划分日期 (可选 YYYY-MM-DD)",
                    key="portfolio_split_date",
                )
                e2.number_input("精确前沿点数", 0, 100, step=5, key="portfolio_frontier_points")
                e3.number_input(
                    "随机散点模拟数 (0 = 仅精确前沿)",
                    min_value=0,
                    max_value=3000,
                    step=100,
                    key="portfolio_num_random",
                    help="默认 0（仅计算 SLSQP 数学精确前沿）；设为 >0 添加可行域散点云。",
                )
                e4.number_input(
                    "年化期数覆盖 (0 = 自动)",
                    min_value=0,
                    max_value=366,
                    step=1,
                    key="portfolio_trading_days_override",
                )

                c_opt1, c_opt2 = st_module.columns(2)
                c_opt1.checkbox(
                    "计算参数稳健性敏感性矩阵（协方差 / 交易成本 / 调仓频率）",
                    key="portfolio_include_sensitivity",
                )
                c_opt2.checkbox(
                    "显示附加图表（标的归一化、水下回撤、动态权重演化）",
                    key="tab3_show_extra_charts",
                )
                st_module.text_area(
                    "批量资产列表（高级文本编辑，每行或分号分隔 `类型:代码`）",
                    key="tab3_ef_legs_input",
                    height=96,
                )

            if existing_result is not None:
                b_run, b_close = st_module.columns([3, 1])
                run_clicked = bool(
                    b_run.button(
                        "运行组合优化与回测",
                        key="tab3_btn_ef",
                        type="primary",
                        width="stretch",
                        disabled=bool(
                            not cfg["dates_valid"] or len(parsed_items) < 2 or parse_error
                        ),
                    )
                )
                if b_close.button("收起参数", key="portfolio_collapse_editor_btn", width="stretch"):
                    state["editor_open"] = False
                    st_module.rerun()
            else:
                run_clicked = bool(
                    st_module.button(
                        "运行组合优化与回测",
                        key="tab3_btn_ef",
                        type="primary",
                        width="stretch",
                        disabled=bool(
                            not cfg["dates_valid"] or len(parsed_items) < 2 or parse_error
                        ),
                    )
                )

        save_widget_inputs(st_module, PORTFOLIO_WIDGET_KEYS)
        inputs = read_persisted_inputs(st_module, PORTFOLIO_INPUT_DEFAULTS)
        cur_sig, cur_meta = build_portfolio_sig_and_meta(inputs, cfg)
        if invalidate_if_signature_changed(state, cur_sig, cur_meta=cur_meta):
            state["_keep_adv_open"] = True
            st_module.rerun()
    else:
        strat_short = STRATEGY_SHORT_LABELS.get(cur_meta["strategy"], cur_meta["strategy"])
        val_short = VALIDATION_SHORT_LABELS.get(
            cur_meta["validation_mode"], cur_meta["validation_mode"]
        )
        cov_short = COV_SHORT_LABELS.get(cur_meta["cov_method"], cur_meta["cov_method"])
        reb_short = REBALANCE_SHORT_LABELS.get(
            cur_meta["rebalance_freq"], cur_meta["rebalance_freq"]
        )
        c_sum, c_edit = st_module.columns([3.2, 1.0])
        c_sum.markdown(
            f'<div class="qa-summary-inline">'
            f'<span class="qa-summary-strong">{html.escape(strat_short)} · {html.escape(val_short)}</span>'
            f'<span class="qa-summary-meta">协方差 {html.escape(cov_short)} · Rf {html.escape(cur_meta["rf_pct"])} · {html.escape(reb_short)}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if c_edit.button("调整参数", key="portfolio_toggle_editor_btn", width="stretch"):
            state["editor_open"] = True
            st_module.rerun()
        c_cmp, c_exp = st_module.columns([1.3, 1.3])
        if c_cmp.button("去比较资产 →", key="portfolio_to_compare_btn", width="stretch"):
            queue_transition(
                st_module,
                view="compare",
                updates={
                    "compare_assets_input": serialize_legs(_extract_result_pairs(existing_result))
                },
            )
            st_module.rerun()

    strategy = str(inputs["portfolio_strategy_input"])
    validation_mode = str(inputs["tab3_val_mode"])
    show_extra_charts = bool(inputs["tab3_show_extra_charts"])

    if run_clicked or recompute_clicked:
        legs_text = str(inputs["tab3_ef_legs_input"])
        benchmark_text = str(inputs["portfolio_benchmark_input"])
        custom_weights_raw = str(inputs["portfolio_custom_weights_input"])
        calendar_mode = str(inputs["portfolio_calendar_mode"])
        base_currency = str(inputs["portfolio_base_currency"])
        cov_method = str(inputs["portfolio_cov_method"])
        rebalance_freq = str(inputs["portfolio_rebalance_freq"])
        tx_cost_bps = float(inputs["portfolio_tx_bps"])
        min_weight = float(inputs["portfolio_min_weight"])
        max_weight = float(inputs["portfolio_max_weight"])
        train_ratio = float(inputs["portfolio_train_ratio"])
        train_window = int(inputs["portfolio_train_window"])
        split_date_raw = str(inputs["portfolio_split_date"] or "").strip()
        frontier_points = int(inputs["portfolio_frontier_points"])
        num_random_portfolios = int(inputs["portfolio_num_random"])
        trading_days_override = int(inputs["portfolio_trading_days_override"])
        include_sensitivity = bool(inputs["portfolio_include_sensitivity"])

        if not cfg["dates_valid"]:
            store_workspace_error(
                state,
                signature=cur_sig,
                error_info=make_validation_error(str(cfg["date_error"] or "请选择有效日期区间。")),
            )
        else:
            apply_default_network_proxy_policy()
            loader_fn, fx_fn = make_service_loaders(
                force_refresh=cfg["force_refresh"],
                cached_loader=cached_loader,
            )
            with st_module.spinner("正在执行组合协方差估计、SLSQP 优化与样本外回测..."):
                try:
                    specs = parse_asset_list([legs_text], syntax="web")
                    bench_spec = (
                        parse_asset_token(benchmark_text, syntax="web")
                        if benchmark_text.strip()
                        else None
                    )
                    custom_w = (
                        parse_weights(custom_weights_raw, web=True)
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
                    store_workspace_error(
                        state,
                        signature=cur_sig,
                        error_info=classify_exception(exc),
                    )
                else:
                    res_is_oos = str(
                        res.methodology.get("validation_mode") or validation_mode
                    ) != "in_sample" and bool(res.methodology.get("is_out_of_sample", True))
                    summary = _build_portfolio_summary(res, cur_meta, is_oos=res_is_oos)
                    store_workspace_result(
                        state,
                        signature=cur_sig,
                        result=res,
                        meta=cur_meta,
                        summary=summary,
                    )
                    record_basket_assets(st_module, _extract_result_pairs(res))
                    st_module.rerun()

    if state.get("error_info") or state.get("error"):
        render_error_banner(st_module, state.get("error_info") or state.get("error"))

    result: MultiAssetResult | None = state.get("result")
    if result is None:
        render_empty_state(
            st_module,
            title="等待运行投资组合优化与回测",
            subtitle=(
                "确认资产、优化目标与验证方式后点击「运行组合优化与回测」，系统将计算配置权重、风险贡献、有效前沿与样本外净值。"
                if not state.get("change_notice")
                else "分析参数或区间已变更，点击上方「重新计算」即可更新回测并对比前后结果。"
            ),
            chips=[
                f"资产数: {len(parsed_items)}",
                f"策略: {strategy}",
                f"协方差: {cur_meta['cov_method']}",
                f"调仓: {cur_meta['rebalance_freq']}",
            ],
        )
        return

    chart_selection = list(PORTFOLIO_CORE_CHARTS)
    if show_extra_charts:
        chart_selection.extend(PORTFOLIO_EXTRA_CHARTS)
        if "weight_history" in result.frames:
            chart_selection.append("weight_history")

    figs, skipped = get_or_build_figures(
        state,
        result,
        chart_types=tuple(chart_selection),
        on_error=lambda exc: render_error_banner(st_module, classify_exception(exc)),
    )

    file_prefix = f"portfolio_{strategy}_{validation_mode}"
    if not show_editor:
        with c_exp:
            render_export_popover(
                st_module,
                state,
                result,
                file_prefix=file_prefix,
                skipped_charts=skipped,
                key_prefix="portfolio",
            )
    else:
        c_act1, c_act2 = st_module.columns([2, 1.4])
        if c_act1.button("返回比较资产 →", key="portfolio_to_compare_btn", width="stretch"):
            queue_transition(
                st_module,
                view="compare",
                updates={"compare_assets_input": serialize_legs(_extract_result_pairs(result))},
            )
            st_module.rerun()
        with c_act2:
            render_export_popover(
                st_module,
                state,
                result,
                file_prefix=file_prefix,
                skipped_charts=skipped,
                key_prefix="portfolio",
            )

    is_oos, has_lookahead, _ = render_validation_protocol_banner(
        st_module, result, fallback_mode=validation_mode
    )
    render_quality_and_warnings(st_module, result.warnings)

    backtest_metrics = result.metrics.get("backtest") or {}
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

    _render_portfolio_takeaway(
        st_module,
        result,
        primary_title=primary_title,
        primary_perf=primary_perf,
        ew_perf=ew_perf,
    )
    render_delta_comparison(
        st_module,
        prev_summary=state.get("prev_summary"),
        cur_summary=state.get("last_summary"),
    )

    render_metric_cards(
        st_module,
        [
            (
                f"{primary_title} CAGR",
                format_percent(primary_perf.get("cagr")),
                KPI_HELP_TEXTS["portfolio_cagr"],
            ),
            (
                f"{primary_title}最大回撤",
                format_drawdown_percent(primary_perf.get("max_drawdown")),
                KPI_HELP_TEXTS["portfolio_mdd"],
            ),
            (
                f"{primary_title}夏普比率",
                format_ratio(primary_perf.get("sharpe_ratio"), decimals=3),
                KPI_HELP_TEXTS["portfolio_sharpe"],
            ),
        ],
    )

    port_meta = result.metrics.get("portfolio") or {}
    rc_meta = result.metrics.get("risk_contributions") or {}
    cov_meta = result.metrics.get("covariance") or {}
    cum_ret = primary_perf.get("cumulative_return")
    ann_vol = primary_perf.get("annualized_volatility")
    dr_val = rc_meta.get("diversification_ratio")
    enb_w = rc_meta.get("effective_n_weights")
    enb_r = rc_meta.get("effective_n_risk")
    cond_num = cov_meta.get("condition_number")

    render_secondary_kpi_strip(
        st_module,
        [
            KPIItem(
                label=f"{primary_title}累计收益",
                value=format_percent(cum_ret, signed=True),
                sub=f"年化波动 {format_percent(ann_vol)}",
                tone=tone_for_signed_number(cum_ret),
            ),
            KPIItem(
                label="分散化比率 (DR)",
                value=format_ratio(dr_val, decimals=3),
                sub=f"求解：{port_meta.get('solver_message', 'ok')}",
                tone="pos" if is_valid_number(dr_val) and float(dr_val) > 1.05 else "neutral",
            ),
            KPIItem(
                label="有效资产数 ENB (权重 / 风险)",
                value=f"{format_ratio(enb_w, decimals=2)} / {format_ratio(enb_r, decimals=2)}",
                sub="分散度结构诊断",
                tone="neutral",
            ),
            KPIItem(
                label="换手率与估算交易成本",
                value=f"{format_ratio(primary_perf.get('total_turnover'), decimals=2)} / {format_percent(primary_perf.get('total_cost_ratio'))}",
                sub=f"协方差条件数 {format_ratio(cond_num, decimals=2)}",
                tone="neutral",
            ),
        ],
    )

    st_module.caption(
        f"求解状态：`{port_meta.get('solver_message', 'ok')}` | "
        f"协方差：`{result.methodology.get('cov_method', cur_meta['cov_method'])}` | "
        f"调仓：`{result.parameters.get('rebalance_freq', cur_meta['rebalance_freq'])}` | "
        f"Rf：{float(result.parameters.get('risk_free_rate', cfg['rf_rate'])) * 100.0:.2f}%"
    )

    if figs is not None:
        render_chart_explorer(
            st_module,
            figs,
            skipped,
            order=chart_selection,
            labels=PORTFOLIO_CHART_LABELS,
            key_prefix="portfolio",
        )

    render_portfolio_backtest_table(
        st_module,
        result,
        primary_title=primary_title,
        strategy=strategy,
        primary_perf=primary_perf,
        ew_perf=ew_perf,
    )
    render_portfolio_weights_table(st_module, result)
    render_portfolio_sensitivity_drawer(st_module, result)
    render_portfolio_methodology_audit(st_module, result)
    render_report_preview_drawer(st_module, state, result, skipped_charts=skipped)
