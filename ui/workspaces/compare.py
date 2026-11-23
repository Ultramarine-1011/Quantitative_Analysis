"""Workspace 2: Multi-Asset Comparison with visual picker, results-first view, and handoff."""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from asset_catalog import PRESET_PORTFOLIOS
from asset_inputs import parse_asset_list, parse_asset_token
from contracts import (
    SUPPORTED_CALENDAR_MODES,
    SUPPORTED_COV_METHODS,
    CompareRequest,
    MultiAssetResult,
)
from data_sources.network import apply_default_network_proxy_policy
from reporting import (
    format_drawdown_percent,
    format_integer,
    format_percent,
    format_ratio,
    is_valid_number,
)
from services.multi_asset_service import compare_assets
from ui.asset_directory import format_asset_display, parse_legs_loose, serialize_legs
from ui.components.alerts import (
    render_change_notice,
    render_delta_comparison,
    render_empty_state,
    render_error_banner,
    render_quality_and_warnings,
    render_takeaway_card,
)
from ui.components.asset_picker import render_visual_asset_picker
from ui.components.charts import render_chart_explorer
from ui.components.exports import render_export_popover, render_report_preview_drawer
from ui.components.header import render_workspace_topbar
from ui.components.metrics import KPIItem, render_metric_cards, render_secondary_kpi_strip
from ui.components.tables import (
    render_compare_benchmark_table,
    render_compare_distribution_table,
    render_compare_matrices_audit,
    render_compare_performance_table,
)
from ui.errors import classify_exception, make_validation_error
from ui.labels import (
    BASE_CURRENCY_LABELS,
    CALENDAR_MODE_LABELS,
    CALENDAR_SHORT_LABELS,
    COMPARE_CHART_LABELS,
    COMPARE_CHART_ORDER,
    COV_METHOD_LABELS,
    COV_SHORT_LABELS,
    RETURN_CONVENTION_LABELS,
)
from ui.services_bridge import get_or_build_figures, make_service_loaders
from ui.state import (
    WORKSPACE_COMPARE_KEY,
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

COMPARE_INPUT_DEFAULTS: dict[str, Any] = {
    "compare_preset_template": next(iter(PRESET_PORTFOLIOS.keys())),
    "compare_assets_input": "\n".join(next(iter(PRESET_PORTFOLIOS.values()))),
    "compare_benchmark_input": "",
    "compare_calendar_mode": "common_trading_dates",
    "compare_base_currency": "CNY",
    "compare_return_convention": "simple",
    "compare_cov_method": "ledoit_wolf",
    "compare_rolling_window": 20,
    "compare_trading_days_override": 0,
}

COMPARE_WIDGET_KEYS: tuple[str, ...] = tuple(COMPARE_INPUT_DEFAULTS.keys())


def _sync_rf_from_compare(st_module: Any) -> None:
    val = float(st_module.session_state.get("compare_rf_pct_input", 1.6))
    set_mirrored_input(st_module, "rf_pct_input", val)
    set_mirrored_input(st_module, "portfolio_rf_pct_input", val)


def build_compare_sig_and_meta(
    inputs: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build deterministic cache signature and human-readable meta for Compare workspace."""
    assets_text = str(inputs.get("compare_assets_input", ""))
    benchmark_text = str(inputs.get("compare_benchmark_input", ""))
    calendar_mode = str(inputs.get("compare_calendar_mode", "common_trading_dates"))
    base_currency = str(inputs.get("compare_base_currency", "CNY"))
    return_convention = str(inputs.get("compare_return_convention", "simple"))
    cov_method = str(inputs.get("compare_cov_method", "ledoit_wolf"))
    rolling_window = int(inputs.get("compare_rolling_window", 20))
    trading_days_override = int(inputs.get("compare_trading_days_override", 0))

    sig = (
        assets_text,
        benchmark_text,
        calendar_mode,
        base_currency,
        return_convention,
        cov_method,
        rolling_window,
        trading_days_override,
        round(float(cfg.get("rf_rate", 0.016)), 8),
        cfg.get("start_str"),
        cfg.get("end_str"),
        bool(cfg.get("allow_usdt_peg", False)),
        cfg.get("proxy_url"),
    )
    parsed = tuple(parse_legs_loose(assets_text))
    date_window = (
        f"{cfg.get('start_str')} ~ {cfg.get('end_str')}"
        if cfg.get("dates_valid") and cfg.get("start_str") and cfg.get("end_str")
        else "区间待修正"
    )
    meta = {
        "scope_key": parsed,
        "date_window": date_window,
        "rf_pct": f"{float(cfg.get('rf_rate', 0.016)) * 100.0:.2f}%",
        "calendar_mode": calendar_mode,
        "base_currency": base_currency,
        "return_convention": return_convention,
        "cov_method": cov_method,
        "rolling_window": rolling_window,
        "benchmark": benchmark_text.strip() or "无",
    }
    return sig, meta


def invalidate_compare_from_persisted(st_module: Any, cfg: Mapping[str, Any]) -> bool:
    """Invalidate Compare workspace if global settings changed while another view is active."""
    state = workspace_state(st_module, WORKSPACE_COMPARE_KEY)
    if (
        state.get("result") is None
        and state.get("error") is None
        and state.get("error_info") is None
    ):
        return False
    inputs = read_persisted_inputs(st_module, COMPARE_INPUT_DEFAULTS)
    sig, meta = build_compare_sig_and_meta(inputs, cfg)
    return invalidate_if_signature_changed(state, sig, cur_meta=meta)


def _build_compare_summary(res: MultiAssetResult, meta: Mapping[str, Any]) -> dict[str, Any]:
    per_asset = res.metrics.get("per_asset") or {}
    cagrs = [float(m["cagr"]) for m in per_asset.values() if is_valid_number(m.get("cagr"))]
    sharpes = [
        float(m["sharpe_ratio"])
        for m in per_asset.values()
        if is_valid_number(m.get("sharpe_ratio"))
    ]
    mdds = [
        float(m["max_drawdown"])
        for m in per_asset.values()
        if is_valid_number(m.get("max_drawdown"))
    ]
    return {
        "scope_key": meta["scope_key"],
        "context_label": str(meta["date_window"]),
        "metrics": [
            {
                "key": "best_cagr",
                "label": "最高资产 CAGR",
                "value": max(cagrs) if cagrs else None,
                "fmt": "pct",
            },
            {
                "key": "best_sharpe",
                "label": "最高资产夏普",
                "value": max(sharpes) if sharpes else None,
                "fmt": "ratio",
            },
            {
                "key": "best_mdd",
                "label": "最浅最大回撤",
                "value": max(mdds) if mdds else None,
                "fmt": "dd",
            },
        ],
    }


def _render_compare_takeaway(st_module: Any, result: MultiAssetResult) -> None:
    per_asset = result.metrics.get("per_asset") or {}
    if not per_asset:
        return
    cagr_pairs = [
        (leg, float(m["cagr"])) for leg, m in per_asset.items() if is_valid_number(m.get("cagr"))
    ]
    dd_pairs = [
        (leg, float(m["max_drawdown"]))
        for leg, m in per_asset.items()
        if is_valid_number(m.get("max_drawdown"))
    ]
    sharpe_pairs = [
        (leg, float(m["sharpe_ratio"]))
        for leg, m in per_asset.items()
        if is_valid_number(m.get("sharpe_ratio"))
    ]
    parts: list[str] = []
    if cagr_pairs:
        top = max(cagr_pairs, key=lambda x: x[1])
        parts.append(f"收益领先：{top[0]}（年化 {format_percent(top[1], signed=True)}）")
    if sharpe_pairs:
        top_s = max(sharpe_pairs, key=lambda x: x[1])
        parts.append(f"风险收益比最优：{top_s[0]}（夏普 {format_ratio(top_s[1], decimals=3)}）")
    if dd_pairs:
        # max_drawdown is stored as non-negative depth (>= 0), so min() is shallowest drawdown.
        best_dd = min(dd_pairs, key=lambda x: x[1])
        parts.append(f"回撤控制最佳：{best_dd[0]}（{format_drawdown_percent(best_dd[1])}）")

    eff_start = result.data_summary.get("effective_start") or ""
    eff_end = result.data_summary.get("effective_end") or ""
    render_takeaway_card(
        st_module,
        headline="；".join(parts) + "。"
        if parts
        else f"已完成 {len(result.asset_order)} 项资产统一口径比较。",
        detail=f"统一估值货币 {result.methodology.get('base_currency', 'CNY')} · 共同有效区间 {eff_start} ~ {eff_end}",
    )


def _render_compare_highlights(st_module: Any, result: MultiAssetResult) -> None:
    per_asset = result.metrics.get("per_asset") or {}
    if not per_asset:
        return
    cagr_pairs = [
        (leg, float(m["cagr"])) for leg, m in per_asset.items() if is_valid_number(m.get("cagr"))
    ]
    sharpe_pairs = [
        (leg, float(m["sharpe_ratio"]))
        for leg, m in per_asset.items()
        if is_valid_number(m.get("sharpe_ratio"))
    ]
    dd_pairs = [
        (leg, float(m["max_drawdown"]))
        for leg, m in per_asset.items()
        if is_valid_number(m.get("max_drawdown"))
    ]
    items: list[KPIItem] = []
    if cagr_pairs:
        best_cagr = max(cagr_pairs, key=lambda x: x[1])
        items.append(
            KPIItem(
                label="区间最高 CAGR",
                value=format_percent(best_cagr[1], signed=True),
                sub=best_cagr[0],
                tone="pos" if best_cagr[1] > 0 else "neg",
            )
        )
    if sharpe_pairs:
        best_sh = max(sharpe_pairs, key=lambda x: x[1])
        items.append(
            KPIItem(
                label="最优风险调整收益 (Sharpe)",
                value=format_ratio(best_sh[1], decimals=3),
                sub=best_sh[0],
                tone="pos" if best_sh[1] > 0 else "neg",
            )
        )
    if dd_pairs:
        min_dd = min(dd_pairs, key=lambda x: x[1])
        items.append(
            KPIItem(
                label="最小历史回撤",
                value=format_drawdown_percent(min_dd[1]),
                sub=min_dd[0],
                tone="neutral",
            )
        )
    items.append(
        KPIItem(
            label="对比资产数量",
            value=str(len(result.asset_order)),
            sub=f"收益口径：{result.methodology.get('return_convention', 'simple')}",
            tone="neutral",
        )
    )
    render_secondary_kpi_strip(st_module, items)


def _extract_result_pairs(result: MultiAssetResult) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for leg in result.asset_order:
        meta = result.assets.get(leg)
        if meta is None:
            continue
        sym = meta.symbol if meta.asset_type == "global_index" else meta.canonical_symbol
        pairs.append((meta.asset_type, sym))
    return pairs


def _handoff_to_portfolio(
    st_module: Any,
    pairs: list[tuple[str, str]],
    inputs: Mapping[str, Any],
) -> None:
    record_basket_assets(st_module, pairs)
    queue_transition(
        st_module,
        view="portfolio",
        updates={
            "tab3_ef_legs_input": serialize_legs(pairs),
            "portfolio_base_currency": str(inputs.get("compare_base_currency", "CNY")),
            "portfolio_calendar_mode": str(
                inputs.get("compare_calendar_mode", "common_trading_dates")
            ),
            "portfolio_cov_method": str(inputs.get("compare_cov_method", "ledoit_wolf")),
            "portfolio_benchmark_input": str(inputs.get("compare_benchmark_input", "")),
        },
    )
    st_module.rerun()


def render_compare_workspace(
    st_module: Any,
    cfg: dict[str, Any],
    *,
    cached_loader: Callable[..., pd.DataFrame],
) -> None:
    """Render Workspace 2: Multi-Asset Comparison backed by ``compare_assets``."""
    if st_module is None:
        return
    state = workspace_state(st_module, WORKSPACE_COMPARE_KEY)
    inputs = read_persisted_inputs(st_module, COMPARE_INPUT_DEFAULTS)

    cur_sig, cur_meta = build_compare_sig_and_meta(inputs, cfg)
    invalidate_if_signature_changed(state, cur_sig, cur_meta=cur_meta)

    parsed_items = parse_legs_loose(str(inputs.get("compare_assets_input", "")))
    ctx_labels = [format_asset_display(a, s) for a, s in parsed_items[:3]]
    ctx_text = " vs ".join(ctx_labels) + (" ..." if len(parsed_items) > 3 else "")

    render_workspace_topbar(
        st_module,
        view_title="比较资产",
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
                button_key="compare_recompute_btn",
                disabled=bool(not cfg["dates_valid"] or len(parsed_items) < 2),
            )

    existing_result: MultiAssetResult | None = state.get("result")
    show_editor = (existing_result is None) or bool(state.get("editor_open"))

    run_clicked = False
    parse_error: str | None = None
    if show_editor:
        hydrate_widget_inputs(st_module, COMPARE_INPUT_DEFAULTS)
        st_module.session_state["compare_rf_pct_input"] = float(cfg.get("rf_pct", 1.6))
        adv_expanded = bool(state.pop("_keep_adv_open", False))
        with st_module.container(border=True):
            parsed_items, parse_error = render_visual_asset_picker(
                st_module,
                text_key="compare_assets_input",
                preset_key="compare_preset_template",
                key_prefix="compare",
                min_assets=2,
            )

            if len(parsed_items) < 2 and not parse_error:
                st_module.info(
                    f"输入未完成：当前已选 {len(parsed_items)} 项资产，请再添加至少 {2 - len(parsed_items)} 项资产后即可运行比较。"
                )

            with st_module.expander(
                "高级参数：计价货币、日历对齐、协方差估计、对比基准与批量代码",
                expanded=adv_expanded,
            ):
                c1, c2, c3 = st_module.columns(3)
                c1.selectbox(
                    "统一估值货币",
                    options=("CNY", "USD", "LOCAL"),
                    format_func=lambda c: BASE_CURRENCY_LABELS.get(c, c),
                    key="compare_base_currency",
                    help="跨币种比较默认统一折算为 CNY，确保不同市场资产站在同一货币尺度。",
                )
                c2.selectbox(
                    "跨市场日历对齐方式",
                    options=SUPPORTED_CALENDAR_MODES,
                    format_func=lambda m: CALENDAR_MODE_LABELS.get(m, m),
                    key="compare_calendar_mode",
                )
                c3.selectbox(
                    "比较收益率口径",
                    options=("simple", "log"),
                    format_func=lambda m: RETURN_CONVENTION_LABELS.get(m, m),
                    key="compare_return_convention",
                )

                adv1, adv2, adv3, adv4 = st_module.columns(4)
                adv1.selectbox(
                    "协方差估计方法",
                    options=SUPPORTED_COV_METHODS,
                    format_func=lambda m: COV_METHOD_LABELS.get(m, m),
                    key="compare_cov_method",
                )
                adv2.number_input(
                    "无风险年化利率 Rf (%)",
                    min_value=0.0,
                    max_value=20.0,
                    step=0.1,
                    key="compare_rf_pct_input",
                    on_change=lambda: _sync_rf_from_compare(st_module),
                )
                adv3.number_input("滚动计算窗口期", 5, 252, step=5, key="compare_rolling_window")
                adv4.number_input(
                    "年化期数覆盖（0 = 自动）",
                    min_value=0,
                    max_value=366,
                    step=1,
                    key="compare_trading_days_override",
                )

                st_module.text_input(
                    "对比基准（可选，例如 `etf:510300` 或 `us_stock:SPY`）",
                    key="compare_benchmark_input",
                    help="留空时不计算相对基准 Alpha/Beta、Tracking Error 与 Memmel Sharpe 差异检验。",
                )
                st_module.text_area(
                    "批量资产列表（高级文本编辑，每行或分号分隔 `类型:代码`）",
                    key="compare_assets_input",
                    height=96,
                    help="支持 `etf,510300` 或 `us_stock:AAPL`；与上方可视化资产选择器双向同步。",
                )

            if existing_result is not None:
                b_run, b_close = st_module.columns([3, 1])
                run_clicked = bool(
                    b_run.button(
                        "运行多资产比较",
                        key="compare_run_btn",
                        type="primary",
                        width="stretch",
                        disabled=bool(
                            not cfg["dates_valid"] or len(parsed_items) < 2 or parse_error
                        ),
                    )
                )
                if b_close.button("收起编辑", key="compare_collapse_editor_btn", width="stretch"):
                    state["editor_open"] = False
                    st_module.rerun()
            else:
                run_clicked = bool(
                    st_module.button(
                        "运行多资产比较",
                        key="compare_run_btn",
                        type="primary",
                        width="stretch",
                        disabled=bool(
                            not cfg["dates_valid"] or len(parsed_items) < 2 or parse_error
                        ),
                    )
                )

        save_widget_inputs(st_module, COMPARE_WIDGET_KEYS)
        inputs = read_persisted_inputs(st_module, COMPARE_INPUT_DEFAULTS)
        cur_sig, cur_meta = build_compare_sig_and_meta(inputs, cfg)
        if invalidate_if_signature_changed(state, cur_sig, cur_meta=cur_meta):
            state["_keep_adv_open"] = True
            st_module.rerun()
    else:
        cal_short = CALENDAR_SHORT_LABELS.get(cur_meta["calendar_mode"], cur_meta["calendar_mode"])
        cov_short = COV_SHORT_LABELS.get(cur_meta["cov_method"], cur_meta["cov_method"])
        c_sum, c_edit = st_module.columns([3.0, 1.2])
        c_sum.markdown(
            f'<div class="qa-summary-inline">'
            f'<span class="qa-summary-strong">{len(existing_result.asset_order)} 项资产对比</span>'
            f'<span class="qa-summary-meta">{html.escape(cur_meta["base_currency"])} · {html.escape(cal_short)} · 协方差 {html.escape(cov_short)}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if c_edit.button("编辑资产与参数", key="compare_toggle_editor_btn", width="stretch"):
            state["editor_open"] = True
            st_module.rerun()
        c_port, c_exp = st_module.columns([1.6, 1.4])
        if c_port.button(
            "用这些资产构建组合 →",
            key="compare_to_portfolio_btn",
            type="primary",
            width="stretch",
        ):
            _handoff_to_portfolio(st_module, _extract_result_pairs(existing_result), inputs)

    if run_clicked or recompute_clicked:
        assets_text = str(inputs["compare_assets_input"])
        benchmark_text = str(inputs["compare_benchmark_input"])
        calendar_mode = str(inputs["compare_calendar_mode"])
        base_currency = str(inputs["compare_base_currency"])
        return_convention = str(inputs["compare_return_convention"])
        cov_method = str(inputs["compare_cov_method"])
        rolling_window = int(inputs["compare_rolling_window"])
        trading_days_override = int(inputs["compare_trading_days_override"])

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
            with st_module.spinner("正在对齐多市场日历、折算汇率并运行比较统计..."):
                try:
                    specs = parse_asset_list([assets_text], syntax="web")
                    bench_spec = (
                        parse_asset_token(benchmark_text, syntax="web")
                        if benchmark_text.strip()
                        else None
                    )
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
                    store_workspace_error(
                        state,
                        signature=cur_sig,
                        error_info=classify_exception(exc),
                    )
                else:
                    summary = _build_compare_summary(res, cur_meta)
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
            title="等待运行多资产比较",
            subtitle=(
                "确认 2 项及以上资产后点击「运行多资产比较」，系统将统一交易日历与计价货币并输出净值曲线、回撤、相关性与绩效对照。"
                if not state.get("change_notice")
                else "分析参数或区间已变更，点击上方「重新计算」即可更新并对比变化。"
            ),
            chips=[
                f"资产数: {len(parsed_items)}",
                f"日历: {cur_meta['calendar_mode']}",
                f"货币: {cur_meta['base_currency']}",
                f"协方差: {cur_meta['cov_method']}",
            ],
        )
        return

    figs, skipped = get_or_build_figures(
        state,
        result,
        theme_mode=cfg.get("resolved_theme_mode") or cfg.get("theme_mode", "auto"),
        on_error=lambda exc: render_error_banner(st_module, classify_exception(exc)),
    )

    if not show_editor:
        with c_exp:
            render_export_popover(
                st_module,
                state,
                result,
                file_prefix="compare_assets",
                skipped_charts=skipped,
                key_prefix="compare",
            )
    else:
        c_act1, c_act2 = st_module.columns([2, 1.4])
        if c_act1.button(
            "用这些资产构建组合 →",
            key="compare_to_portfolio_btn",
            width="stretch",
        ):
            _handoff_to_portfolio(st_module, _extract_result_pairs(result), inputs)
        with c_act2:
            render_export_popover(
                st_module,
                state,
                result,
                file_prefix="compare_assets",
                skipped_charts=skipped,
                key_prefix="compare",
            )

    summary = result.data_summary
    methodology = result.methodology
    render_quality_and_warnings(st_module, result.warnings)
    _render_compare_takeaway(st_module, result)
    render_delta_comparison(
        st_module,
        prev_summary=state.get("prev_summary"),
        cur_summary=state.get("last_summary"),
    )

    render_metric_cards(
        st_module,
        [
            ("对齐估值样本数", format_integer(summary.get("valuation_points"))),
            ("同期收益观测期数", format_integer(summary.get("concurrent_return_periods"))),
            (
                "有效年化期数 / 年",
                format_ratio(methodology.get("effective_periods_per_year"), decimals=1),
            ),
        ],
    )
    _render_compare_highlights(st_module, result)

    eff_start = summary.get("effective_start") or summary.get("requested_start") or ""
    eff_end = summary.get("effective_end") or summary.get("requested_end") or ""
    st_module.caption(
        f"估值货币：`{methodology.get('base_currency', cur_meta['base_currency'])}` | "
        f"区间：{eff_start} ~ {eff_end} | "
        f"日历口径：`{methodology.get('calendar_mode', cur_meta['calendar_mode'])}` | "
        f"协方差估计：`{methodology.get('cov_method', cur_meta['cov_method'])}`"
    )

    if figs is not None:
        render_chart_explorer(
            st_module,
            figs,
            skipped,
            order=COMPARE_CHART_ORDER,
            labels=COMPARE_CHART_LABELS,
            key_prefix="compare",
        )

    render_compare_performance_table(st_module, result)
    render_compare_distribution_table(st_module, result)
    render_compare_benchmark_table(st_module, result)
    render_compare_matrices_audit(st_module, result)
    render_report_preview_drawer(st_module, state, result, skipped_charts=skipped)
