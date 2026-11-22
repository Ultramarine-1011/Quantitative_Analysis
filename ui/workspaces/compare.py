"""Workspace 2: Multi-Asset Comparison backed by ``compare_assets``."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import pandas as pd

from asset_catalog import PRESET_PORTFOLIOS
from asset_inputs import parse_asset_list, parse_asset_token
from contracts import CompareRequest, MultiAssetResult
from data_sources.network import apply_default_network_proxy_policy
from reporting import (
    format_drawdown_percent,
    format_integer,
    format_percent,
    format_ratio,
    is_valid_number,
)
from services.multi_asset_service import compare_assets
from ui.components.alerts import (
    render_empty_state,
    render_error_banner,
    render_quality_and_warnings,
)
from ui.components.charts import render_chart_explorer
from ui.components.exports import render_report_and_download_section
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
    COMPARE_CHART_LABELS,
    COMPARE_CHART_ORDER,
    COV_METHOD_LABELS,
    RETURN_CONVENTION_LABELS,
)
from ui.services_bridge import (
    get_or_build_figures,
    make_service_loaders,
    preview_multi_asset_specs,
)
from ui.state import (
    WORKSPACE_COMPARE_KEY,
    invalidate_if_signature_changed,
    store_workspace_error,
    store_workspace_result,
    workspace_state,
)


def _apply_compare_preset(st_module: Any) -> None:
    preset = str(st_module.session_state.get("compare_preset_template", ""))
    rows = PRESET_PORTFOLIOS.get(preset)
    if rows:
        st_module.session_state["compare_assets_input"] = "\n".join(rows)


def _render_compare_highlights(st_module: Any, result: MultiAssetResult) -> None:
    """Summarize cross-asset leaders directly from backend ``per_asset`` metrics."""
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

    if "compare_assets_input" not in st_module.session_state:
        first_preset = next(iter(PRESET_PORTFOLIOS.values()))
        st_module.session_state["compare_assets_input"] = "\n".join(first_preset)

    st_module.subheader("⚖️ 跨市场多资产历史对比与统计检验")
    col_left, col_right = st_module.columns([3, 2])
    with col_left:
        st_module.selectbox(
            "预设跨市场资产模板（可选）",
            options=tuple(PRESET_PORTFOLIOS.keys()),
            key="compare_preset_template",
            on_change=lambda: _apply_compare_preset(st_module),
        )
        assets_text = st_module.text_area(
            "对比资产列表（每行或分号分隔 `类型,代码` 或 `类型:代码`）",
            key="compare_assets_input",
            height=115,
            help="示例：etf,510300 / us_stock:AAPL / crypto,BTC/USDT。注：不支持将利率水平纳入净值对比。",
        )
        parsed_specs, preview_err = preview_multi_asset_specs(assets_text)
        if parsed_specs:
            chips_html = "".join(
                f'<span class="qa-chip">{html.escape(s.asset_type)}:{html.escape(s.symbol)}</span>'
                for s in parsed_specs
            )
            st_module.markdown(
                f'<div class="qa-chip-row">{chips_html}</div>', unsafe_allow_html=True
            )
        elif preview_err:
            st_module.caption(f"⚠️ 格式预览：{preview_err}")

        benchmark_text = st_module.text_input(
            "对比基准（可选，例如 `etf:510300` 或 `us_stock:SPY`）",
            value="",
            key="compare_benchmark_input",
            help="留空时不计算相对基准 Alpha/Beta、Tracking Error 与 Memmel Sharpe 差异检验。",
        )
    with col_right:
        from contracts import SUPPORTED_CALENDAR_MODES, SUPPORTED_COV_METHODS

        calendar_mode = str(
            st_module.selectbox(
                "跨市场日历对齐方式",
                options=SUPPORTED_CALENDAR_MODES,
                index=0,
                format_func=lambda m: CALENDAR_MODE_LABELS.get(m, m),
                key="compare_calendar_mode",
            )
        )
        base_currency = str(
            st_module.selectbox(
                "统一估值货币",
                options=("CNY", "USD", "LOCAL"),
                index=0,
                format_func=lambda c: BASE_CURRENCY_LABELS.get(c, c),
                key="compare_base_currency",
                help="跨币种比较默认统一折算为 CNY（与 `quant compare` 一致）。",
            )
        )
        return_convention = str(
            st_module.selectbox(
                "比较收益率口径",
                options=("simple", "log"),
                index=0,
                format_func=lambda m: RETURN_CONVENTION_LABELS.get(m, m),
                key="compare_return_convention",
            )
        )

    with st_module.expander("⚙️ 高级统计与滚动检验参数", expanded=False):
        adv1, adv2, adv3 = st_module.columns(3)
        cov_method = str(
            adv1.selectbox(
                "协方差估计方法",
                options=SUPPORTED_COV_METHODS,
                index=list(SUPPORTED_COV_METHODS).index("ledoit_wolf"),
                format_func=lambda m: COV_METHOD_LABELS.get(m, m),
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

    run_clicked = st_module.button(
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
        cfg["proxy_url"],
    )
    invalidate_if_signature_changed(state, cur_sig)

    if run_clicked:
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
                    store_workspace_result(state, signature=cur_sig, result=res)

    if state.get("error_info") or state.get("error"):
        render_error_banner(st_module, state.get("error_info") or state.get("error"))

    result: MultiAssetResult | None = state.get("result")
    if result is None:
        render_empty_state(
            st_module,
            title="⚖️ 多资产跨市场对比尚未运行",
            subtitle="配置 2 个及以上资产后点击「⚖️ 运行多资产比较」，系统将统一日历与计价货币并输出归一化净值、分布正态性检验与相关性面板。",
            chips=[
                f"资产数: {len(parsed_specs)}",
                f"日历: {calendar_mode}",
                f"货币: {base_currency}",
                f"收益口径: {return_convention}",
            ],
        )
        return

    summary = result.data_summary
    methodology = result.methodology
    render_quality_and_warnings(st_module, result.warnings)

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
        f"估值货币：`{methodology.get('base_currency', base_currency)}` | "
        f"区间：{eff_start} ~ {eff_end} | "
        f"日历口径：`{methodology.get('calendar_mode', calendar_mode)}` | "
        f"协方差估计：`{methodology.get('cov_method', cov_method)}`"
    )

    render_compare_performance_table(st_module, result)
    render_compare_distribution_table(st_module, result)
    render_compare_benchmark_table(st_module, result)

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
            order=COMPARE_CHART_ORDER,
            labels=COMPARE_CHART_LABELS,
            key_prefix="compare",
        )

    render_compare_matrices_audit(st_module, result)
    render_report_and_download_section(
        st_module,
        state,
        result,
        file_prefix="compare_assets",
        skipped_charts=skipped,
        key_prefix="compare",
    )
