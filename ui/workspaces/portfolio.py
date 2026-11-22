"""Workspace 3: Portfolio Optimization, Efficient Frontier & OOS Backtest via ``analyze_portfolio``."""

from __future__ import annotations

import html
from collections.abc import Callable
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
from ui.components.alerts import (
    render_empty_state,
    render_error_banner,
    render_quality_and_warnings,
    render_validation_protocol_banner,
)
from ui.components.charts import render_chart_explorer
from ui.components.exports import render_report_and_download_section
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
    PORTFOLIO_CHART_LABELS,
    PORTFOLIO_CORE_CHARTS,
    PORTFOLIO_EXTRA_CHARTS,
    REBALANCE_FREQ_LABELS,
    STRATEGY_HINTS,
    STRATEGY_LABELS,
    VALIDATION_MODE_LABELS,
)
from ui.services_bridge import (
    get_or_build_figures,
    make_service_loaders,
    preview_custom_weights,
    preview_multi_asset_specs,
)
from ui.state import (
    WORKSPACE_PORTFOLIO_KEY,
    invalidate_if_signature_changed,
    store_workspace_error,
    store_workspace_result,
    workspace_state,
)


def _apply_portfolio_preset(st_module: Any) -> None:
    preset = str(st_module.session_state.get("tab3_preset_template", ""))
    rows = PRESET_PORTFOLIOS.get(preset)
    if rows:
        st_module.session_state["tab3_ef_legs_input"] = "\n".join(rows)


def render_portfolio_workspace(
    st_module: Any,
    cfg: dict[str, Any],
    *,
    cached_loader: Callable[..., pd.DataFrame],
) -> None:
    """Render Workspace 3: Portfolio Optimization, Frontier & Backtesting via ``analyze_portfolio``."""
    if st_module is None:
        return
    state = workspace_state(st_module, WORKSPACE_PORTFOLIO_KEY)

    if "tab3_ef_legs_input" not in st_module.session_state:
        first_preset = next(iter(PRESET_PORTFOLIOS.values()))
        st_module.session_state["tab3_ef_legs_input"] = "\n".join(first_preset)

    st_module.subheader("🎯 投资组合优化、有效前沿与样本外回测研究")
    col_left, col_right = st_module.columns([3, 2])
    with col_left:
        st_module.selectbox(
            "预设组合模板（可选）",
            options=tuple(PRESET_PORTFOLIOS.keys()),
            key="tab3_preset_template",
            on_change=lambda: _apply_portfolio_preset(st_module),
        )
        legs_text = st_module.text_area(
            "组合标的列表（至少 2 个，`类型,代码` 或 `类型:代码`）",
            key="tab3_ef_legs_input",
            height=115,
        )
        parsed_specs, preview_err = preview_multi_asset_specs(legs_text)
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
            "组合对比基准（可选，例如 `etf:510300`）",
            value="",
            key="portfolio_benchmark_input",
        )
    with col_right:
        strategy = str(
            st_module.selectbox(
                "资产配置 / 优化策略",
                options=SUPPORTED_PORTFOLIO_STRATEGIES,
                index=list(SUPPORTED_PORTFOLIO_STRATEGIES).index("max_sharpe"),
                format_func=lambda s: STRATEGY_LABELS.get(s, s),
                key="portfolio_strategy_input",
            )
        )
        if strategy in STRATEGY_HINTS:
            st_module.caption(STRATEGY_HINTS[strategy])

        custom_weights_raw = ""
        if strategy == "custom":
            custom_weights_raw = str(
                st_module.text_input(
                    "自定义资产权重（逗号分隔，按上方资产顺序）",
                    value="",
                    placeholder="0.4, 0.3, 0.3",
                    key="portfolio_custom_weights_input",
                )
            )
            parsed_w, w_err = preview_custom_weights(
                custom_weights_raw,
                expected_count=len(parsed_specs) if parsed_specs else None,
            )
            if w_err:
                st_module.caption(f"⚠️ {w_err}")
            elif parsed_w is not None:
                total_w = sum(parsed_w)
                st_module.caption(
                    f"已识别 {len(parsed_w)} 项权重（原始合计 {total_w:.4f}，求解时将自动归一化）。"
                )

        validation_mode = str(
            st_module.selectbox(
                "回测与样本划分协议",
                options=SUPPORTED_VALIDATION_MODES,
                index=list(SUPPORTED_VALIDATION_MODES).index("train_test"),
                format_func=lambda m: VALIDATION_MODE_LABELS.get(m, m),
                key="tab3_val_mode",
            )
        )
        c_cal, c_ccy = st_module.columns(2)
        calendar_mode = str(
            c_cal.selectbox(
                "日历对齐",
                options=SUPPORTED_CALENDAR_MODES,
                index=0,
                format_func=lambda m: CALENDAR_MODE_LABELS.get(m, m),
                key="portfolio_calendar_mode",
            )
        )
        base_currency = str(
            c_ccy.selectbox(
                "组合计价币种",
                options=("CNY", "USD", "LOCAL"),
                index=0,
                format_func=lambda c: BASE_CURRENCY_LABELS.get(c, c),
                key="portfolio_base_currency",
            )
        )

    with st_module.expander("⚙️ 回测约束、有效前沿、蒙特卡洛散点与敏感性参数", expanded=False):
        p1, p2, p3 = st_module.columns(3)
        cov_method = str(
            p1.selectbox(
                "协方差矩阵估计",
                options=SUPPORTED_COV_METHODS,
                index=list(SUPPORTED_COV_METHODS).index("ledoit_wolf"),
                format_func=lambda m: COV_METHOD_LABELS.get(m, m),
                key="portfolio_cov_method",
            )
        )
        rebalance_freq = str(
            p2.selectbox(
                "再平衡调仓频率",
                options=SUPPORTED_REBALANCE_FREQS,
                index=list(SUPPORTED_REBALANCE_FREQS).index("monthly"),
                format_func=lambda f: REBALANCE_FREQ_LABELS.get(f, f),
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

        b1, b2, b3, b4 = st_module.columns(4)
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

        e1, e2, e3, e4 = st_module.columns(4)
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

        c_opt1, c_opt2 = st_module.columns(2)
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

    run_clicked = st_module.button(
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
            with st_module.spinner("正在执行组合协方差估计、SLSQP 优化与回测验证..."):
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
                    store_workspace_result(state, signature=cur_sig, result=res)

    if state.get("error_info") or state.get("error"):
        render_error_banner(st_module, state.get("error_info") or state.get("error"))

    result: MultiAssetResult | None = state.get("result")
    if result is None:
        render_empty_state(
            st_module,
            title="🎯 投资组合优化与样本外回测尚未运行",
            subtitle="选择配置策略与样本划分协议后点击「🎯 运行组合优化与回测」，系统将计算目标权重、Euler 风险贡献、SLSQP 精确有效前沿与样本外净值表现。",
            chips=[
                f"资产数: {len(parsed_specs)}",
                f"策略: {strategy}",
                f"调仓: {rebalance_freq}",
                f"计价: {base_currency}",
            ],
        )
        return

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

    render_metric_cards(
        st_module,
        [
            (f"{primary_title} CAGR", format_percent(primary_perf.get("cagr"))),
            (f"{primary_title}最大回撤", format_drawdown_percent(primary_perf.get("max_drawdown"))),
            (
                f"{primary_title}夏普比率",
                format_ratio(primary_perf.get("sharpe_ratio"), decimals=3),
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
        f"分散化比率 (Diversification Ratio)：{format_ratio(dr_val, decimals=3)} | "
        f"有效资产数 ENB (权重 / 风险)：{format_ratio(enb_w, decimals=2)} / "
        f"{format_ratio(enb_r, decimals=2)} | "
        f"协方差条件数：{format_ratio(cond_num, decimals=2)}"
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

    chart_selection = list(PORTFOLIO_CORE_CHARTS)
    if show_extra_charts:
        chart_selection.extend(PORTFOLIO_EXTRA_CHARTS)
        if "weight_history" in result.frames:
            chart_selection.append("weight_history")

    figs, skipped = get_or_build_figures(
        state,
        result,
        chart_types=tuple(chart_selection),
        theme_mode=cfg.get("resolved_theme_mode") or cfg.get("theme_mode", "auto"),
        on_error=lambda exc: render_error_banner(st_module, classify_exception(exc)),
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

    render_portfolio_sensitivity_drawer(st_module, result)
    render_portfolio_methodology_audit(st_module, result)
    render_report_and_download_section(
        st_module,
        state,
        result,
        file_prefix=f"portfolio_{strategy}_{validation_mode}",
        skipped_charts=skipped,
        key_prefix="portfolio",
    )
