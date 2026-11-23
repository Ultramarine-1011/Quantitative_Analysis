"""Reusable financial tables, matrix viewers, and methodology audit drawers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from asset_catalog import ASSET_RETURN_BASIS_NOTES
from contracts import AnalysisResult, MultiAssetResult
from reporting import (
    format_drawdown_percent,
    format_p_value,
    format_percent,
    format_ratio,
    is_valid_number,
)


def render_section_heading(st_module: Any, title: str, subtitle: str = "") -> None:
    """Render a crisp section header with optional right-aligned subtitle."""
    if st_module is None:
        return
    sub_html = f'<span class="qa-section-sub">{subtitle}</span>' if subtitle else ""
    st_module.markdown(
        f'<div class="qa-section-head"><span class="qa-section-title">{title}</span>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_single_asset_audit_and_preview(st_module: Any, result: AnalysisResult) -> None:
    """Render data quality diagnostics, return-basis disclosure, recent OHLCV rows, and disclaimers."""
    if st_module is None:
        return
    with st_module.expander("数据质量审计、收益率口径与最近样本预览", expanded=False):
        basis_note = ASSET_RETURN_BASIS_NOTES.get(result.asset.return_basis, "")
        if basis_note:
            st_module.markdown(f"- **价格/收益口径 (`{result.asset.return_basis}`)**: {basis_note}")
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
        st_module.dataframe(pd.DataFrame([q_dict]), width="stretch", hide_index=True)

        df = result.data_frame
        if isinstance(df, pd.DataFrame) and not df.empty:
            base_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
            if base_cols:
                tail_df = df[base_cols].tail(15).copy()
                tail_df.index = pd.Index(
                    [
                        idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                        for idx in tail_df.index
                    ],
                    name="Date",
                )
                st_module.markdown("**最近 15 个交易日标准化 OHLCV 明细**")
                st_module.dataframe(tail_df.round(4), width="stretch")

        for disc in result.disclaimers:
            st_module.caption(f"• {disc}")


def render_compare_performance_table(st_module: Any, result: MultiAssetResult) -> None:
    """Render historical realized performance table for multi-asset comparison."""
    if st_module is None:
        return
    render_section_heading(
        st_module,
        "历史实现绩效对照",
        "Historical Realized Performance · 统一货币与周期口径",
    )
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
    st_module.dataframe(pd.DataFrame(perf_rows), width="stretch", hide_index=True)


def render_compare_distribution_table(st_module: Any, result: MultiAssetResult) -> None:
    """Render statistical inference (Jarque-Bera / skewness / kurtosis) table."""
    if st_module is None:
        return
    dist_map = result.metrics.get("distribution") or {}
    if not dist_map:
        return
    render_section_heading(
        st_module,
        "收益分布统计推断",
        "Statistical Inference · Jarque-Bera 正态性检验",
    )
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
    st_module.dataframe(pd.DataFrame(dist_rows), width="stretch", hide_index=True)


def render_compare_benchmark_table(st_module: Any, result: MultiAssetResult) -> None:
    """Render benchmark relative attribution and Memmel Sharpe test table."""
    if st_module is None:
        return
    bench_comp = result.metrics.get("benchmark_comparison") or {}
    if not (result.benchmark and bench_comp):
        return
    bench_name = result.benchmark.get("leg_name", "Benchmark")
    render_section_heading(
        st_module,
        "相对基准归因与 Memmel Sharpe 检验",
        f"对比基准：{bench_name}",
    )
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
    st_module.dataframe(pd.DataFrame(rel_rows), width="stretch", hide_index=True)


def render_compare_matrices_audit(st_module: Any, result: MultiAssetResult) -> None:
    """Render correlation/covariance matrices and methodology JSON in an expander."""
    if st_module is None:
        return
    with st_module.expander("相关性 / 协方差矩阵与方法学审计", expanded=False):
        c1, c2 = st_module.columns(2)
        corr_df = result.frames.get("correlation")
        cov_df = result.frames.get("covariance")
        if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
            c1.markdown("**同期收益率相关性矩阵**")
            c1.dataframe(corr_df.round(4), width="stretch")
        if isinstance(cov_df, pd.DataFrame) and not cov_df.empty:
            c2.markdown("**年化协方差矩阵**")
            c2.dataframe(cov_df.round(6), width="stretch")
        st_module.json(result.methodology)


def render_portfolio_backtest_table(
    st_module: Any,
    result: MultiAssetResult,
    *,
    primary_title: str,
    strategy: str,
    primary_perf: Mapping[str, Any],
    ew_perf: Mapping[str, Any],
) -> None:
    """Render portfolio vs equal-weight vs hindsight/benchmark performance comparison."""
    if st_module is None:
        return
    backtest_metrics = result.metrics.get("backtest") or {}
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
    render_section_heading(
        st_module,
        "组合回测与基准绩效对比",
        "策略表现与等权/外部基准对照组",
    )
    st_module.dataframe(pd.DataFrame(perf_comp_rows), width="stretch", hide_index=True)


def render_portfolio_weights_table(st_module: Any, result: MultiAssetResult) -> None:
    """Render portfolio asset weights, expected stats, and Euler risk decomposition."""
    if st_module is None:
        return
    weights_df = result.frames.get("weights")
    if not (isinstance(weights_df, pd.DataFrame) and not weights_df.empty):
        return
    render_section_heading(
        st_module,
        "资产权重与 Euler 风险贡献拆解",
        "资金配比与边际风险占比对照",
    )
    disp_w = pd.DataFrame(
        {
            "资产": weights_df.index.astype(str),
            "策略权重": [format_percent(v) for v in weights_df["weight"]],
            "风险贡献占比": [format_percent(v) for v in weights_df["pct_risk_contribution"]],
            "年化预期收益(估计)": [format_percent(v) for v in weights_df["expected_return_ann"]],
            "年化波动率(估计)": [format_percent(v) for v in weights_df["volatility_ann"]],
            "等权 1/N": [format_percent(v) for v in weights_df["equal_weight"]],
            "GMV 权重": [format_percent(v) for v in weights_df["min_variance_weight"]],
            "Tangency 权重": [format_percent(v) for v in weights_df["max_sharpe_weight"]],
        }
    )
    st_module.dataframe(disp_w, width="stretch", hide_index=True)


def _format_sensitivity_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    rename_map = {
        "cov_method": "协方差方法",
        "transaction_cost_bps": "单边费率 (bps)",
        "rebalance_freq": "再平衡频率",
        "cagr": "CAGR",
        "annualized_volatility": "年化波动",
        "max_drawdown": "最大回撤",
        "sharpe_ratio": "夏普比率",
        "total_turnover": "总换手",
        "total_cost_ratio": "成本占比",
    }
    for pct_col in ("cagr", "annualized_volatility", "total_cost_ratio"):
        if pct_col in df.columns:
            df[pct_col] = [format_percent(v) if is_valid_number(v) else v for v in df[pct_col]]
    if "max_drawdown" in df.columns:
        df["max_drawdown"] = [
            format_drawdown_percent(v) if is_valid_number(v) else v for v in df["max_drawdown"]
        ]
    for ratio_col, dec in (("sharpe_ratio", 3), ("total_turnover", 2)):
        if ratio_col in df.columns:
            df[ratio_col] = [
                format_ratio(v, decimals=dec) if is_valid_number(v) else v for v in df[ratio_col]
            ]
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def render_portfolio_sensitivity_drawer(st_module: Any, result: MultiAssetResult) -> None:
    """Render collapsible parameter sensitivity tables."""
    if st_module is None:
        return
    sens = result.metrics.get("sensitivity") or {}
    if not sens:
        return
    with st_module.expander("参数与交易成本敏感性分析", expanded=False):
        for sens_key, sens_title in (
            ("covariance_method", "协方差估计方法敏感性"),
            ("transaction_cost_bps", "交易成本敏感性 (bps)"),
            ("rebalance_freq", "再平衡频率敏感性"),
        ):
            rows = sens.get(sens_key)
            if isinstance(rows, list) and rows:
                st_module.markdown(f"**{sens_title}**")
                st_module.dataframe(
                    _format_sensitivity_frame(rows), width="stretch", hide_index=True
                )


def render_portfolio_methodology_audit(st_module: Any, result: MultiAssetResult) -> None:
    """Render portfolio covariance matrix, relative benchmark test, and methodology JSON."""
    if st_module is None:
        return
    with st_module.expander("协方差矩阵、基准归因检验与方法学审计", expanded=False):
        if result.benchmark and isinstance(result.benchmark.get("relative_to_strategy"), Mapping):
            rel = result.benchmark["relative_to_strategy"]
            mem = rel.get("memmel_sharpe_test") or {}
            st_module.markdown("**策略相对基准统计检验**")
            st_module.dataframe(
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
            st_module.markdown("**年化协方差矩阵**")
            st_module.dataframe(cov_df.round(6), width="stretch")
        st_module.json(result.methodology)
