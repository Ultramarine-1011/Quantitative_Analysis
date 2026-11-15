"""Independent chart renderer consuming AnalysisResult / MultiAssetResult.

Provides lazy-loaded Plotly (interactive HTML) and Matplotlib/Mplfinance (PNG/SVG) rendering
without duplicating data fetching or financial computations.
"""

from __future__ import annotations

import importlib
import io
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts import AnalysisResult, MultiAssetResult
from quant_models import normalize_close_to_base_one
from reporting.formatting import is_valid_number

SUPPORTED_CHART_FORMATS: tuple[str, ...] = ("html", "png", "svg")

COMMAND_CHART_TYPES: dict[str, tuple[str, ...]] = {
    "analyze": ("price", "normalized", "drawdown", "rolling"),
    "compare": ("normalized", "drawdown", "rolling", "correlation"),
    "portfolio": (
        "backtest",
        "weights",
        "frontier",
        "correlation",
        "normalized",
        "drawdown",
        "weight_history",
    ),
}

_SERIES_PALETTE: tuple[str, ...] = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4f46e5",
)

_PREFERRED_CJK_FONTS: tuple[str, ...] = (
    "Heiti SC",
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
)

_CHART_TITLES_ZH: dict[str, str] = {
    "price": "价格与技术指标走势",
    "normalized": "归一化净值走势（期初 = 1.0）",
    "drawdown": "历史水下回撤",
    "rolling": "滚动风险与收益统计",
    "correlation": "资产收益率相关性矩阵 (Pearson)",
    "backtest": "投资组合回测净值与水下回撤",
    "weights": "组合权重配置与 Euler 风险贡献",
    "frontier": "均值-方差有效前沿与可行域",
    "weight_history": "Walk-Forward 再平衡权重演化",
}


@dataclass(frozen=True)
class ChartArtifact:
    """Metadata for a single rendered chart file."""

    chart_type: str
    title: str
    description: str
    format: str
    path: Path
    artifact_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "description": self.description,
            "format": self.format,
            "path": str(self.path),
            "artifact_key": self.artifact_key,
        }


@dataclass(frozen=True)
class ChartRenderOutput:
    """Rendered chart artifacts alongside skipped-chart explanations and warnings."""

    charts: tuple[ChartArtifact, ...] = ()
    skipped: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def parse_chart_formats(raw: str | None) -> tuple[str, ...]:
    """Parse ``--charts`` CLI/API value into ordered unique format strings."""
    if raw is None:
        return ()
    cleaned = str(raw).strip().lower()
    if not cleaned or cleaned == "none":
        return ()
    if cleaned == "all":
        return SUPPORTED_CHART_FORMATS

    tokens = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not tokens:
        return ()
    if "none" in tokens and len(tokens) > 1:
        raise ValueError("Invalid --charts value: 'none' cannot be combined with other formats.")

    resolved: list[str] = []
    for token in tokens:
        if token == "all":
            for fmt in SUPPORTED_CHART_FORMATS:
                if fmt not in resolved:
                    resolved.append(fmt)
            continue
        if token not in SUPPORTED_CHART_FORMATS:
            raise ValueError(
                f"Unsupported chart format {token!r}. Allowed values: "
                f"{', '.join(SUPPORTED_CHART_FORMATS)}, all, none."
            )
        if token not in resolved:
            resolved.append(token)
    return tuple(resolved)


def parse_chart_types(command: str, raw: str | None) -> tuple[str, ...]:
    """Parse and validate ``--chart-types`` against the supported charts for *command*."""
    supported = COMMAND_CHART_TYPES.get(command)
    if supported is None:
        raise ValueError(f"Charts are not supported for command {command!r}.")

    if raw is None:
        return supported
    cleaned = str(raw).strip().lower()
    if not cleaned or cleaned in {"auto", "all"}:
        return supported

    tokens = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not tokens:
        return supported

    resolved: list[str] = []
    for token in tokens:
        if token in {"auto", "all"}:
            for ct in supported:
                if ct not in resolved:
                    resolved.append(ct)
            continue
        if token not in supported:
            raise ValueError(
                f"Unsupported chart type {token!r} for command {command!r}. "
                f"Allowed values: {', '.join(supported)}, auto."
            )
        if token not in resolved:
            resolved.append(token)
    return tuple(resolved)


def _load_matplotlib() -> tuple[Any, Any, Any]:
    """Lazy-import matplotlib with non-interactive Agg backend configured first."""
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg", force=True)
    plt = importlib.import_module("matplotlib.pyplot")
    mticker = importlib.import_module("matplotlib.ticker")
    return mpl, plt, mticker


def _get_mpl_rc_overrides() -> dict[str, Any]:
    """Build matplotlib rcParams preferring DejaVu Sans (regular+bold) with CJK fallbacks."""
    font_manager = importlib.import_module("matplotlib.font_manager")
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen_fonts = ["DejaVu Sans"]
    for name in _PREFERRED_CJK_FONTS:
        if name in available and name not in chosen_fonts:
            chosen_fonts.append(name)
    return {
        "font.sans-serif": chosen_fonts,
        "axes.unicode_minus": False,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "grid.color": "#cbd5e1",
        "grid.alpha": 0.45,
        "grid.linestyle": "--",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    }


def _save_mpl_figure_bytes(plt: Any, fig: Any, fmt: str) -> bytes:
    """Render a Matplotlib Figure into in-memory PNG/SVG bytes and close the figure."""
    buf = io.BytesIO()
    try:
        save_kwargs: dict[str, Any] = {
            "format": fmt,
            "bbox_inches": "tight",
            "facecolor": "#ffffff",
        }
        if fmt == "png":
            save_kwargs["dpi"] = 160
        fig.savefig(buf, **save_kwargs)
        return buf.getvalue()
    finally:
        plt.close(fig)


def _export_plotly_html_bytes(figure: Any, *, title: str) -> bytes:
    """Export a Plotly Figure as a fully self-contained offline HTML document."""
    export_fig = importlib.import_module("copy").deepcopy(figure)
    export_fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font={"color": "#0f172a"},
        autosize=True,
    )
    html_body = export_fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    safe_title = importlib.import_module("html").escape(title)
    doc = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        f"  <title>{safe_title}</title>\n"
        "  <style>\n"
        "    body { margin: 0; padding: 16px 20px; background: #ffffff; color: #0f172a;\n"
        "           font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',\n"
        "           'Microsoft YaHei', sans-serif; }\n"
        "    .chart-shell { max-width: 1200px; margin: 0 auto; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '  <div class="chart-shell">\n'
        f"    {html_body}\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )
    return doc.encode("utf-8")


def _prepare_single_asset_df(result: AnalysisResult) -> pd.DataFrame:
    """Copy ``result.data_frame`` with canonical attrs restored for chart builders."""
    if result.data_frame is None or result.data_frame.empty:
        return pd.DataFrame()
    df = result.data_frame.copy()
    existing_summary = (
        dict(df.attrs.get("feature_summary", {}))
        if isinstance(df.attrs.get("feature_summary"), Mapping)
        else {}
    )
    vol_cols = [c for c in df.columns if str(c).startswith("volatility_") and str(c)[11:].isdigit()]
    inferred_vol_win = int(str(vol_cols[0])[11:]) if vol_cols else 20
    sma_short = int(
        result.parameters.get(
            "sma_short",
            result.parameters.get("sma_window", existing_summary.get("sma_short_window", 20)),
        )
    )
    sma_long = int(result.parameters.get("sma_long", existing_summary.get("sma_long_window", 60)))
    bb_win = int(result.parameters.get("bb_window", existing_summary.get("bollinger_window", 20)))
    vol_win = int(
        result.parameters.get(
            "volatility_window",
            existing_summary.get("volatility_window", inferred_vol_win),
        )
    )
    df.attrs["asset_type"] = result.asset.asset_type
    df.attrs["series_kind"] = result.asset.series_kind
    df.attrs["canonical_symbol"] = result.asset.canonical_symbol
    df.attrs["output_currency"] = result.asset.output_currency
    if is_valid_number(result.metrics.get("max_drawdown")):
        df.attrs["max_drawdown"] = float(result.metrics["max_drawdown"])
    df.attrs["feature_summary"] = {
        **existing_summary,
        "sma_short_window": sma_short,
        "sma_long_window": sma_long,
        "bollinger_window": bb_win,
        "volatility_window": vol_win,
    }
    return df


def _resolve_single_asset_specs(
    result: AnalysisResult,
    requested_types: Sequence[str],
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """Determine applicable charts and explicit skip reasons for ``analyze``."""
    df = result.data_frame
    if df is None or df.empty:
        return [], {ct: "缺少可绘制的时间序列数据 (data_frame 为空)。" for ct in requested_types}

    is_yield = result.asset.series_kind == "yield_level"
    specs: list[tuple[str, str, str]] = []
    skipped: dict[str, str] = {}
    symbol = result.asset.canonical_symbol
    ccy = result.asset.output_currency or ""

    for ct in requested_types:
        if ct == "price":
            title = (
                f"{symbol} 收益率水平走势"
                if is_yield
                else f"{symbol} 价格与均线趋势 ({ccy})"
                if ccy
                else f"{symbol} 价格与均线趋势"
            )
            desc = (
                "展示收益率原始水平序列（非价格与非资产收益），叠加移动均值参考线。"
                if is_yield
                else "展示历史价格/净值走势及技术均线与布林带参考。"
            )
            specs.append((ct, title, desc))
            continue

        if is_yield:
            skipped[ct] = (
                "yield_level_not_return: 利率/收益率水平序列不能转换为归一化净值、资产回撤或收益率波动率。"
            )
            continue

        if ct == "normalized":
            close = pd.to_numeric(df.get("Close"), errors="coerce")
            if not (close > 0).any():
                skipped[ct] = "非正价格序列无法构造期初 = 1.0 的归一化净值。"
                continue
            specs.append(
                (
                    ct,
                    f"{symbol} 归一化净值走势",
                    f"以首个有效收盘价为 1.00 进行归一化展示（计价口径：{ccy or '原生'}）。",
                )
            )
        elif ct == "drawdown":
            if (
                "drawdown" not in df.columns
                or not pd.to_numeric(df["drawdown"], errors="coerce").notna().any()
            ):
                skipped[ct] = "缺少有效的历史回撤序列。"
                continue
            specs.append(
                (
                    ct,
                    f"{symbol} 历史水下回撤",
                    "自历史滚动峰值向下的回撤幅度（统一以非正百分比 <= 0% 展示）。",
                )
            )
        elif ct == "rolling":
            vol_win = int(result.parameters.get("volatility_window", 20))
            vol_col = f"volatility_{vol_win}"
            if vol_col not in df.columns and "drawdown" not in df.columns:
                skipped[ct] = "缺少滚动波动率与回撤列。"
                continue
            specs.append(
                (
                    ct,
                    f"{symbol} 滚动波动率与风险面板 ({vol_win}D)",
                    "展示滚动年化波动率、水下回撤及可选 RSI(14) 动量指标。",
                )
            )
    return specs, skipped


def _resolve_compare_specs(
    result: MultiAssetResult,
    requested_types: Sequence[str],
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """Determine applicable charts and explicit skip reasons for ``compare``."""
    specs: list[tuple[str, str, str]] = []
    skipped: dict[str, str] = {}
    frames = result.frames
    base_ccy = str(result.methodology.get("base_currency") or "LOCAL")

    for ct in requested_types:
        if ct == "normalized":
            prices = frames.get("prices") if "prices" in frames else result.data_frame
            if prices is None or prices.empty:
                skipped[ct] = "缺少对齐后的多资产价格序列。"
                continue
            specs.append(
                (
                    ct,
                    f"多资产归一化净值对比 ({base_ccy})",
                    "各标的分别从其有效区间起点归一化为 1.00，便于对比区间累积增长路径。",
                )
            )
        elif ct == "drawdown":
            prices = frames.get("prices") if "prices" in frames else result.data_frame
            if prices is None or prices.empty:
                skipped[ct] = "缺少对齐后的价格序列，无法计算水下回撤。"
                continue
            specs.append(
                (
                    ct,
                    "多资产水下回撤对比",
                    "各资产从各自历史滚动峰值回落的幅度对比（<= 0%）。",
                )
            )
        elif ct == "rolling":
            rolling_df = frames.get("rolling_metrics")
            if rolling_df is None or rolling_df.empty:
                skipped[ct] = "缺少滚动统计数据 (rolling_metrics 为空)。"
                continue
            win = result.methodology.get("rolling_window")
            win_str = f" ({win}D)" if win else ""
            specs.append(
                (
                    ct,
                    f"多资产滚动风险收益统计{win_str}",
                    "滚动年化收益率、滚动年化波动率、滚动 Sharpe 及相对基准/两两滚动相关性。",
                )
            )
        elif ct == "correlation":
            corr_df = frames.get("correlation")
            if corr_df is None or corr_df.empty:
                skipped[ct] = "缺少相关性矩阵 (correlation 为空)。"
                continue
            specs.append(
                (
                    ct,
                    "资产收益率相关性矩阵 (Pearson)",
                    "基于共同对齐区间的全样本线性相关系数热力图。",
                )
            )
    return specs, skipped


def _resolve_portfolio_specs(
    result: MultiAssetResult,
    requested_types: Sequence[str],
    *,
    explicit_selection: bool,
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """Determine applicable charts and explicit skip reasons for ``portfolio``."""
    specs: list[tuple[str, str, str]] = []
    skipped: dict[str, str] = {}
    frames = result.frames
    strategy = str(result.methodology.get("strategy") or "")
    mode = str(result.methodology.get("validation_mode") or "")
    has_bias = bool(result.methodology.get("has_lookahead_bias", False))

    for ct in requested_types:
        if ct == "backtest":
            bt = frames.get("backtest") if "backtest" in frames else result.data_frame
            if bt is None or bt.empty:
                skipped[ct] = "缺少组合回测净值数据 (backtest 为空)。"
                continue
            bias_tag = " [含样本内后视参考]" if has_bias else " (严格因果样本外)"
            specs.append(
                (
                    ct,
                    f"组合回测净值与水下回撤 ({strategy} / {mode}){bias_tag}",
                    "对比策略组合、等权 (1/N) 与可选基准净值及水下回撤。",
                )
            )
        elif ct == "weights":
            w_df = frames.get("weights")
            if w_df is None or w_df.empty:
                skipped[ct] = "缺少组合权重与风险贡献明细 (weights 为空)。"
                continue
            specs.append(
                (
                    ct,
                    f"目标权重配置与 Euler 风险贡献 ({strategy})",
                    "对比当前策略权重与等权/GMV/MaxSharpe 参考权重，以及资金占比与风险贡献占比。",
                )
            )
        elif ct == "frontier":
            f_df = frames.get("frontier")
            if f_df is None or f_df.empty:
                if explicit_selection or int(result.parameters.get("frontier_points", 25)) > 0:
                    skipped[ct] = "当前运行未计算有效前沿 (--frontier-points 0)。"
                continue
            specs.append(
                (
                    ct,
                    "均值-方差有效前沿与可行域",
                    "精确 SLSQP 有效前沿包络线（实线）与 Dirichlet 随机采样可行域（散点，仅作参考）。",
                )
            )
        elif ct == "correlation":
            corr_df = frames.get("correlation")
            if corr_df is None or corr_df.empty:
                skipped[ct] = "缺少资产相关性矩阵。"
                continue
            specs.append(
                (
                    ct,
                    "组合成分资产相关性矩阵 (Pearson)",
                    "估计窗口/对齐样本内底层资产日度收益率的线性相关性。",
                )
            )
        elif ct == "normalized":
            prices_df = frames.get("prices")
            if prices_df is None or prices_df.empty:
                skipped[ct] = "缺少对齐后的底层资产价格序列。"
                continue
            specs.append(
                (
                    ct,
                    "底层资产归一化表现对比",
                    "组合成分标的在对齐估值窗口下的期初 = 1.0 净值对比。",
                )
            )
        elif ct == "drawdown":
            bt = frames.get("backtest") if "backtest" in frames else result.data_frame
            if bt is None or bt.empty:
                skipped[ct] = "缺少组合回测净值序列。"
                continue
            specs.append(
                (
                    ct,
                    "组合与基准水下回撤对比",
                    "各组合净值曲线自历史高点向下的水下回撤幅度（<= 0%）。",
                )
            )
        elif ct == "weight_history":
            wh_df = frames.get("weight_history")
            if wh_df is None or wh_df.empty:
                if explicit_selection or mode.startswith("walk_forward"):
                    skipped[ct] = (
                        "仅 Walk-Forward 验证模式生成多节点再平衡权重历史 (weight_history)。"
                    )
                continue
            specs.append(
                (
                    ct,
                    "Walk-Forward 各再平衡节点权重演化",
                    "样本外滚动/扩展窗口每次再平衡求解得到的目标权重变化。",
                )
            )
    return specs, skipped


def _build_plotly_figure_for_type(
    result: AnalysisResult | MultiAssetResult,
    chart_type: str,
) -> Any:
    """Instantiate a Plotly Figure for *chart_type* reusing ``charts.py``."""
    charts_mod = importlib.import_module("charts")

    if isinstance(result, AnalysisResult):
        plot_df = _prepare_single_asset_df(result)
        if chart_type == "price":
            return charts_mod.build_price_volume_figure(plot_df, include_risk_panels=False)
        if chart_type == "normalized":
            norm = normalize_close_to_base_one(plot_df)
            fig = charts_mod.build_tab3_normalize_figure(norm.index, norm)
            fig.update_layout(title=f"{result.asset.canonical_symbol} 归一化净值走势（期初 = 1.0）")
            return fig
        if chart_type == "drawdown":
            dd_df = plot_df[["drawdown"]].rename(
                columns={"drawdown": result.asset.canonical_symbol}
            )
            return charts_mod.build_drawdown_figure(
                dd_df,
                title=f"{result.asset.canonical_symbol} 历史水下回撤",
                from_nav=False,
            )
        if chart_type == "rolling":
            return charts_mod.build_tab1_risk_figure(plot_df)

    if isinstance(result, MultiAssetResult):
        if result.command == "compare":
            base_ccy = str(result.methodology.get("base_currency") or "LOCAL")
            if chart_type == "normalized":
                prices_df = result.frames.get("prices", result.data_frame)
                return charts_mod.build_tab3_multi_normalize_figure(
                    prices_df, base_currency=base_ccy
                )
            if chart_type == "drawdown":
                prices_df = result.frames.get("prices", result.data_frame)
                return charts_mod.build_drawdown_figure(
                    prices_df, title="多资产水下回撤对比", from_nav=True
                )
            if chart_type == "rolling":
                bench_col = (
                    str(result.benchmark.get("leg_name"))
                    if isinstance(result.benchmark, Mapping) and result.benchmark.get("leg_name")
                    else None
                )
                return charts_mod.build_compare_rolling_figure(
                    result.frames["rolling_metrics"],
                    rolling_window=result.methodology.get("rolling_window"),
                    benchmark_col=bench_col,
                )
            if chart_type == "correlation":
                return charts_mod.build_tab3_correlation_heatmap(result.frames["correlation"])

        if result.command == "portfolio":
            strategy = str(result.methodology.get("strategy") or "")
            mode = str(result.methodology.get("validation_mode") or "")
            has_bias = bool(result.methodology.get("has_lookahead_bias", False))
            base_ccy = str(result.methodology.get("base_currency") or "LOCAL")
            bench_label = (
                str(result.benchmark.get("leg_name"))
                if isinstance(result.benchmark, Mapping) and result.benchmark.get("leg_name")
                else None
            )
            if chart_type == "backtest":
                bt = result.frames.get("backtest", result.data_frame)
                return charts_mod.build_portfolio_backtest_with_drawdown_figure(
                    bt,
                    strategy=strategy,
                    validation_mode=mode,
                    has_lookahead_bias=has_bias,
                    base_currency=base_ccy,
                    benchmark_label=bench_label,
                )
            if chart_type == "weights":
                rc = result.metrics.get("risk_contributions", {})
                dr = rc.get("diversification_ratio") if isinstance(rc, Mapping) else None
                pvol = rc.get("portfolio_volatility") if isinstance(rc, Mapping) else None
                return charts_mod.build_portfolio_weights_risk_figure(
                    result.frames["weights"],
                    strategy=strategy,
                    diversification_ratio=float(dr) if is_valid_number(dr) else None,
                    portfolio_volatility=float(pvol) if is_valid_number(pvol) else None,
                )
            if chart_type == "frontier":
                p_metrics = result.metrics.get("portfolio", {})
                p_vol = p_metrics.get("estimation_volatility")
                p_ret = p_metrics.get("estimation_expected_return")
                point = (
                    (float(p_vol), float(p_ret))
                    if is_valid_number(p_vol) and is_valid_number(p_ret)
                    else None
                )
                return charts_mod.build_efficient_frontier_from_frame(
                    result.frames["frontier"],
                    cov_method=str(result.methodology.get("cov_method") or ""),
                    frontier_summary=result.metrics.get("efficient_frontier"),
                    portfolio_point=point,
                    portfolio_label=f"目标组合 ({strategy})",
                )
            if chart_type == "correlation":
                return charts_mod.build_tab3_correlation_heatmap(result.frames["correlation"])
            if chart_type == "normalized":
                return charts_mod.build_tab3_multi_normalize_figure(
                    result.frames["prices"], base_currency=base_ccy
                )
            if chart_type == "drawdown":
                bt = result.frames.get("backtest", result.data_frame)
                return charts_mod.build_drawdown_figure(
                    bt, title="组合与基准水下回撤对比", from_nav=True
                )
            if chart_type == "weight_history":
                return charts_mod.build_tab3_weight_history_figure(result.frames["weight_history"])

    raise ValueError(
        f"Unsupported chart type {chart_type!r} for result command {result.command!r}."
    )


def _style_mpl_ax(ax: Any) -> None:
    ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(labelsize=9, colors="#334155")


def _safe_ascii_label(text: str, fallback: str) -> str:
    """Keep Matplotlib labels ASCII-clean on systems lacking CJK fonts."""
    if text.isascii():
        return text
    ascii_part = "".join(ch for ch in text if ch.isascii()).strip()
    return ascii_part if ascii_part else fallback


def _build_mpl_figure_for_type(
    result: AnalysisResult | MultiAssetResult,
    chart_type: str,
) -> tuple[Any, Any]:
    """Render a clean, publication-grade Matplotlib Figure for PNG/SVG export."""
    _, plt, mticker = _load_matplotlib()
    rc_overrides = _get_mpl_rc_overrides()

    with plt.rc_context(rc_overrides):
        if isinstance(result, AnalysisResult):
            return _build_single_mpl(result, chart_type, plt, mticker)
        return _build_multi_mpl(result, chart_type, plt, mticker)


def _build_single_mpl(
    result: AnalysisResult,
    chart_type: str,
    plt: Any,
    mticker: Any,
) -> tuple[Any, Any]:
    plot_df = _prepare_single_asset_df(result)
    symbol = result.asset.canonical_symbol
    ccy = result.asset.output_currency or ""
    feat_sum = plot_df.attrs.get("feature_summary", {})
    sma_win = int(feat_sum.get("sma_short_window", 20))
    vol_win = int(feat_sum.get("volatility_window", 20))

    if chart_type == "price" and result.asset.series_kind != "yield_level":
        try:
            vis_mod = importlib.import_module("visualizer")
            vis = vis_mod.MplfinanceVisualizer()
            vol_series = pd.to_numeric(plot_df.get("Volume"), errors="coerce").fillna(0.0)
            show_vol = bool(
                result.asset.asset_type != "mutual_fund" and float(vol_series.abs().max()) > 0.0
            )
            panel_cols: list[str] = []
            sma_col = f"sma_{sma_win}"
            if sma_col in plot_df.columns and plot_df[sma_col].notna().any():
                panel_cols.append(sma_col)
            fig = vis.plot(
                plot_df,
                title=f"{symbol} Price Diagnostics ({ccy})"
                if ccy
                else f"{symbol} Price Diagnostics",
                volume=show_vol,
                main_panel_cols=panel_cols,
                lower_panel_cols=[],
                show=False,
            )
            return fig, plt
        except Exception:
            # Fall back to standard Matplotlib line chart if candlestick/mplfinance fails on degenerate slices
            pass

    if chart_type == "price":
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        close = pd.to_numeric(plot_df["Close"], errors="coerce")
        label = "Yield Level (%)" if result.asset.series_kind == "yield_level" else "Close Price"
        ax.plot(plot_df.index, close, color="#2563eb", linewidth=1.8, label=label)
        sma_col = f"sma_{sma_win}"
        if sma_col in plot_df.columns and plot_df[sma_col].notna().any():
            ax.plot(
                plot_df.index,
                pd.to_numeric(plot_df[sma_col], errors="coerce"),
                color="#d97706",
                linewidth=1.4,
                linestyle="--",
                label=f"SMA {sma_win}",
            )
        if result.asset.series_kind == "yield_level":
            ax.set_ylabel("Yield (%)")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
            ax.set_title(
                f"{symbol} Yield Level Series (Non-Return)", fontsize=12, fontweight="bold"
            )
        else:
            unit_lbl = f" ({ccy})" if ccy else ""
            ax.set_ylabel(f"Price{unit_lbl}")
            ax.set_title(f"{symbol} Price Series{unit_lbl}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        _style_mpl_ax(ax)
        fig.autofmt_xdate()
        return fig, plt

    if chart_type == "normalized":
        norm = normalize_close_to_base_one(plot_df)
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        ax.plot(
            norm.index, norm.values, color="#2563eb", linewidth=1.9, label=f"{symbol} (Base=1.0)"
        )
        ax.axhline(1.0, color="#94a3b8", linewidth=1.0, linestyle=":")
        ax.set_title(f"{symbol} Normalized NAV (Base = 1.0)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized NAV")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        _style_mpl_ax(ax)
        fig.autofmt_xdate()
        return fig, plt

    if chart_type == "drawdown":
        dd = -pd.to_numeric(plot_df["drawdown"], errors="coerce").abs() * 100.0
        fig, ax = plt.subplots(figsize=(10.5, 4.4))
        ax.plot(plot_df.index, dd, color="#dc2626", linewidth=1.6, label="Underwater Drawdown")
        ax.fill_between(plot_df.index, dd, 0.0, color="#dc2626", alpha=0.18)
        ax.axhline(0.0, color="#64748b", linewidth=0.9)
        ax.set_title(f"{symbol} Historical Underwater Drawdown", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Date")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        _style_mpl_ax(ax)
        fig.autofmt_xdate()
        return fig, plt

    if chart_type == "rolling":
        vol_col = f"volatility_{vol_win}"
        has_rsi = "rsi_14" in plot_df.columns and bool(
            pd.to_numeric(plot_df["rsi_14"], errors="coerce").notna().any()
        )
        n_rows = 3 if has_rsi else 2
        fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10.5, 7.6 if has_rsi else 5.8))
        if n_rows == 1:
            axes = [axes]
        if vol_col in plot_df.columns:
            vol_vals = pd.to_numeric(plot_df[vol_col], errors="coerce") * 100.0
            if vol_vals.notna().any():
                axes[0].plot(plot_df.index, vol_vals, color="#0284c7", linewidth=1.6)
            else:
                axes[0].text(
                    0.5,
                    0.5,
                    "Insufficient samples for rolling window",
                    transform=axes[0].transAxes,
                    ha="center",
                    va="center",
                    color="#64748b",
                )
        axes[0].set_title(
            f"{symbol} Rolling Ann. Volatility ({vol_win}D)", fontsize=11, fontweight="bold"
        )
        axes[0].set_ylabel("Ann. Vol (%)")
        axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        _style_mpl_ax(axes[0])

        if "drawdown" in plot_df.columns:
            dd = -pd.to_numeric(plot_df["drawdown"], errors="coerce").abs() * 100.0
            axes[1].plot(plot_df.index, dd, color="#dc2626", linewidth=1.6)
            axes[1].fill_between(plot_df.index, dd, 0.0, color="#dc2626", alpha=0.18)
        axes[1].set_title("Historical Underwater Drawdown", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        _style_mpl_ax(axes[1])

        if has_rsi:
            rsi = pd.to_numeric(plot_df["rsi_14"], errors="coerce")
            axes[2].plot(plot_df.index, rsi, color="#7c3aed", linewidth=1.5)
            axes[2].axhline(70.0, color="#ef4444", linestyle="--", linewidth=0.9, alpha=0.8)
            axes[2].axhline(30.0, color="#10b981", linestyle="--", linewidth=0.9, alpha=0.8)
            axes[2].set_ylim(0, 100)
            axes[2].set_title("RSI (14)", fontsize=11, fontweight="bold")
            axes[2].set_ylabel("RSI")
            _style_mpl_ax(axes[2])

        axes[-1].set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, plt

    raise ValueError(f"Unsupported single-asset Matplotlib chart type: {chart_type!r}")


def _draw_correlation_mpl(corr_df: pd.DataFrame, plt: Any, *, title: str) -> Any:
    labels = [str(c) for c in corr_df.columns]
    z = corr_df.values.astype(float, copy=False)
    n = len(labels)
    side = max(5.8, min(11.0, 1.3 * n + 2.5))
    fig, ax = plt.subplots(figsize=(side + 1.0, side))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = float(z[i, j])
            txt = f"{val:.2f}" if math.isfinite(val) else "N/A"
            color = "#ffffff" if math.isfinite(val) and abs(val) >= 0.55 else "#0f172a"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def _build_multi_mpl(
    result: MultiAssetResult,
    chart_type: str,
    plt: Any,
    mticker: Any,
) -> tuple[Any, Any]:
    frames = result.frames
    base_ccy = str(result.methodology.get("base_currency") or "LOCAL")

    if chart_type == "normalized":
        prices_df = frames.get("prices", result.data_frame)
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        for idx, col in enumerate(prices_df.columns):
            sub = prices_df[[col]].dropna()
            if sub.empty:
                continue
            norm = normalize_close_to_base_one(sub, price_col=col)
            color = _SERIES_PALETTE[idx % len(_SERIES_PALETTE)]
            ax.plot(norm.index, norm.values, color=color, linewidth=1.8, label=str(col))
        ax.axhline(1.0, color="#94a3b8", linewidth=0.9, linestyle=":")
        ax.set_title(
            f"Normalized Performance Comparison ({base_ccy}, Base = 1.0)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Normalized NAV")
        ax.set_xlabel("Date")
        ax.legend(
            loc="upper left", frameon=False, fontsize=9, ncol=min(4, max(1, len(prices_df.columns)))
        )
        _style_mpl_ax(ax)
        fig.autofmt_xdate()
        return fig, plt

    if chart_type == "drawdown":
        source_df = (
            frames.get("backtest", result.data_frame)
            if result.command == "portfolio"
            else frames.get("prices", result.data_frame)
        )
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        for idx, col in enumerate(source_df.columns):
            series = pd.to_numeric(source_df[col], errors="coerce").dropna()
            if series.empty:
                continue
            dd = ((series / series.cummax()) - 1.0) * 100.0
            color = _SERIES_PALETTE[idx % len(_SERIES_PALETTE)]
            lbl = _safe_ascii_label(str(col), f"Series {idx + 1}")
            ax.plot(dd.index, dd.values, color=color, linewidth=1.6, label=lbl)
        ax.axhline(0.0, color="#64748b", linewidth=0.9)
        ax.set_title("Underwater Drawdown Comparison", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Date")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(
            loc="lower left", frameon=False, fontsize=9, ncol=min(4, max(1, len(source_df.columns)))
        )
        _style_mpl_ax(ax)
        fig.autofmt_xdate()
        return fig, plt

    if chart_type == "correlation":
        fig = _draw_correlation_mpl(
            frames["correlation"], plt, title="Return Correlation Matrix (Pearson)"
        )
        return fig, plt

    if chart_type == "rolling":
        rolling_df = frames["rolling_metrics"]
        cols = [str(c) for c in rolling_df.columns]
        ret_cols = [c for c in cols if c.endswith("__rolling_return_ann")]
        vol_cols = [c for c in cols if c.endswith("__rolling_vol_ann")]
        sharpe_cols = [c for c in cols if c.endswith("__rolling_sharpe")]
        corr_bench_cols = [c for c in cols if c.endswith("__rolling_corr_vs_benchmark")]
        corr_pair_cols = [c for c in cols if c.endswith("__rolling_corr")]
        corr_cols = corr_bench_cols if corr_bench_cols else corr_pair_cols
        win = result.methodology.get("rolling_window")
        win_tag = f" ({win}D)" if win else ""

        n_rows = 4 if corr_cols else 3
        fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10.5, 2.35 * n_rows + 0.8))
        panel_specs: list[tuple[int, list[str], str, str, bool]] = [
            (0, ret_cols, "__rolling_return_ann", "Ann. Return (%)", True),
            (1, vol_cols, "__rolling_vol_ann", "Ann. Vol (%)", True),
            (2, sharpe_cols, "__rolling_sharpe", "Sharpe Ratio", False),
        ]
        if corr_cols:
            corr_suf = "__rolling_corr_vs_benchmark" if corr_bench_cols else "__rolling_corr"
            panel_specs.append((3, corr_cols, corr_suf, "Rolling Corr", False))

        for row_idx, pcols, suffix, ylabel, is_pct in panel_specs:
            ax = axes[row_idx]
            any_valid = False
            for idx, col in enumerate(pcols):
                vals = pd.to_numeric(rolling_df[col], errors="coerce")
                if is_pct:
                    vals = vals * 100.0
                if vals.notna().any():
                    any_valid = True
                lbl = col[: -len(suffix)].replace("__vs__", " vs ")
                ax.plot(
                    rolling_df.index,
                    vals,
                    color=_SERIES_PALETTE[idx % len(_SERIES_PALETTE)],
                    linewidth=1.5,
                    label=lbl,
                )
            if not any_valid:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient samples for rolling window",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#64748b",
                    fontsize=9,
                )
            ax.set_ylabel(ylabel)
            if is_pct:
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            if row_idx == 0 or row_idx == 3:
                ax.legend(
                    loc="upper left", frameon=False, fontsize=8, ncol=min(4, max(1, len(pcols)))
                )
            _style_mpl_ax(ax)

        axes[0].set_title(
            f"Multi-Asset Rolling Risk & Return{win_tag}", fontsize=12, fontweight="bold"
        )
        axes[-1].set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, plt

    if chart_type == "backtest":
        bt = frames.get("backtest", result.data_frame)
        strategy = str(result.methodology.get("strategy") or "")
        mode = str(result.methodology.get("validation_mode") or "")
        has_bias = bool(result.methodology.get("has_lookahead_bias", False))
        fig, (ax_nav, ax_dd) = plt.subplots(
            2, 1, sharex=True, figsize=(10.5, 6.4), gridspec_kw={"height_ratios": [1.8, 1.0]}
        )
        primary_col = "OOS_Portfolio" if "OOS_Portfolio" in bt.columns else "Optimal_Portfolio"
        trace_specs = [
            (primary_col, f"{strategy} ({mode})", "#dc2626", "-", 2.0),
            ("Equal_Weight_Portfolio", "Equal Weight (1/N)", "#64748b", "-", 1.6),
            ("Benchmark", "Benchmark", "#d97706", "-.", 1.6),
            ("In_Sample_Portfolio", "In-Sample Ref [Hindsight Bias]", "#2563eb", "--", 1.4),
        ]
        for col, label, color, ls, lw in trace_specs:
            if col not in bt.columns or not bt[col].notna().any():
                continue
            s = pd.to_numeric(bt[col], errors="coerce").dropna()
            ax_nav.plot(s.index, s.values, color=color, linestyle=ls, linewidth=lw, label=label)
            if col != "In_Sample_Portfolio":
                dd = ((s / s.cummax()) - 1.0) * 100.0
                ax_dd.plot(
                    dd.index, dd.values, color=color, linestyle=ls, linewidth=1.4, label=label
                )
                if col == primary_col:
                    ax_dd.fill_between(dd.index, dd.values, 0.0, color=color, alpha=0.14)

        bias_note = " [LOOKAHEAD BIAS WARNING]" if has_bias else ""
        ax_nav.set_title(
            f"Portfolio Backtest ({strategy} | {mode} | {base_ccy}){bias_note}",
            fontsize=11,
            fontweight="bold",
        )
        ax_nav.set_ylabel("Normalized NAV")
        ax_nav.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
        _style_mpl_ax(ax_nav)

        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Date")
        ax_dd.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        _style_mpl_ax(ax_dd)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, plt

    if chart_type == "weights":
        w_df = frames["weights"]
        assets = [str(idx) for idx in w_df.index]
        x = np.arange(len(assets))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

        ref_cols = [
            ("weight", "Target", "#2563eb"),
            ("equal_weight", "Equal", "#64748b"),
            ("min_variance_weight", "MinVar", "#10b981"),
            ("max_sharpe_weight", "MaxSharpe", "#dc2626"),
        ]
        avail = [(c, l, clr) for c, l, clr in ref_cols if c in w_df.columns]
        width = 0.78 / max(1, len(avail))
        for i, (col, label, color) in enumerate(avail):
            offset = (i - (len(avail) - 1) / 2.0) * width
            vals = pd.to_numeric(w_df[col], errors="coerce").fillna(0.0).values * 100.0
            ax1.bar(x + offset, vals, width=width, color=color, label=label)
        ax1.set_xticks(x)
        ax1.set_xticklabels(assets, rotation=25, ha="right", fontsize=9)
        ax1.set_ylabel("Weight (%)")
        ax1.set_title("Strategy vs Reference Weights", fontsize=11, fontweight="bold")
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax1.legend(loc="upper right", frameon=False, fontsize=8.5)
        _style_mpl_ax(ax1)

        pair_cols = [
            ("weight", "Weight %", "#2563eb"),
            ("pct_risk_contribution", "Euler Risk Contrib %", "#d97706"),
        ]
        p_avail = [(c, l, clr) for c, l, clr in pair_cols if c in w_df.columns]
        pw = 0.72 / max(1, len(p_avail))
        for i, (col, label, color) in enumerate(p_avail):
            offset = (i - (len(p_avail) - 1) / 2.0) * pw
            vals = pd.to_numeric(w_df[col], errors="coerce").fillna(0.0).values * 100.0
            ax2.bar(x + offset, vals, width=pw, color=color, label=label)
        ax2.set_xticks(x)
        ax2.set_xticklabels(assets, rotation=25, ha="right", fontsize=9)
        ax2.set_ylabel("Share (%)")
        ax2.set_title("Capital Weight vs Euler Risk Contribution", fontsize=11, fontweight="bold")
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax2.legend(loc="upper right", frameon=False, fontsize=8.5)
        _style_mpl_ax(ax2)

        fig.tight_layout()
        return fig, plt

    if chart_type == "frontier":
        f_df = frames["frontier"]
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        exact_df = (
            f_df[f_df["frontier_type"] == "slsqp_exact"]
            if "frontier_type" in f_df.columns
            else f_df
        )
        rand_df = (
            f_df[f_df["frontier_type"] == "dirichlet_random"]
            if "frontier_type" in f_df.columns
            else f_df.iloc[0:0]
        )
        if not rand_df.empty:
            sc = ax.scatter(
                pd.to_numeric(rand_df["volatility"], errors="coerce") * 100.0,
                pd.to_numeric(rand_df["expected_return"], errors="coerce") * 100.0,
                c=pd.to_numeric(rand_df["sharpe"], errors="coerce"),
                cmap="viridis",
                s=14,
                alpha=0.55,
                label="Dirichlet Random Portfolios (Ref)",
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Sharpe Ratio", fontsize=9)
        if not exact_df.empty:
            ordered = exact_df.sort_values("volatility")
            ax.plot(
                pd.to_numeric(ordered["volatility"], errors="coerce") * 100.0,
                pd.to_numeric(ordered["expected_return"], errors="coerce") * 100.0,
                color="#2563eb",
                linewidth=2.2,
                marker="o",
                markersize=4,
                label="Exact Efficient Frontier (SLSQP)",
            )
        ef_summary = result.metrics.get("efficient_frontier", {})
        exact_sum = ef_summary.get("exact_slsqp", {}) if isinstance(ef_summary, Mapping) else {}
        gmv = (
            exact_sum.get("min_variance") or exact_sum.get("global_min_variance")
            if isinstance(exact_sum, Mapping)
            else None
        )
        if (
            isinstance(gmv, Mapping)
            and is_valid_number(gmv.get("volatility"))
            and is_valid_number(gmv.get("expected_return"))
        ):
            ax.scatter(
                [float(gmv["volatility"]) * 100.0],
                [float(gmv["expected_return"]) * 100.0],
                marker="D",
                s=85,
                color="#10b981",
                edgecolors="#ffffff",
                linewidths=1.0,
                zorder=5,
                label="GMV",
            )
        tang = (
            exact_sum.get("max_sharpe") or exact_sum.get("max_sharpe_tangency")
            if isinstance(exact_sum, Mapping)
            else None
        )
        if (
            isinstance(tang, Mapping)
            and is_valid_number(tang.get("volatility"))
            and is_valid_number(tang.get("expected_return"))
        ):
            ax.scatter(
                [float(tang["volatility"]) * 100.0],
                [float(tang["expected_return"]) * 100.0],
                marker="*",
                s=150,
                color="#dc2626",
                edgecolors="#ffffff",
                linewidths=0.9,
                zorder=6,
                label="Max Sharpe (Tangency)",
            )
        p_metrics = result.metrics.get("portfolio", {})
        p_vol = p_metrics.get("estimation_volatility")
        p_ret = p_metrics.get("estimation_expected_return")
        if is_valid_number(p_vol) and is_valid_number(p_ret):
            ax.scatter(
                [float(p_vol) * 100.0],
                [float(p_ret) * 100.0],
                marker="o",
                s=95,
                color="#d97706",
                edgecolors="#111827",
                linewidths=1.3,
                zorder=7,
                label=f"Selected ({result.methodology.get('strategy', '')})",
            )

        cov_m = result.methodology.get("cov_method", "")
        ax.set_title(f"Mean-Variance Efficient Frontier ({cov_m})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Expected Annualized Return (%)")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(loc="best", frameon=True, facecolor="#ffffff", framealpha=0.88, fontsize=8.5)
        _style_mpl_ax(ax)
        fig.tight_layout()
        return fig, plt

    if chart_type == "weight_history":
        wh_df = frames["weight_history"]
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        x_labels = [
            idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
            for idx in wh_df.index
        ]
        x = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels), dtype=float)
        for idx, col in enumerate(wh_df.columns):
            vals = pd.to_numeric(wh_df[col], errors="coerce").fillna(0.0).values * 100.0
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=_SERIES_PALETTE[idx % len(_SERIES_PALETTE)],
                label=str(col),
                width=0.72,
            )
            bottom = bottom + vals
        step = max(1, len(x_labels) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(x_labels[::step], rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel("Weight Allocation (%)")
        ax.set_xlabel("Rebalance Date")
        ax.set_title("Walk-Forward Rebalance Weight History", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend(
            loc="upper left", frameon=False, fontsize=8.5, ncol=min(4, max(1, len(wh_df.columns)))
        )
        _style_mpl_ax(ax)
        fig.tight_layout()
        return fig, plt

    raise ValueError(f"Unsupported multi-asset Matplotlib chart type: {chart_type!r}")


def _resolve_specs(
    result: AnalysisResult | MultiAssetResult,
    chart_types: Sequence[str] | None,
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    supported = COMMAND_CHART_TYPES.get(result.command)
    if supported is None:
        raise ValueError(f"Charts are not supported for command {result.command!r}.")

    explicit = chart_types is not None
    requested = tuple(chart_types) if chart_types is not None else supported

    if isinstance(result, AnalysisResult):
        return _resolve_single_asset_specs(result, requested)
    if result.command == "compare":
        return _resolve_compare_specs(result, requested)
    if result.command == "portfolio":
        return _resolve_portfolio_specs(result, requested, explicit_selection=explicit)
    raise ValueError(f"Unsupported command {result.command!r}.")


def build_plotly_figures(
    result: AnalysisResult | MultiAssetResult,
    *,
    chart_types: Sequence[str] | None = None,
) -> tuple[dict[str, tuple[str, str, Any]], dict[str, str]]:
    """Build in-memory Plotly figures keyed by chart_type for Streamlit M5 or callers.

    Returns ``({chart_type: (title, description, fig)}, skipped_reasons)``.
    """
    specs, skipped = _resolve_specs(result, chart_types)
    built: dict[str, tuple[str, str, Any]] = {}
    for ct, title, desc in specs:
        fig = _build_plotly_figure_for_type(result, ct)
        built[ct] = (title, desc, fig)
    return built, skipped


def render_charts(
    result: AnalysisResult | MultiAssetResult,
    output_dir: str | Path,
    *,
    formats: Sequence[str],
    chart_types: Sequence[str] | None = None,
    file_writer: Callable[[Path, bytes], Path | None] | None = None,
) -> ChartRenderOutput:
    """Render requested chart formats under ``<output_dir>/charts`` without recalculating data.

    Parameters
    ----------
    file_writer:
        Optional staging callback ``(target_path, content_bytes) -> final_path`` used by the
        atomic 3-phase CLI committer so chart writes participate in rollback/cleanup.
    """
    norm_formats = parse_chart_formats(",".join(formats)) if formats else ()
    if not norm_formats:
        return ChartRenderOutput()

    out_dir = Path(output_dir).expanduser().resolve()
    charts_dir = out_dir / "charts"
    specs, skipped = _resolve_specs(result, chart_types)

    warnings: list[str] = []
    for ct, reason in skipped.items():
        warnings.append(f"图表 `{ct}` 已跳过：{reason}")

    artifacts: list[ChartArtifact] = []
    need_html = "html" in norm_formats
    need_mpl = [fmt for fmt in norm_formats if fmt in {"png", "svg"}]

    for ct, title, desc in specs:
        if need_html:
            fig = _build_plotly_figure_for_type(result, ct)
            html_bytes = _export_plotly_html_bytes(fig, title=title)
            target = charts_dir / f"{ct}.html"
            if file_writer is not None:
                written = file_writer(target, html_bytes)
                final_path = Path(written) if written is not None else target.resolve()
            else:
                charts_dir.mkdir(parents=True, exist_ok=True)
                target.write_bytes(html_bytes)
                final_path = target.resolve()
            artifacts.append(
                ChartArtifact(
                    chart_type=ct,
                    title=_CHART_TITLES_ZH.get(ct, title),
                    description=desc,
                    format="html",
                    path=final_path,
                    artifact_key=f"chart_{ct}_html",
                )
            )

        for fmt in need_mpl:
            fig, plt = _build_mpl_figure_for_type(result, ct)
            img_bytes = _save_mpl_figure_bytes(plt, fig, fmt)
            target = charts_dir / f"{ct}.{fmt}"
            if file_writer is not None:
                written = file_writer(target, img_bytes)
                final_path = Path(written) if written is not None else target.resolve()
            else:
                charts_dir.mkdir(parents=True, exist_ok=True)
                target.write_bytes(img_bytes)
                final_path = target.resolve()
            artifacts.append(
                ChartArtifact(
                    chart_type=ct,
                    title=_CHART_TITLES_ZH.get(ct, title),
                    description=desc,
                    format=fmt,
                    path=final_path,
                    artifact_key=f"chart_{ct}_{fmt}",
                )
            )

    return ChartRenderOutput(
        charts=tuple(artifacts),
        skipped=skipped,
        warnings=tuple(warnings),
    )


__all__ = [
    "COMMAND_CHART_TYPES",
    "SUPPORTED_CHART_FORMATS",
    "ChartArtifact",
    "ChartRenderOutput",
    "build_plotly_figures",
    "parse_chart_formats",
    "parse_chart_types",
    "render_charts",
]
