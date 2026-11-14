"""Plotly figure builders; no Streamlit or data-fetching dependencies."""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pandas as pd

from quant_models import (
    EfficientFrontierResult,
    ExactEfficientFrontierResult,
    MonteCarloGBMResult,
    WalkForwardResult,
    normalize_close_to_base_one,
)

_GRID_COLOR = "rgba(148, 163, 184, 0.22)"
_ZEROLINE_COLOR = "rgba(148, 163, 184, 0.35)"
_SERIES_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4f46e5",
)


def _load_optional_dependency(module_name: str, install_name: str | None = None) -> Any:
    """按需导入三方依赖，避免模块导入阶段就因缺包失败。"""
    package_name = install_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "缺少运行依赖 `%s`，请先安装：`pip install %s`。" % (module_name, package_name)
        ) from exc


def _load_plotly_modules() -> tuple[Any, Any]:
    """按需导入 Plotly，避免模块导入阶段缺包即失败。"""
    go = _load_optional_dependency("plotly.graph_objects", install_name="plotly")
    subplots = _load_optional_dependency("plotly.subplots", install_name="plotly")
    return go, subplots


def apply_theme_layout(
    figure: Any,
    *,
    height: int = 500,
    margin: dict[str, int] | None = None,
    hovermode: str = "x unified",
    title: dict[str, Any] | str | None = None,
    extra_layout: dict[str, Any] | None = None,
) -> Any:
    """应用自适应明暗主题的透明背景与柔和网格线布局。"""
    layout_kwargs: dict[str, Any] = {
        "height": height,
        "margin": margin or {"l": 40, "r": 24, "t": 52, "b": 36},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "hovermode": hovermode,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0.0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    }
    if title is not None:
        layout_kwargs["title"] = (
            {"text": title, "font": {"size": 15}} if isinstance(title, str) else title
        )
    if extra_layout:
        layout_kwargs.update(extra_layout)
    figure.update_layout(**layout_kwargs)
    figure.update_xaxes(gridcolor=_GRID_COLOR, zerolinecolor=_ZEROLINE_COLOR)
    figure.update_yaxes(gridcolor=_GRID_COLOR, zerolinecolor=_ZEROLINE_COLOR)
    return figure


def build_price_volume_figure(
    df: pd.DataFrame,
    *,
    include_risk_panels: bool = True,
) -> Any:
    """绘制 K 线 + 成交量 + 可选滚动波动率与水下回撤副图；净值/收益率水平用折线。"""
    go, subplots = _load_plotly_modules()
    make_subplots = subplots.make_subplots
    asset_key = str(df.attrs.get("asset_type", "")).strip().lower()
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    vol_max = float(pd.to_numeric(df["Volume"], errors="coerce").fillna(0).abs().max())
    use_line = asset_key == "mutual_fund" or series_kind == "yield_level"
    show_volume_panel = not use_line and vol_max > 0.0

    feature_summary = df.attrs.get("feature_summary", {})
    short_window = int(feature_summary.get("sma_short_window", 20))
    long_window = int(feature_summary.get("sma_long_window", 60))
    volatility_window = int(feature_summary.get("volatility_window", 20))
    bollinger_window = int(feature_summary.get("bollinger_window", 20))

    vol_col = f"volatility_{volatility_window}"
    has_vol_panel = (
        include_risk_panels
        and vol_col in df.columns
        and bool(pd.to_numeric(df[vol_col], errors="coerce").notna().any())
    )
    has_dd_panel = (
        include_risk_panels
        and "drawdown" in df.columns
        and bool(pd.to_numeric(df["drawdown"], errors="coerce").notna().any())
    )

    panel_ratios = [0.54 if (has_vol_panel or has_dd_panel) else 0.72]
    if show_volume_panel:
        panel_ratios.append(0.16 if (has_vol_panel or has_dd_panel) else 0.28)
    if has_vol_panel:
        panel_ratios.append(0.15)
    if has_dd_panel:
        panel_ratios.append(0.15)

    n_rows = len(panel_ratios)
    if n_rows > 1:
        total_w = sum(panel_ratios)
        norm_heights = [r / total_w for r in panel_ratios]
        figure = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            row_heights=norm_heights,
        )
    else:
        figure = make_subplots(rows=1, cols=1)

    if use_line:
        line_name = "累计净值" if asset_key == "mutual_fund" else "收盘序列"
        if series_kind == "yield_level":
            line_name = "收益率水平"
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=line_name,
                line={"width": 2.0, "color": "#2563eb"},
            ),
            row=1,
            col=1,
        )
    else:
        figure.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K线",
            ),
            row=1,
            col=1,
        )

    bb_upper = f"bollinger_upper_{bollinger_window}"
    bb_lower = f"bollinger_lower_{bollinger_window}"
    if bb_upper in df.columns and bb_lower in df.columns:
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df[bb_upper],
                mode="lines",
                name="布林上轨",
                line={"width": 1.3, "color": "#9467bd"},
                opacity=0.65,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df[bb_lower],
                mode="lines",
                name="布林下轨",
                line={"width": 1.3, "color": "#9467bd"},
                fill="tonexty",
                fillcolor="rgba(148, 103, 189, 0.10)",
                opacity=0.65,
            ),
            row=1,
            col=1,
        )

    for column, trace_name, color in (
        (f"sma_{short_window}", f"SMA {short_window}", "#1f77b4"),
        (f"sma_{long_window}", f"SMA {long_window}", "#ff7f0e"),
    ):
        if column in df.columns:
            figure.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode="lines",
                    name=trace_name,
                    line={"width": 1.6, "color": color},
                    opacity=0.9,
                ),
                row=1,
                col=1,
            )

    current_row = 2
    y1 = "净值" if asset_key == "mutual_fund" else "价格"
    if series_kind == "yield_level":
        y1 = "收益率水平 (%)"
    figure.update_yaxes(title_text=y1, row=1, col=1)

    if show_volume_panel:
        volume_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
        figure.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="成交量",
                marker={"color": volume_colors},
            ),
            row=current_row,
            col=1,
        )
        figure.update_yaxes(title_text="成交量", row=current_row, col=1)
        current_row += 1

    if has_vol_panel:
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df[vol_col],
                mode="lines",
                name=f"年化波动 ({volatility_window}D)",
                line={"width": 1.5, "color": "#0284c7"},
            ),
            row=current_row,
            col=1,
        )
        figure.update_yaxes(title_text="年化波动", tickformat=".0%", row=current_row, col=1)
        current_row += 1

    if has_dd_panel:
        underwater = -pd.to_numeric(df["drawdown"], errors="coerce").abs()
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=underwater,
                mode="lines",
                name="水下回撤",
                line={"width": 1.5, "color": "#dc2626"},
                fill="tozeroy",
                fillcolor="rgba(220, 38, 38, 0.20)",
            ),
            row=current_row,
            col=1,
        )
        figure.update_yaxes(title_text="回撤", tickformat=".0%", row=current_row, col=1)

    chart_height = 740 if n_rows >= 3 else 620
    apply_theme_layout(
        figure,
        height=chart_height,
        margin={"l": 44, "r": 20, "t": 54, "b": 24},
        hovermode="x unified",
    )
    figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    return figure


def build_tab1_risk_figure(df: pd.DataFrame) -> Any:
    """绘制独立的多面板风险与动量图（年化波动率、水下回撤与可选 RSI-14）。"""
    go, subplots = _load_plotly_modules()
    feature_summary = df.attrs.get("feature_summary", {})
    volatility_window = int(feature_summary.get("volatility_window", 20))
    vol_col = f"volatility_{volatility_window}"
    has_rsi = "rsi_14" in df.columns and bool(
        pd.to_numeric(df["rsi_14"], errors="coerce").notna().any()
    )
    n_rows = 3 if has_rsi else 2
    figure = subplots.make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["滚动年化波动率", "历史水下回撤"] + (["RSI(14) 动量"] if has_rsi else []),
    )
    if vol_col in df.columns:
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df[vol_col],
                mode="lines",
                name=f"年化波动 ({volatility_window}D)",
                line={"width": 1.8, "color": "#0284c7"},
            ),
            row=1,
            col=1,
        )
    if "drawdown" in df.columns:
        underwater = -pd.to_numeric(df["drawdown"], errors="coerce").abs()
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=underwater,
                mode="lines",
                name="水下回撤",
                line={"width": 1.8, "color": "#dc2626"},
                fill="tozeroy",
                fillcolor="rgba(220, 38, 38, 0.22)",
            ),
            row=2,
            col=1,
        )
    if has_rsi:
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi_14"],
                mode="lines",
                name="RSI(14)",
                line={"width": 1.6, "color": "#7c3aed"},
            ),
            row=3,
            col=1,
        )
        figure.add_hline(y=70, line_dash="dot", line_color="rgba(220, 38, 38, 0.5)", row=3, col=1)
        figure.add_hline(y=30, line_dash="dot", line_color="rgba(5, 150, 105, 0.5)", row=3, col=1)
        figure.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    figure.update_yaxes(title_text="波动率", tickformat=".0%", row=1, col=1)
    figure.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    return apply_theme_layout(
        figure,
        height=520 if has_rsi else 400,
        margin={"l": 44, "r": 24, "t": 54, "b": 28},
    )


def build_tab3_normalize_figure(dates: pd.Index, normalized: pd.Series) -> Any:
    """侧栏单一标的：收盘价归一化为期初=1 的净值曲线。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=normalized,
            mode="lines",
            name="Nₜ = Pₜ / P₀",
            line={"width": 2.2, "color": "#2563eb"},
        )
    )
    return apply_theme_layout(
        figure,
        height=420,
        margin={"l": 40, "r": 24, "t": 44, "b": 32},
        extra_layout={"yaxis_title": "归一化净值", "xaxis_title": "日期"},
    )


def build_tab3_multi_normalize_figure(
    base_prices_df: pd.DataFrame,
    *,
    base_currency: str = "CNY",
) -> Any:
    """多标的统一基准货币与估值时点下的期初=1.0 归一化对比曲线。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()
    for idx, col in enumerate(base_prices_df.columns):
        sub = base_prices_df[[col]].dropna()
        if sub.empty:
            continue
        norm = normalize_close_to_base_one(sub, price_col=col)
        color = _SERIES_PALETTE[idx % len(_SERIES_PALETTE)]
        figure.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm,
                mode="lines",
                name=str(col),
                line={"width": 2.0, "color": color},
            )
        )
    return apply_theme_layout(
        figure,
        height=440,
        margin={"l": 44, "r": 24, "t": 56, "b": 32},
        title=f"多资产归一化走势对比（统一口径：{base_currency}，期初 = 1.0）",
        extra_layout={"yaxis_title": "归一化净值", "xaxis_title": "估值日期"},
    )


def build_tab3_gbm_figure(result: MonteCarloGBMResult) -> Any:
    """GBM 蒙特卡洛：历史收盘价、未来分位带、中位数与若干示例路径。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.historical_dates,
            y=result.historical_close,
            mode="lines",
            name="历史收盘价",
            line={"width": 2.0, "color": "#2563eb"},
        )
    )
    fut_x = result.future_dates
    figure.add_trace(
        go.Scatter(
            x=fut_x,
            y=result.quantiles_p95,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=fut_x,
            y=result.quantiles_p05,
            mode="lines",
            name="5%–95% 分位带",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line={"width": 0},
        )
    )
    n_paths = int(result.sample_paths.shape[0])
    for i in range(n_paths):
        figure.add_trace(
            go.Scatter(
                x=fut_x,
                y=result.sample_paths[i],
                mode="lines",
                name="示例路径" if i == 0 else None,
                legendgroup="samples",
                showlegend=(i == 0),
                line={"width": 0.9, "color": "rgba(100, 116, 139, 0.65)"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=fut_x,
            y=result.quantiles_p50,
            mode="lines",
            name="中位数路径",
            line={"width": 2.2, "color": "#d97706", "dash": "dash"},
        )
    )
    return apply_theme_layout(
        figure,
        height=520,
        margin={"l": 44, "r": 24, "t": 54, "b": 32},
        extra_layout={
            "yaxis_title": "价格",
            "xaxis_title": "日期",
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.05,
                "x": 0.0,
                "bgcolor": "rgba(0,0,0,0)",
            },
        },
    )


def build_tab3_efficient_frontier_figure(
    ef: ExactEfficientFrontierResult | EfficientFrontierResult,
) -> Any:
    """绘制精确 SLSQP 有效前沿包络线、GMV、最大夏普组合，并可叠加随机组合可行域散点。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()

    if isinstance(ef, ExactEfficientFrontierResult):
        if ef.random_volatility is not None and ef.random_returns is not None:
            figure.add_trace(
                go.Scatter(
                    x=ef.random_volatility,
                    y=ef.random_returns,
                    mode="markers",
                    name="随机可行组合（参考）",
                    marker={
                        "size": 5,
                        "color": ef.random_sharpe,
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "夏普"},
                        "opacity": 0.55,
                    },
                )
            )
        figure.add_trace(
            go.Scatter(
                x=ef.frontier_volatility,
                y=ef.frontier_returns,
                mode="lines+markers",
                name="精确有效前沿 (SLSQP)",
                line={"width": 3.0, "color": "#2563eb"},
                marker={"size": 5, "color": "#2563eb"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[ef.min_variance.volatility],
                y=[ef.min_variance.expected_return],
                mode="markers",
                name="最小方差组合 (GMV)",
                marker={
                    "size": 15,
                    "symbol": "diamond",
                    "color": "#10b981",
                    "line": {"width": 1.5, "color": "#ffffff"},
                },
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[ef.max_sharpe.volatility],
                y=[ef.max_sharpe.expected_return],
                mode="markers",
                name="最大夏普组合 (Tangency)",
                marker={
                    "size": 20,
                    "symbol": "star",
                    "color": "#ff2a2a",
                    "line": {"width": 1.4, "color": "#ffffff"},
                },
            )
        )
        title_text = f"均值-方差有效前沿 ({ef.cov_method})"
    else:
        vol = ef.volatility
        ret = ef.returns
        sharpe = ef.sharpe
        idx = int(ef.best_sharpe_idx)
        figure.add_trace(
            go.Scatter(
                x=vol,
                y=ret,
                mode="markers",
                name="随机权重组合",
                marker={
                    "size": 6,
                    "color": sharpe,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "夏普"},
                    "opacity": 0.85,
                },
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[float(vol[idx])],
                y=[float(ret[idx])],
                mode="markers",
                name="最大夏普组合",
                marker={
                    "size": 20,
                    "symbol": "star",
                    "color": "#ff2a2a",
                    "line": {"width": 1.2, "color": "#ffffff"},
                },
            )
        )
        title_text = "组合有效前沿模拟"

    return apply_theme_layout(
        figure,
        height=520,
        margin={"l": 50, "r": 24, "t": 56, "b": 36},
        hovermode="closest",
        title=title_text,
        extra_layout={
            "xaxis_title": "年化波动率",
            "yaxis_title": "年化预期收益",
            "xaxis_tickformat": ".0%",
            "yaxis_tickformat": ".0%",
            "xaxis_hoverformat": ".2%",
            "yaxis_hoverformat": ".2%",
        },
    )


def _relative_luminance_srgb_255(r: float, g: float, b: float) -> float:
    """WCAG 2.1 相对亮度，输入 sRGB 通道 0–255。"""

    def channel_lin(x: float) -> float:
        x = x / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel_lin(r) + 0.7152 * channel_lin(g) + 0.0722 * channel_lin(b)


def _relative_luminance_from_plotly_color(color: object) -> float:
    """从 Plotly ``sample_colorscale`` 等返回的颜色字符串估计相对亮度 (0–1)。"""
    s = str(color).strip()
    if not s:
        return 0.5
    if s.startswith("#"):
        hex_s = s[1:]
        if len(hex_s) >= 6:
            r = int(hex_s[0:2], 16)
            g = int(hex_s[2:4], 16)
            b = int(hex_s[4:6], 16)
            return _relative_luminance_srgb_255(float(r), float(g), float(b))
        return 0.5
    if s.startswith("rgb"):
        inner = s.split("(", 1)[1].rsplit(")", 1)[0]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 3:
            r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
            return _relative_luminance_srgb_255(r, g, b)
    return 0.45


def _correlation_heatmap_text_colors(
    z: np.ndarray,
    zmin: float,
    zmax: float,
    *,
    colorscale: str,
    reversescale: bool,
) -> list[list[str]]:
    """与 Heatmap 相同色标采样单元填充色；格偏亮则用黑字，否则白字。"""
    pc = _load_optional_dependency("plotly.colors", install_name="plotly")
    nrows, ncols = z.shape
    span = zmax - zmin
    text_colors: list[list[str]] = []
    for i in range(nrows):
        row_colors: list[str] = []
        for j in range(ncols):
            val = float(z[i, j])
            if not math.isfinite(val) or span == 0.0:
                row_colors.append("#fafafa")
                continue
            t = max(0.0, min(1.0, (val - zmin) / span))
            if reversescale:
                t = 1.0 - t
            sampled = pc.sample_colorscale(colorscale, [t])
            bg_lum = _relative_luminance_from_plotly_color(sampled[0])
            row_colors.append("#111111" if bg_lum > 0.56 else "#fafafa")
        text_colors.append(row_colors)
    return text_colors


def build_tab3_correlation_heatmap(corr_df: pd.DataFrame) -> Any:
    """大类资产收益率皮尔逊相关热力图（z∈[-1,1]，RdBu 发散）。"""
    go, _ = _load_plotly_modules()
    labels = list(corr_df.columns)
    z = corr_df.values.astype(float, copy=False)
    zmin, zmax = -1.0, 1.0
    text_colors = _correlation_heatmap_text_colors(
        z,
        zmin,
        zmax,
        colorscale="RdBu",
        reversescale=True,
    )
    nrows, ncols = z.shape
    cell_annotations: list[dict[str, Any]] = []
    for i in range(nrows):
        for j in range(ncols):
            val = float(z[i, j])
            label = "—" if not math.isfinite(val) else "%.2f" % val
            cell_annotations.append(
                {
                    "x": labels[j],
                    "y": labels[i],
                    "xref": "x",
                    "yref": "y",
                    "text": label,
                    "showarrow": False,
                    "font": {"size": 15, "color": text_colors[i][j]},
                }
            )
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                zmin=zmin,
                zmax=zmax,
                colorscale="RdBu",
                reversescale=True,
                hovertemplate="%{x} vs %{y}<br>ρ=%{z:.4f}<extra></extra>",
            )
        ]
    )
    return apply_theme_layout(
        figure,
        height=560,
        margin={"l": 36, "r": 28, "t": 56, "b": 110},
        hovermode="closest",
        title={"text": "资产收益率相关性", "font": {"size": 16}},
        extra_layout={
            "annotations": cell_annotations,
            "xaxis": {
                "side": "bottom",
                "tickangle": -32,
                "tickfont": {"size": 13},
                "gridcolor": _GRID_COLOR,
            },
            "yaxis": {
                "autorange": "reversed",
                "tickfont": {"size": 13},
                "gridcolor": _GRID_COLOR,
            },
        },
    )


def build_tab3_portfolio_backtest_figure(
    bt_df: pd.DataFrame,
    *,
    title: str = "历史净值回测（期初 = 1.0）",
    optimal_label: str = "最优组合",
) -> Any:
    """最优组合与等权组合的归一化净值曲线。"""
    go, _ = _load_plotly_modules()
    idx = bt_df.index
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=idx,
            y=bt_df["Optimal_Portfolio"],
            mode="lines",
            name=optimal_label,
            line={"width": 2.4, "color": "#ef4444"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=idx,
            y=bt_df["Equal_Weight_Portfolio"],
            mode="lines",
            name="等权重组合",
            line={"width": 2.0, "color": "#64748b"},
        )
    )
    return apply_theme_layout(
        figure,
        height=500,
        margin={"l": 56, "r": 28, "t": 64, "b": 84},
        title={
            "text": title,
            "font": {"size": 16},
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        extra_layout={
            "xaxis_title": "日期",
            "yaxis_title": "组合净值",
            "legend": {
                "orientation": "h",
                "yanchor": "top",
                "y": -0.20,
                "xanchor": "center",
                "x": 0.5,
                "bgcolor": "rgba(0,0,0,0)",
            },
        },
    )


def _nav_to_underwater(nav: pd.Series) -> pd.Series:
    s = pd.to_numeric(nav, errors="coerce").astype(float)
    peak = s.cummax()
    return (s / peak) - 1.0


def build_tab3_walk_forward_figure(
    wf_result: WalkForwardResult,
    *,
    base_currency: str = "CNY",
) -> Any:
    """滚动/扩展样本外回测净值与水下回撤双面板对照图。"""
    go, subplots = _load_plotly_modules()
    nav_df = wf_result.nav_df
    figure = subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.66, 0.34],
        subplot_titles=[
            f"无前视泄露样本外净值对比（{wf_result.window_mode} · {base_currency}，期初 = 1.0）",
            "样本外水下回撤对比",
        ],
    )
    figure.add_trace(
        go.Scatter(
            x=nav_df.index,
            y=nav_df["OOS_Portfolio"],
            mode="lines",
            name="样本外 Walk-Forward 组合",
            line={"width": 2.6, "color": "#ef4444"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=nav_df.index,
            y=nav_df["Equal_Weight_Portfolio"],
            mode="lines",
            name="样本外等权基准",
            line={"width": 2.0, "color": "#64748b"},
        ),
        row=1,
        col=1,
    )
    if "In_Sample_Portfolio" in nav_df.columns:
        figure.add_trace(
            go.Scatter(
                x=nav_df.index,
                y=nav_df["In_Sample_Portfolio"],
                mode="lines",
                name="全样本后视参考（非真实可投）",
                line={"width": 1.6, "color": "#2563eb", "dash": "dash"},
            ),
            row=1,
            col=1,
        )

    dd_oos = _nav_to_underwater(nav_df["OOS_Portfolio"])
    dd_eq = _nav_to_underwater(nav_df["Equal_Weight_Portfolio"])
    figure.add_trace(
        go.Scatter(
            x=nav_df.index,
            y=dd_oos,
            mode="lines",
            name="Walk-Forward 回撤",
            line={"width": 1.8, "color": "#dc2626"},
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.18)",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=nav_df.index,
            y=dd_eq,
            mode="lines",
            name="等权基准回撤",
            line={"width": 1.5, "color": "#64748b", "dash": "dot"},
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="净值", row=1, col=1)
    figure.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    return apply_theme_layout(
        figure,
        height=600,
        margin={"l": 56, "r": 24, "t": 62, "b": 44},
    )


def build_tab3_weight_history_figure(weight_history: pd.DataFrame) -> Any:
    """Walk-Forward 各再平衡节点目标权重堆叠演化图。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()
    x_labels = [
        idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
        for idx in weight_history.index
    ]
    for idx, col in enumerate(weight_history.columns):
        color = _SERIES_PALETTE[idx % len(_SERIES_PALETTE)]
        figure.add_trace(
            go.Bar(
                x=x_labels,
                y=weight_history[col],
                name=str(col),
                marker={"color": color},
            )
        )
    return apply_theme_layout(
        figure,
        height=400,
        margin={"l": 50, "r": 24, "t": 54, "b": 48},
        hovermode="x unified",
        title="Walk-Forward 各再平衡节点权重配置",
        extra_layout={
            "barmode": "stack",
            "xaxis_title": "再平衡日期",
            "yaxis_title": "权重占比",
            "yaxis_tickformat": ".0%",
        },
    )


__all__ = [
    "apply_theme_layout",
    "build_price_volume_figure",
    "build_tab1_risk_figure",
    "build_tab3_correlation_heatmap",
    "build_tab3_efficient_frontier_figure",
    "build_tab3_gbm_figure",
    "build_tab3_multi_normalize_figure",
    "build_tab3_normalize_figure",
    "build_tab3_portfolio_backtest_figure",
    "build_tab3_walk_forward_figure",
    "build_tab3_weight_history_figure",
]
