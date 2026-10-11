"""Plotly figure builders; no Streamlit or data-fetching dependencies."""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pandas as pd
from quant_models import EfficientFrontierResult, MonteCarloGBMResult


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


def build_price_volume_figure(df: pd.DataFrame) -> Any:
    """绘制 K 线 + 成交量；净值/收益率水平/零成交量序列用单面板折线。"""
    go, subplots = _load_plotly_modules()
    make_subplots = subplots.make_subplots
    asset_key = str(df.attrs.get("asset_type", "")).strip().lower()
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    vol_max = float(pd.to_numeric(df["Volume"], errors="coerce").fillna(0).abs().max())
    use_line = asset_key == "mutual_fund" or series_kind == "yield_level"
    show_volume_panel = not use_line and vol_max > 0.0

    if use_line:
        # 基金净值和收益率水平没有真实 K 线含义，用折线图更直观。
        line_name = "累计净值" if asset_key == "mutual_fund" else "收盘序列"
        if series_kind == "yield_level":
            line_name = "收益率水平"
        figure = make_subplots(rows=1, cols=1)
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=line_name,
                line={"width": 2.0, "color": "#1f77b4"},
            ),
            row=1,
            col=1,
        )
    elif show_volume_panel:
        # 有成交量的价格资产使用上下双面板：上方 K 线，下方成交量。
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
        )
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
    else:
        figure = make_subplots(rows=1, cols=1)
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

    feature_summary = df.attrs.get("feature_summary", {})
    short_window = int(feature_summary.get("sma_short_window", 20))
    long_window = int(feature_summary.get("sma_long_window", 60))
    bollinger_window = int(feature_summary.get("bollinger_window", 20))

    overlay_columns = [
        ("sma_%d" % short_window, "SMA %d" % short_window, "#1f77b4"),
        ("sma_%d" % long_window, "SMA %d" % long_window, "#ff7f0e"),
        ("bollinger_upper_%d" % bollinger_window, "布林上轨", "#9467bd"),
        ("bollinger_lower_%d" % bollinger_window, "布林下轨", "#9467bd"),
    ]
    for column, trace_name, color in overlay_columns:
        if column not in df.columns:
            continue
        # 均线和布林带都叠加在主图上，帮助观察趋势和价格偏离程度。
        figure.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                mode="lines",
                name=trace_name,
                line={"width": 1.6, "color": color},
                opacity=0.9 if "bollinger" not in column else 0.65,
            ),
            row=1,
            col=1,
        )

    if show_volume_panel:
        # 收盘不低于开盘用绿色，反之用红色，符合常见行情图习惯。
        volume_colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
        figure.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="成交量",
                marker={"color": volume_colors},
            ),
            row=2,
            col=1,
        )

    figure.update_layout(
        height=760,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    y1 = "净值" if asset_key == "mutual_fund" else "价格"
    if series_kind == "yield_level":
        y1 = "收益率水平"
    if use_line or not show_volume_panel:
        figure.update_yaxes(title_text=y1, row=1, col=1)
    if show_volume_panel:
        figure.update_yaxes(title_text=y1, row=1, col=1)
        figure.update_yaxes(title_text="成交量", row=2, col=1)
    return figure


def build_tab3_normalize_figure(dates: pd.Index, normalized: pd.Series) -> Any:
    """侧栏单一标的：收盘价归一化为期初=1 的净值曲线（暗色主题）。"""
    go, _ = _load_plotly_modules()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=normalized,
            mode="lines",
            name="Nₜ = Pₜ / P₀",
            line={"width": 2.2, "color": "#7fdbff"},
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="归一化净值",
        xaxis_title="日期",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return figure


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
            line={"width": 2.0, "color": "#00d4ff"},
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
            fillcolor="rgba(0, 191, 255, 0.22)",
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
                line={"width": 0.9, "color": "rgba(180, 180, 200, 0.45)"},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=fut_x,
            y=result.quantiles_p50,
            mode="lines",
            name="中位数路径",
            line={"width": 2.2, "color": "#ffcc00"},
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="价格",
        xaxis_title="日期",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0.0},
    )
    return figure


def build_tab3_efficient_frontier_figure(ef: EfficientFrontierResult) -> Any:
    """随机权重组合散点（颜色=夏普），最优组合红星。"""
    go, _ = _load_plotly_modules()
    vol = ef.volatility
    ret = ef.returns
    sharpe = ef.sharpe
    idx = int(ef.best_sharpe_idx)
    figure = go.Figure()
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
    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        template="plotly_dark",
        hovermode="closest",
        xaxis_title="年化波动率",
        yaxis_title="年化收益",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return figure


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
            t = (val - zmin) / span
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
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
    # Plotly Heatmap 的 textfont.color 不支持按单元格二维数组，改用 layout.annotations 逐格写字与配色。
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
    figure.update_layout(
        title="大类资产收益率相关性矩阵 (Correlation Matrix)",
        height=560,
        margin={"l": 28, "r": 28, "t": 56, "b": 110},
        template="plotly_dark",
        annotations=cell_annotations,
        xaxis={
            "side": "bottom",
            "tickangle": -32,
            "tickfont": {"size": 13},
        },
        yaxis={
            "autorange": "reversed",
            "tickfont": {"size": 13},
        },
    )
    return figure


def build_tab3_portfolio_backtest_figure(bt_df: pd.DataFrame) -> Any:
    """最优组合与等权组合的归一化净值曲线。"""
    go, _ = _load_plotly_modules()
    idx = bt_df.index
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=idx,
            y=bt_df["Optimal_Portfolio"],
            mode="lines",
            name="最优组合（最大夏普）",
            line={"width": 2.4, "color": "#ff1744"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=idx,
            y=bt_df["Equal_Weight_Portfolio"],
            mode="lines",
            name="等权重组合",
            line={"width": 2.0, "color": "#888888"},
        )
    )
    figure.update_layout(
        title={
            "text": "历史净值回测（样本首日归一为 1）",
            "y": 0.97,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=500,
        margin={"l": 56, "r": 28, "t": 64, "b": 96},
        template="plotly_dark",
        hovermode="x unified",
        xaxis_title="日期",
        yaxis_title="净值",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(0,0,0,0.35)",
        },
    )
    return figure
