"""量化分析系统的高级可视化层。

本模块基于 mplfinance 构建专业级多面板金融图表，用于把 K 线、趋势、
波动率与回撤信号统一渲染到同一张图中。
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ohlcv import REQUIRED_OHLCV_COLUMNS, prepare_ohlcv

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class BaseVisualizer(ABC):
    """可视化层抽象基类。"""

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV without discarding feature columns or metadata."""
        return prepare_ohlcv(df)

    @abstractmethod
    def plot(self, df: pd.DataFrame, **kwargs: object) -> Figure:
        """绘制图形并返回 ``matplotlib.figure.Figure`` 对象。"""


class MplfinanceVisualizer(BaseVisualizer):
    """基于 mplfinance 的专业 K 线信号仪表盘。"""

    def _load_plot_dependencies(self) -> tuple[Any, Any]:
        """延迟导入绘图库，避免模块导入阶段产生硬依赖。"""
        try:
            mpf = importlib.import_module("mplfinance")
        except ImportError as exc:
            raise ImportError(
                "mplfinance is required for plotting. Install it with `pip install mplfinance`."
            ) from exc

        pyplot = importlib.import_module("matplotlib.pyplot")
        return mpf, pyplot

    def _coerce_indicator_columns(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """将指标列标准化为 ``float64``，允许保留滚动窗口产生的前段 NaN。"""
        coerced = df.copy(deep=True)
        if not columns:
            return coerced

        missing_columns = [column for column in columns if column not in coerced.columns]
        if missing_columns:
            raise ValueError(
                "Input DataFrame is missing required indicator columns: %s"
                % ", ".join(missing_columns)
            )

        numeric_block = coerced.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        numeric_block = numeric_block.replace([np.inf, -np.inf], np.nan).astype("float64")

        coerced[list(columns)] = numeric_block
        return coerced

    def _infer_feature_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """根据 ``feature_summary`` 或默认约定推断特征列名。"""
        feature_summary = df.attrs.get("feature_summary", {})

        # 特征工程会把窗口参数写到 attrs；绘图层据此找到实际列名。
        sma_short_window = int(feature_summary.get("sma_short_window", 20))
        sma_long_window = int(feature_summary.get("sma_long_window", 60))
        volatility_window = int(feature_summary.get("volatility_window", 20))
        bollinger_window = int(feature_summary.get("bollinger_window", 20))

        return {
            "sma_short": "sma_%d" % sma_short_window,
            "sma_long": "sma_%d" % sma_long_window,
            "bollinger_upper": "bollinger_upper_%d" % bollinger_window,
            "bollinger_lower": "bollinger_lower_%d" % bollinger_window,
            "volatility": "volatility_%d" % volatility_window,
            "drawdown": "drawdown",
        }

    def _resolve_title(self, df: pd.DataFrame, title: str | None = None) -> str:
        """生成动态标题，包含标的代码和总体最大回撤。"""
        base_title = (
            title
            or str(df.attrs.get("symbol") or df.attrs.get("ticker") or df.attrs.get("code") or "")
        ).strip()
        if not base_title:
            base_title = "Instrument"

        max_drawdown = df.attrs.get("max_drawdown")
        if max_drawdown is None and "drawdown" in df.columns:
            drawdown_series = pd.to_numeric(df["drawdown"], errors="coerce")
            if not drawdown_series.dropna().empty:
                max_drawdown = float(drawdown_series.max())

        if max_drawdown is None or pd.isna(max_drawdown):
            return base_title

        max_drawdown = abs(float(max_drawdown))
        display_drawdown = 0.0 if max_drawdown == 0.0 else -100.0 * max_drawdown
        return "%s | Max Drawdown: %.1f%%" % (base_title, display_drawdown)

    def _build_addplots(
        self,
        df: pd.DataFrame,
        main_panel_cols: Sequence[str] | None = None,
        lower_panel_cols: Sequence[str] | None = None,
        volume: bool = True,
    ) -> tuple[list[object], dict[str, Any]]:
        """构建主图和副图附加图层。"""
        mpf, _ = self._load_plot_dependencies()
        inferred = self._infer_feature_columns(df)

        resolved_main_cols = list(
            main_panel_cols
            if main_panel_cols is not None
            else [
                inferred["sma_short"],
                inferred["sma_long"],
                inferred["bollinger_upper"],
                inferred["bollinger_lower"],
            ]
        )
        resolved_lower_cols = list(
            lower_panel_cols
            if lower_panel_cols is not None
            else [inferred["volatility"], inferred["drawdown"]]
        )

        lower_panels = [2, 3] if volume else [1, 2]
        if len(resolved_lower_cols) > len(lower_panels):
            raise ValueError(
                "At most %d lower-panel indicator columns are supported in the current layout."
                % len(lower_panels)
            )

        all_indicator_columns = resolved_main_cols + resolved_lower_cols
        validated = self._coerce_indicator_columns(df, all_indicator_columns)

        # 颜色和线宽固定下来，保证 CLI 图表在不同运行环境下视觉含义一致。
        color_map = {
            inferred["sma_short"]: "#f39c12",
            inferred["sma_long"]: "#1f4e79",
            inferred["bollinger_upper"]: "#7fb3d5",
            inferred["bollinger_lower"]: "#7fb3d5",
            inferred["volatility"]: "#3498db",
            inferred["drawdown"]: "#c0392b",
        }

        width_map = {
            inferred["sma_short"]: 1.25,
            inferred["sma_long"]: 1.15,
            inferred["bollinger_upper"]: 0.9,
            inferred["bollinger_lower"]: 0.9,
            inferred["volatility"]: 1.1,
            inferred["drawdown"]: 1.1,
        }

        addplots: list[object] = []
        for column in resolved_main_cols:
            if not validated[column].notna().any():
                continue
            # 主图指标与 K 线共用价格轴，适合叠加均线和布林带。
            addplots.append(
                mpf.make_addplot(
                    validated[column],
                    panel=0,
                    type="line",
                    color=color_map.get(column, "#34495e"),
                    width=width_map.get(column, 1.0),
                    alpha=0.95,
                    secondary_y=False,
                )
            )

        drawdown_plot_column = None
        for column, panel in zip(resolved_lower_cols, lower_panels):
            series = validated[column]
            if not series.notna().any():
                continue
            ylabel = None
            if column == inferred["volatility"]:
                ylabel = "Ann.Vol"
            elif column == inferred["drawdown"]:
                # 回撤按负值画在 0 轴下方，更符合“从高点下跌”的视觉直觉。
                series = -series.abs()
                drawdown_plot_column = column
                ylabel = "Drawdown"

            addplots.append(
                mpf.make_addplot(
                    series,
                    panel=panel,
                    type="line",
                    color=color_map.get(column, "#34495e"),
                    width=width_map.get(column, 1.0),
                    alpha=0.95,
                    secondary_y=False,
                    ylabel=ylabel,
                )
            )

        metadata = {
            "validated_df": validated,
            "main_panel_cols": resolved_main_cols,
            "lower_panel_cols": resolved_lower_cols,
            "lower_panels": lower_panels,
            "volume_enabled": volume,
            "bollinger_upper_col": inferred["bollinger_upper"],
            "bollinger_lower_col": inferred["bollinger_lower"],
            "volatility_col": inferred["volatility"],
            "drawdown_col": inferred["drawdown"],
            "drawdown_plot_column": drawdown_plot_column,
        }
        return addplots, metadata

    @staticmethod
    def _get_primary_axis(axlist: Sequence[Any], panel: int) -> Any:
        """从 mplfinance 返回的轴列表中取出某个 panel 的主轴。"""
        return axlist[panel * 2]

    def plot(
        self,
        df: pd.DataFrame,
        title: str | None = None,
        volume: bool = True,
        main_panel_cols: Sequence[str] | None = None,
        lower_panel_cols: Sequence[str] | None = None,
        style: str = "yahoo",
        figratio: tuple[int, int] = (12, 8),
        figscale: float = 1.2,
        show: bool = True,
        savepath: str | None = None,
    ) -> Figure:
        """绘制类似 Bloomberg 终端风格的多面板 K 线/折线与风险仪表盘。

        净值或收益率水平序列自动采用折线模式（``type="line"``）并隐藏无意义成交量。
        """
        validated = self.validate_input(df)
        asset_key = str(validated.attrs.get("asset_type", "") or "").strip().lower()
        series_kind = str(validated.attrs.get("series_kind", "price") or "price").strip().lower()
        use_line = asset_key == "mutual_fund" or series_kind == "yield_level"
        vol_max = float(pd.to_numeric(validated["Volume"], errors="coerce").fillna(0.0).abs().max())
        effective_volume = (
            bool(volume and not use_line and vol_max > 0.0) if use_line else bool(volume)
        )

        mpf, plt = self._load_plot_dependencies()
        # addplots 是 mplfinance 的附加图层机制，用来叠加均线、布林带和副图指标。
        addplots, metadata = self._build_addplots(
            validated,
            main_panel_cols=main_panel_cols,
            lower_panel_cols=lower_panel_cols,
            volume=effective_volume,
        )
        plot_df = metadata["validated_df"]

        num_panels = 4 if effective_volume else 3
        panel_ratios = (6, 2, 2, 2) if effective_volume else (6, 2, 2)
        ylabel = (
            "Yield (%)"
            if series_kind == "yield_level"
            else ("NAV" if asset_key == "mutual_fund" else "Price")
        )

        plot_kwargs = {
            "type": "line" if use_line else "candle",
            "style": style,
            "volume": effective_volume,
            "addplot": addplots,
            "panel_ratios": panel_ratios,
            "num_panels": num_panels,
            "figratio": figratio,
            "figscale": figscale,
            "ylabel": ylabel,
            "ylabel_lower": "Volume",
            "tight_layout": True,
            "returnfig": True,
        }
        if effective_volume:
            plot_kwargs["volume_panel"] = 1
        if not addplots:
            plot_kwargs.pop("addplot")

        # returnfig=True 让函数返回 Figure，便于测试、保存或由调用方继续加工。
        figure, axes = mpf.plot(
            plot_df,
            **plot_kwargs,
        )

        main_axis = self._get_primary_axis(axes, 0)
        x_positions = np.arange(len(plot_df.index), dtype=float)
        empty_indicator = pd.Series(np.nan, index=plot_df.index)
        bollinger_upper = plot_df.get(metadata["bollinger_upper_col"], empty_indicator)
        bollinger_lower = plot_df.get(metadata["bollinger_lower_col"], empty_indicator)
        valid_band_mask = bollinger_upper.notna() & bollinger_lower.notna()
        if valid_band_mask.any():
            # 淡蓝色填充布林带区域，帮助初学者看到“正常波动区间”。
            main_axis.fill_between(
                x_positions,
                bollinger_lower.to_numpy(dtype=float),
                bollinger_upper.to_numpy(dtype=float),
                where=valid_band_mask.to_numpy(dtype=bool),
                color="#b9d8f0",
                alpha=0.18,
                interpolate=True,
                zorder=0,
            )

        ticker = importlib.import_module("matplotlib.ticker")
        for column, panel in zip(metadata["lower_panel_cols"], metadata["lower_panels"]):
            if column not in (metadata["volatility_col"], metadata["drawdown_col"]):
                continue
            axis = self._get_primary_axis(axes, panel)
            axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
            if column == metadata["drawdown_plot_column"]:
                drawdown = -plot_df[column].abs()
                axis.fill_between(
                    x_positions,
                    drawdown.to_numpy(dtype=float),
                    np.zeros(len(plot_df)),
                    where=drawdown.notna().to_numpy(),
                    color="#e74c3c",
                    alpha=0.22,
                    interpolate=True,
                    zorder=0,
                )
                axis.axhline(0.0, color="#7f8c8d", linewidth=0.9, alpha=0.85)

        main_axis.set_title(
            self._resolve_title(plot_df, title=title),
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        if savepath is not None:
            output_path = Path(savepath).expanduser()
            # 保存前创建目录，方便命令行用户直接传入新的输出路径。
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(str(output_path), dpi=160, bbox_inches="tight")

        if show:
            # CLI 场景默认弹出窗口；测试或批处理可传 show=False 禁用。
            plt.show()

        return figure


__all__ = [
    "REQUIRED_OHLCV_COLUMNS",
    "BaseVisualizer",
    "MplfinanceVisualizer",
]
