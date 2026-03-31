"""量化分析系统的高级可视化层。

本模块基于 mplfinance 构建专业级多面板金融图表，用于把 K 线、趋势、
波动率与回撤信号统一渲染到同一张图中。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

if TYPE_CHECKING:
    from matplotlib.figure import Figure


REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


class BaseVisualizer(ABC):
    """可视化层抽象基类。"""

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验并标准化绘图输入。

        Parameters
        ----------
        df : pd.DataFrame
            待绘图的数据表。
            预期索引为 ``pd.DatetimeIndex``，并至少包含 ``Open``、``High``、
            ``Low``、``Close``、``Volume`` 五列；这些列应为数值列，或可安全
            转换为数值列。

        Returns
        -------
        pd.DataFrame
            清洗后的绘图数据副本。
            返回值索引为升序 ``pd.DatetimeIndex``，OHLCV 五列 dtype 统一为
            ``float64``，重复时间戳会保留最后一条记录。

        Raises
        ------
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。
        ValueError
            若输入为空、缺失 OHLCV 必需列、索引无法转为时间索引，或 OHLCV 列
            在前向填充后仍存在缺失值，则抛出该异常。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        validated = df.copy(deep=True)
        missing_columns = [
            column for column in REQUIRED_OHLCV_COLUMNS if column not in validated.columns
        ]
        if missing_columns:
            raise ValueError(
                "Input DataFrame is missing required OHLCV columns: %s"
                % ", ".join(missing_columns)
            )

        if not is_datetime64_any_dtype(validated.index):
            try:
                validated.index = pd.to_datetime(validated.index, errors="raise")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "Input index must be a pandas DatetimeIndex or convertible to datetime."
                ) from exc

        validated = validated.sort_index()
        validated = validated.loc[~validated.index.duplicated(keep="last")]
        validated.index = pd.DatetimeIndex(validated.index)
        validated.index.name = validated.index.name or "Date"

        numeric_ohlcv = (
            validated.loc[:, REQUIRED_OHLCV_COLUMNS]
            .replace([np.inf, -np.inf], np.nan)
            .stack()
            .pipe(pd.to_numeric, errors="coerce")
            .unstack()
            .reindex(columns=list(REQUIRED_OHLCV_COLUMNS))
            .astype("float64")
            .ffill()
        )

        if numeric_ohlcv.isna().any().any():
            invalid_columns = numeric_ohlcv.columns[numeric_ohlcv.isna().any(axis=0)]
            raise ValueError(
                "OHLCV columns still contain NaN values after forward-fill: %s"
                % ", ".join(invalid_columns.astype(str))
            )

        validated.loc[:, REQUIRED_OHLCV_COLUMNS] = numeric_ohlcv
        return validated

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

        numeric_block = (
            coerced.loc[:, list(columns)]
            .replace([np.inf, -np.inf], np.nan)
            .stack()
            .pipe(pd.to_numeric, errors="coerce")
            .unstack()
            .reindex(columns=list(columns))
            .astype("float64")
        )

        all_nan_columns = numeric_block.columns[numeric_block.isna().all(axis=0)]
        if len(all_nan_columns) > 0:
            raise ValueError(
                "Indicator columns contain no plottable numeric values: %s"
                % ", ".join(all_nan_columns.astype(str))
            )

        coerced.loc[:, list(columns)] = numeric_block
        return coerced

    def _infer_feature_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """根据 ``feature_summary`` 或默认约定推断特征列名。"""
        feature_summary = df.attrs.get("feature_summary", {})

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
            ylabel = None
            if column == inferred["volatility"]:
                ylabel = "Ann.Vol"
            elif column == inferred["drawdown"]:
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
        mav: Sequence[int] | None = None,
        volume: bool = True,
        main_panel_cols: Sequence[str] | None = None,
        lower_panel_cols: Sequence[str] | None = None,
        style: str = "yahoo",
        figratio: tuple[int, int] = (12, 8),
        figscale: float = 1.2,
        show: bool = True,
        savepath: str | None = None,
    ) -> Figure:
        """绘制类似 Bloomberg 终端风格的多面板 K 线图。

        Parameters
        ----------
        df : pd.DataFrame
            输入的标准 OHLCV 数据表。
            除 ``Open``、``High``、``Low``、``Close``、``Volume`` 外，还应包含
            特征工程层生成的 ``sma_*``、``volatility_*``、``bollinger_*``、
            ``drawdown`` 等列。
        title : str | None, default None
            图表标题前缀。若为 ``None``，则优先从 ``df.attrs["symbol"]``、
            ``df.attrs["ticker"]`` 或 ``df.attrs["code"]`` 中推断标的代码。
        mav : Sequence[int] | None, default None
            兼容保留参数。当前实现优先使用 ``main_panel_cols`` 中显式给出的
            均线列，而不是依赖 mplfinance 的 ``mav`` 自动生成功能。
        volume : bool, default True
            是否绘制成交量副图。当前推荐保持为 ``True``，以形成 1 个主图 +
            3 个副图的专业布局。
        main_panel_cols : Sequence[str] | None, default None
            主图叠加指标列。若为 ``None``，则自动使用 20 日均线、60 日均线与
            20 日布林带上下轨。
        lower_panel_cols : Sequence[str] | None, default None
            附加副图指标列。若为 ``None``，则自动使用年化波动率与历史回撤。
        style : str, default "yahoo"
            mplfinance 样式名称。
        figratio : tuple[int, int], default (12, 8)
            图像宽高比例。
        figscale : float, default 1.2
            图像缩放比例。
        show : bool, default True
            是否立即展示图像。
        savepath : str | None, default None
            图片保存路径。若给定，函数会自动创建父目录并保存图片。

        Returns
        -------
        Figure
            ``matplotlib.figure.Figure`` 图形对象。

        Raises
        ------
        ImportError
            若运行环境未安装 ``mplfinance``，则抛出该异常。
        TypeError
            若输入对象类型错误，则抛出该异常。
        ValueError
            若数据结构、指标列或绘图参数非法，则抛出该异常。

        Notes
        -----
        图形布局如下：

        - 主图：K 线 + 20/60 日均线 + 布林带上下轨，并对布林带区域淡蓝色填充
        - 副图 1：成交量
        - 副图 2：年化波动率曲线
        - 副图 3：历史回撤曲线，并在 0 轴以下用红色区域填充
        - 标题：动态显示标的代码与总体最大回撤
        """
        del mav

        validated = self.validate_input(df)
        mpf, plt = self._load_plot_dependencies()
        addplots, metadata = self._build_addplots(
            validated,
            main_panel_cols=main_panel_cols,
            lower_panel_cols=lower_panel_cols,
            volume=volume,
        )
        plot_df = metadata["validated_df"]

        num_panels = 4 if volume else 3
        panel_ratios = (6, 2, 2, 2) if volume else (6, 2, 2)

        plot_kwargs = {
            "type": "candle",
            "style": style,
            "volume": volume,
            "addplot": addplots,
            "panel_ratios": panel_ratios,
            "num_panels": num_panels,
            "figratio": figratio,
            "figscale": figscale,
            "ylabel": "Price",
            "ylabel_lower": "Volume",
            "tight_layout": True,
            "returnfig": True,
        }
        if volume:
            plot_kwargs["volume_panel"] = 1

        figure, axes = mpf.plot(
            plot_df,
            **plot_kwargs,
        )

        main_axis = self._get_primary_axis(axes, 0)
        x_positions = np.arange(len(plot_df.index), dtype=float)
        bollinger_upper = plot_df[metadata["bollinger_upper_col"]]
        bollinger_lower = plot_df[metadata["bollinger_lower_col"]]
        valid_band_mask = bollinger_upper.notna() & bollinger_lower.notna()
        if valid_band_mask.any():
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

        volatility_panel = 2 if volume else 1
        drawdown_panel = 3 if volume else 2
        volatility_axis = self._get_primary_axis(axes, volatility_panel)
        drawdown_axis = self._get_primary_axis(axes, drawdown_panel)

        ticker = importlib.import_module("matplotlib.ticker")
        percent_formatter = ticker.PercentFormatter(xmax=1.0, decimals=1)
        volatility_axis.yaxis.set_major_formatter(percent_formatter)
        drawdown_axis.yaxis.set_major_formatter(percent_formatter)

        drawdown_display = -plot_df[metadata["drawdown_col"]].abs()
        valid_drawdown_mask = drawdown_display.notna()
        if valid_drawdown_mask.any():
            drawdown_axis.fill_between(
                x_positions,
                drawdown_display.to_numpy(dtype=float),
                np.zeros(len(plot_df), dtype=float),
                where=valid_drawdown_mask.to_numpy(dtype=bool),
                color="#e74c3c",
                alpha=0.22,
                interpolate=True,
                zorder=0,
            )
        drawdown_axis.axhline(0.0, color="#7f8c8d", linewidth=0.9, alpha=0.85)

        main_axis.set_title(
            self._resolve_title(plot_df, title=title),
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        if savepath is not None:
            output_path = Path(savepath).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(str(output_path), dpi=160, bbox_inches="tight")

        if show:
            plt.show()

        return figure


__all__ = [
    "REQUIRED_OHLCV_COLUMNS",
    "BaseVisualizer",
    "MplfinanceVisualizer",
]
