"""量化分析系统的特征提取层。

本模块面向标准 OHLCV 时间序列，使用 Pandas / NumPy 的向量化运算实现
常见且重要的量化特征，包括：

- 趋势特征：20 日、60 日简单移动平均线
- 波动率特征：基于对数收益率的 20 日历史年化波动率
- 极值特征：20 日布林带上下轨
- 性能特征：每日回撤与整段区间最大回撤
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

if TYPE_CHECKING:
    from typing import Literal


REQUIRED_OHLCV_COLUMNS: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


class BaseFeatureEngineer(ABC):
    """特征提取层抽象基类。

    Notes
    -----
    本层默认输入为标准 OHLCV 数据表，即：

    - 类型为 ``pd.DataFrame``
    - 索引为 ``pd.DatetimeIndex``
    - 至少包含 ``Open``、``High``、``Low``、``Close``、``Volume`` 五列
    - 上述列应可安全转换为 ``float64``
    """

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验并标准化特征工程层输入。

        Parameters
        ----------
        df : pd.DataFrame
            待校验的 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，并至少包含 ``Open``、``High``、
            ``Low``、``Close``、``Volume`` 五列；这些列应为数值列，或可安全
            转换为数值列。

        Returns
        -------
        pd.DataFrame
            清洗后的 OHLCV 数据表副本。
            返回值索引为按时间升序排列的 ``pd.DatetimeIndex``，关键 OHLCV 列
            统一为 ``float64``，并对关键列执行 ``ffill`` 以处理中间缺失值。

        Raises
        ------
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。
        ValueError
            若输入为空、缺失必需列、索引无法转换为 ``DatetimeIndex``，
            或关键 OHLCV 列在前向填充后仍存在缺失值，则抛出该异常。

        Notes
        -----
        该方法仅处理输入健壮性问题，不负责生成任何特征列。
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input OHLCV DataFrame is empty.")

        validated = df.copy(deep=True)
        required_columns = list(REQUIRED_OHLCV_COLUMNS)
        missing_columns = pd.Index(required_columns).difference(validated.columns)
        if not missing_columns.empty:
            raise ValueError(
                "Input OHLCV DataFrame is missing required columns: %s"
                % ", ".join(missing_columns.astype(str))
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
            validated.loc[:, required_columns]
            .replace([np.inf, -np.inf], np.nan)
            .stack()
            .pipe(pd.to_numeric, errors="coerce")
            .unstack()
            .reindex(columns=required_columns)
            .astype("float64")
            .ffill()
        )

        remaining_nan_columns = numeric_ohlcv.columns[numeric_ohlcv.isna().any(axis=0)]
        if len(remaining_nan_columns) > 0:
            raise ValueError(
                "Required OHLCV columns still contain NaN values after forward-fill: %s"
                % ", ".join(remaining_nan_columns.astype(str))
            )

        validated.loc[:, required_columns] = numeric_ohlcv
        return validated

    @abstractmethod
    def transform(self, df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """批量生成特征列。"""


class OHLCVFeatureEngineer(BaseFeatureEngineer):
    """面向标准 OHLCV 行情的特征工程器。

    Notes
    -----
    本类提供两类能力：

    - 单指标方法：便于独立分析收益率、波动率、动量、回撤等量化特征
    - ``transform`` 批量接口：一次性生成本阶段要求的核心特征矩阵
    """

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> int:
        """校验正整数参数。"""
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("%s must be a positive integer." % field_name)
        return int(value)

    @staticmethod
    def _validate_positive_float(value: float, field_name: str) -> float:
        """校验正浮点参数。"""
        numeric_value = float(value)
        if numeric_value <= 0:
            raise ValueError("%s must be positive." % field_name)
        return numeric_value

    def _get_price_series(self, df: pd.DataFrame, price_col: str) -> pd.Series:
        """从已校验数据表中提取价格序列。"""
        if price_col not in df.columns:
            raise ValueError("price_col `%s` does not exist in the input DataFrame." % price_col)

        price = (
            pd.to_numeric(df[price_col].replace([np.inf, -np.inf], np.nan), errors="coerce")
            .astype("float64")
            .ffill()
        )
        if price.isna().any():
            raise ValueError(
                "price_col `%s` still contains NaN values after forward-fill." % price_col
            )

        return price

    def compute_returns(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        periods: int = 1,
        log: bool = False,
    ) -> pd.Series:
        """计算收益率序列。

        Parameters
        ----------
        df : pd.DataFrame
            输入 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，至少包含 ``price_col`` 指定的价格列；
            标准 OHLCV 五列应可转换为 ``float64``。
        price_col : str, default "Close"
            用于计算收益率的价格列名，通常为 ``"Close"``。
        periods : int, default 1
            收益率回看期数，例如 ``1`` 表示 1 期收益率。
        log : bool, default False
            是否计算对数收益率。
            ``False`` 表示普通收益率，``True`` 表示对数收益率。

        Returns
        -------
        pd.Series
            与输入索引逐行对齐的收益率序列，dtype 为 ``float64``。
            当 ``log=True`` 时，返回 ``log(P_t / P_{t-k})``。

        Raises
        ------
        ValueError
            若 ``periods`` 非法、``price_col`` 缺失或价格列无法转换为有效数值，
            则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        该方法完全基于向量化运算实现，不使用 Python 层 ``for`` 循环。
        """
        validated = self.validate_input(df)
        periods = self._validate_positive_int(periods, "periods")
        price = self._get_price_series(validated, price_col)

        if log:
            returns = np.log(price.div(price.shift(periods)))
            returns.name = "log_return_%d" % periods
        else:
            returns = price.pct_change(periods=periods)
            returns.name = "return_%d" % periods

        return returns.astype("float64")

    def compute_sma(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        window: int = 20,
    ) -> pd.Series:
        """计算简单移动平均线。

        Parameters
        ----------
        df : pd.DataFrame
            输入 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，至少包含 ``price_col`` 指定的价格列；
            标准 OHLCV 五列应可转换为 ``float64``。
        price_col : str, default "Close"
            用于计算移动平均线的价格列名，通常为 ``"Close"``。
        window : int, default 20
            移动平均窗口长度，例如 ``20`` 或 ``60``。

        Returns
        -------
        pd.Series
            与输入索引逐行对齐的简单移动平均线序列，dtype 为 ``float64``。

        Raises
        ------
        ValueError
            若 ``window`` 非法或 ``price_col`` 缺失，则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        计算采用 ``rolling(window).mean()``，属于标准向量化时间序列操作。
        """
        validated = self.validate_input(df)
        window = self._validate_positive_int(window, "window")
        price = self._get_price_series(validated, price_col)

        sma = price.rolling(window=window, min_periods=window).mean()
        sma.name = "sma_%d" % window
        return sma.astype("float64")

    def compute_volatility(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """计算历史年化波动率序列。

        Parameters
        ----------
        df : pd.DataFrame
            输入 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，至少包含 ``price_col`` 指定的价格列；
            标准 OHLCV 五列应可转换为 ``float64``。
        price_col : str, default "Close"
            用于计算波动率的价格列名，通常为 ``"Close"``。
        window : int, default 20
            滚动窗口长度，常用 20 个交易日。
        annualize : bool, default True
            是否年化波动率。
        trading_days : int, default 252
            年化因子中的年交易日数量。

        Returns
        -------
        pd.Series
            与输入索引逐行对齐的波动率序列，dtype 为 ``float64``。
            当 ``annualize=True`` 时，返回基于对数收益率滚动标准差乘以
            ``sqrt(trading_days)`` 的历史年化波动率。

        Raises
        ------
        ValueError
            若 ``window``、``trading_days`` 非法，或价格列缺失，则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        波动率以对数收益率为基础计算，这比简单收益率更符合时间可加性假设。
        """
        validated = self.validate_input(df)
        window = self._validate_positive_int(window, "window")
        trading_days = self._validate_positive_int(trading_days, "trading_days")

        log_returns = self.compute_returns(validated, price_col=price_col, periods=1, log=True)
        volatility = log_returns.rolling(window=window, min_periods=window).std()
        if annualize:
            volatility = volatility * np.sqrt(float(trading_days))

        volatility.name = "volatility_%d" % window
        return volatility.astype("float64")

    def compute_momentum(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        window: int = 20,
        method: Literal["pct", "diff"] = "pct",
    ) -> pd.Series:
        """计算动量序列。

        Parameters
        ----------
        df : pd.DataFrame
            输入 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，至少包含 ``price_col`` 指定的价格列；
            标准 OHLCV 五列应可转换为 ``float64``。
        price_col : str, default "Close"
            用于计算动量的价格列名，通常为 ``"Close"``。
        window : int, default 20
            动量回看窗口长度。
        method : {"pct", "diff"}, default "pct"
            ``"pct"`` 表示百分比动量，``"diff"`` 表示绝对价差动量。

        Returns
        -------
        pd.Series
            与输入索引逐行对齐的动量序列，dtype 为 ``float64``。

        Raises
        ------
        ValueError
            若 ``window`` 非法、``method`` 不受支持或价格列缺失，则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        虽然本阶段的 ``transform`` 不默认输出动量列，但该方法保留为后续策略研究
        的独立接口。
        """
        validated = self.validate_input(df)
        window = self._validate_positive_int(window, "window")
        price = self._get_price_series(validated, price_col)

        if method == "pct":
            momentum = price.pct_change(periods=window)
        elif method == "diff":
            momentum = price.diff(periods=window)
        else:
            raise ValueError("method must be either 'pct' or 'diff'.")

        momentum.name = "momentum_%d" % window
        return momentum.astype("float64")

    def compute_bollinger_bands(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """计算布林带上下轨。

        Parameters
        ----------
        df : pd.DataFrame
            输入 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，至少包含 ``price_col`` 指定的价格列；
            标准 OHLCV 五列应可转换为 ``float64``。
        price_col : str, default "Close"
            用于计算布林带的价格列名，通常为 ``"Close"``。
        window : int, default 20
            布林带滚动窗口长度，常用 20。
        num_std : float, default 2.0
            标准差倍数，常用 2.0。

        Returns
        -------
        pd.DataFrame
            包含布林带上下轨的 ``pd.DataFrame``。
            返回值索引与输入一致，包含两列：

            - ``bollinger_upper_{window}``
            - ``bollinger_lower_{window}``

            两列 dtype 均为 ``float64``。

        Raises
        ------
        ValueError
            若 ``window``、``num_std`` 非法或价格列缺失，则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        中轨本质上等于同窗口的简单移动平均线，因此本方法只返回上下轨；
        中轨可直接复用 ``compute_sma(window=window)`` 的结果。
        """
        validated = self.validate_input(df)
        window = self._validate_positive_int(window, "window")
        num_std = self._validate_positive_float(num_std, "num_std")
        price = self._get_price_series(validated, price_col)

        rolling_mean = price.rolling(window=window, min_periods=window).mean()
        rolling_std = price.rolling(window=window, min_periods=window).std()

        bands = pd.DataFrame(index=price.index)
        bands["bollinger_upper_%d" % window] = (
            rolling_mean + num_std * rolling_std
        ).astype("float64")
        bands["bollinger_lower_%d" % window] = (
            rolling_mean - num_std * rolling_std
        ).astype("float64")
        return bands

    def compute_drawdown(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
    ) -> pd.Series:
        """计算每日回撤幅度序列。

        Parameters
        ----------
        df : pd.DataFrame
            输入 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，至少包含 ``price_col`` 指定的价格列；
            标准 OHLCV 五列应可转换为 ``float64``。
        price_col : str, default "Close"
            用于计算回撤的价格列名，通常为 ``"Close"``。

        Returns
        -------
        pd.Series
            与输入索引逐行对齐的每日回撤幅度序列，dtype 为 ``float64``。
            该序列按 ``1 - P_t / cummax(P_t)`` 计算，因此理论上位于 ``[0, 1]``
            区间内，数值越大表示距离历史高点越远。

        Raises
        ------
        ValueError
            若价格列缺失或无法形成有效数值序列，则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        返回的是逐日风险路径，而不是单个摘要统计量。
        """
        validated = self.validate_input(df)
        price = self._get_price_series(validated, price_col)

        running_peak = price.cummax()
        drawdown = 1.0 - price.div(running_peak)
        drawdown.name = "drawdown"
        return drawdown.astype("float64")

    def compute_max_drawdown(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
    ) -> float:
        """计算整段时间内的最大回撤。

        Parameters
        ----------
        df : pd.DataFrame
            输入 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，至少包含 ``price_col`` 指定的价格列；
            标准 OHLCV 五列应可转换为 ``float64``。
        price_col : str, default "Close"
            用于计算最大回撤的价格列名，通常为 ``"Close"``。

        Returns
        -------
        float
            最大回撤标量，dtype 语义等价于 ``float64``。
            返回值为非负数，等于每日回撤序列的最大值；例如 ``0.23`` 表示
            样本区间内曾出现过 23% 的最大回撤。

        Raises
        ------
        ValueError
            若价格列缺失或无法形成有效数值序列，则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        该方法返回单个标量风险摘要，不向输出 ``DataFrame`` 追加同名列。
        """
        drawdown = self.compute_drawdown(df, price_col=price_col)
        return float(drawdown.max())

    def transform(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        sma_short_window: int = 20,
        sma_long_window: int = 60,
        volatility_window: int = 20,
        bollinger_window: int = 20,
        bollinger_num_std: float = 2.0,
        annualize: bool = True,
        trading_days: int = 252,
        include_drawdown: bool = True,
    ) -> pd.DataFrame:
        """批量生成本阶段核心量化特征。

        Parameters
        ----------
        df : pd.DataFrame
            输入标准 OHLCV 数据表。
            预期索引为 ``pd.DatetimeIndex``，并至少包含 ``Open``、``High``、
            ``Low``、``Close``、``Volume`` 五列；这些列会被校验并转换为
            ``float64``。
        price_col : str, default "Close"
            特征计算所使用的价格列名，通常为 ``"Close"``。
        sma_short_window : int, default 20
            短周期简单移动平均窗口，默认生成 ``sma_20``。
        sma_long_window : int, default 60
            长周期简单移动平均窗口，默认生成 ``sma_60``。
        volatility_window : int, default 20
            历史波动率滚动窗口，默认生成 ``volatility_20``。
        bollinger_window : int, default 20
            布林带滚动窗口，默认生成 ``bollinger_upper_20`` 与
            ``bollinger_lower_20``。
        bollinger_num_std : float, default 2.0
            布林带标准差倍数。
        annualize : bool, default True
            是否将历史波动率年化。
        trading_days : int, default 252
            年化时使用的年交易日数量。
        include_drawdown : bool, default True
            是否在输出表中附加每日回撤列 ``drawdown``。

        Returns
        -------
        pd.DataFrame
            附加核心特征后的新 ``DataFrame``。
            返回值保留原始 OHLCV 列与索引，并新增以下列：

            - ``sma_{sma_short_window}``
            - ``sma_{sma_long_window}``
            - ``volatility_{volatility_window}``
            - ``bollinger_upper_{bollinger_window}``
            - ``bollinger_lower_{bollinger_window}``
            - ``drawdown``（当 ``include_drawdown=True``）

            返回值的 ``attrs["max_drawdown"]`` 中额外保存整个样本区间的最大回撤
            标量，便于下游回测报告直接读取。

        Raises
        ------
        ValueError
            若窗口参数非法、价格列缺失，或输入数据无法通过校验，则抛出该异常。
        TypeError
            若 ``df`` 不是 ``pd.DataFrame``，则抛出该异常。

        Notes
        -----
        本方法内部所有指标计算均采用 Pandas / NumPy 向量化实现，不使用
        Python 层 ``for`` 循环。
        """
        sma_short_window = self._validate_positive_int(
            sma_short_window, "sma_short_window"
        )
        sma_long_window = self._validate_positive_int(sma_long_window, "sma_long_window")
        volatility_window = self._validate_positive_int(
            volatility_window, "volatility_window"
        )
        bollinger_window = self._validate_positive_int(
            bollinger_window, "bollinger_window"
        )
        bollinger_num_std = self._validate_positive_float(
            bollinger_num_std, "bollinger_num_std"
        )
        trading_days = self._validate_positive_int(trading_days, "trading_days")

        if sma_short_window >= sma_long_window:
            raise ValueError("sma_short_window must be smaller than sma_long_window.")

        transformed = self.validate_input(df)
        short_sma = self.compute_sma(
            transformed, price_col=price_col, window=sma_short_window
        )
        long_sma = self.compute_sma(
            transformed, price_col=price_col, window=sma_long_window
        )
        volatility = self.compute_volatility(
            transformed,
            price_col=price_col,
            window=volatility_window,
            annualize=annualize,
            trading_days=trading_days,
        )
        bollinger_bands = self.compute_bollinger_bands(
            transformed,
            price_col=price_col,
            window=bollinger_window,
            num_std=bollinger_num_std,
        )

        transformed[short_sma.name] = short_sma
        transformed[long_sma.name] = long_sma
        transformed[volatility.name] = volatility
        transformed = transformed.join(bollinger_bands)

        if include_drawdown:
            transformed["drawdown"] = self.compute_drawdown(
                transformed, price_col=price_col
            )

        transformed.attrs["max_drawdown"] = self.compute_max_drawdown(
            transformed, price_col=price_col
        )
        transformed.attrs["feature_summary"] = {
            "price_col": price_col,
            "sma_short_window": sma_short_window,
            "sma_long_window": sma_long_window,
            "volatility_window": volatility_window,
            "bollinger_window": bollinger_window,
            "bollinger_num_std": bollinger_num_std,
            "annualize": annualize,
            "trading_days": trading_days,
            "include_drawdown": include_drawdown,
        }
        return transformed


__all__ = [
    "REQUIRED_OHLCV_COLUMNS",
    "BaseFeatureEngineer",
    "OHLCVFeatureEngineer",
]
