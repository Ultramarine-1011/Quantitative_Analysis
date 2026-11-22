"""Return preparation and normalization."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


def _resolve_price_column(df: pd.DataFrame, price_col: str | None) -> str:
    """解析价格列名：显式列名优先，否则 ``Close`` / ``close`` 二选一。"""
    if price_col is not None:
        if price_col not in df.columns:
            raise ValueError(f"price_col {price_col!r} not in DataFrame columns.")
        return price_col
    if "Close" in df.columns:
        return "Close"
    if "close" in df.columns:
        return "close"
    raise ValueError("DataFrame must contain a 'Close' or 'close' price column.")


def normalize_close_to_base_one(
    df: pd.DataFrame,
    price_col: str | None = None,
) -> pd.Series:
    """将收盘价序列归一化为以首期为 1 的净值曲线。

    Parameters
    ----------
    df
        含收盘价列的表；索引通常为日期。
    price_col
        价格列名；为 ``None`` 时自动选择 ``Close`` 或 ``close``。

    Returns
    -------
    pd.Series
        ``close / close.iloc[0]``，与 ``df`` 索引对齐。

    Raises
    ------
    ValueError
        若首价为零、缺失或非有限值。
    """
    col = _resolve_price_column(df, price_col)
    close = pd.to_numeric(df[col], errors="coerce")
    if close.empty:
        raise ValueError("Close series is empty.")
    first = close.iloc[0]
    if pd.isna(first) or not np.isfinite(first) or first == 0:
        raise ValueError(
            "First closing price must be finite, non-NaN, and non-zero for base-one normalization."
        )
    # 归一化后首日净值为 1，方便把不同价格量级的资产画在同一张图上比较。
    return close.astype(float) / float(first)


def _positive_integer(value: int, field_name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(f"{field_name} must be an integer >= {minimum}.")
    return int(value)


def clean_joint_returns(returns_df: pd.DataFrame, min_history: int = 30) -> pd.DataFrame:
    """对多资产日收益宽表做与有效前沿一致的联合清洗。

    ``to_numeric`` → 按行 ``dropna(how="all")`` → 再 ``dropna()``（任列为 NaN 的行删除），
    并在关键步骤校验 ``min_history`` 与至少 2 列资产。
    """
    # 多资产组合要求每个交易日都有所有资产收益，缺一个就无法正确相乘求组合收益。
    min_history = _positive_integer(min_history, "min_history", minimum=2)
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if not returns_df.columns.is_unique or not returns_df.index.is_unique:
        raise ValueError("Return matrix must have unique columns and observations.")
    r = returns_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    r = r.dropna(how="all")
    if r.shape[0] < min_history:
        raise ValueError(
            f"Need at least {min_history} rows of joint return history after cleaning; got {r.shape[0]}."
        )
    if r.shape[1] < 2:
        raise ValueError("Fewer than 2 asset columns remain after dropping all-NaN rows.")
    r = r.dropna()
    if r.empty:
        raise ValueError("Returns are empty after dropping rows with any NaN.")
    if r.shape[0] < min_history:
        raise ValueError(
            f"Need at least {min_history} complete return rows; got {r.shape[0]} after dropna."
        )
    return r


def returns_correlation_matrix(returns_df: pd.DataFrame, min_history: int = 30) -> pd.DataFrame:
    """皮尔逊相关矩阵；样本与列顺序与 ``clean_joint_returns`` 一致。"""
    r = clean_joint_returns(returns_df, min_history=min_history)
    return r.corr(method="pearson")
