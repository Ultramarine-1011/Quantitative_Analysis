"""基础价格因子研究工具。"""

from __future__ import annotations

import pandas as pd


def _clean_wide(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("%s must be a non-empty DataFrame." % name)
    out = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    if out.shape[1] < 2:
        raise ValueError("%s requires at least 2 assets." % name)
    return out


def make_price_factors(
    close_wide: pd.DataFrame,
    *,
    momentum_window: int = 20,
    volatility_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """从价格宽表生成动量、波动率和短期反转因子。"""
    prices = _clean_wide(close_wide, "close_wide")
    if momentum_window < 1 or volatility_window < 2:
        raise ValueError("momentum_window must be >=1 and volatility_window must be >=2.")
    returns = prices.pct_change()
    return {
        "momentum_%d" % momentum_window: prices.pct_change(momentum_window),
        "volatility_%d" % volatility_window: returns.rolling(volatility_window).std(),
        "reversal_1": -returns,
    }


def compute_information_coefficient(
    factor_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
) -> pd.Series:
    """逐日横截面 Spearman IC；调用方需传入已向未来错位的收益。"""
    factor = _clean_wide(factor_df, "factor_df")
    forward = _clean_wide(forward_returns_df, "forward_returns_df")
    common_idx = factor.index.intersection(forward.index)
    common_cols = factor.columns.intersection(forward.columns)
    values: dict[pd.Timestamp, float] = {}
    for dt in common_idx:
        f = factor.loc[dt, common_cols]
        r = forward.loc[dt, common_cols]
        sample = pd.concat([f, r], axis=1).dropna()
        if sample.shape[0] >= 2:
            left = sample.iloc[:, 0].rank(method="average")
            right = sample.iloc[:, 1].rank(method="average")
            values[dt] = float(left.corr(right, method="pearson"))
    out = pd.Series(values, name="information_coefficient", dtype="float64")
    return out.dropna()


def compute_quantile_returns(
    factor_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    *,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """按因子横截面分组，计算各组未来收益均值。"""
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")
    factor = _clean_wide(factor_df, "factor_df")
    forward = _clean_wide(forward_returns_df, "forward_returns_df")
    if factor.shape[1] < n_quantiles:
        raise ValueError("factor_df needs at least %d assets for quantile grouping." % n_quantiles)
    common_idx = factor.index.intersection(forward.index)
    common_cols = factor.columns.intersection(forward.columns)
    rows: list[pd.Series] = []
    for dt in common_idx:
        sample = pd.DataFrame({"factor": factor.loc[dt, common_cols], "forward": forward.loc[dt, common_cols]}).dropna()
        if sample.shape[0] < n_quantiles:
            continue
        labels = pd.qcut(sample["factor"].rank(method="first"), q=n_quantiles, labels=False) + 1
        grouped = sample.groupby(labels)["forward"].mean()
        grouped.index = ["Q%d" % int(i) for i in grouped.index]
        grouped.name = dt
        rows.append(grouped)
    if not rows:
        raise ValueError("No dates have enough complete assets for quantile grouping.")
    return pd.DataFrame(rows).sort_index()


__all__ = [
    "compute_information_coefficient",
    "compute_quantile_returns",
    "make_price_factors",
]
