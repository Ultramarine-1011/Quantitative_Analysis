"""家庭组合分析：权重、净值曲线、风险指标与再平衡提醒。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from portfolio import compute_holdings_values, summarize_assets


def _normalize_weights(weights: pd.Series, columns: pd.Index | list[str] | None = None) -> pd.Series:
    w = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).astype("float64")
    if columns is not None:
        w = w.reindex(list(columns)).fillna(0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must contain a positive total weight.")
    return w / total


def build_position_weights(holdings_df: pd.DataFrame) -> pd.Series:
    """按可交易持仓市值生成权重，排除现金、空代码和负债。"""
    valued = compute_holdings_values(holdings_df)
    assets = valued[valued["item_type"] == "asset"].copy()
    assets = assets[assets["symbol"].astype(str).str.strip() != ""]
    assets = assets[~assets["asset_type"].astype(str).str.lower().isin({"cash", "deposit"})]
    if assets.empty:
        raise ValueError("没有可用于组合收益分析的有代码资产。")
    grouped = assets.groupby("symbol")["market_value"].sum().astype("float64")
    return _normalize_weights(grouped)


def compute_portfolio_curve(returns_df: pd.DataFrame, weights: pd.Series | dict[str, float]) -> pd.Series:
    """根据简单收益率和权重生成组合净值曲线，首日归一为 1。"""
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        raise ValueError("returns_df must be a non-empty DataFrame.")
    r = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna()
    if r.empty:
        raise ValueError("returns_df has no complete return rows.")
    w = _normalize_weights(pd.Series(weights), r.columns)
    daily = r @ w
    raw = (1.0 + daily).cumprod()
    nav = raw / raw.iloc[0]
    nav.name = "portfolio_nav"
    return nav.astype("float64")


def compute_portfolio_metrics(
    nav: pd.Series,
    *,
    trading_days: int = 252,
    risk_free_rate: float = 0.016,
) -> dict[str, float]:
    """计算组合净值的累计收益、最大回撤、年化波动率和夏普。"""
    series = pd.to_numeric(nav, errors="coerce").dropna().astype("float64")
    if series.shape[0] < 2:
        raise ValueError("nav must contain at least 2 valid observations.")
    cumulative_return = float(series.iloc[-1] / series.iloc[0] - 1.0)
    drawdown = 1.0 - series / series.cummax()
    max_drawdown = float(drawdown.max())
    returns = series.pct_change().dropna()
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(float(trading_days)))
    daily_rf = np.power(1.0 + float(risk_free_rate), 1.0 / float(trading_days)) - 1.0
    excess = returns - daily_rf
    if annualized_volatility == 0.0 or not np.isfinite(annualized_volatility):
        sharpe_ratio = float("nan")
    else:
        sharpe_ratio = float(np.sqrt(float(trading_days)) * excess.mean() / excess.std(ddof=1))
    return {
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
    }


def current_asset_class_weights(holdings_df: pd.DataFrame) -> pd.Series:
    """从台账计算资产类别当前权重，负债不参与配置权重。"""
    summary = summarize_assets(holdings_df)
    by_class = summary["asset_class_summary"]
    if by_class.empty:
        return pd.Series(dtype="float64")
    return by_class["asset_ratio"].astype("float64")


def target_asset_class_weights(holdings_df: pd.DataFrame) -> pd.Series:
    """从台账目标权重字段聚合资产类别目标配置。"""
    valued = compute_holdings_values(holdings_df)
    assets = valued[valued["item_type"] == "asset"]
    if assets.empty:
        return pd.Series(dtype="float64")
    target = assets.groupby("asset_class")["target_weight"].sum().astype("float64")
    total = float(target.sum())
    if total > 0:
        target = target / total
    return target


def compute_rebalance_suggestions(
    current_by_class: pd.Series | dict[str, float],
    target_by_class: pd.Series | dict[str, float],
    *,
    tolerance: float = 0.05,
    total_value: float = 0.0,
) -> pd.DataFrame:
    """比较当前配置和目标配置，输出偏离与建议调整金额。"""
    current = pd.to_numeric(pd.Series(current_by_class), errors="coerce").fillna(0.0)
    target = pd.to_numeric(pd.Series(target_by_class), errors="coerce").fillna(0.0)
    labels = sorted(set(current.index.astype(str)) | set(target.index.astype(str)))
    current = current.reindex(labels).fillna(0.0).astype("float64")
    target = target.reindex(labels).fillna(0.0).astype("float64")
    drift = current - target
    out = pd.DataFrame(
        {
            "current_weight": current,
            "target_weight": target,
            "drift": drift,
            "needs_rebalance": drift.abs() > float(tolerance),
            "suggested_trade_value": (target - current) * float(total_value),
        }
    )
    out.index.name = "asset_class"
    return out


__all__ = [
    "build_position_weights",
    "compute_portfolio_curve",
    "compute_portfolio_metrics",
    "compute_rebalance_suggestions",
    "current_asset_class_weights",
    "target_asset_class_weights",
]

