"""轻量真实组合回测：调仓频率、线性成本和换手率。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    strategy: str = "fixed_weight"
    rebalance_frequency: str = "D"
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_nav: float = 1.0


def _normalize_weights(weights: pd.Series, columns: Sequence[str]) -> pd.Series:
    w = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0).astype("float64")
    w = w.reindex(list(columns)).fillna(0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("target_weights must have positive total weight.")
    return w / total


def make_equal_weights(columns: Sequence[str]) -> pd.Series:
    labels = list(columns)
    if not labels:
        raise ValueError("columns must not be empty.")
    return pd.Series(1.0 / len(labels), index=labels, dtype="float64")


def compute_turnover(current_weights: pd.Series, target_weights: pd.Series) -> float:
    labels = sorted(set(current_weights.index.astype(str)) | set(target_weights.index.astype(str)))
    cur = pd.to_numeric(current_weights.reindex(labels), errors="coerce").fillna(0.0)
    tgt = pd.to_numeric(target_weights.reindex(labels), errors="coerce").fillna(0.0)
    return float((tgt - cur).abs().sum())


def _rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    freq = str(frequency).strip().upper()
    if freq in {"D", "DAILY", "DAY"}:
        return pd.Series(True, index=index)
    periods = pd.Series(index.to_period({"W": "W", "M": "M", "Q": "Q"}.get(freq, freq)), index=index)
    return periods.ne(periods.shift(1)).fillna(True)


def run_weighted_backtest(
    returns_df: pd.DataFrame,
    target_weights: pd.Series | dict[str, float],
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """用日末收益更新权重，调仓日先扣交易成本再应用当日收益。"""
    cfg = config or BacktestConfig()
    if cfg.initial_nav <= 0:
        raise ValueError("initial_nav must be positive.")
    if cfg.commission_bps < 0 or cfg.slippage_bps < 0:
        raise ValueError("commission_bps and slippage_bps must be non-negative.")

    r = returns_df.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna()
    if r.empty:
        raise ValueError("returns_df has no complete rows.")
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index, errors="raise")
    r = r.sort_index()

    strategy = str(cfg.strategy).strip().lower()
    if strategy == "equal_weight":
        target = make_equal_weights(r.columns)
    elif strategy in {"fixed_weight", "target_weight"}:
        target = _normalize_weights(pd.Series(target_weights), r.columns)
    else:
        raise ValueError("strategy must be equal_weight, fixed_weight, or target_weight.")

    current = target.copy()
    mask = _rebalance_mask(pd.DatetimeIndex(r.index), cfg.rebalance_frequency)
    cost_rate = (float(cfg.commission_bps) + float(cfg.slippage_bps)) / 10000.0
    nav = float(cfg.initial_nav)
    rows: list[dict[str, float | bool]] = []

    for i, (dt, row) in enumerate(r.iterrows()):
        is_rebalance = bool(mask.loc[dt])
        turnover = 0.0
        cost = 0.0
        if is_rebalance:
            turnover = 0.0 if i == 0 else compute_turnover(current, target)
            cost = nav * turnover * cost_rate
            nav -= cost
            current = target.copy()
        daily_return = float(row @ current)
        nav *= 1.0 + daily_return
        grown = current * (1.0 + row)
        denom = float(grown.sum())
        current = grown / denom if denom > 0 else target.copy()
        rows.append(
            {
                "nav": nav,
                "daily_return": daily_return,
                "turnover": turnover,
                "cost": cost,
                "cumulative_cost": 0.0,
                "is_rebalance": is_rebalance,
            }
        )

    out = pd.DataFrame(rows, index=r.index)
    out["cumulative_cost"] = out["cost"].cumsum()
    return out


__all__ = [
    "BacktestConfig",
    "compute_turnover",
    "make_equal_weights",
    "run_weighted_backtest",
]
