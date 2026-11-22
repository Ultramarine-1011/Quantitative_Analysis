"""Exact and sampled efficient frontiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis_metrics import RISK_FREE_RATE
from models.covariance import CovMethod, estimate_covariance_and_mean
from models.optimization import (
    PortfolioSolution,
    _greedy_extreme_return_weights,
    _project_capped_simplex,
    _solve_min_variance_core,
    _solve_tangency_core,
    _validate_weight_bounds,
)
from models.returns import _positive_integer


@dataclass
class EfficientFrontierResult:
    """随机权重有效前沿模拟结果。"""

    volatility: np.ndarray
    returns: np.ndarray
    sharpe: np.ndarray
    weights: np.ndarray
    best_sharpe_idx: int


@dataclass
class ExactEfficientFrontierResult:
    """SLSQP 精确约束有效前沿、关键组合与可选随机散点。"""

    asset_names: list[str]
    frontier_volatility: np.ndarray
    frontier_returns: np.ndarray
    frontier_sharpe: np.ndarray
    frontier_weights: np.ndarray
    frontier_converged: np.ndarray
    min_variance: PortfolioSolution
    max_sharpe: PortfolioSolution
    cov_method: CovMethod
    shrinkage: float | None
    condition_number: float
    risk_free_rate: float
    min_weight: float
    max_weight: float
    random_volatility: np.ndarray | None = None
    random_returns: np.ndarray | None = None
    random_sharpe: np.ndarray | None = None
    random_weights: np.ndarray | None = None


def _sample_constrained_dirichlet(
    rng: np.random.Generator,
    num_portfolios: int,
    n_assets: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    if lower == 0.0 and upper == 1.0:
        return rng.dirichlet(np.ones(n_assets), size=num_portfolios).astype(float, copy=False)
    accepted: list[np.ndarray] = []
    batch = max(num_portfolios * 4, 256)
    for _ in range(8):
        draws = rng.dirichlet(np.ones(n_assets), size=batch)
        mask = np.all((draws >= lower - 1e-12) & (draws <= upper + 1e-12), axis=1)
        if np.any(mask):
            accepted.append(draws[mask])
            if sum(len(chunk) for chunk in accepted) >= num_portfolios:
                break
    if accepted:
        merged = np.vstack(accepted)
        if len(merged) >= num_portfolios:
            return merged[:num_portfolios].astype(float, copy=False)
    raw = rng.dirichlet(np.ones(n_assets), size=num_portfolios)
    projected = np.vstack([_project_capped_simplex(row, lower, upper) for row in raw])
    return projected.astype(float, copy=False)


def solve_exact_efficient_frontier(
    returns_df: pd.DataFrame,
    *,
    n_points: int = 30,
    cov_method: CovMethod = "ledoit_wolf",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    num_random_portfolios: int = 0,
    random_state: int | np.random.Generator | None = None,
) -> ExactEfficientFrontierResult:
    """使用 SLSQP 求解真实均值-方差有效前沿包络线、GMV 与最大夏普组合。"""
    n_points = _positive_integer(n_points, "n_points", minimum=2)
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    n_assets = len(est.asset_names)
    w_min, w_max = _validate_weight_bounds(n_assets, min_weight, max_weight)
    rf = float(risk_free_rate)

    gmv = _solve_min_variance_core(est.mean_returns, est.cov_matrix, rf, w_min, w_max)
    w_max_ret = _greedy_extreme_return_weights(est.mean_returns, w_min, w_max, maximize=True)
    max_ret = float(est.mean_returns @ w_max_ret)

    frontier_vols: list[float] = []
    frontier_rets: list[float] = []
    frontier_sharpes: list[float] = []
    frontier_weights: list[np.ndarray] = []
    frontier_conv: list[bool] = []

    if max_ret - gmv.expected_return <= 1e-10:
        frontier_vols.append(gmv.volatility)
        frontier_rets.append(gmv.expected_return)
        frontier_sharpes.append(gmv.sharpe)
        frontier_weights.append(gmv.weights)
        frontier_conv.append(gmv.converged)
    else:
        targets = np.linspace(gmv.expected_return, max_ret, n_points)
        prev_w = gmv.weights
        for idx, tgt in enumerate(targets):
            if idx == 0:
                sol = gmv
            else:
                try:
                    sol = _solve_min_variance_core(
                        est.mean_returns,
                        est.cov_matrix,
                        rf,
                        w_min,
                        w_max,
                        target_return=float(tgt),
                        x0=prev_w,
                    )
                except RuntimeError:
                    frontier_conv.append(False)
                    continue
            prev_w = sol.weights
            frontier_vols.append(sol.volatility)
            frontier_rets.append(sol.expected_return)
            frontier_sharpes.append(sol.sharpe)
            frontier_weights.append(sol.weights)
            frontier_conv.append(sol.converged)

    if not frontier_weights:
        raise RuntimeError(
            "Exact efficient frontier failed to converge at all target return points."
        )

    rand_vol: np.ndarray | None = None
    rand_ret: np.ndarray | None = None
    rand_sharpe: np.ndarray | None = None
    rand_weights: np.ndarray | None = None
    extra_seeds: list[np.ndarray] = list(frontier_weights)
    if num_random_portfolios > 0:
        num_rand = _positive_integer(num_random_portfolios, "num_random_portfolios")
        rng = np.random.default_rng(random_state)
        rw = _sample_constrained_dirichlet(rng, num_rand, n_assets, w_min, w_max)
        r_ret = np.einsum("ij,j->i", rw, est.mean_returns)
        r_var = np.maximum(np.einsum("ij,jk,ik->i", rw, est.cov_matrix, rw), 0.0)
        r_vol = np.sqrt(r_var)
        r_sharpe = np.where(r_vol > 0.0, (r_ret - rf) / r_vol, np.nan)
        rand_vol = r_vol.astype(float, copy=False)
        rand_ret = r_ret.astype(float, copy=False)
        rand_sharpe = r_sharpe.astype(float, copy=False)
        rand_weights = rw
        if np.any(np.isfinite(r_sharpe)):
            extra_seeds.append(rw[int(np.nanargmax(r_sharpe))])

    max_sharpe = _solve_tangency_core(
        est.mean_returns,
        est.cov_matrix,
        rf,
        w_min,
        w_max,
        gmv_weights=gmv.weights,
        extra_starts=extra_seeds,
    )

    return ExactEfficientFrontierResult(
        asset_names=est.asset_names,
        frontier_volatility=np.asarray(frontier_vols, dtype=float),
        frontier_returns=np.asarray(frontier_rets, dtype=float),
        frontier_sharpe=np.asarray(frontier_sharpes, dtype=float),
        frontier_weights=np.vstack(frontier_weights),
        frontier_converged=np.asarray(frontier_conv, dtype=bool),
        min_variance=gmv,
        max_sharpe=max_sharpe,
        cov_method=cov_method,
        shrinkage=est.shrinkage,
        condition_number=est.condition_number,
        risk_free_rate=rf,
        min_weight=w_min,
        max_weight=w_max,
        random_volatility=rand_vol,
        random_returns=rand_ret,
        random_sharpe=rand_sharpe,
        random_weights=rand_weights,
    )


def generate_efficient_frontier(
    returns_df: pd.DataFrame,
    num_portfolios: int = 5000,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    random_state: int | np.random.Generator | None = None,
    min_history: int = 30,
    cov_method: CovMethod = "sample",
) -> EfficientFrontierResult:
    """在简单收益率宽表上随机生成权重，计算年化收益、波动与夏普比率。

    权重独立服从 Dirichlet(1,...,1)，保证非负且和为 1；主要用于配合精确包络线做散点云可视化。
    年化约定：组合日均乘以 ``trading_days``、日标准差乘以 ``sqrt(trading_days)``；
    夏普为 ``(年化收益 - risk_free_rate) / 年化波动``（统一默认 ``RISK_FREE_RATE = 0.016``）。
    """
    num_portfolios = _positive_integer(num_portfolios, "num_portfolios")
    trading_days = _positive_integer(trading_days, "trading_days")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if returns_df.shape[1] < 2:
        raise ValueError("Efficient frontier requires at least 2 assets (columns).")

    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    n_assets = len(est.asset_names)

    rng = np.random.default_rng(random_state)
    # Dirichlet 抽样天然生成非负且总和为 1 的随机权重。
    w = rng.dirichlet(np.ones(n_assets), size=num_portfolios).astype(float, copy=False)

    ann_ret = np.einsum("ij,j->i", w, est.mean_returns)
    variance = np.einsum("ij,jk,ik->i", w, est.cov_matrix, w)
    ann_vol = np.sqrt(np.maximum(variance, 0.0))
    sharpe = np.where(ann_vol > 0, (ann_ret - float(risk_free_rate)) / ann_vol, np.nan)

    best_sharpe_idx = int(np.nanargmax(sharpe))

    return EfficientFrontierResult(
        volatility=ann_vol.astype(float, copy=False),
        returns=ann_ret.astype(float, copy=False),
        sharpe=sharpe.astype(float, copy=False),
        weights=w,
        best_sharpe_idx=best_sharpe_idx,
    )
