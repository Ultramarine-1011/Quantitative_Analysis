"""Constrained portfolio solvers and risk contributions."""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd

from analysis_metrics import RISK_FREE_RATE
from models.covariance import (
    CovMethod,
    _estimate_covariance_from_array,
    estimate_covariance_and_mean,
)

OptimizerObjective = Literal[
    "max_sharpe",
    "min_variance",
    "risk_parity",
    "inverse_volatility",
    "equal_weight",
]


@dataclass
class PortfolioSolution:
    """单次约束组合优化求解结果。"""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    converged: bool
    solver_message: str


def _validate_weight_bounds(
    n_assets: int, min_weight: float, max_weight: float
) -> tuple[float, float]:
    for val, name in ((min_weight, "min_weight"), (max_weight, "max_weight")):
        if isinstance(val, bool) or not isinstance(val, Real) or not np.isfinite(float(val)):
            raise ValueError(f"{name} must be a finite real number.")
    w_min = float(min_weight)
    w_max = float(max_weight)
    if w_min < 0.0 or w_max > 1.0 or w_min > w_max:
        raise ValueError("Long-only weight bounds require 0.0 <= min_weight <= max_weight <= 1.0.")
    if w_min * float(n_assets) > 1.0 + 1e-9:
        raise ValueError(
            f"Infeasible weight bounds: min_weight ({w_min:.4g}) * {n_assets} assets > 1.0."
        )
    if w_max * float(n_assets) < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible weight bounds: max_weight ({w_max:.4g}) * {n_assets} assets < 1.0."
        )
    return w_min, w_max


def _project_capped_simplex(
    vector: np.ndarray,
    lower: float,
    upper: float,
    total: float = 1.0,
) -> np.ndarray:
    """欧氏投影到带上下界单纯形 {w : sum(w) == total, lower <= w_i <= upper}。"""
    v = np.asarray(vector, dtype=float)
    n = v.size
    if abs(upper - lower) <= 1e-12:
        return np.full(n, total / float(n), dtype=float)
    lo = float(np.min(v) - upper - 1.0)
    hi = float(np.max(v) - lower + 1.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = float(np.sum(np.clip(v - mid, lower, upper)))
        if s > total:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - 0.5 * (lo + hi), lower, upper)
    diff = total - float(np.sum(w))
    if abs(diff) > 0.0:
        room = (upper - w) if diff > 0.0 else (w - lower)
        room_sum = float(np.sum(room))
        if room_sum > 1e-15:
            w = np.clip(w + diff * (room / room_sum), lower, upper)
    return w.astype(float, copy=False)


def _greedy_extreme_return_weights(
    mean_ret: np.ndarray,
    lower: float,
    upper: float,
    *,
    maximize: bool,
) -> np.ndarray:
    n = mean_ret.size
    w = np.full(n, lower, dtype=float)
    rem = 1.0 - float(n) * lower
    order = np.argsort(-mean_ret if maximize else mean_ret, kind="mergesort")
    span = upper - lower
    for idx in order:
        if rem <= 1e-15:
            break
        add = min(span, rem)
        w[idx] += add
        rem -= add
    return _project_capped_simplex(w, lower, upper)


def _portfolio_stats(
    weights: np.ndarray,
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    ret = float(mean_ret @ weights)
    var = max(float(weights @ cov_mat @ weights), 0.0)
    vol = math.sqrt(var)
    sharpe = (ret - risk_free_rate) / vol if vol > 0.0 else float("nan")
    return ret, vol, sharpe


def _is_feasible_solution(
    weights: np.ndarray,
    lower: float,
    upper: float,
    mean_ret: np.ndarray | None = None,
    target_return: float | None = None,
    tol: float = 1e-5,
) -> bool:
    if not np.isfinite(weights).all():
        return False
    if abs(float(np.sum(weights)) - 1.0) > tol:
        return False
    if float(np.min(weights)) < lower - tol or float(np.max(weights)) > upper + tol:
        return False
    if target_return is not None and mean_ret is not None:
        if float(mean_ret @ weights) < float(target_return) - tol:
            return False
    return True


def _deduplicate_starts(
    candidates: list[np.ndarray], lower: float, upper: float
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for cand in candidates:
        proj = _project_capped_simplex(cand, lower, upper)
        if not any(np.allclose(proj, prev, atol=1e-6) for prev in unique):
            unique.append(proj)
    return unique


def _slsqp_minimize(scipy_optimize: object, *args: object, **kwargs: object) -> object:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Values in x were outside bounds during a minimize step.*",
            category=RuntimeWarning,
        )
        return scipy_optimize.minimize(*args, **kwargs)  # type: ignore[attr-defined]


def _solve_min_variance_core(
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
    lower: float,
    upper: float,
    *,
    target_return: float | None = None,
    x0: np.ndarray | None = None,
) -> PortfolioSolution:
    scipy_optimize = importlib.import_module("scipy.optimize")
    n = mean_ret.size
    if abs(upper - lower) <= 1e-12:
        w = np.full(n, 1.0 / float(n), dtype=float)
        if target_return is not None and float(mean_ret @ w) < float(target_return) - 1e-8:
            raise ValueError("Target return exceeds feasible portfolio bound under fixed weights.")
        ret, vol, sharpe = _portfolio_stats(w, mean_ret, cov_mat, risk_free_rate)
        return PortfolioSolution(w, ret, vol, sharpe, True, "Degenerate singleton box bounds.")

    scale = max(float(np.mean(np.diag(cov_mat))), 1e-8)
    cov_scaled = cov_mat / scale
    bounds = [(lower, upper)] * n
    constraints: list[dict[str, object]] = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones(n, dtype=float),
        }
    ]
    if target_return is not None:
        tgt = float(target_return)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, t=tgt: float(mean_ret @ w - t),
                "jac": lambda w: mean_ret,
            }
        )

    w_eq = np.full(n, 1.0 / float(n), dtype=float)
    inv_std = 1.0 / np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
    w_inv = inv_std / float(np.sum(inv_std))
    w_max_ret = _greedy_extreme_return_weights(mean_ret, lower, upper, maximize=True)
    raw_starts = [s for s in (x0, w_eq, w_inv, w_max_ret) if s is not None]
    starts = _deduplicate_starts(raw_starts, lower, upper)

    best_w: np.ndarray | None = None
    best_var = float("inf")
    last_msg = "No optimization run executed."

    for start in starts:
        res = _slsqp_minimize(
            scipy_optimize,
            lambda w: 0.5 * float(w @ cov_scaled @ w),
            start,
            jac=lambda w: cov_scaled @ w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500, "disp": False},
        )
        last_msg = str(res.message)
        if res.success and _is_feasible_solution(
            res.x, lower, upper, mean_ret=mean_ret, target_return=target_return
        ):
            cand = _project_capped_simplex(res.x, lower, upper)
            if target_return is not None and float(mean_ret @ cand) < float(target_return) - 1e-5:
                cand = np.asarray(res.x, dtype=float)
            var = float(cand @ cov_mat @ cand)
            if var < best_var:
                best_var = var
                best_w = cand
            if x0 is not None and start is starts[0]:
                # 凸二次规划从上一前沿解热启动收敛后即为全局极小。
                break

    if best_w is None and target_return is not None:
        max_ret_val = float(mean_ret @ w_max_ret)
        if abs(float(target_return) - max_ret_val) <= 1e-6:
            ret, vol, sharpe = _portfolio_stats(w_max_ret, mean_ret, cov_mat, risk_free_rate)
            return PortfolioSolution(
                weights=w_max_ret,
                expected_return=ret,
                volatility=vol,
                sharpe=sharpe,
                converged=True,
                solver_message="Boundary vertex reached at maximum feasible return.",
            )

    if best_w is None:
        raise RuntimeError(f"Minimum variance optimization failed to converge: {last_msg}")

    ret, vol, sharpe = _portfolio_stats(best_w, mean_ret, cov_mat, risk_free_rate)
    return PortfolioSolution(
        weights=best_w,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        converged=True,
        solver_message=last_msg,
    )


def _solve_tangency_core(
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
    lower: float,
    upper: float,
    *,
    gmv_weights: np.ndarray | None = None,
    extra_starts: Sequence[np.ndarray] = (),
) -> PortfolioSolution:
    scipy_optimize = importlib.import_module("scipy.optimize")
    n = mean_ret.size
    if abs(upper - lower) <= 1e-12:
        w = np.full(n, 1.0 / float(n), dtype=float)
        ret, vol, sharpe = _portfolio_stats(w, mean_ret, cov_mat, risk_free_rate)
        return PortfolioSolution(w, ret, vol, sharpe, True, "Degenerate singleton box bounds.")

    cov_scale = max(float(np.mean(np.diag(cov_mat))), 1e-12)
    cov_scaled = cov_mat / cov_scale
    bounds = [(lower, upper)] * n
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones(n, dtype=float),
        }
    ]

    w_eq = np.full(n, 1.0 / float(n), dtype=float)
    w_max_ret = _greedy_extreme_return_weights(mean_ret, lower, upper, maximize=True)
    inv_std = 1.0 / np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
    w_inv = inv_std / float(np.sum(inv_std))
    single_sharpes = (mean_ret - risk_free_rate) / np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
    best_single = np.zeros(n, dtype=float)
    best_single[int(np.argmax(single_sharpes))] = 1.0
    best_ret_idx = np.zeros(n, dtype=float)
    best_ret_idx[int(np.argmax(mean_ret))] = 1.0

    raw_starts: list[np.ndarray] = []
    for cand in (gmv_weights, w_eq, w_max_ret, w_inv, best_single, best_ret_idx, *extra_starts):
        if cand is not None:
            raw_starts.append(np.asarray(cand, dtype=float))
    starts = _deduplicate_starts(raw_starts, lower, upper)

    def neg_scaled_sharpe_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        w_arr = np.asarray(w, dtype=float)
        excess = float(mean_ret @ w_arr) - risk_free_rate
        cov_w = cov_scaled @ w_arr
        var = max(float(w_arr @ cov_w), 1e-24)
        std = math.sqrt(var)
        obj = -excess / std
        grad = (-mean_ret * var + excess * cov_w) / (var * std)
        return obj, grad

    best_sol: PortfolioSolution | None = None
    last_msg = "No optimization run executed."

    for start in starts:
        res = _slsqp_minimize(
            scipy_optimize,
            lambda w: neg_scaled_sharpe_and_grad(w)[0],
            start,
            jac=lambda w: neg_scaled_sharpe_and_grad(w)[1],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 600, "disp": False},
        )
        last_msg = str(res.message)
        if not res.success or not _is_feasible_solution(res.x, lower, upper):
            continue
        w_cand = _project_capped_simplex(res.x, lower, upper)
        ret, vol, sharpe = _portfolio_stats(w_cand, mean_ret, cov_mat, risk_free_rate)
        if not np.isfinite(sharpe):
            continue
        if (
            best_sol is None
            or sharpe > best_sol.sharpe + 1e-12
            or (abs(sharpe - best_sol.sharpe) <= 1e-12 and vol < best_sol.volatility)
        ):
            best_sol = PortfolioSolution(
                weights=w_cand,
                expected_return=ret,
                volatility=vol,
                sharpe=sharpe,
                converged=True,
                solver_message=last_msg,
            )

    if best_sol is None:
        raise RuntimeError(f"Tangency portfolio optimization failed to converge: {last_msg}")
    return best_sol


def solve_minimum_variance_portfolio(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "sample",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_return: float | None = None,
) -> PortfolioSolution:
    """通过 SLSQP 求解约束最小方差组合（可选给定目标年化收益下限）。"""
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if target_return is not None and not np.isfinite(float(target_return)):
        raise ValueError("target_return must be finite when provided.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    return _solve_min_variance_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
        target_return=target_return,
    )


def solve_tangency_portfolio(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "sample",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    """通过多起点 SLSQP 求解约束最大夏普（切线）组合。"""
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    gmv = _solve_min_variance_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
    )
    return _solve_tangency_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
        gmv_weights=gmv.weights,
    )


def compute_risk_contributions(
    weights: np.ndarray | Sequence[float],
    cov_matrix: np.ndarray,
    asset_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """计算 Euler 风险分解：边际风险贡献 (MCR)、绝对风险贡献 (RC)、百分比风险贡献 (PRC) 与分散度指标。"""
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(cov_matrix, dtype=float)
    n = w.size
    if n == 0:
        raise ValueError("weights must be non-empty.")
    if cov.shape != (n, n):
        raise ValueError(f"cov_matrix shape {cov.shape} must match weights length ({n}, {n}).")
    if not np.isfinite(w).all() or not np.isfinite(cov).all():
        raise ValueError("weights and cov_matrix must be finite.")

    sym_cov = 0.5 * (cov + cov.T)
    names = (
        [str(x) for x in asset_names]
        if asset_names is not None
        else [f"asset_{i}" for i in range(n)]
    )
    if len(names) != n:
        raise ValueError("asset_names length must match weights length.")

    port_var = max(float(w @ sym_cov @ w), 0.0)
    port_vol = math.sqrt(port_var)
    asset_vols = np.sqrt(np.maximum(np.diag(sym_cov), 0.0))
    weighted_asset_vol = float(np.sum(np.abs(w) * asset_vols))

    if port_vol <= 1e-15:
        mcr = np.full(n, float("nan"), dtype=float)
        rc = np.zeros(n, dtype=float)
        prc = np.full(n, float("nan"), dtype=float)
        div_ratio = float("nan")
        eff_n_risk = float("nan")
    else:
        marginal = sym_cov @ w
        mcr = marginal / port_vol
        rc = w * mcr
        prc = rc / port_vol
        div_ratio = float(weighted_asset_vol / port_vol)
        prc_sq_sum = float(np.sum(prc * prc))
        eff_n_risk = float(1.0 / prc_sq_sum) if prc_sq_sum > 1e-18 else float("nan")

    abs_w_sum = float(np.sum(np.abs(w)))
    if abs_w_sum > 1e-18:
        norm_abs_w = np.abs(w) / abs_w_sum
        eff_n_weights = float(1.0 / float(np.sum(norm_abs_w * norm_abs_w)))
    else:
        eff_n_weights = float("nan")

    per_asset: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        per_asset[name] = {
            "weight": float(w[idx]),
            "asset_volatility": float(asset_vols[idx]),
            "marginal_risk_contribution": float(mcr[idx]),
            "risk_contribution": float(rc[idx]),
            "pct_risk_contribution": float(prc[idx]),
        }

    max_prc_deviation = (
        float(np.max(np.abs(prc - (1.0 / float(n))))) if np.isfinite(prc).all() else float("nan")
    )
    return {
        "asset_names": names,
        "portfolio_volatility": float(port_vol),
        "weighted_asset_volatility": float(weighted_asset_vol),
        "diversification_ratio": float(div_ratio),
        "effective_n_weights": float(eff_n_weights),
        "effective_n_risk": float(eff_n_risk),
        "max_equal_risk_deviation": float(max_prc_deviation),
        "weights": {names[i]: float(w[i]) for i in range(n)},
        "marginal_risk_contribution": {names[i]: float(mcr[i]) for i in range(n)},
        "risk_contribution": {names[i]: float(rc[i]) for i in range(n)},
        "pct_risk_contribution": {names[i]: float(prc[i]) for i in range(n)},
        "per_asset": per_asset,
    }


def _solve_equal_weight_core(
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
    lower: float,
    upper: float,
) -> PortfolioSolution:
    n = len(mean_ret)
    weights = _project_capped_simplex(np.full(n, 1.0 / float(n), dtype=float), lower, upper)
    ret, vol, sharpe = _portfolio_stats(weights, mean_ret, cov_mat, risk_free_rate)
    return PortfolioSolution(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        converged=True,
        solver_message="equal_weight_analytical",
    )


def _solve_inverse_volatility_core(
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
    lower: float,
    upper: float,
) -> PortfolioSolution:
    diag_vol = np.sqrt(np.maximum(np.diag(cov_mat), 0.0))
    if np.any(diag_vol <= 1e-15) or not np.isfinite(diag_vol).all():
        # Clamp tiny variances smoothly while preserving finite ratios.
        safe_vol = np.where(diag_vol > 1e-15, diag_vol, 1e-15)
        raw = 1.0 / safe_vol
    else:
        raw = 1.0 / diag_vol
    raw = raw / float(np.sum(raw))
    weights = _project_capped_simplex(raw, lower, upper)
    ret, vol, sharpe = _portfolio_stats(weights, mean_ret, cov_mat, risk_free_rate)
    return PortfolioSolution(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        converged=True,
        solver_message="inverse_volatility_analytical",
    )


def _solve_risk_parity_core(
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
    lower: float,
    upper: float,
) -> PortfolioSolution:
    scipy_optimize = importlib.import_module("scipy.optimize")
    n = len(mean_ret)
    sym_cov = 0.5 * (cov_mat + cov_mat.T)
    target_prc = np.full(n, 1.0 / float(n), dtype=float)
    inv_vol = _solve_inverse_volatility_core(mean_ret, sym_cov, risk_free_rate, lower, upper)
    eq_w = _solve_equal_weight_core(mean_ret, sym_cov, risk_free_rate, lower, upper)
    gmv_w = _solve_min_variance_core(mean_ret, sym_cov, risk_free_rate, lower, upper)

    if abs(upper - lower) <= 1e-12:
        return eq_w

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ sym_cov @ w)
        if not np.isfinite(port_var) or port_var <= 1e-18:
            return 1e6
        rc_ratio = (w * (sym_cov @ w)) / port_var
        diff = rc_ratio - target_prc
        # Scale up O(1e4) so SLSQP ftol stops only after PRC converges tightly.
        return 1e4 * float(np.dot(diff, diff))

    starts = _deduplicate_starts(
        [inv_vol.weights, eq_w.weights, gmv_w.weights],
        lower,
        upper,
    )
    best_w: np.ndarray | None = None
    best_obj = float("inf")
    best_msg = "risk_parity_fallback"
    converged = False

    bounds = [(lower, upper)] * n
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)

    for s in starts:
        res = _slsqp_minimize(
            scipy_optimize,
            objective,
            s,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 600, "disp": False},
        )
        cand = np.asarray(res.x, dtype=float)  # type: ignore[attr-defined]
        if _is_feasible_solution(cand, lower, upper):
            cand = _project_capped_simplex(cand, lower, upper)
            val = objective(cand)
            if np.isfinite(val) and (best_w is None or val < best_obj):
                best_w = cand
                best_obj = val
                best_msg = str(getattr(res, "message", "slsqp_ok"))
                converged = converged or bool(getattr(res, "success", False))

    for fallback in (inv_vol.weights, eq_w.weights):
        val = objective(fallback)
        if np.isfinite(val) and (best_w is None or val < best_obj):
            best_w = fallback.copy()
            best_obj = val

    assert best_w is not None
    ret, vol, sharpe = _portfolio_stats(best_w, mean_ret, sym_cov, risk_free_rate)
    return PortfolioSolution(
        weights=best_w,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        converged=bool(converged),
        solver_message=best_msg,
    )


def solve_equal_weight_portfolio(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "ledoit_wolf",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    """求解等权（1/N）基准组合及其预期收益、波动率与 Sharpe。"""
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    return _solve_equal_weight_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
    )


def solve_inverse_volatility_portfolio(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "ledoit_wolf",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    """求解按资产年化波动率倒数加权（并投影到上下界单纯形）的组合。"""
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    return _solve_inverse_volatility_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
    )


def solve_risk_parity_portfolio(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "ledoit_wolf",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    """求解约束等风险贡献（ERC / Risk Parity）组合。"""
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    return _solve_risk_parity_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
    )


def _optimize_weights_for_window(
    window_df: pd.DataFrame,
    *,
    objective: OptimizerObjective,
    cov_method: CovMethod,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    trading_days: int,
) -> PortfolioSolution:
    if objective not in (
        "max_sharpe",
        "min_variance",
        "risk_parity",
        "inverse_volatility",
        "equal_weight",
    ):
        raise ValueError(f"Unsupported objective: {objective!r}")
    est = _estimate_covariance_from_array(
        window_df.to_numpy(dtype=float, copy=False),
        [str(c) for c in window_df.columns],
        cov_method=cov_method,
        trading_days=trading_days,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    if objective == "equal_weight":
        return _solve_equal_weight_core(
            est.mean_returns,
            est.cov_matrix,
            risk_free_rate,
            w_min,
            w_max,
        )
    if objective == "inverse_volatility":
        return _solve_inverse_volatility_core(
            est.mean_returns,
            est.cov_matrix,
            risk_free_rate,
            w_min,
            w_max,
        )
    if objective == "risk_parity":
        return _solve_risk_parity_core(
            est.mean_returns,
            est.cov_matrix,
            risk_free_rate,
            w_min,
            w_max,
        )
    gmv = _solve_min_variance_core(
        est.mean_returns,
        est.cov_matrix,
        risk_free_rate,
        w_min,
        w_max,
    )
    if objective == "min_variance":
        return gmv
    return _solve_tangency_core(
        est.mean_returns,
        est.cov_matrix,
        risk_free_rate,
        w_min,
        w_max,
        gmv_weights=gmv.weights,
    )
