"""Covariance and expected-return estimation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from models.returns import _positive_integer, clean_joint_returns

CovMethod = Literal["sample", "ledoit_wolf", "diagonal"]


@dataclass
class CovarianceEstimate:
    """预期年化收益与协方差估计结果及条件数诊断。"""

    asset_names: list[str]
    mean_returns: np.ndarray
    cov_matrix: np.ndarray
    cov_method: CovMethod
    shrinkage: float | None
    condition_number: float
    min_eigenvalue: float
    ridge_applied: float


def _ledoit_wolf_cov_daily(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 收缩协方差（优先调用 scikit-learn，缺省退回解析公式）。"""
    try:
        sklearn_cov = importlib.import_module("sklearn.covariance")
        lw = sklearn_cov.LedoitWolf(assume_centered=False).fit(x)
        return np.asarray(lw.covariance_, dtype=float), float(lw.shrinkage_)
    except ImportError:
        pass

    n_obs, n_assets = x.shape
    x_c = x - np.mean(x, axis=0)
    emp_cov = (x_c.T @ x_c) / float(n_obs)
    mu_iso = float(np.trace(emp_cov) / float(n_assets))
    delta_mat = emp_cov.copy()
    delta_mat.flat[:: n_assets + 1] -= mu_iso
    delta = float(np.sum(delta_mat * delta_mat) / float(n_assets))
    x2 = x_c * x_c
    # 用 ||x_k||^4 均值等价计算 (X^2)^T (X^2) 总和，时间复杂度 O(n*p)。
    beta_raw = (float(np.mean(np.sum(x2, axis=1) ** 2)) - float(np.sum(emp_cov * emp_cov))) / (
        float(n_assets) * float(n_obs)
    )
    beta = max(0.0, min(beta_raw, delta))
    shrinkage = 0.0 if delta <= 1e-30 else float(np.clip(beta / delta, 0.0, 1.0))
    shrunk = (1.0 - shrinkage) * emp_cov + shrinkage * mu_iso * np.eye(n_assets, dtype=float)
    return shrunk, shrinkage


def _estimate_covariance_from_array(
    x: np.ndarray,
    asset_names: list[str],
    *,
    cov_method: CovMethod = "sample",
    trading_days: int = 252,
) -> CovarianceEstimate:
    if cov_method not in ("sample", "ledoit_wolf", "diagonal"):
        raise ValueError(f"Unsupported cov_method: {cov_method!r}")
    trading_days = _positive_integer(trading_days, "trading_days")
    _, n_assets = x.shape
    if n_assets < 2:
        raise ValueError("Portfolio optimization requires at least 2 assets (columns).")

    mean_ann = np.mean(x, axis=0) * float(trading_days)
    shrinkage: float | None = None
    ridge_applied = 0.0

    if cov_method == "sample":
        cov_daily = np.cov(x, rowvar=False, ddof=1)
        if not np.isfinite(cov_daily).all():
            raise ValueError("Return covariance matrix contains non-finite values.")
        if cov_daily.shape != (n_assets, n_assets):
            raise ValueError("Internal error: covariance shape mismatch.")
        eff_rank = np.linalg.matrix_rank(cov_daily, tol=1e-10)
        if eff_rank < n_assets or bool(np.any(np.diag(cov_daily) <= 1e-15)):
            raise ValueError(
                "Return covariance matrix is singular or rank-deficient; check for redundant or constant series."
            )
        cond_daily = float(np.linalg.cond(cov_daily))
        if not np.isfinite(cond_daily) or cond_daily > 1e12:
            raise ValueError(
                f"Return covariance matrix is ill-conditioned (cond={cond_daily:.2e}); cannot stabilize frontier simulation."
            )
    elif cov_method == "ledoit_wolf":
        cov_daily, shrinkage = _ledoit_wolf_cov_daily(x)
        if not np.isfinite(cov_daily).all() or float(np.trace(cov_daily)) <= 1e-18:
            raise ValueError(
                "All assets have zero return variance; cannot estimate covariance matrix."
            )
        cov_daily = 0.5 * (cov_daily + cov_daily.T)
        eigs_d = np.linalg.eigvalsh(cov_daily)
        mean_var = float(np.mean(np.diag(cov_daily)))
        if float(eigs_d[0]) <= 1e-12 * max(mean_var, 1e-18):
            ridge_applied = max(1e-10 * mean_var, float(-eigs_d[0]) + 1e-12 * mean_var)
            cov_daily = cov_daily + ridge_applied * np.eye(n_assets, dtype=float)
    else:
        raw_cov = np.cov(x, rowvar=False, ddof=1)
        diag_var = np.diag(raw_cov).astype(float, copy=True)
        if not np.isfinite(diag_var).all() or float(np.max(diag_var)) <= 1e-18:
            raise ValueError(
                "All assets have zero return variance; cannot estimate covariance matrix."
            )
        pos = diag_var[diag_var > 1e-15]
        floor = 1e-8 * float(np.mean(pos))
        bad = diag_var <= 1e-15
        if bool(np.any(bad)):
            ridge_applied = floor
            diag_var = np.where(bad, floor, diag_var)
        cov_daily = np.diag(diag_var)

    cov_ann = 0.5 * (cov_daily + cov_daily.T) * float(trading_days)
    eigs_ann = np.linalg.eigvalsh(cov_ann)
    min_eig = float(eigs_ann[0])
    cond = float(eigs_ann[-1] / min_eig) if min_eig > 0.0 else float("inf")

    return CovarianceEstimate(
        asset_names=list(asset_names),
        mean_returns=mean_ann.astype(float, copy=False),
        cov_matrix=cov_ann.astype(float, copy=False),
        cov_method=cov_method,
        shrinkage=shrinkage,
        condition_number=cond,
        min_eigenvalue=min_eig,
        ridge_applied=ridge_applied * float(trading_days),
    )


def estimate_covariance_and_mean(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "sample",
    trading_days: int = 252,
    min_history: int = 30,
) -> CovarianceEstimate:
    """估计年化收益向量与协方差矩阵，支持 sample、Ledoit-Wolf 收缩与对角估计。"""
    r = clean_joint_returns(returns_df, min_history=min_history)
    x = r.to_numpy(dtype=float, copy=False)
    return _estimate_covariance_from_array(
        x,
        [str(c) for c in r.columns],
        cov_method=cov_method,
        trading_days=trading_days,
    )
