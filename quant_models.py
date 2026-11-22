"""Compatibility API for the numerical algorithms in models."""

from models.backtesting import (
    RebalanceFreq,
    TrainTestSplitResult,
    WalkForwardResult,
    WindowMode,
    backtest_portfolio,
    compute_backtest_metrics,
    run_walk_forward_backtest,
    split_train_test_returns,
)
from models.covariance import (
    CovarianceEstimate,
    CovMethod,
    _ledoit_wolf_cov_daily as _ledoit_wolf_cov_daily,
    estimate_covariance_and_mean,
)
from models.frontier import (
    EfficientFrontierResult,
    ExactEfficientFrontierResult,
    generate_efficient_frontier,
    solve_exact_efficient_frontier,
)
from models.optimization import (
    OptimizerObjective,
    PortfolioSolution,
    compute_risk_contributions,
    solve_equal_weight_portfolio,
    solve_inverse_volatility_portfolio,
    solve_minimum_variance_portfolio,
    solve_risk_parity_portfolio,
    solve_tangency_portfolio,
)
from models.returns import (
    clean_joint_returns,
    normalize_close_to_base_one,
    returns_correlation_matrix,
)
from models.simulation import MonteCarloGBMResult, run_monte_carlo_gbm

__all__ = [
    "CovMethod",
    "CovarianceEstimate",
    "EfficientFrontierResult",
    "ExactEfficientFrontierResult",
    "MonteCarloGBMResult",
    "OptimizerObjective",
    "PortfolioSolution",
    "RebalanceFreq",
    "TrainTestSplitResult",
    "WalkForwardResult",
    "WindowMode",
    "backtest_portfolio",
    "clean_joint_returns",
    "compute_backtest_metrics",
    "compute_risk_contributions",
    "estimate_covariance_and_mean",
    "generate_efficient_frontier",
    "normalize_close_to_base_one",
    "returns_correlation_matrix",
    "run_monte_carlo_gbm",
    "run_walk_forward_backtest",
    "solve_equal_weight_portfolio",
    "solve_exact_efficient_frontier",
    "solve_inverse_volatility_portfolio",
    "solve_minimum_variance_portfolio",
    "solve_risk_parity_portfolio",
    "solve_tangency_portfolio",
    "split_train_test_returns",
]
