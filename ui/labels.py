"""Centralized UI display labels, options, and financial methodology captions."""

from __future__ import annotations

WORKSPACE_TAB_LABELS: tuple[str, str, str] = (
    "📈 单资产探索",
    "⚖️ 多资产比较",
    "🎯 投资组合研究",
)

ADJUST_LABELS: dict[str, str] = {
    "hfq": "后复权 (hfq)",
    "qfq": "前复权 (qfq)",
    "": "不复权 (raw)",
}

BASE_CURRENCY_LABELS: dict[str, str] = {
    "LOCAL": "原生计价货币 (LOCAL)",
    "CNY": "人民币统一折算 (CNY)",
    "USD": "美元统一折算 (USD)",
}

RETURN_CONVENTION_LABELS: dict[str, str] = {
    "log": "对数收益率 (log)",
    "simple": "简单算术收益率 (simple)",
}

CALENDAR_MODE_LABELS: dict[str, str] = {
    "common_trading_dates": "共同交易日交集 (common_trading_dates)",
    "weekly_friday": "周频周五重采样 (weekly_friday)",
    "mtm_outer": "跨市场外连接按市值重估 (mtm_outer)",
}

COV_METHOD_LABELS: dict[str, str] = {
    "ledoit_wolf": "Ledoit-Wolf 收缩估计 (ledoit_wolf)",
    "sample": "样本协方差 (sample)",
    "diagonal": "对角方差阵 (diagonal)",
}

STRATEGY_LABELS: dict[str, str] = {
    "max_sharpe": "最大夏普切点组合 (max_sharpe)",
    "min_variance": "全局最小方差组合 GMV (min_variance)",
    "risk_parity": "风险平价 ERC (risk_parity)",
    "inverse_volatility": "倒数波动率加权 (inverse_volatility)",
    "equal_weight": "等权重基准 1/N (equal_weight)",
    "custom": "自定义权重组合 (custom)",
}

STRATEGY_HINTS: dict[str, str] = {
    "max_sharpe": "最大化切点组合夏普比率，受预期收益估计误差影响较大，建议配合样本外验证。",
    "min_variance": "仅依赖协方差矩阵求全局最小方差，避免均值预测噪音。",
    "risk_parity": "使各成分资产对组合总波动率的 Euler 边际风险贡献相等。",
    "inverse_volatility": "按各资产年化波动率倒数加权，不依赖相关性矩阵，稳健性高。",
    "equal_weight": "1/N 等权再平衡基准，用于衡量主动优化配置是否提供样本外增量。",
    "custom": "使用手动指定权重执行静态/再平衡回测与风险贡献分解。",
}

VALIDATION_MODE_LABELS: dict[str, str] = {
    "train_test": "静态样本外划分 OOS (train_test)",
    "walk_forward_rolling": "滚动窗口 Walk-Forward OOS (walk_forward_rolling)",
    "walk_forward_expanding": "扩展窗口 Walk-Forward OOS (walk_forward_expanding)",
    "in_sample": "全样本拟合 In-Sample (含前视偏差警告)",
}

REBALANCE_FREQ_LABELS: dict[str, str] = {
    "monthly": "月度再平衡 (monthly)",
    "quarterly": "季度再平衡 (quarterly)",
    "daily": "日度再平衡 (daily)",
    "buy_and_hold": "买入持有不调仓 (buy_and_hold)",
}

SINGLE_CHART_LABELS: dict[str, str] = {
    "price": "价格 / 均线 / 布林带与成交量",
    "rolling": "滚动波动率、VaR 与技术动量",
    "drawdown": "历史水下回撤曲线",
    "normalized": "区间归一化净值 (Base=1)",
}

COMPARE_CHART_LABELS: dict[str, str] = {
    "normalized": "归一化净值曲线 (Base=1)",
    "drawdown": "水下回撤对比",
    "rolling": "滚动风险与相关性",
    "correlation": "收益率相关性热力图",
}

PORTFOLIO_CHART_LABELS: dict[str, str] = {
    "backtest": "策略回测净值与回撤",
    "weights": "权重分布与 Euler 风险贡献",
    "frontier": "SLSQP 精确有效前沿",
    "correlation": "标的同期相关性热力图",
    "normalized": "组合标的归一化净值",
    "drawdown": "组合标的水下回撤",
    "weight_history": "Walk-Forward / 调仓权重演化",
}

SINGLE_CHART_ORDER: tuple[str, ...] = ("price", "rolling", "drawdown", "normalized")
COMPARE_CHART_ORDER: tuple[str, ...] = ("normalized", "drawdown", "rolling", "correlation")
PORTFOLIO_CORE_CHARTS: tuple[str, ...] = ("backtest", "weights", "frontier", "correlation")
PORTFOLIO_EXTRA_CHARTS: tuple[str, ...] = ("normalized", "drawdown")
