"""Centralized UI display labels, plain-language options, and tooltips."""

from __future__ import annotations

WORKSPACE_TAB_LABELS: tuple[str, str, str] = (
    "分析资产",
    "比较资产",
    "构建组合",
)

NAV_LABELS: dict[str, str] = {
    "home": "开始",
    "single": "分析资产",
    "compare": "比较资产",
    "portfolio": "构建组合",
}

NAV_SUBTITLES: dict[str, str] = {
    "home": "搜索标的或一键启动量化研究",
    "single": "收益、风险回撤与价格走势诊断",
    "compare": "统一货币与日历的跨资产横向比较",
    "portfolio": "资产权重优化、有效前沿与样本外回测",
}

DATE_PRESETS: tuple[tuple[str, str, int], ...] = (
    ("6M", "近6月", 183),
    ("1Y", "近1年", 365),
    ("3Y", "近3年", 1096),
    ("5Y", "近5年", 1826),
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
    "ledoit_wolf": "Ledoit-Wolf 收缩估计（推荐，抗噪音）",
    "sample": "样本协方差 (sample)",
    "diagonal": "对角方差阵 (diagonal)",
}

STRATEGY_LABELS: dict[str, str] = {
    "max_sharpe": "最大夏普比率（追求最高风险收益比）",
    "min_variance": "最小波动组合（追求最稳健回撤控制）",
    "risk_parity": "风险平价（各资产承担均衡风险）",
    "inverse_volatility": "倒数波动率加权（低波高配）",
    "equal_weight": "等权重基准 1/N（均分资金）",
    "custom": "自定义权重组合（手动指定比例）",
}

STRATEGY_HINTS: dict[str, str] = {
    "max_sharpe": "寻找风险调整收益最高的切点配置；对历史预期收益敏感，建议结合样本外回测验证。",
    "min_variance": "仅依赖协方差矩阵最小化组合总波动，不依赖收益预测，通常更抗过拟合。",
    "risk_parity": "使每个资产对组合总风险的贡献比例相同，避免单一高波动资产主导组合。",
    "inverse_volatility": "按波动率倒数分配权重，不依赖资产间相关性估计，在样本较短时更稳健。",
    "equal_weight": "各资产均匀分配 1/N 权重并定期再平衡，常用作检验主动优化是否跑赢朴素分散的基准。",
    "custom": "按手动输入的固定权重执行回测与风险贡献分解。",
}

VALIDATION_MODE_LABELS: dict[str, str] = {
    "train_test": "样本外独立验证（前后切分，推荐）",
    "walk_forward_rolling": "滚动窗口样本外回测 (Walk-Forward)",
    "walk_forward_expanding": "扩展窗口样本外回测 (Expanding)",
    "in_sample": "全样本拟合（含前视偏差，仅作参考）",
}

VALIDATION_MODE_HINTS: dict[str, str] = {
    "train_test": "用前段历史训练权重、在后段严格未见数据上验证表现，避免用未来信息选权重。",
    "walk_forward_rolling": "随时间滚动窗口定期重算权重并在下一期真实持有，模拟动态策略运行。",
    "walk_forward_expanding": "保留起点至今的全部历史动态重估权重，并在下一期验证。",
    "in_sample": "在整段历史同时优化与评估，存在前视偏差（Lookahead Bias），不可直接视为可实现收益。",
}

REBALANCE_FREQ_LABELS: dict[str, str] = {
    "monthly": "月度再平衡 (monthly)",
    "quarterly": "季度再平衡 (quarterly)",
    "daily": "日度再平衡 (daily)",
    "buy_and_hold": "买入持有不调仓 (buy_and_hold)",
}

STRATEGY_SHORT_LABELS: dict[str, str] = {
    "max_sharpe": "最大夏普",
    "min_variance": "最小波动",
    "risk_parity": "风险平价",
    "inverse_volatility": "倒数波动率",
    "equal_weight": "等权重 1/N",
    "custom": "自定义权重",
}

VALIDATION_SHORT_LABELS: dict[str, str] = {
    "train_test": "样本外切分",
    "walk_forward_rolling": "滚动样本外",
    "walk_forward_expanding": "扩展样本外",
    "in_sample": "全样本拟合",
}

COV_SHORT_LABELS: dict[str, str] = {
    "ledoit_wolf": "Ledoit-Wolf 收缩",
    "sample": "样本协方差",
    "diagonal": "对角方差阵",
}

CALENDAR_SHORT_LABELS: dict[str, str] = {
    "common_trading_dates": "共同交易日",
    "weekly_friday": "周频周五",
    "mtm_outer": "跨市外连接",
}

REBALANCE_SHORT_LABELS: dict[str, str] = {
    "monthly": "月度调仓",
    "quarterly": "季度调仓",
    "daily": "日度调仓",
    "buy_and_hold": "买入持有",
}

KPI_HELP_TEXTS: dict[str, str] = {
    "cagr": "年化复合收益率：假设收益持续复利，平均每年增长的速度。",
    "max_drawdown": "最大回撤：区间内从历史高点回落到最低点的最大跌幅，越小越稳健。",
    "sharpe": "夏普比率：每承担一单位总波动所获得的超额收益（扣除无风险利率 Rf 后），越高越好。",
    "portfolio_cagr": "组合策略在验证区间内的年化复合收益率（已扣除估算交易成本）。",
    "portfolio_mdd": "组合策略在验证区间内的最大净值回撤深度。",
    "portfolio_sharpe": "组合策略的超额收益与年化波动率之比，衡量风险调整后表现。",
}

SINGLE_CHART_LABELS: dict[str, str] = {
    "price": "价格走势与均线",
    "rolling": "滚动波动率与风险",
    "drawdown": "历史水下回撤",
    "normalized": "归一化净值 (Base=1)",
}

COMPARE_CHART_LABELS: dict[str, str] = {
    "normalized": "归一化净值对比",
    "drawdown": "水下回撤对比",
    "rolling": "滚动风险与相关性",
    "correlation": "相关性热力图",
}

PORTFOLIO_CHART_LABELS: dict[str, str] = {
    "backtest": "策略回测净值与回撤",
    "weights": "权重与风险贡献",
    "frontier": "有效前沿",
    "correlation": "资产相关性",
    "normalized": "标的归一化净值",
    "drawdown": "标的水下回撤",
    "weight_history": "动态权重演化",
}

SINGLE_CHART_ORDER: tuple[str, ...] = ("price", "rolling", "drawdown", "normalized")
COMPARE_CHART_ORDER: tuple[str, ...] = ("normalized", "drawdown", "rolling", "correlation")
PORTFOLIO_CORE_CHARTS: tuple[str, ...] = ("backtest", "weights", "frontier", "correlation")
PORTFOLIO_EXTRA_CHARTS: tuple[str, ...] = ("normalized", "drawdown")
