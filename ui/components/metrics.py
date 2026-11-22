"""Metric cards and responsive secondary KPI grids with semantic tone coding."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from contracts import AnalysisResult
from reporting import (
    format_drawdown_percent,
    format_metric_value,
    format_percent,
    format_price_with_unit,
    format_ratio,
    format_reason_label,
    is_valid_number,
)

MetricTone = Literal["neutral", "pos", "neg", "warn", "muted"]


@dataclass(frozen=True)
class KPIItem:
    """Display specification for a single KPI tile."""

    label: str
    value: str
    sub: str = ""
    tone: MetricTone = "neutral"


def tone_for_signed_number(value: Any, *, invert: bool = False) -> MetricTone:
    """Determine semantic tone (``pos`` / ``neg`` / ``muted``) from a numeric value."""
    if not is_valid_number(value):
        return "muted"
    num = float(value)
    if abs(num) < 1e-12:
        return "neutral"
    is_good = num < 0 if invert else num > 0
    return "pos" if is_good else "neg"


def render_metric_cards(st_module: Any, items: Sequence[tuple[str, str]]) -> None:
    """Render primary ``st.metric`` headline cards (3 columns by default)."""
    if st_module is None or not items:
        return
    cols = st_module.columns(len(items))
    for col, (label, val) in zip(cols, items, strict=False):
        col.metric(label, val)


def render_secondary_kpi_strip(st_module: Any, items: Sequence[KPIItem]) -> None:
    """Render compact tabular-numeral HTML KPI grid without inflating ``st.metric`` counts."""
    if st_module is None or not items:
        return
    cards: list[str] = []
    for item in items:
        tone_cls = f"qa-kpi-value--{item.tone}" if item.tone != "neutral" else ""
        sub_html = f'<div class="qa-kpi-sub">{html.escape(item.sub)}</div>' if item.sub else ""
        cards.append(
            f'<div class="qa-kpi-card">'
            f'<div class="qa-kpi-label">{html.escape(item.label)}</div>'
            f'<div class="qa-kpi-value {tone_cls}">{html.escape(item.value)}</div>'
            f"{sub_html}"
            f"</div>"
        )
    st_module.markdown(
        f'<div class="qa-kpi-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_single_asset_kpis(st_module: Any, result: AnalysisResult) -> None:
    """Render single-asset headline KPIs and secondary metrics honoring ``yield_level`` blocking."""
    if st_module is None:
        return
    metrics = result.metrics
    reasons = result.metric_reasons
    series_kind = result.asset.series_kind
    output_ccy = result.asset.output_currency or result.asset.native_currency

    if series_kind == "yield_level":
        st_module.info(
            "利率水平序列提示：当前标的为利率/收益率水平 (%)，并非总收益资产价格，"
            "按金融严谨性禁止计算复利回报、CAGR、回撤百分比或夏普比率。"
        )
        lo = format_metric_value(
            "level_min",
            metrics.get("level_min"),
            series_kind=series_kind,
            output_currency=output_ccy,
        )
        hi = format_metric_value(
            "level_max",
            metrics.get("level_max"),
            series_kind=series_kind,
            output_currency=output_ccy,
        )
        render_metric_cards(
            st_module,
            [
                (
                    "最新利率水平",
                    format_metric_value(
                        "latest_level",
                        metrics.get("latest_level"),
                        series_kind=series_kind,
                        output_currency=output_ccy,
                    ),
                ),
                (
                    "区间绝对变动",
                    format_metric_value(
                        "level_change",
                        metrics.get("level_change"),
                        series_kind=series_kind,
                        output_currency=output_ccy,
                    ),
                ),
                ("区间高低范围", f"{lo} ~ {hi}"),
            ],
        )

        rows = [
            {
                "指标": label,
                "数值": "N/A（口径禁用）",
                "禁用原因": format_reason_label(reasons.get(key, "yield_level_not_return")),
            }
            for key, label in (
                ("cumulative_return", "累计收益率"),
                ("cagr", "年化复合收益 (CAGR)"),
                ("max_drawdown", "最大回撤"),
                ("annualized_volatility", "年化波动率"),
                ("sharpe_ratio", "夏普比率"),
                ("sortino_ratio", "Sortino 比率"),
                ("calmar_ratio", "Calmar 比率"),
            )
        ]
        st_module.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        return

    cagr_val = metrics.get("cagr")
    mdd_val = metrics.get("max_drawdown")
    sharpe_val = metrics.get("sharpe_ratio")

    render_metric_cards(
        st_module,
        [
            (
                "CAGR",
                format_percent(cagr_val)
                if is_valid_number(cagr_val)
                else format_reason_label(reasons.get("cagr")),
            ),
            (
                "最大回撤",
                format_drawdown_percent(mdd_val)
                if is_valid_number(mdd_val)
                else format_reason_label(reasons.get("max_drawdown")),
            ),
            (
                "夏普比率",
                format_ratio(sharpe_val, decimals=3)
                if is_valid_number(sharpe_val)
                else format_reason_label(reasons.get("sharpe_ratio")),
            ),
        ],
    )

    cum_ret = metrics.get("cumulative_return")
    ann_vol = metrics.get("annualized_volatility")
    sortino = metrics.get("sortino_ratio")
    calmar = metrics.get("calmar_ratio")
    var95 = metrics.get("var_95")
    cvar95 = metrics.get("cvar_95")

    secondary_items = [
        KPIItem(
            label="累计收益率",
            value=format_percent(cum_ret, signed=True),
            tone=tone_for_signed_number(cum_ret),
        ),
        KPIItem(
            label="年化波动率",
            value=format_percent(ann_vol),
            sub=f"{result.asset.trading_days} 期/年 · {metrics.get('return_convention', 'log')}",
            tone="neutral",
        ),
        KPIItem(
            label="Sortino / Calmar",
            value=f"{format_ratio(sortino, decimals=2)} / {format_ratio(calmar, decimals=2)}",
            sub="下行风险与回撤补偿比",
            tone=tone_for_signed_number(sortino),
        ),
        KPIItem(
            label="95% 日 VaR / CVaR",
            value=f"{format_percent(var95)} / {format_percent(cvar95)}",
            sub="尾部单日风险分位数",
            tone="neg" if is_valid_number(var95) and float(var95) < 0 else "neutral",
        ),
        KPIItem(
            label="最新收盘价",
            value=format_price_with_unit(
                metrics.get("latest_close"),
                series_kind=series_kind,
                output_currency=output_ccy,
            ),
            sub=str(metrics.get("latest_signal", "N/A")),
            tone="neutral",
        ),
    ]
    render_secondary_kpi_strip(st_module, secondary_items)
