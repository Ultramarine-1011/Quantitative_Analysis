"""Structured status alerts, change notices, takeaways, delta strips, and empty states."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from contracts import AnalysisResult, MultiAssetResult
from reporting import format_drawdown_percent, format_percent, format_ratio, is_valid_number
from ui.errors import UIError, format_ui_error_markdown
from ui.services_bridge import format_data_quality_warning


def render_error_banner(st_module: Any, error_obj: UIError | str | None) -> None:
    """Render a single classified error banner so fetch, validation, and compute failures stay distinct."""
    if st_module is None or not error_obj:
        return
    st_module.error(format_ui_error_markdown(error_obj))


def render_change_notice(
    st_module: Any,
    *,
    notice: str | None,
    button_label: str = "重新计算",
    button_key: str,
    disabled: bool = False,
) -> bool:
    """Render a clear pending-recalculation banner with an inline primary action button."""
    if st_module is None or not notice:
        return False
    col_msg, col_btn = st_module.columns([4, 1])
    col_msg.warning(notice)
    return bool(
        col_btn.button(
            button_label,
            key=button_key,
            type="primary",
            disabled=disabled,
            width="stretch",
        )
    )


def render_takeaway_card(
    st_module: Any,
    *,
    headline: str,
    detail: str = "",
) -> None:
    """Render a calm plain-language conclusion strip above KPI cards."""
    if st_module is None or not headline:
        return
    detail_html = f'<div class="qa-takeaway-detail">{html.escape(detail)}</div>' if detail else ""
    st_module.markdown(
        f"""
        <div class="qa-takeaway">
          <div class="qa-takeaway-head">{html.escape(headline)}</div>
          {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_metric_scalar(val: Any, fmt: str) -> str:
    if not is_valid_number(val):
        return "N/A"
    if fmt == "pct":
        return format_percent(val, signed=True)
    if fmt == "dd":
        return format_drawdown_percent(val)
    return format_ratio(val, decimals=3)


def _format_metric_delta(old_val: Any, new_val: Any, fmt: str) -> tuple[str, str]:
    if not (is_valid_number(old_val) and is_valid_number(new_val)):
        return "—", "neutral"
    if fmt == "dd":
        # max_drawdown is stored as non-negative depth (>= 0) and displayed as negative %.
        # Shallower depth (e.g. 0.25 -> 0.18) means +7.00 pp improvement (positive tone).
        diff = float(old_val) - float(new_val)
    else:
        diff = float(new_val) - float(old_val)
    if abs(diff) < 1e-9:
        return "持平", "neutral"
    if fmt in {"pct", "dd"}:
        sign = "+" if diff > 0 else ""
        text = f"{sign}{diff * 100.0:.2f} pp"
    else:
        sign = "+" if diff > 0 else ""
        text = f"{sign}{diff:.3f}"
    if fmt == "neutral_ratio":
        return text, "neutral"
    tone = "pos" if diff > 0 else "neg"
    return text, tone


def render_delta_comparison(
    st_module: Any,
    *,
    prev_summary: Mapping[str, Any] | None,
    cur_summary: Mapping[str, Any] | None,
) -> None:
    """Render a compact before/after comparison strip when parameters or dates changed."""
    if (
        st_module is None
        or not isinstance(prev_summary, Mapping)
        or not isinstance(cur_summary, Mapping)
    ):
        return
    if prev_summary.get("scope_key") != cur_summary.get("scope_key"):
        return

    prev_metrics = {
        str(m["key"]): m for m in (prev_summary.get("metrics") or ()) if isinstance(m, Mapping)
    }
    cur_metrics = [m for m in (cur_summary.get("metrics") or ()) if isinstance(m, Mapping)]
    if not prev_metrics or not cur_metrics:
        return

    changed_items = prev_summary.get("changed_items") or []
    changes_note = (
        "变更项：" + " · ".join(str(x) for x in changed_items)
        if changed_items
        else f"{prev_summary.get('context_label', '上一轮')} → {cur_summary.get('context_label', '当前')}"
    )

    cards: list[str] = []
    for cur_m in cur_metrics:
        m_key = str(cur_m["key"])
        old_m = prev_metrics.get(m_key)
        if old_m is None:
            continue
        fmt = str(cur_m.get("fmt", "ratio"))
        old_txt = _format_metric_scalar(old_m.get("value"), fmt)
        new_txt = _format_metric_scalar(cur_m.get("value"), fmt)
        delta_txt, tone = _format_metric_delta(old_m.get("value"), cur_m.get("value"), fmt)
        cards.append(
            f'<div class="qa-delta-item">'
            f'<div class="qa-delta-label">{html.escape(str(cur_m.get("label", m_key)))}</div>'
            f'<div class="qa-delta-values">{html.escape(old_txt)} → <strong>{html.escape(new_txt)}</strong></div>'
            f'<div class="qa-delta-diff qa-kpi-value--{tone}">{html.escape(delta_txt)}</div>'
            f"</div>"
        )

    if not cards:
        return

    st_module.markdown(
        f"""
        <div class="qa-delta-strip">
          <div class="qa-delta-header">
            <span class="qa-delta-title">前后结果对比</span>
            <span class="qa-delta-meta">{html.escape(changes_note)}</span>
          </div>
          <div class="qa-delta-grid">{"".join(cards)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_and_warnings(
    st_module: Any,
    warnings_list: Sequence[str],
    *,
    single_result: AnalysisResult | None = None,
) -> None:
    """Display formatted data-quality banner and deduplicated service warnings."""
    if st_module is None:
        return
    shown_messages: set[str] = set()
    if single_result is not None:
        formatted_q = format_data_quality_warning(single_result)
        if formatted_q:
            st_module.warning(formatted_q)
            shown_messages.add(formatted_q)

    for raw_warn in warnings_list:
        msg = str(raw_warn).strip()
        if not msg or msg in shown_messages:
            continue
        if (
            single_result is not None
            and shown_messages
            and msg.startswith("数据质量提示：dropped=")
        ):
            continue
        shown_messages.add(msg)
        st_module.warning(msg)


def render_validation_protocol_banner(
    st_module: Any,
    result: MultiAssetResult,
    *,
    fallback_mode: str,
) -> tuple[bool, bool, str]:
    """Render portfolio validation-mode banner and return ``(is_oos, has_lookahead, mode)``."""
    methodology = result.methodology
    validation_meta = result.metrics.get("validation") or {}
    res_val_mode = str(methodology.get("validation_mode") or fallback_mode)
    is_oos = res_val_mode != "in_sample" and bool(methodology.get("is_out_of_sample", True))
    has_lookahead = bool(methodology.get("has_lookahead_bias", False))

    if st_module is None:
        return is_oos, has_lookahead, res_val_mode

    if not is_oos:
        if has_lookahead:
            st_module.warning(
                "当前为 **全样本拟合 (In-Sample Fit)**：权重估计与评估使用全区间事后信息，"
                "存在前视偏差（Lookahead Bias），不代表真实样本外可执行收益。"
            )
        else:
            st_module.warning(
                "当前为 **全样本固定权重回测 (In-Sample Non-OOS Backtest)**："
                "采用外生自定义权重在完整历史区间直接回测，未经过时间切分或滚动样本外隔离，"
                "非样本外评估（OOS）；若权重基于事后观察手工选定，仍可能存在选择偏差。"
            )
    elif res_val_mode == "train_test":
        st_module.info(
            f"样本外独立测试协议 (`train_test`)：训练集 {validation_meta.get('train_start')} ~ "
            f"{validation_meta.get('train_end')} ({validation_meta.get('train_rows')} 期) | "
            f"严格样本外测试集 {validation_meta.get('test_start')} ~ "
            f"{validation_meta.get('test_end')} ({validation_meta.get('test_rows')} 期)"
        )
    else:
        st_module.info(
            f"因果 Walk-Forward 样本外协议 (`{res_val_mode}`)："
            f"OOS 区间 {validation_meta.get('oos_start')} ~ {validation_meta.get('oos_end')} "
            f"({validation_meta.get('oos_periods')} 期，共执行 {validation_meta.get('rebalance_count')} 次滚动优化调仓)"
        )
    return is_oos, has_lookahead, res_val_mode


def render_empty_state(
    st_module: Any,
    *,
    title: str,
    subtitle: str,
    chips: Sequence[str] = (),
) -> None:
    """Render an informative empty-state card with active parameter summary chips."""
    if st_module is None:
        return
    chip_html = ""
    if chips:
        items = "".join(f'<span class="qa-chip">{html.escape(str(c))}</span>' for c in chips if c)
        if items:
            chip_html = f'<div class="qa-chip-row">{items}</div>'
    st_module.markdown(
        f"""
        <div class="qa-empty-card">
          <div class="qa-empty-title">{html.escape(title)}</div>
          <p class="qa-empty-sub">{html.escape(subtitle)}</p>
          {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
