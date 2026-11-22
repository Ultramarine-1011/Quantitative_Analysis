"""Structured error banners, quality warnings, validation protocol notices, and empty states."""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

from contracts import AnalysisResult, MultiAssetResult
from ui.errors import UIError, format_ui_error_markdown
from ui.services_bridge import format_data_quality_warning


def render_error_banner(st_module: Any, error_obj: UIError | str | None) -> None:
    """Render a single classified error banner so fetch, validation, and compute failures stay distinct."""
    if st_module is None or not error_obj:
        return
    st_module.error(format_ui_error_markdown(error_obj))


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
        # Suppress raw English summary duplicate when clean Chinese quality warning already shown.
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
                "⚠️ 当前为 **全样本拟合 (In-Sample Fit)**：权重估计与评估使用全区间事后信息，"
                "存在前视偏差（Lookahead Bias），不代表真实样本外可执行收益。"
            )
        else:
            st_module.warning(
                "⚠️ 当前为 **全样本固定权重回测 (In-Sample Non-OOS Backtest)**："
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
