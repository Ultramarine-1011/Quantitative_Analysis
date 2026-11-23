"""Compact workspace topbar and collapsible methodology/reference content."""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

from asset_catalog import _TAB2_SYSTEM_MATH_MARKDOWN, build_ef_multi_asset_codes_reference_df
from contracts import SOFTWARE_VERSION

_STATUS_PILL_STYLE: dict[str, tuple[str, str]] = {
    "ready": ("qa-pill qa-pill--ok", "已计算"),
    "pending_rerun": ("qa-pill qa-pill--warn", "待重新计算"),
    "incomplete": ("qa-pill", "待补充输入"),
    "empty_data": ("qa-pill qa-pill--warn", "暂无行情数据"),
    "error": ("qa-pill qa-pill--danger", "计算或数据异常"),
    "idle": ("qa-pill", "准备就绪"),
}


def render_workspace_topbar(
    st_module: Any,
    *,
    view_title: str,
    context_label: str = "",
    status_kind: str = "idle",
    status_text: str | None = None,
    cfg: Mapping[str, Any],
) -> None:
    """Render a minimal single-line header bar with status pill and active window context."""
    if st_module is None:
        return

    if cfg.get("dates_valid") and cfg.get("start_str") and cfg.get("end_str"):
        window_text = f"{cfg['start_str']} ~ {cfg['end_str']}"
        window_cls = "qa-pill"
    else:
        window_text = "日期区间待修正"
        window_cls = "qa-pill qa-pill--warn"

    rf_pct = float(cfg.get("rf_rate", 0.016)) * 100.0
    pill_cls, default_status = _STATUS_PILL_STYLE.get(status_kind, ("qa-pill", "准备就绪"))
    pill_label = status_text or default_status
    ctx_html = (
        f'<span class="qa-topbar-sep">/</span>'
        f'<span class="qa-topbar-context">{html.escape(context_label)}</span>'
        if context_label
        else ""
    )

    st_module.markdown(
        f"""
        <div class="qa-topbar">
          <div class="qa-topbar-left">
            <span class="qa-topbar-title">{html.escape(view_title)}</span>
            {ctx_html}
          </div>
          <div class="qa-topbar-badges">
            <span class="{pill_cls}">{html.escape(pill_label)}</span>
            <span class="{window_cls}">{html.escape(window_text)}</span>
            <span class="qa-pill">Rf {rf_pct:.2f}%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_masthead(
    st_module: Any,
    cfg: Mapping[str, Any],
    *,
    theme_mode: str = "auto",
) -> None:
    """Backward-compatible alias rendering the compact workspace topbar."""
    render_workspace_topbar(
        st_module,
        view_title="Quant Insight",
        context_label=f"V{SOFTWARE_VERSION}",
        status_kind="idle",
        cfg=cfg,
    )


def render_methodology_content(st_module: Any) -> None:
    """Render financial conventions, formulas, and 13-class asset code reference."""
    if st_module is None:
        return
    st_module.markdown(
        "- **口径说明**：单资产探索默认保留资产原生货币 (`LOCAL`) 与对数收益率 (`log`)；"
        "多资产比较与投资组合默认折算为人民币 (`CNY`) 与简单收益率 (`simple`，保证权重加和线性性)。\n"
        "- **验证区分**：界面显式区分 **历史已实现绩效**、**正态性/Memmel 检验**、"
        "**样本内拟合（含前视偏差警告）** 与 **严格因果样本外回测（OOS）**。"
    )
    st_module.markdown(_TAB2_SYSTEM_MATH_MARKDOWN)
    ref_df = build_ef_multi_asset_codes_reference_df()
    st_module.dataframe(
        ref_df,
        width="stretch",
        hide_index=True,
        height=min(420, 42 + 36 * len(ref_df)),
    )


def render_methodology_drawer(st_module: Any) -> None:
    """Backward-compatible collapsible methodology drawer."""
    if st_module is None:
        return
    with st_module.expander("计算口径、数学原理与 13 类资产代码参考", expanded=False):
        render_methodology_content(st_module)
