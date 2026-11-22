"""Masthead banner and collapsible methodology/reference drawer."""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

from asset_catalog import _TAB2_SYSTEM_MATH_MARKDOWN, build_ef_multi_asset_codes_reference_df
from contracts import SOFTWARE_VERSION
from ui.theme import THEME_MODE_LABELS, normalize_theme_mode


def render_masthead(
    st_module: Any,
    cfg: Mapping[str, Any],
    *,
    theme_mode: str = "auto",
) -> None:
    """Render the compact terminal masthead with live global runtime badges."""
    if st_module is None:
        return

    mode = normalize_theme_mode(theme_mode)
    theme_label = THEME_MODE_LABELS.get(mode, mode)

    if cfg.get("dates_valid") and cfg.get("start_str") and cfg.get("end_str"):
        window_text = f"{cfg['start_str']} ~ {cfg['end_str']}"
        window_cls = "qa-pill qa-pill--accent"
    else:
        window_text = "日期区间待修正"
        window_cls = "qa-pill qa-pill--warn"

    rf_pct = float(cfg.get("rf_rate", 0.016)) * 100.0
    cache_text = "实时刷新" if cfg.get("force_refresh") else "共享缓存"
    cache_cls = "qa-pill qa-pill--warn" if cfg.get("force_refresh") else "qa-pill qa-pill--ok"

    masthead_html = f"""
    <div class="qa-masthead">
      <div class="qa-masthead-left">
        <div class="qa-masthead-kicker">QUANT INSIGHT · MULTI-ASSET RESEARCH TERMINAL</div>
        <div class="qa-masthead-title">多资产量化分析与组合研究工作台</div>
        <p class="qa-masthead-sub">
          与 <code>quant</code> CLI 共用统一金融内核、语义防线、跨市场日历对齐与无前视样本外回测引擎。
        </p>
      </div>
      <div class="qa-masthead-badges">
        <span class="{window_cls}">窗口 {html.escape(window_text)}</span>
        <span class="qa-pill">Rf {rf_pct:.2f}%</span>
        <span class="{cache_cls}">{html.escape(cache_text)}</span>
        <span class="qa-pill">主题 {html.escape(theme_label)}</span>
        <span class="qa-pill">V{html.escape(str(SOFTWARE_VERSION))}</span>
      </div>
    </div>
    """
    st_module.markdown(masthead_html, unsafe_allow_html=True)


def render_methodology_drawer(st_module: Any) -> None:
    """Render collapsible financial conventions, mathematical formulas, and symbol reference."""
    if st_module is None:
        return
    with st_module.expander(
        "📐 计算口径、数学原理与 13 类资产代码参考（点击展开）", expanded=False
    ):
        st_module.markdown(
            "- **单资产与多资产默认口径差异**：单资产探索默认保留原生货币 (`LOCAL`) 与对数收益率 (`log`)；"
            "多资产比较与组合研究默认折算为统一货币 (`CNY`) 与离散简单收益率 (`simple`，保证组合加权线性性)。\n"
            "- **统计严格区分**：界面与导出报告均显式区分 **历史已实现绩效**、**正态性/Memmel 统计推断**、"
            "**样本内拟合（含前视偏差警告）** 与 **严格因果样本外回测（OOS）**。"
        )
        st_module.markdown(_TAB2_SYSTEM_MATH_MARKDOWN)
        ref_df = build_ef_multi_asset_codes_reference_df()
        st_module.dataframe(
            ref_df,
            width="stretch",
            hide_index=True,
            height=min(560, 42 + 36 * len(ref_df)),
        )
