"""Workspace 0: Intuitive Home launcher with Chinese asset search, quick starts, and resume cards."""

from __future__ import annotations

import html
from typing import Any

from asset_catalog import SUPPORTED_ADJUSTMENTS, get_asset_trading_days
from reporting import format_percent, is_valid_number
from ui.asset_directory import (
    POPULAR_ASSETS,
    QUICK_START_ASSETS,
    format_asset_display,
    format_asset_picker_option,
)
from ui.components.header import render_workspace_topbar
from ui.state import (
    WORKSPACE_COMPARE_KEY,
    WORKSPACE_PORTFOLIO_KEY,
    WORKSPACE_SINGLE_KEY,
    queue_transition,
    record_basket_assets,
    workspace_state,
)


def _launch_single_asset(st_module: Any, asset_type: str, symbol: str) -> None:
    a_type = str(asset_type).strip().lower()
    sym = str(symbol).strip()
    updates: dict[str, Any] = {
        "asset_type": a_type,
        "_prev_asset_type": a_type,
        "symbol_input": sym,
        "single_trading_days_input": get_asset_trading_days(a_type),
    }
    supported = SUPPORTED_ADJUSTMENTS.get(a_type)
    if supported:
        updates["adjust_input"] = supported[0]
    record_basket_assets(st_module, [(a_type, sym)])
    queue_transition(st_module, view="single", updates=updates, auto_run_view="single")
    st_module.rerun()


def render_home_workspace(st_module: Any, cfg: dict[str, Any]) -> None:
    """Render the calm, search-first Home screen guiding new users by task."""
    if st_module is None:
        return

    s_state = workspace_state(st_module, WORKSPACE_SINGLE_KEY)
    c_state = workspace_state(st_module, WORKSPACE_COMPARE_KEY)
    p_state = workspace_state(st_module, WORKSPACE_PORTFOLIO_KEY)
    has_any_result = any(st_obj.get("result") is not None for st_obj in (s_state, c_state, p_state))

    render_workspace_topbar(
        st_module,
        view_title="开始",
        context_label="定量研究工作台",
        status_kind="ready" if has_any_result else "idle",
        status_text="已有研究会话" if has_any_result else "准备就绪",
        cfg=cfg,
    )

    st_module.markdown(
        """
        <div class="qa-home-hero">
          <div class="qa-home-title">今天想研究什么资产？</div>
          <p class="qa-home-sub">
            无需记忆代码前缀或金融术语。搜索中文名称即可诊断表现，或按任务直接比较资产与构建投资组合。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st_module.container(border=True):
        col_sel, col_go = st_module.columns([4, 1])
        with col_sel:
            selected_entry = st_module.selectbox(
                "搜索资产（支持中文名称或代码，如 沪深300、AAPL、黄金）",
                options=POPULAR_ASSETS,
                index=0,
                format_func=format_asset_picker_option,
                key="home_asset_select",
            )
        with col_go:
            st_module.markdown('<div class="qa-btn-spacer"></div>', unsafe_allow_html=True)
            if st_module.button(
                "立即分析",
                key="home_analyze_btn",
                type="primary",
                disabled=not cfg["dates_valid"],
                width="stretch",
            ):
                _launch_single_asset(st_module, selected_entry.asset_type, selected_entry.symbol)

        st_module.caption("一键体验热门标的（自动使用近 3 年与默认稳健参数）：")
        q_cols = st_module.columns(len(QUICK_START_ASSETS))
        for idx, (col, entry) in enumerate(zip(q_cols, QUICK_START_ASSETS, strict=False)):
            short_title = entry.name.split(" (", 1)[0]
            if short_title == "苹果 Apple":
                short_title = "苹果 AAPL"
            elif short_title == "标普500 ETF":
                short_title = "标普500 SPY"
            if col.button(
                short_title,
                key=f"home_quick_{idx}",
                disabled=not cfg["dates_valid"],
                width="stretch",
            ):
                _launch_single_asset(st_module, entry.asset_type, entry.symbol)

    st_module.markdown(
        '<div class="qa-section-head"><span class="qa-section-title">按研究任务开始</span><span class="qa-section-sub">分析结果与日期区间会在各步骤间自动衔接</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st_module.columns(3)
    with c1:
        with st_module.container(border=True):
            st_module.markdown("**1. 分析单个资产**")
            st_module.caption(
                "查看收益走势、最大回撤、夏普比率与波动率表现，支持 K 线与风险图表诊断。"
            )
            if st_module.button("进入分析资产", key="home_go_single", width="stretch"):
                queue_transition(st_module, view="single")
                st_module.rerun()
    with c2:
        with st_module.container(border=True):
            st_module.markdown("**2. 比较多个资产**")
            st_module.caption(
                "把不同市场、不同货币的资产折算到同一标尺下，对比累计收益、回撤与相关性。"
            )
            if st_module.button("进入比较资产", key="home_go_compare", width="stretch"):
                queue_transition(st_module, view="compare")
                st_module.rerun()
    with c3:
        with st_module.container(border=True):
            st_module.markdown("**3. 构建投资组合**")
            st_module.caption("在风险与收益间寻找最优配比，运行无前视样本外回测并导出研究报告。")
            if st_module.button("进入构建组合", key="home_go_portfolio", width="stretch"):
                queue_transition(st_module, view="portfolio")
                st_module.rerun()

    # Active session resume strip when at least one workspace has results or pending updates.
    resume_items: list[tuple[str, str, str, str]] = []
    if s_state.get("result") is not None:
        s_res = s_state["result"]
        cagr_val = s_res.metrics.get("cagr")
        cagr_txt = format_percent(cagr_val) if is_valid_number(cagr_val) else "已就绪"
        label = format_asset_display(s_res.asset.asset_type, s_res.asset.canonical_symbol)
        resume_items.append(("single", "分析资产", label, f"CAGR {cagr_txt}"))
    if c_state.get("result") is not None:
        c_res = c_state["result"]
        resume_items.append(
            (
                "compare",
                "比较资产",
                f"{len(c_res.asset_order)} 项资产对比",
                f"{c_res.methodology.get('base_currency', 'CNY')} 口径",
            )
        )
    if p_state.get("result") is not None:
        p_res = p_state["result"]
        resume_items.append(
            (
                "portfolio",
                "构建组合",
                f"{len(p_res.asset_order)} 项资产组合",
                f"策略 {p_res.parameters.get('strategy', '')}",
            )
        )

    if resume_items:
        st_module.markdown(
            '<div class="qa-section-head"><span class="qa-section-title">继续当前研究</span><span class="qa-section-sub">点击直接回到已有结果</span></div>',
            unsafe_allow_html=True,
        )
        r_cols = st_module.columns(len(resume_items))
        for col, (v_key, title, main_txt, meta_txt) in zip(r_cols, resume_items, strict=False):
            with col:
                with st_module.container(border=True):
                    st_module.markdown(
                        f'<div class="qa-resume-title">{html.escape(title)}</div>'
                        f'<div class="qa-resume-main">{html.escape(main_txt)}</div>'
                        f'<div class="qa-resume-meta">{html.escape(meta_txt)}</div>',
                        unsafe_allow_html=True,
                    )
                    if st_module.button(
                        "查看结果 →",
                        key=f"home_resume_{v_key}",
                        width="stretch",
                    ):
                        queue_transition(st_module, view=v_key)
                        st_module.rerun()
