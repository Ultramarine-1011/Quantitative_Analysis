"""Left navigation rail, shared analysis date window, research basket, and global settings."""

from __future__ import annotations

import datetime as dt
import html
from typing import Any

from asset_resolver import resolve_analysis_date_window
from ui.asset_directory import format_asset_display, parse_legs_loose
from ui.components.header import render_methodology_content
from ui.labels import DATE_PRESETS, NAV_LABELS
from ui.state import (
    NAV_VIEWS,
    WORKSPACE_COMPARE_KEY,
    WORKSPACE_PORTFOLIO_KEY,
    WORKSPACE_SINGLE_KEY,
    get_active_view,
    get_research_basket,
    hydrate_widget_inputs,
    queue_transition,
    set_mirrored_input,
    workspace_state,
    workspace_status,
)
from ui.theme import THEME_MODES, THEME_SIDEBAR_LABELS, normalize_theme_mode


def _sync_rf_from_global(st_module: Any) -> None:
    val = float(st_module.session_state.get("rf_pct_input", 1.6))
    set_mirrored_input(st_module, "compare_rf_pct_input", val)
    set_mirrored_input(st_module, "portfolio_rf_pct_input", val)


def _nav_button_label(st_module: Any, view: str) -> str:
    base = NAV_LABELS[view]
    if view == "home":
        return base
    if view == "single":
        s_state = workspace_state(st_module, WORKSPACE_SINGLE_KEY)
        st_status = workspace_status(s_state)
        res = s_state.get("result")
        if res is not None:
            return f"{base} · {res.asset.canonical_symbol}"
        if st_status == "pending_rerun":
            return f"{base} · 待更新"
        return base
    if view == "compare":
        c_state = workspace_state(st_module, WORKSPACE_COMPARE_KEY)
        st_status = workspace_status(c_state)
        res = c_state.get("result")
        if res is not None:
            return f"{base} · {len(res.asset_order)}项"
        if st_status == "pending_rerun":
            return f"{base} · 待更新"
        cnt = len(parse_legs_loose(str(st_module.session_state.get("compare_assets_input", ""))))
        return f"{base} · {cnt}项" if cnt > 0 else base
    if view == "portfolio":
        p_state = workspace_state(st_module, WORKSPACE_PORTFOLIO_KEY)
        st_status = workspace_status(p_state)
        if p_state.get("result") is not None:
            return f"{base} · 已回测"
        if st_status == "pending_rerun":
            return f"{base} · 待更新"
        return base
    return base


def render_sidebar(st_module: Any) -> dict[str, Any]:
    """Render minimal left navigation rail, shared date window, basket, and global settings."""
    default_start, default_end = resolve_analysis_date_window(None, None)
    active_view = get_active_view(st_module)

    hydrate_widget_inputs(
        st_module,
        {
            "start_date_input": default_start.date(),
            "end_date_input": default_end.date(),
            "rf_pct_input": 1.6,
            "ui_theme_mode": "auto",
            "proxy_url_input": "",
            "force_refresh_input": False,
            "allow_usdt_peg_input": False,
        },
    )

    st_module.sidebar.markdown(
        """
        <div class="qa-sidebar-brand">
          <div class="qa-sidebar-brand-title">Quant Insight</div>
          <div class="qa-sidebar-brand-sub">多资产量化研究工作台</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st_module.sidebar.markdown(
        '<div class="qa-sidebar-section">研究导航</div>', unsafe_allow_html=True
    )
    for view_key in NAV_VIEWS:
        is_active = active_view == view_key
        label = _nav_button_label(st_module, view_key)
        if st_module.sidebar.button(
            label,
            key=f"nav_btn_{view_key}",
            type="primary" if is_active else "secondary",
            width="stretch",
        ):
            if not is_active:
                queue_transition(st_module, view=view_key)
                st_module.rerun()

    st_module.sidebar.markdown(
        '<div class="qa-sidebar-section">分析区间</div>', unsafe_allow_html=True
    )
    preset_cols = st_module.sidebar.columns(len(DATE_PRESETS))
    cur_end_raw = st_module.session_state.get("end_date_input")
    base_end = cur_end_raw if isinstance(cur_end_raw, dt.date) else default_end.date()
    for col, (code, label, days) in zip(preset_cols, DATE_PRESETS, strict=False):
        if col.button(code, key=f"date_preset_{code}", help=f"切换为{label}区间", width="stretch"):
            new_start = base_end - dt.timedelta(days=days)
            queue_transition(
                st_module,
                updates={
                    "start_date_input": new_start,
                    "end_date_input": base_end,
                },
            )
            st_module.rerun()

    start_val = st_module.sidebar.date_input("开始日期", key="start_date_input")
    end_val = st_module.sidebar.date_input("结束日期", key="end_date_input")

    date_error: str | None = None
    start_str: str | None = None
    end_str: str | None = None
    if (
        start_val is None
        or end_val is None
        or isinstance(start_val, tuple)
        or isinstance(end_val, tuple)
    ):
        date_error = "请选择开始和结束日期。"
    elif isinstance(start_val, dt.date) and isinstance(end_val, dt.date):
        if start_val > end_val:
            date_error = "开始日期不能晚于结束日期，请调整后再运行。"
        else:
            start_str = start_val.strftime("%Y-%m-%d")
            end_str = end_val.strftime("%Y-%m-%d")
    else:
        date_error = "请选择开始和结束日期。"

    if date_error:
        st_module.sidebar.error(date_error)

    basket = get_research_basket(st_module)
    if basket:
        st_module.sidebar.markdown(
            '<div class="qa-sidebar-section">当前研究篮子</div>', unsafe_allow_html=True
        )
        chips = "".join(
            f'<span class="qa-chip">{html.escape(format_asset_display(a_type, sym))}</span>'
            for a_type, sym in basket[:8]
        )
        st_module.sidebar.markdown(
            f'<div class="qa-chip-row">{chips}</div>', unsafe_allow_html=True
        )

    cur_rf_pct = float(st_module.session_state.get("rf_pct_input", 1.6))
    cur_theme = normalize_theme_mode(st_module.session_state.get("ui_theme_mode", "auto"))
    cur_cache_lbl = "实时刷新" if st_module.session_state.get("force_refresh_input") else "本地缓存"

    st_module.sidebar.markdown(
        '<div class="qa-sidebar-section">系统与口径</div>', unsafe_allow_html=True
    )
    st_module.sidebar.caption(
        f"Rf {cur_rf_pct:.2f}% · {THEME_SIDEBAR_LABELS.get(cur_theme, cur_theme)} · {cur_cache_lbl}"
    )

    with st_module.sidebar.expander("全局参数与设置", expanded=False):
        rf_pct = float(
            st_module.number_input(
                "无风险年化利率 Rf (%)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                key="rf_pct_input",
                on_change=lambda: _sync_rf_from_global(st_module),
                help="用于夏普比率、Sortino 与切线组合计算；默认 1.6%（可在组合专业参数中同步修改）。",
            )
        )
        if hasattr(st_module, "segmented_control"):
            raw_theme = st_module.segmented_control(
                "界面外观",
                options=THEME_MODES,
                format_func=lambda m: THEME_SIDEBAR_LABELS.get(m, m),
                key="ui_theme_mode",
                width="stretch",
            )
        else:
            raw_theme = st_module.radio(
                "界面外观",
                options=THEME_MODES,
                horizontal=True,
                format_func=lambda m: THEME_SIDEBAR_LABELS.get(m, m),
                key="ui_theme_mode",
            )
        theme_mode = normalize_theme_mode(raw_theme)

        proxy_raw = st_module.text_input(
            "HTTP/SOCKS 代理地址 (可选)",
            placeholder="http://127.0.0.1:7890",
            key="proxy_url_input",
        )
        force_refresh = st_module.checkbox(
            "跳过本地缓存（强制在线刷新）",
            key="force_refresh_input",
        )
        allow_usdt_peg = st_module.checkbox(
            "允许加密货币按 USDT=USD (1:1) 折算",
            key="allow_usdt_peg_input",
            help="默认严格抓取真实 USDT/USD 汇率；若不可得时可勾选近似。",
        )
        if hasattr(st_module, "popover"):
            with st_module.popover("计算口径与代码参考", width="stretch"):
                render_methodology_content(st_module)
        else:
            with st_module.expander("计算口径与代码参考", expanded=False):
                render_methodology_content(st_module)

    return {
        "active_view": active_view,
        "theme_mode": theme_mode,
        "resolved_theme_mode": theme_mode,
        "dates_valid": date_error is None,
        "date_error": date_error,
        "start_str": start_str,
        "end_str": end_str,
        "rf_pct": rf_pct,
        "rf_rate": rf_pct / 100.0,
        "proxy_url": proxy_raw.strip() or None,
        "force_refresh": force_refresh,
        "allow_usdt_peg": allow_usdt_peg,
    }
