"""Visual multi-asset picker with chip removal, quick search, shortcuts, and presets."""

from __future__ import annotations

import html
from typing import Any

from asset_catalog import PRESET_PORTFOLIOS, normalize_symbol_alias
from asset_inputs import parse_asset_token
from ui.asset_directory import (
    POPULAR_ASSETS,
    AssetEntry,
    format_asset_display,
    format_asset_picker_option,
    parse_legs_loose,
    serialize_legs,
)
from ui.state import queue_transition, record_basket_assets, set_mirrored_input


def _find_unparsed_reason(raw_text: str) -> str | None:
    for line in str(raw_text or "").splitlines():
        for part in line.split(";"):
            tok = part.strip()
            if not tok:
                continue
            try:
                parse_asset_token(tok, syntax="web")
            except ValueError as exc:
                return str(exc)
    return None


def render_visual_asset_picker(
    st_module: Any,
    *,
    text_key: str,
    preset_key: str,
    key_prefix: str,
    min_assets: int = 2,
) -> tuple[list[tuple[str, str]], str | None]:
    """Render visual asset chips, searchable add bar, quick chips, and preset selector."""
    if st_module is None:
        return [], None

    raw_text = str(st_module.session_state.get(text_key, "") or "")
    items = parse_legs_loose(raw_text)
    parse_error = _find_unparsed_reason(raw_text)

    existing_keys = {
        (
            a_type,
            (
                normalize_symbol_alias(a_type, sym) if a_type != "global_index" else sym.upper()
            ).upper(),
        )
        for a_type, sym in items
    }

    st_module.markdown(
        f'<div class="qa-picker-label">已选资产（共 {len(items)} 项，至少需要 {min_assets} 项）</div>',
        unsafe_allow_html=True,
    )

    if not items:
        st_module.markdown(
            '<div class="qa-picker-empty">暂未添加资产。可从下方搜索添加常用标的，或直接套用预设组合。</div>',
            unsafe_allow_html=True,
        )
    else:
        cols_per_row = 2
        for start_idx in range(0, len(items), cols_per_row):
            row_items = items[start_idx : start_idx + cols_per_row]
            row_cols = st_module.columns(cols_per_row)
            for offset, (a_type, sym) in enumerate(row_items):
                idx = start_idx + offset
                display_name = format_asset_display(a_type, sym)
                with row_cols[offset]:
                    c_name, c_rm = st_module.columns([3.4, 1.6])
                    c_name.markdown(
                        f'<div class="qa-asset-pill" title="{html.escape(f"{a_type}:{sym}")}">'
                        f'<span class="qa-asset-pill-name">{html.escape(display_name)}</span>'
                        f'<span class="qa-asset-pill-code">{html.escape(a_type)}:{html.escape(sym)}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if c_rm.button(
                        "移除",
                        key=f"{key_prefix}_rm_{idx}",
                        help=f"移除 {display_name}",
                        width="stretch",
                    ):
                        remaining = [pair for j, pair in enumerate(items) if j != idx]
                        queue_transition(
                            st_module,
                            updates={text_key: serialize_legs(remaining)},
                        )
                        st_module.rerun()

    # Searchable dropdown + preset row
    available_entries: list[AssetEntry] = [
        entry
        for entry in POPULAR_ASSETS
        if (
            entry.asset_type,
            (
                normalize_symbol_alias(entry.asset_type, entry.symbol)
                if entry.asset_type != "global_index"
                else entry.symbol.upper()
            ).upper(),
        )
        not in existing_keys
        and entry.asset_type != "bond_us_yield"
    ]

    col_search, col_add = st_module.columns([3.2, 1.2])
    with col_search:
        if available_entries:
            picked_entry = st_module.selectbox(
                "搜索并添加资产（中文名或代码）",
                options=available_entries,
                format_func=format_asset_picker_option,
                key=f"{key_prefix}_add_select",
                label_visibility="collapsed",
            )
        else:
            picked_entry = None
            st_module.selectbox(
                "搜索并添加资产",
                options=("常用资产已全部加入",),
                disabled=True,
                key=f"{key_prefix}_add_select_empty",
                label_visibility="collapsed",
            )
    with col_add:
        if st_module.button(
            "+ 添加资产",
            key=f"{key_prefix}_add_btn",
            disabled=picked_entry is None,
            width="stretch",
        ):
            if picked_entry is not None:
                updated = [*items, (picked_entry.asset_type, picked_entry.symbol)]
                record_basket_assets(st_module, [(picked_entry.asset_type, picked_entry.symbol)])
                queue_transition(
                    st_module,
                    updates={text_key: serialize_legs(updated)},
                )
                st_module.rerun()

    # Quick shortcut chips: prioritize diverse asset classes (equity, bond, gold, US)
    if available_entries:
        preferred_order = (
            "etf:510500",
            "etf:511010",
            "etf:518880",
            "us_stock:AAPL",
            "us_stock:SPY",
            "gold:GC",
            "etf:510300",
            "hk_stock:00700",
        )
        priority_map = {tok: rank for rank, tok in enumerate(preferred_order)}
        quick_candidates = sorted(
            available_entries,
            key=lambda e: (
                priority_map.get(e.token, len(preferred_order)),
                POPULAR_ASSETS.index(e),
            ),
        )[:4]
        q_cols = st_module.columns(len(quick_candidates))
        for idx, entry in enumerate(quick_candidates):
            short_base = entry.name.split(" (", 1)[0]
            if len(short_base) <= 7:
                short_label = f"+ {short_base}"
            else:
                first_word = short_base.split(" ", 1)[0]
                short_label = (
                    f"+ {first_word}" if len(first_word) >= 4 else f"+ {first_word} {entry.symbol}"
                )
            if q_cols[idx].button(
                short_label,
                key=f"{key_prefix}_quick_{entry.asset_type}_{entry.symbol}",
                width="stretch",
            ):
                updated = [*items, (entry.asset_type, entry.symbol)]
                record_basket_assets(st_module, [(entry.asset_type, entry.symbol)])
                queue_transition(
                    st_module,
                    updates={text_key: serialize_legs(updated)},
                )
                st_module.rerun()

    def _on_preset_change() -> None:
        preset_name = str(st_module.session_state.get(preset_key, ""))
        rows = PRESET_PORTFOLIOS.get(preset_name)
        if rows:
            set_mirrored_input(st_module, text_key, "\n".join(rows))

    st_module.selectbox(
        "或套用预设组合模板",
        options=tuple(PRESET_PORTFOLIOS.keys()),
        key=preset_key,
        on_change=_on_preset_change,
        help="快速加载预设跨市场标的组合",
    )

    if parse_error:
        st_module.caption(f"批量列表存在待修正格式：{parse_error}")

    return items, parse_error
