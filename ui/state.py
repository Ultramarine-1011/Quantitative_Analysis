"""Session state management, workspace transitions, and cache-invalidation helpers."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from asset_catalog import normalize_symbol_alias
from ui.errors import UIError, format_ui_error_markdown

WORKSPACE_SINGLE_KEY = "_workspace_single"
WORKSPACE_COMPARE_KEY = "_workspace_compare"
WORKSPACE_PORTFOLIO_KEY = "_workspace_portfolio"
LEGACY_RESULTS_KEY = "_analysis_results"

NAV_VIEWS: tuple[str, ...] = ("home", "single", "compare", "portfolio")
ACTIVE_VIEW_KEY = "active_view"
RESEARCH_BASKET_KEY = "_research_basket"
PENDING_TRANSITION_KEY = "_pending_transition"
INPUT_MIRROR_KEY = "_input_mirror"
_AUTO_RUN_PREFIX = "_auto_run_"

_META_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("date_window", "分析区间"),
    ("rf_pct", "无风险利率"),
    ("adjust", "复权方式"),
    ("base_currency", "估值货币"),
    ("return_convention", "收益率口径"),
    ("calendar_mode", "日历对齐"),
    ("cov_method", "协方差估计"),
    ("strategy", "优化策略"),
    ("validation_mode", "回测方式"),
    ("rebalance_freq", "调仓频率"),
    ("tx_cost_bps", "交易成本"),
    ("rolling_window", "滚动窗口"),
    ("train_ratio", "训练比例"),
    ("train_window", "训练窗口"),
    ("min_weight", "权重下界"),
    ("max_weight", "权重上界"),
    ("benchmark", "对比基准"),
    ("assets_label", "标的列表"),
)


def _resolve_session_map(st_or_state: Any) -> MutableMapping[str, Any]:
    if hasattr(st_or_state, "session_state"):
        return st_or_state.session_state
    return st_or_state


def _default_workspace_dict() -> dict[str, Any]:
    return {
        "signature": None,
        "fetch_signature": None,
        "result": None,
        "result_version": 0,
        "figures": None,
        "figures_cache_key": None,
        "figures_requested": None,
        "skipped": None,
        "exports": None,
        "exports_cache_key": None,
        "error": None,
        "error_info": None,
        "editor_open": False,
        "last_meta": None,
        "last_summary": None,
        "prev_summary": None,
        "change_notice": None,
    }


def workspace_state(st_or_state: Any, key: str) -> dict[str, Any]:
    """Retrieve or initialize an isolated workspace dictionary in session state."""
    session = _resolve_session_map(st_or_state)
    state = session.get(key)
    if not isinstance(state, dict):
        state = _default_workspace_dict()
        session[key] = state
        return state
    for field_key, default_val in _default_workspace_dict().items():
        if field_key not in state:
            state[field_key] = default_val
    return state


def queue_transition(
    st_or_state: Any,
    *,
    view: str | None = None,
    updates: Mapping[str, Any] | None = None,
    auto_run_view: str | None = None,
) -> None:
    """Schedule view navigation and session state updates before next widget render."""
    session = _resolve_session_map(st_or_state)
    pending = session.get(PENDING_TRANSITION_KEY)
    if not isinstance(pending, dict):
        pending = {"updates": {}}
    if view is not None and view in NAV_VIEWS:
        pending["view"] = view
    if auto_run_view is not None and auto_run_view in NAV_VIEWS:
        pending["auto_run_view"] = auto_run_view
    if updates:
        merged = dict(pending.get("updates") or {})
        merged.update(updates)
        pending["updates"] = merged
    session[PENDING_TRANSITION_KEY] = pending


def apply_pending_transitions(st_or_state: Any) -> None:
    """Apply queued state transitions at the very top of ``app.main()`` before widgets mount."""
    session = _resolve_session_map(st_or_state)
    if ACTIVE_VIEW_KEY not in session or session.get(ACTIVE_VIEW_KEY) not in NAV_VIEWS:
        session[ACTIVE_VIEW_KEY] = "home"

    pending = session.pop(PENDING_TRANSITION_KEY, None)
    if not isinstance(pending, Mapping):
        return

    view = pending.get("view")
    if isinstance(view, str) and view in NAV_VIEWS:
        session[ACTIVE_VIEW_KEY] = view

    updates = pending.get("updates")
    if isinstance(updates, Mapping):
        mirror = session.get(INPUT_MIRROR_KEY)
        if not isinstance(mirror, dict):
            mirror = {}
            session[INPUT_MIRROR_KEY] = mirror
        for key, value in updates.items():
            session[key] = value
            mirror[key] = value

    auto_run_view = pending.get("auto_run_view")
    if isinstance(auto_run_view, str) and auto_run_view in NAV_VIEWS:
        session[f"{_AUTO_RUN_PREFIX}{auto_run_view}"] = True


def _get_input_mirror(session: MutableMapping[str, Any]) -> dict[str, Any]:
    mirror = session.get(INPUT_MIRROR_KEY)
    if not isinstance(mirror, dict):
        mirror = {}
        session[INPUT_MIRROR_KEY] = mirror
    return mirror


def hydrate_widget_inputs(st_or_state: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure ``defaults`` exist in both ``session_state`` and persistent mirror before widgets mount."""
    session = _resolve_session_map(st_or_state)
    mirror = _get_input_mirror(session)
    resolved: dict[str, Any] = {}
    for key, default_val in defaults.items():
        if key in session:
            mirror[key] = session[key]
        elif key in mirror:
            session[key] = mirror[key]
        else:
            mirror[key] = default_val
            session[key] = default_val
        resolved[key] = session[key]
    return resolved


def save_widget_inputs(st_or_state: Any, keys: Sequence[str]) -> None:
    """Persist mounted widget values into the non-widget mirror."""
    session = _resolve_session_map(st_or_state)
    mirror = _get_input_mirror(session)
    for key in keys:
        if key in session:
            mirror[key] = session[key]


def set_mirrored_input(st_or_state: Any, key: str, value: Any) -> None:
    """Update a key in both ``session_state`` and the persistent mirror."""
    session = _resolve_session_map(st_or_state)
    session[key] = value
    _get_input_mirror(session)[key] = value


def read_persisted_inputs(st_or_state: Any, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Read persisted workspace inputs without requiring widgets to be mounted."""
    session = _resolve_session_map(st_or_state)
    mirror = _get_input_mirror(session)
    out: dict[str, Any] = {}
    for key, default_val in defaults.items():
        if key in session:
            out[key] = session[key]
        elif key in mirror:
            out[key] = mirror[key]
        else:
            out[key] = default_val
    return out


def consume_auto_run(st_or_state: Any, view: str) -> bool:
    """Return True once when a view has a queued auto-run request."""
    session = _resolve_session_map(st_or_state)
    key = f"{_AUTO_RUN_PREFIX}{view}"
    if session.get(key):
        session[key] = False
        return True
    return False


def get_active_view(st_or_state: Any) -> str:
    """Return validated active workspace key (``home``, ``single``, ``compare``, or ``portfolio``)."""
    session = _resolve_session_map(st_or_state)
    view = str(session.get(ACTIVE_VIEW_KEY, "home") or "home").strip().lower()
    return view if view in NAV_VIEWS else "home"


def get_research_basket(st_or_state: Any) -> list[tuple[str, str]]:
    """Return deduplicated list of ``(asset_type, symbol)`` currently in the research basket."""
    session = _resolve_session_map(st_or_state)
    raw = session.get(RESEARCH_BASKET_KEY)
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        a_type = str(item[0] or "").strip().lower()
        sym = str(item[1] or "").strip()
        if not a_type or not sym:
            continue
        canon = normalize_symbol_alias(a_type, sym) if a_type != "global_index" else sym.upper()
        key = (a_type, canon.upper())
        if key not in seen:
            seen.add(key)
            out.append((a_type, canon))
    return out


def record_basket_assets(
    st_or_state: Any,
    items: Sequence[tuple[str, str]],
    *,
    replace: bool = False,
) -> list[tuple[str, str]]:
    """Add or replace assets in the cross-workspace research basket."""
    session = _resolve_session_map(st_or_state)
    current = [] if replace else get_research_basket(session)
    seen: set[tuple[str, str]] = {
        (
            a_type,
            (
                normalize_symbol_alias(a_type, sym) if a_type != "global_index" else sym.upper()
            ).upper(),
        )
        for a_type, sym in current
    }
    for a_type_raw, sym_raw in items:
        a_type = str(a_type_raw or "").strip().lower()
        sym = str(sym_raw or "").strip()
        if not a_type or not sym:
            continue
        canon = normalize_symbol_alias(a_type, sym) if a_type != "global_index" else sym.upper()
        key = (a_type, canon.upper())
        if key not in seen:
            seen.add(key)
            current.append((a_type, canon))
    session[RESEARCH_BASKET_KEY] = current
    return current


def describe_meta_changes(
    prev_meta: Mapping[str, Any] | None,
    cur_meta: Mapping[str, Any] | None,
) -> list[str]:
    """Produce concise Chinese change labels comparing previous and current workspace meta."""
    if not isinstance(prev_meta, Mapping) or not isinstance(cur_meta, Mapping):
        return []
    changes: list[str] = []
    for field_key, label in _META_FIELD_LABELS:
        old_val = prev_meta.get(field_key)
        new_val = cur_meta.get(field_key)
        if old_val != new_val:
            if old_val is not None and new_val is not None:
                changes.append(f"{label}: {old_val} → {new_val}")
            else:
                changes.append(label)
    return changes


def reset_workspace_outputs(state: dict[str, Any], *, keep_error: bool = False) -> None:
    """Clear derived results, figures, and exports and bump ``result_version``."""
    state["result"] = None
    state["fetch_signature"] = None
    state["result_version"] = int(state.get("result_version", 0)) + 1
    state["figures"] = None
    state["figures_cache_key"] = None
    state["figures_requested"] = None
    state["skipped"] = None
    state["exports"] = None
    state["exports_cache_key"] = None
    if not keep_error:
        state["error"] = None
        state["error_info"] = None


def invalidate_if_signature_changed(
    state: dict[str, Any],
    cur_sig: Any,
    *,
    cur_meta: Mapping[str, Any] | None = None,
) -> bool:
    """Invalidate stale workspace outputs when input parameters differ from ``state['signature']``."""
    has_payload = (
        state.get("result") is not None
        or state.get("error") is not None
        or state.get("error_info") is not None
    )
    if has_payload and state.get("signature") != cur_sig:
        had_result = state.get("result") is not None
        prev_meta = state.get("last_meta")
        prev_summary = state.get("last_summary")
        reset_workspace_outputs(state, keep_error=False)
        if had_result and isinstance(cur_meta, Mapping) and isinstance(prev_meta, Mapping):
            same_scope = prev_meta.get("scope_key") == cur_meta.get("scope_key")
            changes = describe_meta_changes(prev_meta, cur_meta)
            if same_scope and isinstance(prev_summary, Mapping):
                state["prev_summary"] = dict(prev_summary)
                if changes:
                    state["change_notice"] = (
                        f"已变更（{'；'.join(changes)}），结果已自动清空。点击「重新计算」更新并对比变化。"
                    )
                else:
                    state["change_notice"] = (
                        "参数已更新，结果已自动清空。点击「重新计算」更新结果。"
                    )
            else:
                state["prev_summary"] = None
                state["change_notice"] = "分析对象已切换，请点击「重新计算」获取新结果。"
        else:
            state["prev_summary"] = None
            state["change_notice"] = None
        return True

    # Keep change_notice accurate across multiple sequential edits before re-running.
    if (
        state.get("result") is None
        and state.get("change_notice")
        and isinstance(cur_meta, Mapping)
        and isinstance(state.get("last_meta"), Mapping)
    ):
        prev_meta = state["last_meta"]
        if prev_meta.get("scope_key") == cur_meta.get("scope_key"):
            changes = describe_meta_changes(prev_meta, cur_meta)
            if changes:
                state["change_notice"] = (
                    f"已变更（{'；'.join(changes)}），点击「重新计算」更新并对比变化。"
                )
        else:
            state["prev_summary"] = None
            state["change_notice"] = "分析对象已切换，请点击「重新计算」获取新结果。"
    return False


def store_workspace_result(
    state: dict[str, Any],
    *,
    signature: Any,
    result: Any,
    fetch_signature: Any = None,
    meta: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> None:
    """Persist a newly computed result and reset downstream figure/export memoization."""
    prev_meta = state.get("last_meta")
    prev_sum = state.get("prev_summary")
    reset_workspace_outputs(state, keep_error=False)
    state["signature"] = signature
    state["fetch_signature"] = fetch_signature
    state["result"] = result

    keep_prev = (
        isinstance(prev_sum, Mapping)
        and isinstance(meta, Mapping)
        and prev_sum.get("scope_key") == meta.get("scope_key")
    )
    if keep_prev:
        updated_prev = dict(prev_sum)
        updated_prev["changed_items"] = describe_meta_changes(prev_meta, meta)
        state["prev_summary"] = updated_prev
    else:
        state["prev_summary"] = None

    state["last_meta"] = dict(meta) if isinstance(meta, Mapping) else None
    state["last_summary"] = dict(summary) if isinstance(summary, Mapping) else None
    state["change_notice"] = None
    state["editor_open"] = False


def store_workspace_error(
    state: dict[str, Any],
    *,
    signature: Any,
    error_info: UIError | str,
) -> None:
    """Discard any previous result and store the active error payload."""
    reset_workspace_outputs(state, keep_error=False)
    state["signature"] = signature
    state["change_notice"] = None
    if isinstance(error_info, UIError):
        state["error_info"] = error_info
        state["error"] = format_ui_error_markdown(error_info)
    else:
        state["error_info"] = None
        state["error"] = str(error_info)


def workspace_status(state: Mapping[str, Any], *, inputs_complete: bool = True) -> str:
    """Classify current workspace state for status pills and sidebar badges."""
    err_info = state.get("error_info")
    if isinstance(err_info, UIError):
        if err_info.category == "empty_data":
            return "empty_data"
        return "error"
    if state.get("error"):
        return "error"
    if state.get("result") is not None:
        return "ready"
    if state.get("change_notice"):
        return "pending_rerun"
    if not inputs_complete:
        return "incomplete"
    return "idle"


def sync_legacy_analysis_results(st_or_state: Any) -> dict[str, Any]:
    """Keep ``session_state['_analysis_results']`` synchronized across workspaces."""
    session = _resolve_session_map(st_or_state)
    summary: dict[str, Any] = {}
    s_state = session.get(WORKSPACE_SINGLE_KEY) or {}
    if s_state.get("result") is not None:
        summary["single"] = s_state["result"]
    if s_state.get("norm_fig") is not None:
        summary["normalize"] = s_state["norm_fig"]
    if s_state.get("gbm_result") is not None:
        summary["gbm"] = s_state["gbm_result"]
    c_state = session.get(WORKSPACE_COMPARE_KEY) or {}
    if c_state.get("result") is not None:
        summary["compare"] = c_state["result"]
    p_state = session.get(WORKSPACE_PORTFOLIO_KEY) or {}
    if p_state.get("result") is not None:
        summary["frontier"] = p_state["result"]
    session[LEGACY_RESULTS_KEY] = summary
    return summary
