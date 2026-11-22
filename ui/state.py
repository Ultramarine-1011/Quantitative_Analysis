"""Session state management and cache-invalidation helpers for all workspaces."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from ui.errors import UIError, format_ui_error_markdown

WORKSPACE_SINGLE_KEY = "_workspace_single"
WORKSPACE_COMPARE_KEY = "_workspace_compare"
WORKSPACE_PORTFOLIO_KEY = "_workspace_portfolio"
LEGACY_RESULTS_KEY = "_analysis_results"


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


def invalidate_if_signature_changed(state: dict[str, Any], cur_sig: Any) -> bool:
    """Invalidate stale workspace outputs when input parameters differ from ``state['signature']``."""
    has_payload = (
        state.get("result") is not None
        or state.get("error") is not None
        or state.get("error_info") is not None
    )
    if has_payload and state.get("signature") != cur_sig:
        reset_workspace_outputs(state, keep_error=False)
        return True
    return False


def store_workspace_result(
    state: dict[str, Any],
    *,
    signature: Any,
    result: Any,
    fetch_signature: Any = None,
) -> None:
    """Persist a newly computed result and reset downstream figure/export memoization."""
    reset_workspace_outputs(state, keep_error=False)
    state["signature"] = signature
    state["fetch_signature"] = fetch_signature
    state["result"] = result


def store_workspace_error(
    state: dict[str, Any],
    *,
    signature: Any,
    error_info: UIError | str,
) -> None:
    """Discard any previous result and store the active error payload."""
    reset_workspace_outputs(state, keep_error=False)
    state["signature"] = signature
    if isinstance(error_info, UIError):
        state["error_info"] = error_info
        state["error"] = format_ui_error_markdown(error_info)
    else:
        state["error_info"] = None
        state["error"] = str(error_info)


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
