"""研究和回测结果的 Markdown 报告导出。"""

from __future__ import annotations

from typing import Any


def build_markdown_report(result: dict[str, Any]) -> str:
    """把指标、假设和附加说明组织成 Markdown 文本。"""
    title = str(result.get("title") or "回测报告")
    metrics = dict(result.get("metrics") or {})
    assumptions = list(result.get("assumptions") or [])
    lines = ["# %s" % title, "", "## 核心指标", ""]
    if metrics:
        for key, value in metrics.items():
            lines.append("- `%s`: %s" % (key, value))
    else:
        lines.append("- 暂无指标")
    lines.extend(["", "## 关键假设", ""])
    if assumptions:
        for item in assumptions:
            lines.append("- %s" % item)
    else:
        lines.append("- 未提供额外假设")
    notes = result.get("notes")
    if notes:
        lines.extend(["", "## 说明", "", str(notes)])
    return "\n".join(lines) + "\n"


__all__ = ["build_markdown_report"]

