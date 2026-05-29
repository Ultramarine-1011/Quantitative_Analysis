"""回测/因子报告导出。"""

from __future__ import annotations

from reporting import build_markdown_report


def test_markdown_report_contains_metrics_and_assumptions() -> None:
    text = build_markdown_report(
        {
            "title": "组合回测报告",
            "metrics": {"max_drawdown": 0.12, "sharpe_ratio": 1.1},
            "assumptions": ["手续费 10bps", "无税费"],
        }
    )
    assert "# 组合回测报告" in text
    assert "max_drawdown" in text
    assert "手续费 10bps" in text

