"""相关性热力图：Plotly 不接受 textfont 二维颜色矩阵，用 annotations 兜底后应可构建。"""

from __future__ import annotations

import pandas as pd

from app import build_tab3_correlation_heatmap


def test_build_tab3_correlation_heatmap_creates_figure_with_annotations() -> None:
    corr = pd.DataFrame(
        [[1.0, 0.3, -0.8], [0.3, 1.0, 0.1], [-0.8, 0.1, 1.0]],
        columns=["etf:510300", "etf:510500", "us_stock:AAPL"],
        index=["etf:510300", "etf:510500", "us_stock:AAPL"],
    )
    fig = build_tab3_correlation_heatmap(corr)
    assert fig.data
    assert len(fig.layout.annotations) == 9
    assert fig.data[0].type == "heatmap"
