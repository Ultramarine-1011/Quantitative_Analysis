"""app.get_quant_metrics：收益率水平序列标签。"""

from __future__ import annotations

import pandas as pd

from app import get_quant_metrics


def test_get_quant_metrics_yield_level_labels() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame(
        {
            "Open": [4.0, 4.1, 4.15],
            "High": [4.05, 4.12, 4.2],
            "Low": [3.95, 4.08, 4.1],
            "Close": [4.0, 4.1, 4.2],
            "Volume": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    df.attrs["series_kind"] = "yield_level"
    metrics = get_quant_metrics(df)
    assert metrics["series_kind"] == "yield_level"
    assert "水平" in metrics["cumulative_return_label"]
    assert "水平" in metrics["cagr_label"]
    assert metrics["metric_disclaimer"]
