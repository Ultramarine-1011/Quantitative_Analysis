"""数据质量检查。"""

from __future__ import annotations

import pandas as pd

from data_quality import inspect_price_wide


def test_inspect_price_wide_detects_missing_duplicates_and_extreme_returns() -> None:
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"a": [1.0, None, 10.0, 30.0], "b": [2.0, 2.0, 2.0, 2.0]}, index=idx)
    report = inspect_price_wide(prices, extreme_return_threshold=1.0)
    assert report["duplicate_index_count"] == 1
    assert report["missing_ratio_by_column"]["a"] > 0
    assert report["extreme_return_count"] >= 1
    assert report["valid_rows"] == 3

