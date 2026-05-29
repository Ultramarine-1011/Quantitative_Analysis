"""行情宽表数据质量检查。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def inspect_price_wide(
    close_wide: pd.DataFrame,
    *,
    extreme_return_threshold: float = 0.5,
) -> dict[str, Any]:
    """检查缺失、重复日期、极端收益和有效样本数量。"""
    if not isinstance(close_wide, pd.DataFrame) or close_wide.empty:
        raise ValueError("close_wide must be a non-empty DataFrame.")
    work = close_wide.apply(pd.to_numeric, errors="coerce")
    duplicate_count = int(work.index.duplicated().sum())
    work_no_dup = work.loc[~work.index.duplicated(keep="last")].sort_index()
    returns = work_no_dup.pct_change(fill_method=None)
    extreme = returns.abs() > float(extreme_return_threshold)
    return {
        "row_count": int(work.shape[0]),
        "valid_rows": int(work_no_dup.dropna(how="all").shape[0]),
        "column_count": int(work.shape[1]),
        "duplicate_index_count": duplicate_count,
        "missing_ratio_by_column": work.isna().mean().to_dict(),
        "extreme_return_count": int(extreme.sum().sum()),
        "first_date": None if work_no_dup.empty else str(work_no_dup.index.min().date()),
        "last_date": None if work_no_dup.empty else str(work_no_dup.index.max().date()),
    }


__all__ = ["inspect_price_wide"]

