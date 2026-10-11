"""Deterministic daily OHLCV fixtures; tests never need vendor services."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def sample_ohlcv():
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0004, 0.01, 120)))
    frame = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), 1000.0),
        },
        index=pd.date_range("2024-01-02", periods=len(close), freq="B"),
    )
    frame.index = frame.index.as_unit("ns")
    frame.index.name = "Date"
    frame.attrs.update(symbol="510300", asset_type="etf", series_kind="price")
    return frame
