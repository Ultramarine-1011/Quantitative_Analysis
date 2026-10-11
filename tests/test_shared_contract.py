"""Shared validation and feature calculations retain the daily-data contract."""

import numpy as np
import pandas as pd
import pytest

from data_fetcher import AKShareFetcher
from feature_engineering import OHLCVFeatureEngineer
from ohlcv import normalize_ohlcv, prepare_ohlcv, validate_ohlcv
from visualizer import MplfinanceVisualizer


def test_cleaning_preserves_features_metadata_and_float_dtypes(sample_ohlcv):
    frame = sample_ohlcv.astype(str)
    frame["custom"] = np.arange(len(frame))
    frame.loc[frame.index[1], ["Open", "High", "Low", "Close", "Volume"]] = None
    before = frame.copy(deep=True)
    clean = prepare_ohlcv(frame)
    assert (clean.iloc[1][:5] == clean.iloc[0][:5]).all()
    assert (clean.iloc[:, :5].dtypes == "float64").all()
    assert clean["custom"].equals(before["custom"])
    assert clean.attrs == before.attrs
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(OHLCVFeatureEngineer().validate_input(frame), clean)
    pd.testing.assert_frame_equal(MplfinanceVisualizer().validate_input(frame), clean)


def test_duplicate_dates_keep_last_and_timezone_is_removed(sample_ohlcv):
    raw = sample_ohlcv.iloc[:3].copy()
    duplicate = raw.iloc[[0]].assign(Close=123.0)
    raw = pd.concat([raw, duplicate]).iloc[::-1].iloc[::-1]
    raw.index = raw.index.tz_localize("Asia/Shanghai")
    clean = normalize_ohlcv(raw, {})
    assert len(clean) == 3
    assert clean.iloc[0]["Close"] == 123.0
    assert clean.index.tz is None
    assert str(clean.index.dtype) == "datetime64[ns]"


def test_infinite_and_bad_values_are_forward_filled(sample_ohlcv):
    raw = sample_ohlcv.iloc[:4].astype(object)
    raw.iloc[1, 3] = "inf"
    raw.iloc[2, 3] = "bad"
    out = normalize_ohlcv(raw, {})
    assert out["Close"].iloc[1] == out["Close"].iloc[0]
    assert out["Close"].iloc[2] == out["Close"].iloc[0]
    fetcher = AKShareFetcher(max_retries=1, use_system_proxy=True)
    pd.testing.assert_frame_equal(fetcher._normalize_ohlcv(raw, {}), out)


def test_leading_missing_rows_drop_only_at_vendor_boundary(sample_ohlcv):
    raw = sample_ohlcv.copy()
    raw.iloc[0, 3] = np.nan
    assert len(normalize_ohlcv(raw, {})) == len(raw) - 1
    with pytest.raises(ValueError, match="NaN"):
        prepare_ohlcv(raw)


def test_rejects_invalid_index_and_mapping_collisions(sample_ohlcv):
    raw = sample_ohlcv.iloc[:3].copy()
    raw.index = pd.DatetimeIndex([pd.NaT, *raw.index[1:]])
    with pytest.raises(ValueError, match="invalid datetime"):
        prepare_ohlcv(raw)
    raw = sample_ohlcv.assign(close=1.0)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_ohlcv(raw, {"close": "Close"})


def test_contract_rejects_infinity_and_non_float64(sample_ohlcv):
    with pytest.raises(TypeError, match="float64"):
        validate_ohlcv(sample_ohlcv.astype("float32"))
    frame = sample_ohlcv.copy()
    frame.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_ohlcv(frame)


def test_features_match_independent_formulas_and_are_idempotent(sample_ohlcv):
    engineer = OHLCVFeatureEngineer()
    out = engineer.transform(sample_ohlcv)
    price = sample_ohlcv["Close"]
    expected_vol = np.log(price / price.shift()).rolling(20).std() * np.sqrt(252)
    pd.testing.assert_series_equal(out["sma_20"], price.rolling(20).mean(), check_names=False)
    pd.testing.assert_series_equal(out["volatility_20"], expected_vol, check_names=False)
    pd.testing.assert_series_equal(
        out["bollinger_upper_20"],
        price.rolling(20).mean() + 2 * price.rolling(20).std(),
        check_names=False,
    )
    pd.testing.assert_series_equal(out["drawdown"], 1 - price / price.cummax(), check_names=False)
    assert out.attrs["max_drawdown"] == pytest.approx((1 - price / price.cummax()).max())
    assert out.attrs["symbol"] == sample_ohlcv.attrs["symbol"]
    pd.testing.assert_frame_equal(engineer.transform(out), out)
    assert list(sample_ohlcv.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_batch_validates_only_once_and_custom_windows_match(sample_ohlcv, monkeypatch):
    engineer = OHLCVFeatureEngineer()
    calls = []
    original = engineer.validate_input

    def validate(frame):
        calls.append(1)
        return original(frame)

    monkeypatch.setattr(engineer, "validate_input", validate)
    out = engineer.transform(
        sample_ohlcv,
        sma_short_window=5,
        sma_long_window=10,
        volatility_window=7,
        bollinger_window=8,
        annualize=False,
    )
    assert len(calls) == 1
    pd.testing.assert_series_equal(
        out["volatility_7"], engineer.compute_volatility(sample_ohlcv, window=7, annualize=False)
    )
    pd.testing.assert_frame_equal(
        out[["bollinger_upper_8", "bollinger_lower_8"]],
        engineer.compute_bollinger_bands(sample_ohlcv, window=8),
    )


@pytest.mark.parametrize("value", [0, -1, True, 2.5, np.inf, np.nan])
def test_invalid_window_parameters_raise_value_error(sample_ohlcv, value):
    with pytest.raises(ValueError, match="positive integer"):
        OHLCVFeatureEngineer().compute_sma(sample_ohlcv, window=value)


def test_logs_reject_nonpositive_prices(sample_ohlcv):
    frame = sample_ohlcv.copy()
    frame.iloc[2, 3] = 0
    with pytest.raises(ValueError, match="positive prices"):
        OHLCVFeatureEngineer().compute_returns(frame, log=True)
