from dataclasses import replace

import pytest

from contracts import AnalysisRequest, AssetSpec, CompareRequest
from services import compare_assets, fetch_single_asset


def test_actual_adjustment_and_conflicting_volume_units_are_visible(sample_ohlcv):
    req = AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-31", adjust="hfq")
    frame = sample_ohlcv.copy()
    frame.attrs.update(adjust="", data_source="eastmoney_direct", volume_unit="shares_or_units")
    result = fetch_single_asset(req, raw_df=frame)
    assert result.status == "degraded"
    assert {"volume_unit_mismatch", "adjustment_mismatch"} <= set(result.warnings)
    assert result.asset.adjust == ""
    assert result.asset.volume_unit == "shares_or_units"
    del frame.attrs["volume_unit"]
    expected = fetch_single_asset(replace(req, adjust=""), raw_df=frame)
    assert expected.asset.volume_unit == "lots_100"
    assert expected.asset.to_dict()["volume_unit"] == "lots_100"


def test_compare_degrades_on_malformed_futures_candle(sample_ohlcv):
    def loader(asset_type, symbol, **kwargs):
        frame = sample_ohlcv.copy()
        frame.attrs.update(asset_type=asset_type, symbol=symbol, currency="USD")
        if asset_type == "gold":
            frame.iloc[2, frame.columns.get_loc("High")] = 0
        return frame

    result = compare_assets(
        CompareRequest(
            (AssetSpec("gold", "GC"), AssetSpec("us_stock", "SPY")),
            "2024-01-02",
            "2024-05-31",
            base_currency="LOCAL",
        ),
        loader=loader,
    )
    assert result.status == "degraded"
    assert result.quality["gold:GC"].ohlc_anomaly_rows == 1


@pytest.mark.parametrize("symbol", ["DXY", "美元指数", "DINIW", "UDI"])
def test_dollar_index_cannot_be_converted_like_usd(sample_ohlcv, symbol):
    frame = sample_ohlcv.copy()
    frame.attrs.update(asset_type="global_index", symbol=symbol, currency="USD")
    req = AnalysisRequest("global_index", symbol, "2024-01-02", "2024-05-31", base_currency="USD")
    with pytest.raises(ValueError, match="index_points_not_currency"):
        fetch_single_asset(req, raw_df=frame)
