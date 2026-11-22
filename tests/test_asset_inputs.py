import pytest

from asset_inputs import parse_asset_list, parse_weights
from contracts import CompareRequest


@pytest.mark.parametrize("syntax", ["cli", "web", "legacy"])
def test_alias_order_and_duplicate_policy(syntax):
    specs = parse_asset_list(["ETF,sh510300;hk_stock|700\netf,510300.SH"], syntax=syntax)
    ids = [s.canonical_symbol for s in specs]
    if syntax == "cli":
        assert ids == ["510300", "00700", "510300"]
        with pytest.raises(ValueError, match="重复"):
            CompareRequest(specs, "2024-01-01", "2024-02-01")
    else:
        assert ids == ["510300", "00700"]


@pytest.mark.parametrize("web", [True, False])
def test_weight_separators_and_finite_numbers(web):
    assert parse_weights(".1; .2, .7", web=web) == (0.1, 0.2, 0.7)
    with pytest.raises(ValueError):
        parse_weights("1 inf", web=web)
