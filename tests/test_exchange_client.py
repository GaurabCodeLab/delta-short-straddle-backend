import pytest
from datetime import datetime, timezone

from src.exchange_client import DeltaExchangeClient, ExchangeClientError
from src.models import OptionLeg


def test_to_dt_variants():
    # datetime passes through
    now = datetime.now(timezone.utc)
    assert DeltaExchangeClient._to_dt(now) == now

    # timestamp numeric
    ts = 1690000000
    dt = DeltaExchangeClient._to_dt(ts)
    assert isinstance(dt, datetime)

    # ISO string
    s = now.isoformat()
    dt2 = DeltaExchangeClient._to_dt(s)
    assert isinstance(dt2, datetime)

    # unsupported string should raise
    with pytest.raises(ExchangeClientError):
        DeltaExchangeClient._to_dt("not-a-date")


class DummyClient:
    # reuse DeltaExchangeClient methods without running __init__
    pass


@pytest.mark.asyncio
async def test_parse_option_positions_parses_valid_positions():
    client = DummyClient()

    async def get_open_positions_raw():
        return [
            {
                "product_id": 1,
                "size": "-1",
                "entry_price": "1.0",
                "product": {
                    "contract_type": "OPTION_CALL",
                    "strike_price": "100.0",
                    "settlement_time": datetime.now(timezone.utc).isoformat(),
                    "contract_value": "1.0",
                    "symbol": "OPT-1",
                },
            }
        ]

    # bind method
    client.get_open_positions_raw = get_open_positions_raw

    # attach _to_dt helper
    client._to_dt = DeltaExchangeClient._to_dt

    # attach parse_option_positions method from class
    client.parse_option_positions = DeltaExchangeClient.parse_option_positions.__get__(client, client.__class__)

    legs = await client.parse_option_positions()
    assert isinstance(legs, list)
    assert len(legs) == 1
    leg = legs[0]
    assert isinstance(leg, OptionLeg)
    assert leg.product_id == 1
    assert leg.option_type in ("call", "put")


@pytest.mark.asyncio
async def test_get_product_by_symbol_fallback_uses_get_products_raw():
    class ClientFallback(DeltaExchangeClient):
        def __init__(self):
            # do not call super
            pass

        async def _call(self, method_name: str, **kwargs):
            # force initial attempt to fail so fallback runs
            raise Exception("no direct query")

        async def get_products_raw(self):
            return [
                {"id": 1, "symbol": "FOO"},
                {"id": 2, "symbol": "BAR"},
            ]

    client = ClientFallback()
    # bind method from class
    client.get_product_by_symbol = DeltaExchangeClient.get_product_by_symbol.__get__(client, client.__class__)
    res = await client.get_product_by_symbol("BAR")
    assert res is not None
    assert str(res.get("symbol")) == "BAR"
