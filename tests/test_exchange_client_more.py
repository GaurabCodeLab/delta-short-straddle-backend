import asyncio
from types import SimpleNamespace
import pytest

from src.exchange_client import DeltaExchangeClient, ExchangeClientError


@pytest.mark.asyncio
async def test_call_missing_method_raises():
    inst = SimpleNamespace()
    inst._client = SimpleNamespace()
    # bind method
    bound = DeltaExchangeClient._call.__get__(inst, inst.__class__)
    with pytest.raises(ExchangeClientError):
        await bound("nonexistent")


@pytest.mark.asyncio
async def test_call_typeerror_fallback_to_positional():
    class C:
        def method(self):
            return "ok"

    inst = SimpleNamespace()
    inst._client = SimpleNamespace(method=C().method)
    bound = DeltaExchangeClient._call.__get__(inst, inst.__class__)
    # call with kwargs that method doesn't accept; _call should catch TypeError and retry without kwargs
    res = await bound("method", a=1)
    assert res == "ok"


@pytest.mark.asyncio
async def test_get_index_price_parsing_variants():
    inst = SimpleNamespace()

    async def _call_upper(method_name, **kwargs):
        return {"spot_price": "150.0"}

    inst._call = _call_upper
    bound_get = DeltaExchangeClient.get_index_price.__get__(inst, inst.__class__)
    price = await bound_get()
    assert float(price) == 150.0

    async def _call_nested(method_name, **kwargs):
        return {"result": {"data": {"last_price": "250.0"}}}

    inst._call = _call_nested
    price2 = await bound_get()
    assert float(price2) == 250.0

    async def _call_ticker(method_name, **kwargs):
        return {"ticker": {"last_price": "300.0"}}

    inst._call = _call_ticker
    price3 = await bound_get()
    assert float(price3) == 300.0

    # missing price -> should raise ExchangeClientError
    async def _call_bad(method_name, **kwargs):
        return {"result": {}}

    inst._call = _call_bad
    bound_get2 = DeltaExchangeClient.get_index_price.__get__(inst, inst.__class__)
    with pytest.raises(ExchangeClientError):
        await bound_get2()


@pytest.mark.asyncio
async def test_get_best_quote_supports_alternate_payloads_and_errors():
    inst = SimpleNamespace()

    async def _call_quotes(method_name, **kwargs):
        return {"quotes": {"best_bid": "101.5", "best_ask": "102.5"}}

    inst._call = _call_quotes
    bound = DeltaExchangeClient.get_best_quote.__get__(inst, inst.__class__)
    quote = await bound(42)
    assert quote.best_bid == 101.5
    assert quote.best_ask == 102.5

    async def _call_result(method_name, **kwargs):
        return {"result": {"best_bid": "110.0", "best_ask": "111.0"}}

    inst._call = _call_result
    second = await bound(43)
    assert second.best_bid == 110.0
    assert second.best_ask == 111.0

    async def _call_bad(method_name, **kwargs):
        return {"unexpected": "payload"}

    inst._call = _call_bad
    with pytest.raises(ExchangeClientError):
        await bound(44)


@pytest.mark.asyncio
async def test_get_open_positions_raw_and_position_size_handle_failures():
    class C(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            return {"success": False, "message": "blocked"}

    client = C()
    client.get_open_positions_raw = DeltaExchangeClient.get_open_positions_raw.__get__(client, client.__class__)
    client.get_position_size = DeltaExchangeClient.get_position_size.__get__(client, client.__class__)

    positions = await client.get_open_positions_raw()
    assert positions is None
    assert await client.get_position_size(99) == 0.0

    class C2(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            return {"success": True, "result": [{"product_id": 7, "size": "2.5", "product": {"product_id": 7}}]}

    client2 = C2()
    client2.get_open_positions_raw = DeltaExchangeClient.get_open_positions_raw.__get__(client2, client2.__class__)
    client2.get_position_size = DeltaExchangeClient.get_position_size.__get__(client2, client2.__class__)
    assert await client2.get_position_size(7) == 2.5


@pytest.mark.asyncio
async def test_parse_option_positions_skips_invalid_entries_and_keeps_valid_one():
    class C(DeltaExchangeClient):
        def __init__(self):
            pass

        async def get_open_positions_raw(self):
            return [
                {"size": "0", "product_id": 1, "product": {"contract_type": "option_call", "strike_price": "100", "settlement_time": "2026-12-31T00:00:00Z", "contract_value": "1", "symbol": "OPT-1"}},
                {"size": "-1", "product_id": 2, "product": {"contract_type": "futures", "strike_price": "100", "settlement_time": "2026-12-31T00:00:00Z", "contract_value": "1", "symbol": "FUT-2"}},
                {"size": "-1", "product_id": 3, "product": {"contract_type": "option_unknown", "strike_price": "100", "settlement_time": "2026-12-31T00:00:00Z", "contract_value": "1", "symbol": "OPT-3"}},
                {"size": "-1", "product_id": 4, "product": {"contract_type": "option_call", "strike_price": "0", "settlement_time": "2026-12-31T00:00:00Z", "contract_value": "1", "symbol": "OPT-4"}},
                {"size": "-1", "product_id": 5, "entry_price": "9.5", "product": {"contract_type": "option_call", "strike_price": "200", "settlement_time": "2026-12-31T00:00:00Z", "contract_value": "1.5", "symbol": "OPT-5"}},
            ]

    client = C()
    client._to_dt = DeltaExchangeClient._to_dt
    client.parse_option_positions = DeltaExchangeClient.parse_option_positions.__get__(client, client.__class__)
    legs = await client.parse_option_positions()
    assert len(legs) == 1
    assert legs[0].product_id == 5
    assert legs[0].option_type == "call"
    assert legs[0].contract_value == 1.5
