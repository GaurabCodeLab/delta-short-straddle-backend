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
