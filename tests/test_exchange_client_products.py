import asyncio
from types import SimpleNamespace
import pytest

from src.exchange_client import DeltaExchangeClient, ExchangeClientError


@pytest.mark.asyncio
async def test_get_products_raw_pagination_and_dedup():
    # prepare page payloads
    page1 = {"success": True, "result": [{"id": 1}, {"id": 2}]}
    page2 = {"success": True, "result": [{"id": 2}, {"id": 3}]}
    page3 = {"success": True, "result": []}

    calls = [page1, page2, page3]

    async def _call(method_name, **kwargs):
        # simulate a response object with json method
        payload = calls.pop(0)

        def json():
            return payload

        return SimpleNamespace(json=json)

    client = SimpleNamespace()
    client._call = _call
    # bind method
    client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(client, client.__class__)

    products = await client.get_products_raw()
    assert isinstance(products, list)
    # dedup ensures ids 1,2,3 only once
    ids = {int(p.get("id")) for p in products}
    assert ids == {1, 2, 3}


@pytest.mark.asyncio
async def test_get_products_raw_stops_on_duplicate_and_failed_pages():
    calls = [
        {"success": True, "result": [{"id": 1}]},
        {"success": True, "result": [{"id": 1}]},
        {"success": False, "result": []},
    ]

    async def _call(method_name, **kwargs):
        payload = calls.pop(0)

        def json():
            return payload

        return SimpleNamespace(json=json)

    client = SimpleNamespace()
    client._call = _call
    client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(client, client.__class__)

    products = await client.get_products_raw()
    assert [p["id"] for p in products] == [1]


@pytest.mark.asyncio
async def test_place_market_order_and_get_order_success_and_failure():
    # place_market_order expects self._order_type_market and _tif_ioc exist
    async def _call_place(method_name, **kwargs):
        if method_name == "place_order":
            return {"order_id": "abc"}
        if method_name == "order_history":
            return {"result": [{"id": "abc", "status": "filled"}]}
        raise Exception("unexpected")

    client = SimpleNamespace()
    client._call = _call_place
    client._order_type_market = "MARKET"
    client._tif_ioc = "IOC"
    # bind methods
    client.place_market_order = DeltaExchangeClient.place_market_order.__get__(client, client.__class__)
    client.get_order = DeltaExchangeClient.get_order.__get__(client, client.__class__)

    resp = await client.place_market_order(product_id=1, side="buy", size=1.0, reduce_only=True)
    assert isinstance(resp, dict)

    order = await client.get_order(order_id="abc", product_id=1)
    assert isinstance(order, dict)
    assert order.get("status") == "filled"

    # test unexpected response type for place_market_order raises
    async def _call_bad(method_name, **kwargs):
        if method_name == "place_order":
            return "not-a-dict"
        return {"result": []}

    client2 = SimpleNamespace()
    client2._call = _call_bad
    client2._order_type_market = "MARKET"
    client2._tif_ioc = "IOC"
    client2.place_market_order = DeltaExchangeClient.place_market_order.__get__(client2, client2.__class__)

    with pytest.raises(ExchangeClientError):
        await client2.place_market_order(product_id=1, side="buy", size=1.0, reduce_only=False)
