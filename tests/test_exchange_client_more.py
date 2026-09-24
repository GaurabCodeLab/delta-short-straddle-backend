import asyncio
from types import SimpleNamespace
import pytest

from src.exchange_client import DeltaExchangeClient, ExchangeClientError
from src.order_executor import OrderExecutor


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


@pytest.mark.asyncio
async def test_exchange_client_runtime_helpers_and_product_lookup_variants():
    client = object.__new__(DeltaExchangeClient)

    assert DeltaExchangeClient._unwrap_payload({"result": {"data": {"id": 7}}}) == {"id": 7}
    assert DeltaExchangeClient._coerce_float("12.5") == 12.5
    assert DeltaExchangeClient._coerce_float("bad") is None
    with pytest.raises(ExchangeClientError):
        DeltaExchangeClient._to_dt("not-a-date")

    class ProductClient(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            if kwargs.get("path") == "/v2/products" and kwargs.get("query", {}).get("symbol") == "OPT-42":
                return {"success": True, "result": [{"symbol": "OPT-42", "id": 42}]}
            if kwargs.get("path") == "/v2/products" and kwargs.get("query", {}).get("page_number") == 1:
                return {"success": True, "result": [{"id": 1, "symbol": "OPT-1"}, {"id": 2, "symbol": "OPT-2"}]}
            return {"success": True, "result": [{"id": 2, "symbol": "OPT-2"}]}

    product_client = ProductClient()
    product_client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(product_client, product_client.__class__)
    product_client.get_product_by_symbol = DeltaExchangeClient.get_product_by_symbol.__get__(product_client, product_client.__class__)

    assert await product_client.get_product_by_symbol("OPT-42") == {"symbol": "OPT-42", "id": 42}
    products = await product_client.get_products_raw()
    assert [p["id"] for p in products] == [1, 2]

    class OrderClient(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            return {"result": [{"id": "ord-1"}]}

    order_client = OrderClient()
    order_client.get_order = DeltaExchangeClient.get_order.__get__(order_client, order_client.__class__)
    order = await order_client.get_order("ord-1", product_id=9)
    assert order["id"] == "ord-1"


@pytest.mark.asyncio
async def test_exchange_client_handles_nested_result_data_and_empty_results():
    class C(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            return {"success": True, "result": {"data": [{"product_id": "9", "size": "4", "product": {"id": 9}}]}}

    client = C()
    client.get_open_positions_raw = DeltaExchangeClient.get_open_positions_raw.__get__(client, client.__class__)
    client.get_position_size = DeltaExchangeClient.get_position_size.__get__(client, client.__class__)
    assert await client.get_position_size(9) == 4.0

    class P(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            query = kwargs.get("query") or {}
            page_number = query.get("page_number", 1)
            if page_number == 1:
                return {"success": True, "result": [{"id": "1"}, {"id": "2"}]}
            if page_number == 2:
                return {"success": True, "result": [{"id": "2"}, {"id": "3"}]}
            return {"success": True, "result": []}

    product_client = P()
    product_client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(product_client, product_client.__class__)
    products = await product_client.get_products_raw()
    assert [p["id"] for p in products] == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_order_executor_handles_error_branches_and_nested_meta_payloads():
    class E:
        async def get_position_size(self, product_id: int):
            return 0.0

        async def place_market_order(self, **kwargs):
            return {"result": {"id": "ord-9"}}

    executor = OrderExecutor(E())

    with pytest.raises(RuntimeError):
        OrderExecutor._extract_order_id({"result": {}})

    payload = {"result": {"meta_data": {"pnl": "12.5"}, "avg_fill_price": "25.0"}}
    assert OrderExecutor._extract_realized_pnl_from_payload(payload) == pytest.approx(12.5)
    assert OrderExecutor._extract_price_from_payload(payload) == pytest.approx(25.0)

    class E2:
        async def get_position_size(self, product_id: int):
            return 0.0

        async def place_market_order(self, **kwargs):
            raise RuntimeError("no_position_for_reduce_only")

    reduced = OrderExecutor(E2())
    filled = await reduced.execute_market_single_submission_with_fill_confirmation(7, "buy", 2.0, True)
    assert filled == 2.0
    assert reduced.last_fill_details["filled_qty"] == 2.0
    assert reduced.last_fill_details["product_id"] == 7


@pytest.mark.asyncio
async def test_order_executor_uses_order_history_metadata_when_place_response_is_sparse():
    class E:
        def __init__(self):
            self.calls = 0

        async def get_position_size(self, product_id: int):
            self.calls += 1
            if self.calls == 1:
                return 0.0
            return 1.0

        async def place_market_order(self, **kwargs):
            return {"result": {"id": "ord-9"}}

        async def get_order(self, order_id: str, product_id: int | None = None):
            return {"result": [{"meta_data": {"pnl": "7.5", "avg_exit_price": "99.5"}, "avg_fill_price": "98.0"}]}

    exchange = E()
    executor = OrderExecutor(exchange)
    filled = await executor.execute_market_single_submission_with_fill_confirmation(9, "buy", 1.0, False)

    assert filled == 1.0
    assert executor.last_fill_details["realized_pnl"] == pytest.approx(7.5)
    assert executor.last_fill_details["exit_price"] == pytest.approx(99.5)
    assert executor.last_fill_details["avg_fill_price"] == pytest.approx(98.0)


@pytest.mark.asyncio
async def test_delta_client_extracts_nested_price_variants_and_falls_back_across_symbols():
    client = object.__new__(DeltaExchangeClient)

    assert DeltaExchangeClient._extract_price_value(123.0) == pytest.approx(123.0)
    assert DeltaExchangeClient._extract_price_value({"quotes": {"last_price": "55.5"}}) == pytest.approx(55.5)
    assert DeltaExchangeClient._extract_price_value({"result": {"data": {"price": "66.6"}}}) == pytest.approx(66.6)

    async def _call(method_name: str, **kwargs):
        symbol = kwargs.get("identifier") or kwargs.get("symbol")
        if symbol in {"BTCUSD", "BTCUSDT"}:
            raise RuntimeError("unavailable")
        if symbol == ".DEXBTUSD":
            return {"result": {"data": {"last_price": "22000"}}}
        raise RuntimeError(f"unexpected symbol {symbol!r}")

    client._call = _call
    assert await client.get_index_price("BTCUSD") == pytest.approx(22000.0)


def test_order_executor_extracts_nested_close_metadata_and_pnl():
    payload = {"result": {"data": {"fills": [{"fill_price": "44.5"}]}}}
    assert OrderExecutor._extract_fill_price_from_payload(payload) == pytest.approx(44.5)
    assert OrderExecutor._extract_exit_price_from_payload({"meta_data": {"avg_exit_price": "77.25"}}) == pytest.approx(77.25)
    assert OrderExecutor._extract_exit_price_from_payload({"result": {"details": {"average_exit_price": "88.5"}}}) == pytest.approx(88.5)
    assert OrderExecutor._extract_realized_pnl_from_payload({"result": [{"meta_data": {"pnl": "12.5"}}]}) == pytest.approx(12.5)
    assert OrderExecutor._extract_price_from_payload({"result": {"meta_data": {"avg_exit_price": "91.5"}}}) == pytest.approx(91.5)


@pytest.mark.asyncio
async def test_order_executor_rejects_non_positive_size():
    executor = OrderExecutor(exchange=None)
    with pytest.raises(ValueError, match="Order quantity must be positive"):
        await executor.execute_market_single_submission_with_fill_confirmation(1, "buy", 0.0, False)


@pytest.mark.asyncio
async def test_order_executor_uses_meta_data_even_without_fill_data_in_order_history():
    class E:
        async def get_position_size(self, product_id: int):
            return 0.0

        async def place_market_order(self, **kwargs):
            return {"result": {"id": "ord-19"}}

        async def get_order(self, order_id: str, product_id: int | None = None):
            return {"data": [{"meta_data": {"pnl": "6.5", "avg_exit_price": "120.0"}}]}

    executor = OrderExecutor(E())
    filled = await executor.execute_market_single_submission_with_fill_confirmation(19, "sell", 1.0, False)
    assert filled == 1.0
    assert executor.last_fill_details["realized_pnl"] == pytest.approx(6.5)
    assert executor.last_fill_details["exit_price"] == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_delta_client_get_order_and_products_accept_data_wrapped_payloads():
    class C(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            return {"data": [{"id": "ord-10"}]}

    client = C()
    client.get_order = DeltaExchangeClient.get_order.__get__(client, client.__class__)
    assert await client.get_order("ord-10") == {"id": "ord-10"}

    class P(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            return {"data": [{"id": "7", "symbol": "XYZ-7", "contract_type": "put option", "strike_price": "7000", "settlement_time": "2026-12-31T00:00:00Z"}]}

    product_client = P()
    product_client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(product_client, product_client.__class__)
    products = await product_client.get_products_raw()
    assert products[0]["id"] == "7"


@pytest.mark.asyncio
async def test_exchange_client_handles_nested_result_dicts_and_positions_variants():
    class C(DeltaExchangeClient):
        def __init__(self):
            pass

        async def _call(self, method_name: str, **kwargs):
            if kwargs.get("path") == "/v2/products":
                return {"result": {"data": [{"id": "1", "symbol": "X-1"}]}}
            return {"result": {"positions": [{"product_id": "9", "size": "4", "product": {"id": 9}}]}}

    client = C()
    client.get_open_positions_raw = DeltaExchangeClient.get_open_positions_raw.__get__(client, client.__class__)
    client.get_position_size = DeltaExchangeClient.get_position_size.__get__(client, client.__class__)

    assert await client.get_open_positions_raw() == [{"product_id": "9", "size": "4", "product": {"id": 9}}]
    assert await client.get_position_size(9) == 4.0

    client.get_products_raw = DeltaExchangeClient.get_products_raw.__get__(client, client.__class__)
    products = await client.get_products_raw()
    assert products == [{"id": "1", "symbol": "X-1"}]


@pytest.mark.asyncio
async def test_order_executor_handles_order_history_error_and_direct_close_payloads():
    class E:
        def __init__(self):
            self.calls = 0

        async def get_position_size(self, product_id: int):
            self.calls += 1
            if self.calls == 1:
                return 0.0
            return 1.0

        async def place_market_order(self, **kwargs):
            return {"result": {"id": "ord-21"}}

        async def get_order(self, order_id: str, product_id: int | None = None):
            raise RuntimeError("order lookup failed")

    executor = OrderExecutor(E())
    filled = await executor.execute_market_single_submission_with_fill_confirmation(21, "buy", 1.0, False)
    assert filled == 1.0
    assert executor.last_fill_details["avg_fill_price"] is None

    payload = {"pnl": "3.25", "avg_exit_price": "66.0"}
    assert OrderExecutor._extract_realized_pnl_from_payload(payload) == pytest.approx(3.25)
    assert OrderExecutor._extract_exit_price_from_payload(payload) == pytest.approx(66.0)
    assert OrderExecutor._extract_fill_price_from_payload([{"fill_price": "44.4"}, {"avg_fill_price": "45.5"}]) == pytest.approx(44.4)
