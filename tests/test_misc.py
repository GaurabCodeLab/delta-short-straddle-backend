from __future__ import annotations

import asyncio
import json
import logging
import runpy
from datetime import datetime, timezone
from typing import Any

import pytest
import uvicorn
from fastapi import HTTPException

from src.api import BotManager, JsonLogHandler, build_strategy, manager, health, api_start, api_stop, api_status, api_summary, api_logs
from src.exchange_client import DeltaExchangeClient, ExchangeClientError
from src.logging_config import setup_logging
from src.models import OptionLeg, Quote, ShortStraddle
from src.order_executor import OrderExecutor
from src.position_manager import PositionManager
from src.strategy_engine import StrategyEngine, StrategyState


class DummyStrategy:
    def __init__(self) -> None:
        self.state = StrategyState()
        self._run_event = asyncio.Event()

    async def run(self) -> None:
        await self._run_event.wait()


class DummySummaryExchange:
    def __init__(self) -> None:
        self._quotes: dict[int, Quote] = {}

    async def get_index_price(self, symbol: str = "BTCUSDT") -> float:
        return 30000.0

    async def parse_option_positions(self) -> list[OptionLeg]:
        return [
            OptionLeg(
                product_id=1,
                symbol="P-30000",
                option_type="put",
                strike=30000.0,
                expiry=datetime(2026, 12, 31, tzinfo=timezone.utc),
                size=-1.0,
                entry_price=100.0,
                contract_value=1.0,
            )
        ]

    async def get_best_quote(self, product_id: int) -> Quote:
        return Quote(best_bid=1.0, best_ask=110.0)


class DummyDeltaClient(DeltaExchangeClient):
    def __init__(self) -> None:
        self._order_type_market = "MARKET"
        self._tif_ioc = "IOC"

    async def _call(self, method_name: str, **kwargs: Any) -> Any:
        raise Exception("direct call failed")


class DummyOrderExecutor(OrderExecutor):
    def __init__(self, exchange):
        super().__init__(exchange)
        self.executed = []
        self.exit_event = asyncio.Event()

    async def execute_market_single_submission_with_fill_confirmation(
        self,
        product_id: int,
        side: str,
        size: float,
        reduce_only: bool,
    ) -> float:
        self.executed.append((product_id, side, size, reduce_only))
        if reduce_only:
            self.exchange._positions = [
                pos for pos in self.exchange._positions if pos.product_id != product_id
            ]
        self.exit_event.set()
        return size


def test_json_log_handler_parses_json_data():
    handler = JsonLogHandler(max_records=3)
    logger = logging.getLogger("test_json_log_handler")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('{"foo":"bar"}')
    logger.info("plain text")

    records = handler.latest()
    assert len(records) == 2
    assert records[0]["data"] == {"foo": "bar"}
    assert records[1]["message"] == "plain text"
    assert handler.latest(limit=1) == [records[-1]]


def test_setup_logging_configures_root_handler():
    setup_logging("DEBUG")
    root_handlers = logging.getLogger().handlers
    assert any(isinstance(handler, logging.StreamHandler) for handler in root_handlers)


@pytest.mark.asyncio
async def test_bot_manager_start_stop_and_status():
    strategy = DummyStrategy()
    manager = BotManager(strategy)

    result = manager.start()
    assert result["status"] == "started"
    assert result["running"] is True
    assert manager.status()["running"] is True

    stop_result = manager.stop()
    assert stop_result["status"] == "stopped"
    assert stop_result["running"] is False
    await asyncio.sleep(0)
    assert manager.status()["running"] is False


@pytest.mark.asyncio
async def test_bot_manager_start_already_running_behavior():
    strategy = DummyStrategy()
    manager = BotManager(strategy)

    result = manager.start()
    assert result["status"] == "started"
    assert result["running"] is True

    result_again = manager.start()
    assert result_again["status"] == "already_running"
    assert result_again["running"] is True

    stop_result = manager.stop()
    assert stop_result["status"] == "stopped"
    assert stop_result["running"] is False
    await asyncio.sleep(0)


def test_bot_manager_status_exposes_runtime_details():
    strategy = DummyStrategy()
    strategy.state.last_index_price = 65000.0
    strategy.state.previous_index_price = 64000.0
    strategy.state.reference_strike = 64000.0
    strategy.state.current_structure = "straddle"
    strategy.state.threshold = 1000.0
    strategy.state.last_transition = "straddle->strangle"
    manager = BotManager(strategy)

    state = manager.status()["strategy_state"]
    assert state["last_index_price"] == 65000.0
    assert state["previous_index_price"] == 64000.0
    assert state["reference_strike"] == 64000.0
    assert state["current_structure"] == "straddle"
    assert state["threshold"] == 1000.0
    assert state["last_transition"] == "straddle->strangle"
    assert state["combined_pnl"] is None
    assert state["profit_target"] is None
    assert state["stop_loss"] is None
    assert state["exit_reason"] is None


@pytest.mark.asyncio
async def test_api_status_refreshes_live_combined_pnl_state():
    class RefreshableStrategy(DummyStrategy):
        def __init__(self) -> None:
            super().__init__()
            self.state = StrategyState()
            self.state.combined_pnl = 0.0
            self.state.profit_target = 10.0
            self.state.stop_loss = 20.0

        async def refresh_live_state(self) -> None:
            self.state.combined_pnl = 123.45
            self.state.profit_target = 50.0
            self.state.stop_loss = 100.0

    strategy = RefreshableStrategy()
    manager = BotManager(strategy)
    await manager.refresh_live_state()
    payload = manager.status()

    assert payload["strategy_state"]["combined_pnl"] == pytest.approx(123.45)
    assert payload["strategy_state"]["profit_target"] == pytest.approx(50.0)
    assert payload["strategy_state"]["stop_loss"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_bot_manager_collect_summary_aggregates_positions():
    class StrategyWithState:
        def __init__(self) -> None:
            self.state = StrategyState()

        async def run(self) -> None:
            return None

    strategy = StrategyWithState()
    strategy.state.status_message = "ok"
    strategy.state.action = "waiting"
    strategy.state.status = "idle"
    strategy.state.trigger_price = None
    strategy.state.trigger_pnl = None
    manager = BotManager(strategy)
    manager.strategy.exchange = DummySummaryExchange()

    summary = await manager.collect_summary()
    assert summary["index_price"] == 30000.0
    assert summary["position_count"] == 1
    assert summary["unrealized_pnl"] == pytest.approx(-10.0)
    assert summary["strategy_state"]["status"] == "idle"


@pytest.mark.asyncio
async def test_bot_manager_collect_summary_skips_invalid_quotes(settings):
    class InvalidQuoteExchange(DummySummaryExchange):
        async def get_best_quote(self, product_id: int):
            return Quote(best_bid=None, best_ask=None)

    exchange = InvalidQuoteExchange()
    strategy = DummyStrategy()
    strategy.state = StrategyState()
    strategy.exchange = exchange
    manager = BotManager(strategy)

    summary = await manager.collect_summary()
    assert summary["position_count"] == 0
    assert summary["unrealized_pnl"] == 0.0


@pytest.mark.asyncio
async def test_bot_manager_collect_summary_propagates_index_price_errors(settings):
    class FailingIndexExchange(DummySummaryExchange):
        async def get_index_price(self, symbol: str = "BTCUSDT") -> float:
            raise RuntimeError("index failed")

    exchange = FailingIndexExchange()
    strategy = DummyStrategy()
    strategy.state = StrategyState()
    strategy.exchange = exchange
    manager = BotManager(strategy)

    with pytest.raises(RuntimeError, match="index failed"):
        await manager.collect_summary()


@pytest.mark.asyncio
async def test_bot_manager_collect_summary_includes_nested_error_details(settings):
    class FailingIndexExchange(DummySummaryExchange):
        async def get_index_price(self, symbol: str = "BTCUSDT") -> float:
            raise ExchangeClientError(
                "Unable to fetch BTC index price from client methods."
            ) from RuntimeError("IP 1.2.3.4 not whitelisted")

    exchange = FailingIndexExchange()
    strategy = DummyStrategy()
    strategy.state = StrategyState()
    strategy.exchange = exchange
    manager = BotManager(strategy)

    with pytest.raises(ExchangeClientError, match="IP 1.2.3.4 not whitelisted"):
        await manager.collect_summary()


@pytest.mark.asyncio
async def test_api_summary_route_includes_nested_error_details(monkeypatch):
    async def failing_summary():
        raise ExchangeClientError(
            "Unable to collect summary"
        ) from RuntimeError("IP 1.2.3.4 not whitelisted")

    monkeypatch.setattr(manager, "collect_summary", failing_summary)
    with pytest.raises(HTTPException, match="IP 1.2.3.4 not whitelisted"):
        await api_summary()


def test_build_strategy_uses_configured_components(monkeypatch, settings):
    import src.api as api_module

    class DummyExchange:
        def __init__(self, api_key, api_secret, base_url, ssl_verify):
            self.api_key = api_key
            self.api_secret = api_secret
            self.base_url = base_url
            self.ssl_verify = ssl_verify

    class DummyPositions:
        def __init__(self, exchange):
            self.exchange = exchange

    class DummyExecutor:
        def __init__(self, exchange):
            self.exchange = exchange

    class DummyStrategyEngine:
        def __init__(self, exchange, positions, executor, settings):
            self.exchange = exchange
            self.positions = positions
            self.executor = executor
            self.settings = settings

    monkeypatch.setattr(api_module, "load_settings", lambda: settings)
    monkeypatch.setattr(api_module, "DeltaExchangeClient", DummyExchange)
    monkeypatch.setattr(api_module, "PositionManager", DummyPositions)
    monkeypatch.setattr(api_module, "OrderExecutor", DummyExecutor)
    monkeypatch.setattr(api_module, "StrategyEngine", DummyStrategyEngine)

    strategy = api_module.build_strategy()
    assert isinstance(strategy, DummyStrategyEngine)
    assert isinstance(strategy.exchange, DummyExchange)
    assert isinstance(strategy.positions, DummyPositions)
    assert isinstance(strategy.executor, DummyExecutor)
    assert strategy.settings is settings


@pytest.mark.asyncio
async def test_health_route_returns_ok():
    assert await health() == {"status": "ok"}


@pytest.mark.asyncio
async def test_api_start_route_calls_manager_start(monkeypatch):
    monkeypatch.setattr(manager, "start", lambda: {"status": "started", "running": True})
    assert await api_start() == {"status": "started", "running": True}


@pytest.mark.asyncio
async def test_api_stop_route_calls_manager_stop(monkeypatch):
    monkeypatch.setattr(manager, "stop", lambda: {"status": "stopped", "running": False})
    assert await api_stop() == {"status": "stopped", "running": False}


@pytest.mark.asyncio
async def test_api_status_route_calls_manager_status(monkeypatch):
    monkeypatch.setattr(manager, "status", lambda: {"running": False, "strategy_state": {}})
    assert await api_status() == {"running": False, "strategy_state": {}}


@pytest.mark.asyncio
async def test_api_logs_route_returns_current_logs(monkeypatch):
    import src.api as api_module

    handler = JsonLogHandler(max_records=2)
    monkeypatch.setattr(api_module, "log_handler", handler)

    logger = logging.getLogger("test_api_logs")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("hello")
    result = await api_logs(limit=1)
    assert result["count"] == 1
    assert result["logs"][0]["message"] == "hello"


@pytest.mark.asyncio
async def test_api_summary_route_returns_summary(monkeypatch):
    async def fake_collect_summary():
        return {"index_price": 1.0, "unrealized_pnl": 0.0, "position_count": 0, "positions": [], "strategy_state": {}}

    monkeypatch.setattr(manager, "collect_summary", fake_collect_summary)
    assert await api_summary() == {"index_price": 1.0, "unrealized_pnl": 0.0, "position_count": 0, "positions": [], "strategy_state": {}}


@pytest.mark.asyncio
async def test_api_summary_route_raises_http_exception(monkeypatch):
    async def failing_summary():
        raise RuntimeError("summary failure")

    monkeypatch.setattr(manager, "collect_summary", failing_summary)
    with pytest.raises(Exception) as excinfo:
        await api_summary()
    assert "summary failure" in str(excinfo.value)


@pytest.mark.asyncio
async def test_json_log_handler_handles_invalid_json_and_limit():
    handler = JsonLogHandler(max_records=2)
    logger = logging.getLogger("test_json_log_handler_invalid")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('{bad json}')
    logger.info('plain text')

    assert handler.latest()[0]["message"] == "{bad json}"
    assert handler.latest(limit=1)[0]["message"] == "plain text"


@pytest.mark.asyncio
async def test_json_log_handler_records_timestamp_if_formatted(settings):
    handler = JsonLogHandler(max_records=1)
    logger = logging.getLogger("test_json_log_handler_timestamp")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("hello")
    record = handler.latest()[0]
    assert record["level"] == "INFO"
    assert record["message"] == "hello"
    assert record["timestamp"] is not None


@pytest.mark.asyncio
async def test_api_summary_endpoint_calls_collect_summary(monkeypatch):
    async def fake_collect_summary():
        return {"index_price": 1.0, "unrealized_pnl": 0.0, "position_count": 0, "positions": [], "strategy_state": {}}

    monkeypatch.setattr(manager, "collect_summary", fake_collect_summary)
    assert await api_summary() == {"index_price": 1.0, "unrealized_pnl": 0.0, "position_count": 0, "positions": [], "strategy_state": {}}


@pytest.mark.asyncio
async def test_api_start_route_uses_manager_start(monkeypatch):
    monkeypatch.setattr(manager, "start", lambda: {"status": "started", "running": True})
    result = await api_start()
    assert result == {"status": "started", "running": True}

    parsed = DeltaExchangeClient._to_dt("2026-12-31T00:00:00Z")
    assert parsed.date().isoformat() == "2026-12-31"

    parsed_ts = DeltaExchangeClient._to_dt(1703980800)
    assert parsed_ts.date().isoformat() == "2023-12-31"

    with pytest.raises(ExchangeClientError, match="Unsupported datetime value"):
        DeltaExchangeClient._to_dt(object())


@pytest.mark.asyncio
async def test_exchange_client_get_product_by_symbol_fallbacks_to_product_scan():
    client = DummyDeltaClient()

    async def get_products_raw() -> list[dict[str, Any]]:
        return [
            {
                "id": "1",
                "symbol": "P-30000",
                "contract_type": "put option",
                "strike_price": "30000",
                "settlement_time": "2026-12-31T00:00:00Z",
            }
        ]

    client.get_products_raw = get_products_raw
    product = await client.get_product_by_symbol("P-30000")
    assert product is not None
    assert product["id"] == "1"


@pytest.mark.asyncio
async def test_exchange_client_get_best_quote_returns_quote():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"quotes": {"best_bid": 1.0, "best_ask": 2.0}}

    client._call = _call
    quote = await client.get_best_quote(1)
    assert quote.best_bid == 1.0
    assert quote.best_ask == 2.0


@pytest.mark.asyncio
async def test_exchange_client_get_best_quote_raises_when_data_is_missing():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"quotes": {"best_bid": None, "best_ask": None}}

    client._call = _call
    with pytest.raises(ExchangeClientError, match="Quote response missing best_bid or best_ask"):
        await client.get_best_quote(1)


@pytest.mark.asyncio
async def test_exchange_client_get_product_by_symbol_direct_query_returns_match():
    client = DummyDeltaClient()

    class DummyResponse:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    async def _call(method_name: str, **kwargs: Any) -> DummyResponse:
        return DummyResponse({"success": True, "result": [{"id": "1", "symbol": "P-30000"}]})

    client._call = _call
    product = await client.get_product_by_symbol("P-30000")
    assert product is not None
    assert product["id"] == "1"


@pytest.mark.asyncio
async def test_exchange_client_get_products_raw_succeeds_with_duplicate_page_stop():
    client = DummyDeltaClient()

    class DummyResponse:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    async def _call(method_name: str, **kwargs: Any) -> DummyResponse:
        query = kwargs.get("query") or {}
        page_number = query.get("page_number", 1)
        if page_number == 1:
            return DummyResponse({"success": True, "result": [{"id": "1"}, {"id": "2"}]})
        if page_number == 2:
            return DummyResponse({"success": True, "result": [{"id": "2"}]})
        return DummyResponse({"success": False, "result": []})

    client._call = _call
    products = await client.get_products_raw()
    assert isinstance(products, list)
    assert len(products) == 2


@pytest.mark.asyncio
async def test_exchange_client_parse_option_positions_filters_invalid_entries():
    class InvalidPositionsExchange(DummyDeltaClient):
        async def get_open_positions_raw(self):
            return [
                {
                    "product_id": "1",
                    "product": {
                        "symbol": "P-30000",
                        "contract_type": "put option",
                        "strike_price": "30000",
                        "settlement_time": "2026-12-31T00:00:00Z",
                        "contract_value": "1.0",
                    },
                    "size": "-1",
                    "entry_price": "100",
                },
                {
                    "product_id": "2",
                    "product": None,
                    "size": "-1",
                    "entry_price": "100",
                },
                {
                    "product_id": "3",
                    "product": {"contract_type": "spot", "strike_price": "30000", "settlement_time": "2026-12-31T00:00:00Z", "contract_value": "1.0"},
                    "size": "-1",
                    "entry_price": "100",
                },
            ]

    client = InvalidPositionsExchange()
    legs = await client.parse_option_positions()
    assert len(legs) == 1
    assert legs[0].symbol == "P-30000"


@pytest.mark.asyncio
async def test_exchange_client_get_index_price_returns_float():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"spot_price": "40000"}

    client._call = _call
    assert await client.get_index_price() == 40000.0


@pytest.mark.asyncio
async def test_exchange_client_get_index_price_raises_when_client_fails():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    client._call = _call
    with pytest.raises(ExchangeClientError, match="Unable to fetch BTC index price"):
        await client.get_index_price()


@pytest.mark.asyncio
async def test_exchange_client_get_index_price_parses_nested_ticker_response():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"result": {"data": {"last_price": "42000"}}}

    client._call = _call
    assert await client.get_index_price() == 42000.0


@pytest.mark.asyncio
async def test_exchange_client_get_index_price_parses_ticker_payload():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"ticker": {"last_price": "43000"}}

    client._call = _call
    assert await client.get_index_price() == 43000.0


@pytest.mark.asyncio
async def test_exchange_client_get_index_price_returns_delta_error_message():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"success": False, "error": "IP 1.2.3.4 not whitelisted"}

    client._call = _call
    with pytest.raises(ExchangeClientError, match="IP 1.2.3.4 not whitelisted"):
        await client.get_index_price()


@pytest.mark.asyncio
async def test_exchange_client_get_open_positions_raw_returns_result():
    client = DummyDeltaClient()

    class DummyResponse:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    async def _call(method_name: str, **kwargs: Any) -> DummyResponse:
        return DummyResponse({"success": True, "result": [{"product_id": "1"}]})

    client._call = _call
    result = await client.get_open_positions_raw()
    assert result == [{"product_id": "1"}]


@pytest.mark.asyncio
async def test_exchange_client_get_open_positions_raw_raises_on_error_payload():
    client = DummyDeltaClient()

    class DummyResponse:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    async def _call(method_name: str, **kwargs: Any) -> DummyResponse:
        return DummyResponse({"success": False, "error": "bad request"})

    client._call = _call
    assert await client.get_open_positions_raw() is None


@pytest.mark.asyncio
async def test_exchange_client_get_open_positions_raw_raises_on_call_exception():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> Any:
        raise RuntimeError("network failure")

    client._call = _call
    with pytest.raises(ExchangeClientError, match="Unable to fetch open positions.*network failure"):
        await client.get_open_positions_raw()


@pytest.mark.asyncio
async def test_exchange_client_get_position_size_returns_zero_when_no_match():
    client = DummyDeltaClient()

    async def get_open_positions_raw() -> list[dict[str, Any]]:
        return [
            {"product_id": "1", "product": {"id": 1}, "size": "2"},
        ]

    client.get_open_positions_raw = get_open_positions_raw
    assert await client.get_position_size(2) == 0.0


@pytest.mark.asyncio
async def test_exchange_client_place_market_order_raises_on_non_dict_response():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> Any:
        return "unexpected"

    client._call = _call
    with pytest.raises(ExchangeClientError, match="Unexpected place_order response"):
        await client.place_market_order(product_id=1, side="buy", size=1.0, reduce_only=False)


@pytest.mark.asyncio
async def test_exchange_client_get_order_returns_empty_dict_when_result_missing():
    client = DummyDeltaClient()

    async def _call(method_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"result": []}

    client._call = _call
    assert await client.get_order("order-1") == {}


@pytest.mark.asyncio
async def test_exchange_client_get_product_by_symbol_returns_none_for_blank_query():
    client = DummyDeltaClient()
    assert await client.get_product_by_symbol("") is None


@pytest.mark.asyncio
async def test_exchange_client_get_products_raw_falls_back_to_no_query_on_type_error():
    client = DummyDeltaClient()
    call_count = {"count": 0}

    class DummyResponse:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    async def _call(method_name: str, **kwargs: Any) -> DummyResponse:
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise TypeError("unexpected kwargs")
        return DummyResponse({"success": True, "result": [{"id": "1"}]})

    client._call = _call
    products = await client.get_products_raw()
    assert products == [{"id": "1"}]


@pytest.mark.asyncio
async def test_exchange_client_get_products_raw_raises_on_unexpected_products_response():
    client = DummyDeltaClient()

    class DummyResponse:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    async def _call(method_name: str, **kwargs: Any) -> DummyResponse:
        return DummyResponse({"success": True, "result": []})

    client._call = _call
    with pytest.raises(ExchangeClientError, match="Unexpected products response shape"):
        await client.get_products_raw()


def test_order_executor_extract_order_id_raises_for_missing_id():
    executor = OrderExecutor(exchange=DummyDeltaClient())
    with pytest.raises(RuntimeError, match="Cannot extract order id"):
        executor._extract_order_id({"result": {}})


@pytest.mark.asyncio
async def test_order_executor_confirms_sell_fill_from_position_delta():
    class FillExchange(DummyDeltaClient):
        def __init__(self):
            self._sizes = {1: 10.0}
            self.calls = []

        async def get_position_size(self, product_id: int) -> float:
            return self._sizes.get(product_id, 0.0)

        async def place_market_order(self, product_id: int, side: str, size: float, reduce_only: bool):
            self.calls.append((product_id, side, size, reduce_only))
            self._sizes[product_id] = 5.0
            return {"result": {"id": "order-1"}}

    exchange = FillExchange()
    executor = OrderExecutor(exchange)
    confirmed = await executor.execute_market_single_submission_with_fill_confirmation(
        product_id=1,
        side="sell",
        size=5.0,
        reduce_only=True,
    )

    assert confirmed == 5.0
    assert exchange.calls == [(1, "sell", 5.0, True)]


@pytest.fixture
def settings():
    from src.config import Settings

    return Settings(
        api_key="key",
        api_secret="secret",
        base_url="https://test",
        poll_interval_seconds=0.1,
        strike_adjustment_threshold=1000.0,
        profit_target=50.0,
        stop_loss=100.0,
        log_level="INFO",
        ssl_verify=False,
    )


class DummyExchangeForClose(DummyDeltaClient):
    def __init__(self):
        self._positions = []

    async def parse_option_positions(self):
        return self._positions

    async def get_position_size(self, product_id: int) -> float:
        for pos in self._positions:
            if pos.product_id == product_id:
                return pos.size
        return 0.0

    async def place_market_order(self, product_id: int, side: str, size: float, reduce_only: bool):
        self._positions = [pos for pos in self._positions if pos.product_id != product_id]
        return {"result": {"id": f"order-{product_id}"}}


@pytest.mark.asyncio
async def test_strategy_engine_close_all_open_option_positions_closes_positions(settings):
    exchange = DummyExchangeForClose()
    exchange._positions = [
        OptionLeg(
            product_id=1,
            symbol="P-30000",
            option_type="put",
            strike=30000.0,
            expiry=datetime(2026, 12, 31, tzinfo=timezone.utc),
            size=-1.0,
            entry_price=100.0,
            contract_value=1.0,
        )
    ]
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)

    engine = StrategyEngine(exchange, positions, executor, settings)
    await engine._close_all_open_option_positions()
    assert exchange._positions == []
    assert len(executor.executed) == 1


def test_main_module_invokes_uvicorn_run(monkeypatch):
    called: list[tuple[str, str, int, str]] = []

    def fake_run(app: str, host: str, port: int, log_level: str) -> None:
        called.append((app, host, port, log_level))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    runpy.run_module("src.main", run_name="__main__")

    assert called == [("src.api:app", "0.0.0.0", 8000, "info")]


def test_main_module_inserts_project_root_into_sys_path(monkeypatch):
    called: list[tuple[str, str, int, str]] = []

    def fake_run(app: str, host: str, port: int, log_level: str) -> None:
        called.append((app, host, port, log_level))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    original_path = sys.path.copy()
    monkeypatch.setattr(sys, "path", [str(project_root)] + original_path)

    main_path = project_root / "src" / "main.py"
    runpy.run_path(str(main_path), run_name="__main__")

    assert called == [("src.api:app", "0.0.0.0", 8000, "info")]
    assert sys.path[0] == str(project_root)
