import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.exchange_client import DeltaExchangeClient, ExchangeClientError
from src.models import OptionLeg, Quote, ShortStrangle
from src.order_executor import OrderExecutor
from src.position_manager import PositionManager
from src.strategy_engine import StrategyEngine


@dataclass
class DummyExchange(DeltaExchangeClient):
    def __init__(self):
        self._prices = {}
        self._positions = []
        self._quotes = {}

    async def get_index_price(self, symbol: str = "BTCUSD") -> float:
        return self._prices.get(symbol, 0.0)

    async def get_open_positions_raw(self):
        return []

    async def parse_option_positions(self):
        return self._positions

    async def get_best_quote(self, product_id: int) -> Quote:
        if product_id not in self._quotes:
            raise ExchangeClientError("Missing quote")
        return self._quotes[product_id]

    async def place_market_order(self, product_id: int, side: str, size: float, reduce_only: bool):
        return {"result": {"id": f"order-{product_id}-{side}"}}

    async def get_position_size(self, product_id: int) -> float:
        for pos in self._positions:
            if pos.product_id == product_id:
                return pos.size
        return 0.0

    async def request(self, *args, **kwargs):
        raise NotImplementedError


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
        self.exit_event.set()
        return size


def make_option_leg(
    product_id: int,
    symbol: str,
    opt_type: str,
    strike: float,
    size: float,
    entry_price: float,
    contract_value: float = 1.0,
):
    return OptionLeg(
        product_id=product_id,
        symbol=symbol,
        option_type=opt_type,
        strike=strike,
        expiry=datetime(2026, 12, 31, tzinfo=timezone.utc),
        size=size,
        entry_price=entry_price,
        contract_value=contract_value,
    )


@pytest.fixture
def settings():
    return Settings(
        api_key="key",
        api_secret="secret",
        base_url="https://test",
        poll_interval_seconds=0.1,
        max_single_order_qty=100,
        max_total_option_notional=100000,
        profit_capture_ratio=0.5,
        leg_exit_buffer=10.0,
        log_level="INFO",
        ssl_verify=False,
    )


@pytest.mark.asyncio
async def test_premium_capture_and_profit_threshold(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    strangle = ShortStrangle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        call_leg=make_option_leg(2, "C-32000", "call", 32000.0, -1.0, 60.0),
    )

    total_premium = positions.compute_total_premium_received(strangle)
    assert total_premium == 110.0
    assert engine._compute_profit_threshold(total_premium) == 55.0


@pytest.mark.asyncio
async def test_leg_exit_threshold_is_configurable(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    assert engine._compute_leg_exit_threshold(110.0) == 120.0


@pytest.mark.asyncio
async def test_leg_market_value_is_scaled_by_position_size(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    leg = make_option_leg(1, "P-30000", "put", 30000.0, -100.0, 50.0)
    quote = Quote(best_bid=1.0, best_ask=315.0)

    assert engine._compute_leg_market_value(leg, quote) == 31500.0
    assert engine._compute_leg_exit_threshold(abs(leg.size) * leg.contract_value * leg.entry_price) == 5010.0


@pytest.mark.asyncio
async def test_leg_wise_exit_closes_only_one_leg(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        make_option_leg(2, "C-32000", "call", 32000.0, -1.0, 60.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=120.0),
        2: Quote(best_bid=2.0, best_ask=2.0),
    }
    exchange._prices["BTCUSDT"] = 31000.0

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    task = asyncio.create_task(engine.run())
    try:
        await asyncio.wait_for(executor.exit_event.wait(), timeout=1.0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert len(executor.executed) >= 1
    assert any(call[0] == 1 and call[1] == "buy" for call in executor.executed)
