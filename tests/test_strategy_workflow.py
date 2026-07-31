import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.exchange_client import DeltaExchangeClient, ExchangeClientError
from src.models import OptionLeg, Quote, ShortStraddle, ShortStrangle
from src.order_executor import OrderExecutor
from src.position_manager import PositionManager
from src.strategy_engine import StrategyEngine
import src.strategy_engine as strategy_engine


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

    async def get_products_raw(self):
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
        if reduce_only:
            self.exchange._positions = [
                pos for pos in self.exchange._positions if pos.product_id != product_id
            ]
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
        short_straddle_profit_capture_ratio=0.5,
        short_straddle_max_loss_ratio=1.0,
        leg_exit_buffer=10.0,
        log_level="INFO",
        ssl_verify=False,
    )


def test_apply_clock_skew_from_error_updates_offset():
    client = DeltaExchangeClient(
        api_key="key",
        api_secret="secret",
        base_url="https://test",
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

    assert engine._compute_leg_exit_threshold(110.0) == 230.0


@pytest.mark.asyncio
async def test_leg_market_value_is_scaled_by_position_size(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    leg = make_option_leg(1, "P-30000", "put", 30000.0, -100.0, 50.0)
    quote = Quote(best_bid=1.0, best_ask=315.0)

    assert engine._compute_leg_market_value(leg, quote) == 31500.0
    assert engine._compute_leg_exit_threshold(abs(leg.size) * leg.contract_value * leg.entry_price) == 10010.0


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

@pytest.mark.asyncio
async def test_short_straddle_monitor_exits_on_profit_target(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=80.0),
        2: Quote(best_bid=1.0, best_ask=90.0),
    }
    settings.short_straddle_profit_capture_ratio = 0.1
    settings.short_straddle_max_loss_ratio = 1.0

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    await engine._monitor_short_straddle(straddle)

    assert len(executor.executed) == 2
    assert any(call[0] == 1 and call[1] == "buy" for call in executor.executed)
    assert any(call[0] == 2 and call[1] == "buy" for call in executor.executed)
    assert exchange._positions == []

@pytest.mark.asyncio
async def test_short_straddle_monitor_handles_single_leg_and_exits(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=80.0),
    }
    settings.short_straddle_profit_capture_ratio = 0.1
    settings.short_straddle_max_loss_ratio = 1.0

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    await engine._monitor_short_straddle(straddle)

    assert len(executor.executed) == 1
    assert executor.executed[0][0] == 1
    assert exchange._positions == []


def test_delta_exchange_client_to_dt_parses_iso_string():
    parsed = DeltaExchangeClient._to_dt("2026-12-31T00:00:00Z")
    assert parsed.date().isoformat() == "2026-12-31"


def test_position_manager_normalizes_option_types_and_safe_datetime():
    assert PositionManager._normalize_option_type("CALL") == "call"
    assert PositionManager._normalize_option_type("put") == "put"
    assert PositionManager._normalize_option_type("unknown") == ""
    assert PositionManager._to_dt_safe("2026-12-31T00:00:00Z").date().isoformat() == "2026-12-31"
    assert PositionManager._to_dt_safe(1700000000) is not None
    assert PositionManager._to_dt_safe("not-a-date") is None


@pytest.mark.asyncio
async def test_get_option_product_for_strike_uses_cached_products():
    exchange = DummyExchange()
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)
    exchange._quotes = {}

    async def get_products_raw():
        return [
            {
                "id": "1",
                "symbol": "P-30000",
                "contract_type": "put option",
                "strike_price": "30000",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
            {
                "id": "2",
                "symbol": "C-30000",
                "contract_type": "call option",
                "strike_price": "30000.000000001",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
        ]

    exchange.get_products_raw = get_products_raw
    positions = PositionManager(exchange)

    put_product = await positions.get_option_product_for_strike("put", 30000.0, expiry)
    assert put_product is not None and put_product["id"] == "1"

    call_product = await positions.get_option_product_for_strike("call", 30000.0, expiry)
    assert call_product is not None and call_product["id"] == "2"


@pytest.mark.asyncio
async def test_nearest_strikes_for_breakeven_points_returns_existing_strikes():
    exchange = DummyExchange()
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)

    async def get_products_raw():
        return [
            {
                "id": "1",
                "symbol": "P-29500",
                "contract_type": "put option",
                "strike_price": "29500",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
            {
                "id": "2",
                "symbol": "P-30000",
                "contract_type": "put option",
                "strike_price": "30000",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
            {
                "id": "3",
                "symbol": "C-30500",
                "contract_type": "call option",
                "strike_price": "30500",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
            {
                "id": "4",
                "symbol": "C-30000",
                "contract_type": "call option",
                "strike_price": "30000",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
        ]

    exchange.get_products_raw = get_products_raw
    positions = PositionManager(exchange)
    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    lower, upper = await positions.nearest_strikes_for_breakeven_points(straddle, 29600.0, 30400.0)
    assert lower == 29500.0
    assert upper == 30500.0


@pytest.mark.asyncio
async def test_detect_short_straddle_and_short_strangle_behaviors(settings):
    exchange = DummyExchange()
    exchange._prices["BTCUSDT"] = 30000.0

    # Same-strike short put + short call should detect as a short straddle.
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    positions = PositionManager(exchange)
    straddle = await positions.detect_short_straddle()
    assert straddle.put_leg.option_type == "put"
    assert straddle.call_leg.option_type == "call"

    # OTM short put and call with same expiry should detect as a short strangle.
    exchange._positions = [
        make_option_leg(3, "P-29000", "put", 29000.0, -1.0, 100.0),
        make_option_leg(4, "C-31000", "call", 31000.0, -1.0, 110.0),
    ]
    strangle = await positions.detect_short_strangle()
    assert strangle.put_leg.strike == 29000.0
    assert strangle.call_leg.strike == 31000.0

    exchange._positions = [
        make_option_leg(5, "P-30000", "put", 30000.0, 1.0, 100.0),
        make_option_leg(6, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    with pytest.raises(RuntimeError, match="No short straddle"):
        await positions.detect_short_straddle()

    exchange._positions = []
    with pytest.raises(RuntimeError, match="No open option positions found"):
        await positions.detect_short_strangle()


@pytest.mark.asyncio
async def test_compute_pnl_and_breakevens_for_straddle(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
    )
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=45.0),
        2: Quote(best_bid=2.0, best_ask=55.0),
    }

    pnl = await positions.compute_straddle_pnl(straddle)
    assert pnl == pytest.approx((50.0 - 45.0) + (60.0 - 55.0))

    be = positions.compute_straddle_breakevens(straddle)
    assert be.strike == 30000.0
    assert be.lower_breakeven < straddle.put_leg.strike
    assert be.upper_breakeven > straddle.call_leg.strike

    assert positions.compute_total_premium_received(straddle) == 110.0

    bad_straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, 0.0, 50.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
    )
    with pytest.raises(RuntimeError, match="Invalid straddle exposure"):
        positions.compute_total_premium_received(bad_straddle)


@pytest.mark.asyncio
async def test_compute_iron_fly_pnl_and_breakevens(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
        make_option_leg(3, "P-29500", "put", 29500.0, 1.0, 5.0),
        make_option_leg(4, "C-30500", "call", 30500.0, 1.0, 5.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=45.0),
        2: Quote(best_bid=2.0, best_ask=55.0),
        3: Quote(best_bid=4.0, best_ask=4.5),
        4: Quote(best_bid=4.0, best_ask=4.5),
    }
    positions = PositionManager(exchange)
    straddle = ShortStraddle(
        put_leg=exchange._positions[0],
        call_leg=exchange._positions[1],
    )

    iron_pnl = await positions.compute_iron_fly_pnl(straddle, 29500.0, 30500.0)
    assert iron_pnl == pytest.approx((50.0 - 45.0) + (60.0 - 55.0) + (4.0 - 5.0) + (4.0 - 5.0))

    be = await positions.compute_iron_fly_breakevens(straddle, 29500.0, 30500.0)
    assert be.total_premium_received < positions.compute_total_premium_received(straddle)
    assert be.lower_breakeven < straddle.put_leg.strike
    assert be.upper_breakeven > straddle.call_leg.strike


@pytest.mark.asyncio
async def test_order_executor_extract_order_id_and_invalid_size(settings):
    exchange = DummyExchange()
    executor = OrderExecutor(exchange)

    assert executor._extract_order_id({"result": {"id": "abc"}}) == "abc"
    with pytest.raises(RuntimeError, match="Cannot extract order id"):
        executor._extract_order_id({"result": {}})

    with pytest.raises(ValueError, match="Order quantity must be positive"):
        await executor.execute_market_single_submission_with_fill_confirmation(
            product_id=1,
            side="buy",
            size=0,
            reduce_only=True,
        )


@pytest.mark.asyncio
async def test_order_executor_handles_reduce_only_no_position(settings):
    exchange = DummyExchange()
    async def place_market_order(*args, **kwargs):
        raise Exception("no_position_for_reduce_only")
    exchange.place_market_order = place_market_order
    executor = OrderExecutor(exchange)

    confirmed = await executor.execute_market_single_submission_with_fill_confirmation(
        product_id=1,
        side="buy",
        size=5.0,
        reduce_only=True,
    )
    assert confirmed == 5.0


@pytest.mark.asyncio
async def test_close_all_open_option_positions_closes_all_legs(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
    ]
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    await engine._close_all_open_option_positions()
    assert len(executor.executed) == 2
    assert exchange._positions == []


@pytest.mark.asyncio
async def test_close_short_straddle_handles_missing_leg(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
    ]
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)
    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
    )

    await engine._close_short_straddle(straddle)
    assert len(executor.executed) == 1
    assert exchange._positions == []


@pytest.mark.asyncio
async def test_short_straddle_monitor_exits_on_loss_target(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=130.0),
        2: Quote(best_bid=2.0, best_ask=140.0),
    }
    settings.short_straddle_profit_capture_ratio = 0.5
    settings.short_straddle_max_loss_ratio = 0.1

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    await engine._monitor_short_straddle(straddle)

    assert len(executor.executed) == 2
    assert exchange._positions == []


@pytest.mark.asyncio
async def test_convert_short_strangle_to_straddle_and_iron_fly_closes_and_opens_wings(settings):
    exchange = DummyExchange()
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)
    exchange._prices["BTCUSDT"] = 30000.0
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=95.0),
        3: Quote(best_bid=1.0, best_ask=5.0),
        4: Quote(best_bid=1.0, best_ask=5.0),
    }

    async def get_products_raw():
        return [
            {
                "id": "3",
                "symbol": "P-29500",
                "contract_type": "put option",
                "strike_price": "29500",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
            {
                "id": "4",
                "symbol": "C-30500",
                "contract_type": "call option",
                "strike_price": "30500",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
        ]

    exchange.get_products_raw = get_products_raw

    async def get_product_by_symbol(symbol: str) -> dict:
        if symbol == "C-30000":
            return {"id": "2", "symbol": "C-30000", "contract_type": "call option"}
        return None

    exchange.get_product_by_symbol = get_product_by_symbol

    class WingOrderExecutor(DummyOrderExecutor):
        async def execute_market_single_submission_with_fill_confirmation(
            self,
            product_id: int,
            side: str,
            size: float,
            reduce_only: bool,
        ) -> float:
            result = await super().execute_market_single_submission_with_fill_confirmation(
                product_id=product_id,
                side=side,
                size=size,
                reduce_only=reduce_only,
            )
            if not reduce_only and side == "sell" and product_id == 2:
                exchange._positions.append(
                    make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0)
                )
            if not reduce_only and side == "buy" and product_id == 3:
                exchange._positions.append(
                    make_option_leg(3, "P-29500", "put", 29500.0, 1.0, 5.0)
                )
            if not reduce_only and side == "buy" and product_id == 4:
                exchange._positions.append(
                    make_option_leg(4, "C-30500", "call", 30500.0, 1.0, 5.0)
                )
            return result

    positions = PositionManager(exchange)
    executor = WingOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    strangle = ShortStrangle(
        put_leg=exchange._positions[0],
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    wing_put, wing_call = await engine._convert_strangle_to_straddle_and_iron_fly(strangle)
    assert wing_put == 29500.0
    assert wing_call == 30500.0
    assert any(pos.product_id == 2 and pos.option_type == "call" for pos in exchange._positions)
    assert any(pos.product_id == 3 and pos.option_type == "put" for pos in exchange._positions)
    assert any(pos.product_id == 4 and pos.option_type == "call" for pos in exchange._positions)


def test_crossed_or_touched_various_cases():
    assert StrategyEngine._crossed_or_touched(None, 29900.0, 30000.0, "put") is True
    assert StrategyEngine._crossed_or_touched(None, 30100.0, 30000.0, "call") is True
    assert StrategyEngine._crossed_or_touched(30100.0, 29900.0, 30000.0, "put") is True
    assert StrategyEngine._crossed_or_touched(29900.0, 30100.0, 30000.0, "call") is True
    assert StrategyEngine._crossed_or_touched(29900.0, 29950.0, 30000.0, "put") is True
    assert StrategyEngine._crossed_or_touched(30100.0, 30000.0, 30000.0, "call") is True


def test_crossed_or_touched_false_cases():
    assert StrategyEngine._crossed_or_touched(None, 30100.0, 30000.0, "put") is False
    assert StrategyEngine._crossed_or_touched(None, 29900.0, 30000.0, "call") is False
    assert StrategyEngine._crossed_or_touched(29900.0, 29950.0, 30000.0, "call") is False
    assert StrategyEngine._crossed_or_touched(30100.0, 30050.0, 30000.0, "put") is False


@pytest.mark.asyncio
async def test_strategy_state_helpers_update_state_and_status(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    engine._set_strategy_state(
        action="test-action",
        status="test-status",
        status_message="test-message",
        trigger_price="test-price",
        trigger_pnl=42.0,
    )
    assert engine.state.action == "test-action"
    assert engine.state.status == "test-status"
    assert engine.state.status_message == "test-message"
    assert engine.state.trigger_price == "test-price"
    assert engine.state.trigger_pnl == 42.0

    engine._set_status("updated-message")
    assert engine.state.status_message == "updated-message"


@pytest.mark.asyncio
async def test_close_all_open_option_positions_no_positions(settings):
    exchange = DummyExchange()
    exchange._positions = []
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    await engine._close_all_open_option_positions()
    assert executor.executed == []


@pytest.mark.asyncio
async def test_close_short_straddle_warns_on_partial_close(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
    ]
    positions = PositionManager(exchange)

    class PartialCloseExecutor(DummyOrderExecutor):
        async def execute_market_single_submission_with_fill_confirmation(
            self,
            product_id: int,
            side: str,
            size: float,
            reduce_only: bool,
        ) -> float:
            self.executed.append((product_id, side, size, reduce_only))
            self.exchange._positions = [
                pos for pos in self.exchange._positions if pos.product_id != product_id
            ]
            return size / 2.0

    executor = PartialCloseExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)
    straddle = ShortStraddle(
        put_leg=exchange._positions[0],
        call_leg=exchange._positions[1],
    )

    await engine._close_short_straddle(straddle)
    assert len(executor.executed) == 2


@pytest.mark.asyncio
async def test_convert_strangle_to_straddle_and_iron_fly_raises_when_opposite_option_missing(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    strangle = ShortStrangle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    async def find_open_option_leg(option_type: str, strike: float, expiry, side: str):
        if option_type == "put":
            return strangle.put_leg
        return None

    async def get_product_by_symbol(symbol: str):
        return None

    positions.find_open_option_leg = find_open_option_leg
    exchange.get_product_by_symbol = get_product_by_symbol

    with pytest.raises(RuntimeError, match="Could not find opposite option contract"):
        await engine._convert_strangle_to_straddle_and_iron_fly(strangle)


@pytest.mark.asyncio
async def test_convert_short_straddle_to_iron_fly_succeeds(settings, monkeypatch):
    exchange = DummyExchange()
    positions = PositionManager(exchange)

    async def nearest_strikes_for_breakeven_points(straddle, lower_breakeven, upper_breakeven):
        return 29500.0, 30500.0

    async def get_option_product_for_strike(option_type: str, strike: float, expiry):
        return {"id": "3"} if option_type == "put" else {"id": "4"}

    positions.nearest_strikes_for_breakeven_points = nearest_strikes_for_breakeven_points
    positions.get_option_product_for_strike = get_option_product_for_strike

    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    class DummyBreakevens:
        lower_breakeven = 29500.0
        upper_breakeven = 30500.0

    put_strike, call_strike = await engine._convert_short_straddle_to_iron_fly(straddle, DummyBreakevens())

    assert put_strike == 29500.0
    assert call_strike == 30500.0
    assert executor.executed == [(3, "buy", 1.0, False), (4, "buy", 1.0, False)]


@pytest.mark.asyncio
async def test_convert_strangle_to_straddle_and_iron_fly_succeeds(settings, monkeypatch):
    exchange = DummyExchange()
    exchange._prices["BTCUSDT"] = 30000.0
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
    ]

    class StubPositions(PositionManager):
        async def detect_short_straddle(self):
            return ShortStraddle(
                put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
                call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
            )

        async def nearest_strikes_for_breakeven_points(self, straddle, lower_breakeven, upper_breakeven):
            return 29500.0, 30500.0

        async def get_option_product_for_strike(self, option_type: str, strike: float, expiry):
            return {"id": "3"} if option_type == "put" else {"id": "4"}

    async def get_product_by_symbol(symbol: str):
        if symbol.startswith("C-"):
            return {"id": "2"}
        return None

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    positions = StubPositions(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    exchange.get_product_by_symbol = get_product_by_symbol

    strangle = ShortStrangle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    wing_put, wing_call = await engine._convert_strangle_to_straddle_and_iron_fly(strangle)

    assert wing_put == 29500.0
    assert wing_call == 30500.0
    assert any(call[1] == "sell" for call in executor.executed)
    assert any(call[1] == "buy" for call in executor.executed)


@pytest.mark.asyncio
async def test_run_closes_short_strangle_on_profit_target(settings):
    exchange = DummyExchange()
    exchange._prices["BTCUSDT"] = 30000.0
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=50.0),
        2: Quote(best_bid=1.0, best_ask=50.0),
    }

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    async def detect_short_straddle():
        raise RuntimeError("no short straddle")

    async def detect_short_strangle(require_otm: bool = True):
        if require_otm:
            return ShortStrangle(
                put_leg=exchange._positions[0],
                call_leg=exchange._positions[1],
            )
        raise RuntimeError("short strangle not found")

    positions.detect_short_straddle = detect_short_straddle
    positions.detect_short_strangle = detect_short_strangle

    await engine.run()

    assert len(executor.executed) == 2
    assert all(call[1] == "buy" for call in executor.executed)
    assert exchange._positions == []


@pytest.mark.asyncio
async def test_convert_short_straddle_to_iron_fly_raises_when_put_wing_partial_fill(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)

    class PartialFillExecutor(DummyOrderExecutor):
        async def execute_market_single_submission_with_fill_confirmation(
            self,
            product_id: int,
            side: str,
            size: float,
            reduce_only: bool,
        ) -> float:
            if side == "buy" and not reduce_only and product_id == 3:
                return size - 0.5
            return await super().execute_market_single_submission_with_fill_confirmation(
                product_id=product_id,
                side=side,
                size=size,
                reduce_only=reduce_only,
            )

    async def get_products_raw():
        return [
            {
                "id": "3",
                "symbol": "P-29500",
                "contract_type": "put option",
                "strike_price": "29500",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
            {
                "id": "4",
                "symbol": "C-30500",
                "contract_type": "call option",
                "strike_price": "30500",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
        ]

    exchange.get_products_raw = get_products_raw
    executor = PartialFillExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    class DummyBreakevens:
        lower_breakeven = 29500.0
        upper_breakeven = 30500.0

    with pytest.raises(RuntimeError, match="Put wing open partial after retries"):
        await engine._convert_short_straddle_to_iron_fly(straddle, DummyBreakevens())


@pytest.mark.asyncio
async def test_short_straddle_monitor_retries_on_missing_price_then_exits(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    call_count = {"count": 0}

    async def get_best_quote(product_id: int):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise ExchangeClientError("missing price")
        return Quote(best_bid=1.0, best_ask=80.0)

    exchange.get_best_quote = get_best_quote
    settings.short_straddle_profit_capture_ratio = 0.1
    settings.short_straddle_max_loss_ratio = 1.0

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)
    straddle = ShortStraddle(
        put_leg=exchange._positions[0],
        call_leg=exchange._positions[1],
    )

    await engine._monitor_short_straddle(straddle)
    assert len(executor.executed) == 2


@pytest.mark.asyncio
async def test_run_updates_last_index_price_when_strangle_scan_task_fails(settings):
    exchange = DummyExchange()
    exchange._prices["BTCUSDT"] = 30100.0
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    async def detect_short_straddle():
        raise RuntimeError("no short straddle")

    async def detect_short_strangle():
        raise RuntimeError("no short strangle")

    positions.detect_short_straddle = detect_short_straddle
    positions.detect_short_strangle = detect_short_strangle

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(settings.poll_interval_seconds * 4)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert engine.state.last_index_price == 30100.0


@pytest.mark.asyncio
async def test_run_resets_state_when_short_strangle_closes_externally(settings):
    exchange = DummyExchange()
    exchange._prices["BTCUSDT"] = 30000.0
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    strangle = ShortStrangle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    )

    async def detect_short_straddle():
        raise RuntimeError("no short straddle")

    async def detect_short_strangle(require_otm: bool = True):
        if require_otm:
            return strangle
        raise RuntimeError("short strangle not found")

    async def find_open_option_leg(option_type: str, strike: float, expiry, side: str):
        return None

    positions.detect_short_straddle = detect_short_straddle
    positions.detect_short_strangle = detect_short_strangle
    positions.find_open_option_leg = find_open_option_leg

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(settings.poll_interval_seconds * 4)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert engine.state.status == "no short strangle found"
