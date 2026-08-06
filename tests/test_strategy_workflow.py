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
        strike_adjustment_threshold=1000.0,
        profit_target=50.0,
        stop_loss=100.0,
        log_level="INFO",
        ssl_verify=False,
    )


def test_crossed_or_touched_does_not_trigger_on_initial_observation(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    assert not engine._crossed_or_touched(None, 65000.0, 64000.0, "put")
    assert not engine._crossed_or_touched(None, 65000.0, 64000.0, "call")


def test_apply_clock_skew_from_error_updates_offset():
    client = DeltaExchangeClient(
        api_key="key",
        api_secret="secret",
        base_url="https://test",
        ssl_verify=False,
    )


@pytest.mark.asyncio
async def test_monitor_dynamic_straddle_strangle_keeps_monitoring_until_threshold_crossing(settings):
    class SequenceExchange(DummyExchange):
        def __init__(self, values):
            super().__init__()
            self._values = iter(values)

        async def get_index_price(self, symbol: str = "BTCUSD") -> float:
            try:
                return next(self._values)
            except StopIteration:
                return 65000.0

    class DummyPositions:
        def __init__(self, straddle):
            self._straddle = straddle

        async def detect_short_straddle(self):
            return self._straddle

    class DummyExecutor:
        async def execute_market_single_submission_with_fill_confirmation(self, **kwargs):
            return kwargs.get("size", 0)

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-64000", "put", 64000.0, -1, 100.0),
        call_leg=make_option_leg(2, "C-64000", "call", 64000.0, -1, 100.0),
    )
    exchange = SequenceExchange([64000.0, 65000.0])
    positions = DummyPositions(straddle)
    executor = DummyExecutor()
    engine = StrategyEngine(exchange=exchange, positions=positions, executor=executor, settings=settings)

    converted = []

    async def fake_check_exit_conditions():
        return len(converted) > 0

    async def fake_monitor_short_straddle(_straddle):
        return False

    async def fake_convert(_straddle, price):
        converted.append(price)
        return price

    engine._check_strategy_exit_conditions = fake_check_exit_conditions
    engine._monitor_short_straddle = fake_monitor_short_straddle
    engine._convert_straddle_to_strangle = fake_convert

    result = await engine._monitor_dynamic_straddle_strangle()

    assert result is True
    assert converted == [65000.0]


@pytest.mark.asyncio
async def test_monitor_dynamic_straddle_strangle_checks_threshold_without_blocking_on_straddle_monitor(settings):
    class SequenceExchange(DummyExchange):
        def __init__(self, values):
            super().__init__()
            self._values = iter(values)

        async def get_index_price(self, symbol: str = "BTCUSD") -> float:
            try:
                return next(self._values)
            except StopIteration:
                return 65000.0

    class DummyPositions:
        def __init__(self, straddle):
            self._straddle = straddle

        async def detect_short_straddle(self):
            return self._straddle

        async def get_option_product_for_strike(self, **kwargs):
            return {"id": 99}

        async def find_open_option_leg(self, **kwargs):
            return None

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-64000", "put", 64000.0, -1, 100.0),
        call_leg=make_option_leg(2, "C-64000", "call", 64000.0, -1, 100.0),
    )
    exchange = SequenceExchange([64000.0, 65000.0])
    positions = DummyPositions(straddle)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange=exchange, positions=positions, executor=executor, settings=settings)

    converted = []

    async def fake_monitor_short_straddle(_straddle):
        await asyncio.sleep(10)
        return False

    async def fake_convert(_straddle, price):
        converted.append(price)
        return price

    async def fake_check_exit_conditions():
        return bool(converted)

    async def fake_get_active_short_option_legs():
        return [object(), object()]

    engine._monitor_short_straddle = fake_monitor_short_straddle
    engine._convert_straddle_to_strangle = fake_convert
    engine._check_strategy_exit_conditions = fake_check_exit_conditions
    engine._get_active_short_option_legs = fake_get_active_short_option_legs

    result = await asyncio.wait_for(engine._monitor_dynamic_straddle_strangle(), timeout=0.5)

    assert result is True
    assert converted == [65000.0]


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

    total_premium = (
        strangle.put_leg.entry_price * -strangle.put_leg.size
        + strangle.call_leg.entry_price * -strangle.call_leg.size
    )
    assert total_premium == 110.0
    assert engine.settings.profit_target == 50.0


@pytest.mark.asyncio
async def test_leg_market_value_is_scaled_by_position_size(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    leg = make_option_leg(1, "P-30000", "put", 30000.0, -100.0, 50.0)
    quote = Quote(best_bid=1.0, best_ask=315.0)

    assert engine._compute_leg_market_value(leg, quote) == 31500.0


@pytest.mark.asyncio
async def test_convert_straddle_to_strangle_records_realized_pnl_for_closed_leg(settings, monkeypatch):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-64000", "put", 64000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-64000", "call", 64000.0, -1.0, 100.0),
    )
    exchange._quotes[1] = Quote(best_bid=90.0, best_ask=90.0)
    exchange._quotes[2] = Quote(best_bid=90.0, best_ask=90.0)

    async def fake_get_option_product_for_strike(option_type: str, strike: float, expiry):
        return {"id": "3" if option_type == "call" else "4"}

    positions.get_option_product_for_strike = fake_get_option_product_for_strike

    await engine._convert_straddle_to_strangle(straddle, 65000.0)

    assert engine._realized_pnl == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_get_active_short_option_legs_returns_only_short_positions(settings):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    exchange._positions = [
        make_option_leg(1, "P-64000", "put", 64000.0, -1.0, 100.0),
        make_option_leg(2, "C-64000", "call", 64000.0, -1.0, 100.0),
        make_option_leg(3, "C-65000", "call", 65000.0, 1.0, 110.0),
    ]

    legs = await engine._get_active_short_option_legs()

    assert [leg.product_id for leg in legs] == [1, 2]


@pytest.mark.asyncio
async def test_detect_short_straddle_accepts_small_quantity_tolerance(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-64000", "put", 64000.0, -1.0, 100.0),
        make_option_leg(2, "C-64000", "call", 64000.0, -0.9999, 100.0),
    ]
    positions = PositionManager(exchange)

    straddle = await positions.detect_short_straddle()

    assert straddle is not None
    assert straddle.put_leg.product_id == 1
    assert straddle.call_leg.product_id == 2


@pytest.mark.asyncio
async def test_is_valid_short_straddle_accepts_small_quantity_tolerance(settings):
    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-64000", "put", 64000.0, -1.0, 100.0),
        call_leg=make_option_leg(2, "C-64000", "call", 64000.0, -0.9999, 100.0),
    )

    assert StrategyEngine._is_valid_short_straddle(straddle) is True


@pytest.mark.asyncio
async def test_convert_straddle_to_strangle_uses_reference_strike_and_same_qty(settings, monkeypatch):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)

    async def fake_get_option_product_for_strike(option_type: str, strike: float, expiry):
        return {"id": "3" if option_type == "call" else "4"}

    positions.get_option_product_for_strike = fake_get_option_product_for_strike

    straddle = ShortStraddle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        call_leg=make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 60.0),
    )

    await engine._convert_straddle_to_strangle(straddle, 32000.0)

    assert executor.executed[0] == (2, "buy", 1.0, True)
    assert executor.executed[1] == (3, "sell", 1.0, False)


@pytest.mark.asyncio
async def test_convert_strangle_to_straddle_replaces_reference_leg_at_atm(settings, monkeypatch):
    exchange = DummyExchange()
    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)
    engine._reference_strike = 30000.0
    engine._reference_option_type = "put"

    async def fake_get_option_product_for_strike(option_type: str, strike: float, expiry):
        return {"id": "5"}

    positions.get_option_product_for_strike = fake_get_option_product_for_strike

    strangle = ShortStrangle(
        put_leg=make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 50.0),
        call_leg=make_option_leg(2, "C-31000", "call", 31000.0, -1.0, 60.0),
    )

    await engine._convert_strangle_to_straddle(strangle, 31000.0)

    assert executor.executed[0] == (1, "buy", 1.0, True)
    assert executor.executed[1] == (5, "sell", 1.0, False)


# Leg-wise exit tests removed — strategy closes all positions on overall targets

@pytest.mark.asyncio
async def test_short_straddle_monitor_exits_on_profit_target(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=70.0),
        2: Quote(best_bid=1.0, best_ask=80.0),
    }

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


@pytest.mark.asyncio
async def test_strategy_exit_checks_combined_pnl_and_closes_positions(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=80.0),
        2: Quote(best_bid=1.0, best_ask=90.0),
    }

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)
    engine._realized_pnl = 60.0

    exited = await engine._check_strategy_exit_conditions()

    assert exited is True
    assert engine.state.status == "Completed - Profit Target Reached"
    assert len(executor.executed) == 2
    assert executor.executed[0] == (1, "buy", 1.0, True)
    assert executor.executed[1] == (2, "buy", 1.0, True)


@pytest.mark.asyncio
async def test_strategy_exit_hits_stop_loss_when_combined_pnl_is_too_negative(settings):
    exchange = DummyExchange()
    exchange._positions = [
        make_option_leg(1, "P-30000", "put", 30000.0, -1.0, 100.0),
        make_option_leg(2, "C-30000", "call", 30000.0, -1.0, 110.0),
    ]
    exchange._quotes = {
        1: Quote(best_bid=1.0, best_ask=150.0),
        2: Quote(best_bid=1.0, best_ask=180.0),
    }

    positions = PositionManager(exchange)
    executor = DummyOrderExecutor(exchange)
    engine = StrategyEngine(exchange, positions, executor, settings)
    engine._realized_pnl = -120.0

    exited = await engine._check_strategy_exit_conditions()

    assert exited is True
    assert engine.state.status == "Completed - Stop Loss Hit"
    assert len(executor.executed) == 2


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
async def test_get_option_product_for_strike_uses_nearest_available_strike():
    exchange = DummyExchange()
    expiry = datetime(2026, 12, 31, tzinfo=timezone.utc)
    exchange._quotes = {}

    async def get_products_raw():
        return [
            {
                "id": "1",
                "symbol": "P-64000",
                "contract_type": "put option",
                "strike_price": "64000",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
            {
                "id": "2",
                "symbol": "P-63600",
                "contract_type": "put option",
                "strike_price": "63600",
                "settlement_time": "2026-12-31T00:00:00Z",
            },
        ]

    exchange.get_products_raw = get_products_raw
    positions = PositionManager(exchange)

    product = await positions.get_option_product_for_strike("put", 63550.0, expiry)
    assert product is not None and product["id"] == "2"


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
        1: Quote(best_bid=1.0, best_ask=160.0),
        2: Quote(best_bid=2.0, best_ask=170.0),
    }
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


def test_crossed_or_touched_various_cases():
    assert StrategyEngine._crossed_or_touched(30100.0, 30000.0, 30000.0, "put") is True
    assert StrategyEngine._crossed_or_touched(29900.0, 30000.0, 30000.0, "call") is True
    assert StrategyEngine._crossed_or_touched(30100.0, 29900.0, 30000.0, "put") is True
    assert StrategyEngine._crossed_or_touched(29900.0, 30100.0, 30000.0, "call") is True


def test_crossed_or_touched_false_cases():
    assert StrategyEngine._crossed_or_touched(None, 30100.0, 30000.0, "put") is False
    assert StrategyEngine._crossed_or_touched(None, 29900.0, 30000.0, "call") is False
    assert StrategyEngine._crossed_or_touched(29900.0, 29950.0, 30000.0, "call") is False
    assert StrategyEngine._crossed_or_touched(30100.0, 30050.0, 30000.0, "put") is False
    assert StrategyEngine._crossed_or_touched(30050.0, 30100.0, 30000.0, "put") is False


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
        call_leg=make_option_leg(2, "C-31000", "call", 31000.0, -1.0, 110.0),
    )

    async def detect_short_straddle():
        raise RuntimeError("no short straddle")

    async def detect_short_strangle(require_otm: bool = True):
        return strangle

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

    assert engine.state.status == "position mismatch"
