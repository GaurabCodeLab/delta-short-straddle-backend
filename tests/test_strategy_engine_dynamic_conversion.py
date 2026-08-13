import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, ShortStraddle, ShortStrangle, Quote


class SimpleSettings:
    def __init__(self, strike_adjustment_threshold=5.0, profit_target=1e9, stop_loss=1e9, poll_interval_seconds=0.001):
        self.strike_adjustment_threshold = strike_adjustment_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.poll_interval_seconds = poll_interval_seconds


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only=False):
        self.calls.append((product_id, side, size, reduce_only))
        await asyncio.sleep(0)
        return float(size)


class FakePositions:
    def __init__(self, short_straddle=None, short_strangle=None, new_product_id=1234):
        self._straddle = short_straddle
        self._strangle = short_strangle
        self._new_product_id = new_product_id

    async def detect_short_straddle(self):
        if self._straddle is None:
            raise RuntimeError("No short straddle")
        return self._straddle

    async def detect_short_strangle(self, require_otm=False):
        if self._strangle is None:
            raise RuntimeError("No short strangle")
        return self._strangle

    async def get_option_product_for_strike(self, option_type, strike, expiry):
        return {"id": self._new_product_id, "strike": strike}
    
    async def find_open_option_leg(self, option_type, strike, expiry, side):
        # return matching leg from either straddle or strangle
        legs = []
        if self._straddle is not None:
            legs = [self._straddle.put_leg, self._straddle.call_leg]
        if self._strangle is not None:
            legs = [self._strangle.put_leg, self._strangle.call_leg]
        for leg in legs:
            if leg.option_type == option_type and abs(leg.strike - strike) < 1e-6:
                if side == "short" and leg.size < 0:
                    return leg
                if side == "long" and leg.size > 0:
                    return leg
        return None


class SeqIndexExchange:
    def __init__(self, index_sequence, legs=None):
        self._seq = list(index_sequence)
        self._legs = legs or []
        self._last = self._seq[-1] if self._seq else None

    async def get_index_price(self, symbol="BTCUSDT"):
        # return and pop first element; if empty return last
        if self._seq:
            val = self._seq.pop(0)
            self._last = val
            return val
        if self._last is None:
            raise AttributeError("'SeqIndexExchange' object has no attribute '_last'")
        return self._last

    async def parse_option_positions(self):
        return self._legs

    async def get_best_quote(self, product_id):
        return Quote(best_bid=100.0, best_ask=101.0)


def make_leg(pid, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=pid, symbol=f"OPT-{pid}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=entry_price, contract_value=1.0)


@pytest.mark.asyncio
async def test_straddle_to_strangle_triggers_executor_calls():
    # create short straddle at strike 100
    put = make_leg(1, "put", 100.0, -1)
    call = make_leg(2, "call", 100.0, -1)
    straddle = ShortStraddle(put_leg=put, call_leg=call)

    # index moves immediately above reference+threshold to trigger conversion
    exchange = SeqIndexExchange(index_sequence=[110.0])
    positions = FakePositions(short_straddle=straddle, new_product_id=999)
    executor = FakeExecutor()
    settings = SimpleSettings(strike_adjustment_threshold=5.0, poll_interval_seconds=0.001)

    engine = StrategyEngine(exchange, positions, executor, settings)
    # ensure previous index price is set to avoid None formatting in logs
    engine.state.last_index_price = 100.0

    # run monitor loop briefly; use timeout to stop after conversion
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(engine._monitor_dynamic_straddle_strangle(), timeout=0.05)

    # verify executor received a buy to close existing call and a sell to open new call
    assert len(executor.calls) >= 2
    assert executor.calls[0][1] == "buy"  # close existing call
    assert executor.calls[1][1] == "sell"  # open new call


@pytest.mark.asyncio
async def test_strangle_to_straddle_triggers_executor_calls():
    # create short strangle (put lower, call higher)
    put = make_leg(3, "put", 90.0, -1)
    call = make_leg(4, "call", 110.0, -1)
    strangle = ShortStrangle(put_leg=put, call_leg=call)

    # index moves to or above call strike to trigger strangle->straddle conversion
    exchange = SeqIndexExchange(index_sequence=[115.0])
    # ensure parse_option_positions returns the active short legs so reconciliation passes
    exchange._legs = [put, call]
    positions = FakePositions(short_strangle=strangle, new_product_id=777)
    executor = FakeExecutor()
    settings = SimpleSettings(strike_adjustment_threshold=5.0, poll_interval_seconds=0.001)

    engine = StrategyEngine(exchange, positions, executor, settings)
    # ensure previous index price is set to avoid None formatting in logs
    engine.state.last_index_price = 100.0

    # run monitor loop briefly; expect timeout after conversion side-effects
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(engine._monitor_dynamic_straddle_strangle(), timeout=0.05)

    # verify executor performed close then open
    assert len(executor.calls) >= 2
    assert executor.calls[0][1] == "buy"  # close put
    assert executor.calls[1][1] == "sell"  # open put at new strike


@pytest.mark.asyncio
async def test_straddle_to_strangle_lower_threshold_triggers_put_adjustment():
    # create short straddle at strike 100
    put = make_leg(21, "put", 100.0, -1)
    call = make_leg(22, "call", 100.0, -1)
    straddle = ShortStraddle(put_leg=put, call_leg=call)

    # index moves immediately below reference-threshold to trigger lower conversion
    exchange = SeqIndexExchange(index_sequence=[90.0])
    positions = FakePositions(short_straddle=straddle, new_product_id=555)
    executor = FakeExecutor()
    settings = SimpleSettings(strike_adjustment_threshold=5.0, poll_interval_seconds=0.001)

    engine = StrategyEngine(exchange, positions, executor, settings)
    engine.state.last_index_price = 100.0

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(engine._monitor_dynamic_straddle_strangle(), timeout=0.05)

    # expect executor to buy to close put then sell new put
    assert len(executor.calls) >= 2
    assert executor.calls[0][1] == "buy"
    assert executor.calls[1][1] == "sell"
