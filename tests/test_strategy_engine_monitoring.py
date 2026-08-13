import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, ShortStraddle, Quote


class SimpleSettings:
    def __init__(self, strike_adjustment_threshold=10.0, profit_target=50.0, stop_loss=100.0, poll_interval_seconds=0.01):
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
    def __init__(self, open_legs=None):
        self._open_legs = open_legs or []

    async def find_open_option_leg(self, option_type, strike, expiry, side):
        for leg in self._open_legs:
            if leg.option_type == option_type and leg.expiry.date() == expiry.date() and abs(leg.strike - strike) < 1e-6:
                # match side short/long
                if side == "short" and leg.size < 0:
                    return leg
                if side == "long" and leg.size > 0:
                    return leg
        return None

    async def detect_short_straddle(self):
        shorts = [x for x in self._open_legs if x.size < 0]
        puts = [x for x in shorts if x.option_type == "put"]
        calls = [x for x in shorts if x.option_type == "call"]
        for p in puts:
            for c in calls:
                if abs(p.strike - c.strike) < 1e-3 and p.expiry.date() == c.expiry.date():
                    return ShortStraddle(put_leg=p, call_leg=c)
        raise RuntimeError("No short straddle")

    async def get_option_product_for_strike(self, option_type, strike, expiry):
        return {"id": 999, "strike": strike}


class FakeExchange:
    def __init__(self, legs=None, bid=1.0, ask=2.0, index_price=1000.0):
        self._legs = legs or []
        self._bid = bid
        self._ask = ask
        self._index = index_price

    async def get_best_quote(self, product_id):
        return Quote(best_bid=self._bid, best_ask=self._ask)

    async def parse_option_positions(self):
        return self._legs

    async def get_index_price(self, symbol="BTCUSDT"):
        return self._index


def make_leg(pid, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=pid, symbol=f"OPT-{pid}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=entry_price, contract_value=1.0)


@pytest.mark.asyncio
async def test_monitor_short_straddle_both_missing_returns_false():
    settings = SimpleSettings(poll_interval_seconds=0.001)
    # empty positions
    positions = FakePositions(open_legs=[])
    exchange = FakeExchange(legs=[])
    executor = FakeExecutor()
    engine = StrategyEngine(exchange, positions, executor, settings)

    # Create a dummy straddle
    put = make_leg(1, "put", 100.0, -1)
    call = make_leg(2, "call", 100.0, -1)
    straddle = ShortStraddle(put_leg=put, call_leg=call)

    # When both legs missing in the account, should return False
    result = await engine._monitor_short_straddle(straddle)
    assert result is False


@pytest.mark.asyncio
async def test_monitor_short_straddle_one_missing_triggers_exit():
    settings = SimpleSettings(poll_interval_seconds=0.001)
    # only one leg exists in account (simulate missing call)
    existing_put = make_leg(1, "put", 100.0, -1)
    positions = FakePositions(open_legs=[existing_put])
    # exchange returns one leg
    exchange = FakeExchange(legs=[existing_put])
    executor = FakeExecutor()
    engine = StrategyEngine(exchange, positions, executor, settings)

    put = make_leg(1, "put", 100.0, -1)
    call = make_leg(2, "call", 100.0, -1)
    straddle = ShortStraddle(put_leg=put, call_leg=call)

    result = await engine._monitor_short_straddle(straddle)
    # should trigger exit and return True
    assert result is True
    assert engine.state.exit_reason == "stop_loss"
    assert engine.state.triggered is True


@pytest.mark.asyncio
async def test_check_strategy_exit_conditions_triggers_profit_and_stop():
    # test profit exit
    settings = SimpleSettings(profit_target=5.0, stop_loss=10.0)
    # two legs to compute unrealized pnl
    leg1 = make_leg(1, "call", 100.0, -1, entry_price=1.0)
    leg2 = make_leg(2, "put", 100.0, -1, entry_price=1.0)
    # set exchange quotes so unrealized is large
    exchange = FakeExchange(legs=[leg1, leg2], bid=100.0, ask=101.0)
    positions = FakePositions(open_legs=[leg1, leg2])
    executor = FakeExecutor()
    engine = StrategyEngine(exchange, positions, executor, settings)

    # no realized pnl, but unrealized will be high due to quotes
    triggered = await engine._check_strategy_exit_conditions()
    assert triggered is True
    assert engine.state.exit_reason in {"profit", "stop_loss"}

    # test stop loss negative (simulate large adverse move where shorts lose)
    settings2 = SimpleSettings(profit_target=10000.0, stop_loss=1.0)
    # set quotes very high so short positions incur losses (entry_price - ask becomes negative)
    exchange2 = FakeExchange(legs=[leg1, leg2], bid=100.0, ask=101.0)
    engine2 = StrategyEngine(exchange2, positions, executor, settings2)
    triggered2 = await engine2._check_strategy_exit_conditions()
    assert triggered2 is True
    assert engine2.state.exit_reason == "stop_loss"
