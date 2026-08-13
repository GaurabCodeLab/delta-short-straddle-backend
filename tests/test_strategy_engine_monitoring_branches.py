import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, Quote, ShortStraddle


class SimpleSettings:
    def __init__(self, profit_target=50.0, stop_loss=100.0, poll_interval_seconds=0.001):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.poll_interval_seconds = poll_interval_seconds
        self.strike_adjustment_threshold = 10.0


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

    async def parse_option_positions(self):
        return self._open_legs


class StaticQuoteExchange:
    def __init__(self, quotes_by_pid, index_price=1000.0):
        self._quotes = quotes_by_pid
        self._index = index_price

    async def get_best_quote(self, product_id):
        return self._quotes.get(product_id)

    async def parse_option_positions(self):
        # return legs for strategy pnl calculation
        return []

    async def get_index_price(self, symbol="BTCUSDT"):
        return self._index


def make_leg(pid, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=pid, symbol=f"OPT-{pid}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=entry_price, contract_value=1.0)


def test_calculate_realized_pnl_from_close_long_and_short():
    # long leg realized (size >0): PnL = size * (best_bid - entry)
    long_leg = make_leg(1, "call", 100.0, 1, entry_price=2.0)
    quote = Quote(best_bid=5.0, best_ask=6.0)
    pnl_long = StrategyEngine._calculate_realized_pnl_from_close(long_leg, quote)
    assert pnl_long == pytest.approx(3.0)

    # short leg realized (size <0): PnL = abs(size) * (entry_price - best_ask)
    short_leg = make_leg(2, "put", 100.0, -1, entry_price=4.0)
    quote2 = Quote(best_bid=1.0, best_ask=2.0)
    pnl_short = StrategyEngine._calculate_realized_pnl_from_close(short_leg, quote2)
    assert pnl_short == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_monitor_short_straddle_exits_on_profit_threshold():
    # create short straddle with two short legs
    put = make_leg(10, "put", 100.0, -1, entry_price=1.0)
    call = make_leg(11, "call", 100.0, -1, entry_price=1.0)
    positions = FakePositions(open_legs=[put, call])
    # set quotes such that unrealized pnl is large positive (favourable for shorts)
    quotes = {10: Quote(best_bid=0.0, best_ask=0.1), 11: Quote(best_bid=0.0, best_ask=0.1)}
    exchange = StaticQuoteExchange(quotes_by_pid=quotes)
    executor = FakeExecutor()
    settings = SimpleSettings(profit_target=1.0, stop_loss=100.0, poll_interval_seconds=0.001)

    engine = StrategyEngine(exchange, positions, executor, settings)

    # build a ShortStraddle to pass into monitor
    straddle = ShortStraddle(put_leg=put, call_leg=call)

    result = await engine._monitor_short_straddle(straddle)
    assert result is True
    # executor should have been called to close both legs
    assert any(call_info[1] == "buy" for call_info in executor.calls)


@pytest.mark.asyncio
async def test_monitor_short_straddle_returns_false_when_legs_missing():
    # empty account -> both legs missing -> should return False
    positions = FakePositions(open_legs=[])
    exchange = StaticQuoteExchange(quotes_by_pid={})
    executor = FakeExecutor()
    settings = SimpleSettings()
    engine = StrategyEngine(exchange, positions, executor, settings)

    put = make_leg(20, "put", 100.0, -1)
    call = make_leg(21, "call", 100.0, -1)
    straddle = ShortStraddle(put_leg=put, call_leg=call)

    result = await engine._monitor_short_straddle(straddle)
    assert result is False
