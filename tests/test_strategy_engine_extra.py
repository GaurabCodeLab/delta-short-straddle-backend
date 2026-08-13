import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, ShortStraddle, ShortStrangle, Quote


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
        # Simulate immediate full fill
        await asyncio.sleep(0)
        return float(size)


class FakePositions:
    def __init__(self, new_product_id=None):
        self.new_product_id = new_product_id or 999

    async def get_option_product_for_strike(self, option_type, strike, expiry):
        # return a simple product mapping expected by StrategyEngine
        return {"id": self.new_product_id, "strike": strike, "expiry": expiry.isoformat()}

    async def find_open_option_leg(self, option_type, strike, expiry, side):
        return None

    async def detect_short_straddle(self):
        return None

    async def detect_short_strangle(self, require_otm=True):
        return None


class FakeExchange:
    def __init__(self, quote_bid=1.0, quote_ask=2.0, legs=None):
        self._quote_bid = quote_bid
        self._quote_ask = quote_ask
        self._legs = legs or []

    async def get_best_quote(self, product_id):
        return Quote(best_bid=self._quote_bid, best_ask=self._quote_ask)

    async def parse_option_positions(self):
        return self._legs

    async def get_index_price(self, symbol="BTCUSDT"):
        return 1000.0


def make_leg(product_id, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=product_id, symbol=f"OPT-{product_id}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=float(entry_price), contract_value=1.0)


@pytest.mark.asyncio
async def test_crossed_or_touched_staticmethod():
    # Import statically
    from src.strategy_engine import StrategyEngine

    # prev None should return False for touch (except exact equality)
    assert StrategyEngine._crossed_or_touched(None, 100.0, 100.0, "call") is True
    assert StrategyEngine._crossed_or_touched(None, 101.0, 100.0, "call") is False

    # call crossing from below to above
    assert StrategyEngine._crossed_or_touched(99.0, 101.0, 100.0, "call") is True
    assert StrategyEngine._crossed_or_touched(101.0, 99.0, 100.0, "call") is False

    # put crossing from above to below
    assert StrategyEngine._crossed_or_touched(101.0, 99.0, 100.0, "put") is True
    assert StrategyEngine._crossed_or_touched(99.0, 101.0, 100.0, "put") is False


@pytest.mark.asyncio
async def test_convert_straddle_to_strangle_upper_and_lower_conversion():
    settings = SimpleSettings(strike_adjustment_threshold=5.0)
    # create a straddle with strike 100
    put = make_leg(1, "put", 100.0, -1, entry_price=10.0)
    call = make_leg(2, "call", 100.0, -1, entry_price=8.0)
    straddle = ShortStraddle(put_leg=put, call_leg=call)

    executor = FakeExecutor()
    positions = FakePositions(new_product_id=555)
    exchange = FakeExchange(quote_bid=1.0, quote_ask=1.5)

    engine = StrategyEngine(exchange, positions, executor, settings)

    # Upper conversion: index above reference + threshold
    # ensure atm reference strike will be set to 100, call with index 106 (>100+5)
    new_strike = await engine._convert_straddle_to_strangle(straddle, 106.0)
    assert new_strike == 100.0 + 2 * settings.strike_adjustment_threshold
    assert engine._current_structure == "strangle"

    # Reset current structure and atm reference for lower conversion
    engine._current_structure = "unknown"
    engine._atm_reference_strike = None
    # Lower conversion: index below reference - threshold
    new_strike2 = await engine._convert_straddle_to_strangle(straddle, 94.0)
    assert new_strike2 == 100.0 - 2 * settings.strike_adjustment_threshold
    assert engine._current_structure == "strangle"


@pytest.mark.asyncio
async def test_convert_strangle_to_straddle_call_and_put_conversion():
    settings = SimpleSettings(strike_adjustment_threshold=5.0)
    # create a strangle with put strike 90 and call strike 110
    put = make_leg(1, "put", 90.0, -1, entry_price=5.0)
    call = make_leg(2, "call", 110.0, -1, entry_price=7.0)
    strangle = ShortStrangle(put_leg=put, call_leg=call)

    executor = FakeExecutor()
    positions = FakePositions(new_product_id=777)
    exchange = FakeExchange(quote_bid=1.0, quote_ask=1.5)

    engine = StrategyEngine(exchange, positions, executor, settings)

    # Test conversion when index >= call strike
    new_strike = await engine._convert_strangle_to_straddle(strangle, 111.0)
    assert new_strike == float(strangle.call_leg.strike)
    assert engine._atm_reference_strike == new_strike
    assert engine._current_structure == "straddle"

    # Test conversion when index <= put strike
    engine._atm_reference_strike = None
    engine._current_structure = "unknown"
    new_strike2 = await engine._convert_strangle_to_straddle(strangle, 89.0)
    assert new_strike2 == float(strangle.put_leg.strike)
    assert engine._atm_reference_strike == new_strike2
    assert engine._current_structure == "straddle"


@pytest.mark.asyncio
async def test_calculate_strategy_pnl_combines_realized_and_unrealized():
    settings = SimpleSettings()
    # create legs and set exchange to return quotes
    leg1 = make_leg(1, "call", 100, -1, entry_price=2.0)
    leg2 = make_leg(2, "put", 100, -1, entry_price=3.0)

    exchange = FakeExchange(quote_bid=5.0, quote_ask=6.0, legs=[leg1, leg2])
    executor = FakeExecutor()
    positions = FakePositions()

    engine = StrategyEngine(exchange, positions, executor, settings)
    # simulate some realized pnl
    engine._realized_pnl = 10.0

    realized, unrealized, combined = await engine._calculate_strategy_pnl()
    assert realized == pytest.approx(10.0)
    # unrealized is computed using _calculate_realized_pnl_from_close with best bid/ask
    assert unrealized != 0.0
    assert combined == pytest.approx(realized + unrealized)
