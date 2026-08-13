import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from src.strategy_engine import StrategyEngine
from src.models import OptionLeg, Quote, ShortStraddle, ShortStrangle


class DummySettings:
    def __init__(self):
        self.profit_target = 1000.0
        self.stop_loss = 1000.0
        self.poll_interval_seconds = 0.001
        self.strike_adjustment_threshold = 10.0


class FakeExecutor:
    def __init__(self):
        self.calls = []

    async def execute_market_single_submission_with_fill_confirmation(self, product_id, side, size, reduce_only=False):
        self.calls.append((product_id, side, size, reduce_only))
        await asyncio.sleep(0)
        return float(size)


class ExchangeWithCancel:
    def __init__(self, legs=None, quotes=None, index_price=100.0, cancel_raises=False):
        self._legs = legs or []
        self._quotes = quotes or {}
        self._index = index_price
        self._cancel_raises = cancel_raises

    async def cancel_all_orders(self):
        if self._cancel_raises:
            raise RuntimeError("cancel failed")
        await asyncio.sleep(0)

    async def parse_option_positions(self):
        return self._legs

    async def get_best_quote(self, product_id):
        return self._quotes.get(product_id, Quote(best_bid=0.5, best_ask=0.6))

    async def get_index_price(self, symbol="BTCUSDT"):
        return self._index


class DummyPositions:
    def __init__(self, legs=None):
        self._legs = legs or []

    async def find_open_option_leg(self, option_type, strike, expiry, side):
        for leg in self._legs:
            if leg.option_type == option_type and abs(leg.strike - strike) < 1e-6:
                if side == "short" and leg.size < 0:
                    return leg
                if side == "long" and leg.size > 0:
                    return leg
        return None

    async def get_option_product_for_strike(self, option_type, strike, expiry):
        return {"id": 999, "strike": strike}


def make_leg(pid, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=pid, symbol=f"OPT-{pid}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=entry_price, contract_value=1.0)


@pytest.mark.asyncio
async def test_cancel_pending_orders_noop_and_exception():
    settings = DummySettings()
    executor = FakeExecutor()
    ex = ExchangeWithCancel()
    pos = DummyPositions()
    engine = StrategyEngine(ex, pos, executor, settings)

    # callable present and succeeds
    await engine._cancel_pending_orders()

    # callable present and raises
    ex2 = ExchangeWithCancel(cancel_raises=True)
    engine2 = StrategyEngine(ex2, pos, executor, settings)
    await engine2._cancel_pending_orders()


def test_is_valid_short_straddle_and_strangle():
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    put = OptionLeg(product_id=1, symbol="p", option_type="put", strike=100.0, expiry=expiry, size=-1, entry_price=1.0, contract_value=1.0)
    call = OptionLeg(product_id=2, symbol="c", option_type="call", strike=100.0, expiry=expiry, size=-1, entry_price=1.0, contract_value=1.0)
    straddle = ShortStraddle(put_leg=put, call_leg=call)
    assert StrategyEngine._is_valid_short_straddle(straddle) is True

    # strangle should be invalid if strikes equal
    strangle = ShortStrangle(put_leg=put, call_leg=call)
    assert StrategyEngine._is_valid_short_strangle(strangle) is False

    # modify call strike to make it a valid strangle
    call2 = OptionLeg(product_id=3, symbol="c2", option_type="call", strike=110.0, expiry=expiry, size=-1, entry_price=1.0, contract_value=1.0)
    strangle2 = ShortStrangle(put_leg=put, call_leg=call2)
    assert StrategyEngine._is_valid_short_strangle(strangle2) is True


def test_crossed_or_touched_behaviour():
    # exact touch
    assert StrategyEngine._crossed_or_touched(None, 100.0, 100.0, "call") is True
    # prev None -> no crossing
    assert StrategyEngine._crossed_or_touched(None, 101.0, 100.0, "call") is False
    # call crossing
    assert StrategyEngine._crossed_or_touched(99.0, 101.0, 100.0, "call") is True
    # put crossing
    assert StrategyEngine._crossed_or_touched(101.0, 99.0, 100.0, "put") is True
