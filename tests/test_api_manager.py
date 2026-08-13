import asyncio
from datetime import datetime, timedelta, timezone
import logging

import pytest

from src.api import BotManager, JsonLogHandler
from src.models import OptionLeg, Quote


class DummyState:
    def __init__(self):
        self.triggered = False
        self.last_index_price = None
        self.previous_index_price = None
        self.reference_strike = None
        self.current_structure = None
        self.threshold = None
        self.last_transition = None
        self.combined_pnl = 0.0
        self.profit_target = None
        self.stop_loss = None
        self.exit_reason = None
        self.status_message = None
        self.action = None
        self.status = None
        self.trigger_price = None
        self.trigger_pnl = None


class FakeExchange:
    def __init__(self, legs, quotes, index_price=500.0):
        self._legs = legs
        self._quotes = quotes
        self._index = index_price

    async def get_index_price(self, symbol="BTCUSDT"):
        return self._index

    async def parse_option_positions(self):
        return self._legs

    async def get_best_quote(self, product_id):
        return self._quotes[product_id]


class DummyStrategy:
    def __init__(self, exchange):
        self.exchange = exchange
        self.state = DummyState()


def make_leg(pid, option_type, strike, size, entry_price=1.0):
    expiry = datetime.now(timezone.utc) + timedelta(days=7)
    return OptionLeg(product_id=pid, symbol=f"OPT-{pid}", option_type=option_type, strike=float(strike), expiry=expiry, size=size, entry_price=entry_price, contract_value=1.0)


@pytest.mark.asyncio
async def test_collect_summary_and_status():
    # prepare legs and quotes
    leg1 = make_leg(1, "call", 100.0, -1, entry_price=1.0)
    leg2 = make_leg(2, "put", 100.0, -1, entry_price=1.0)
    quotes = {1: Quote(best_bid=0.5, best_ask=0.6), 2: Quote(best_bid=0.4, best_ask=0.5)}
    exchange = FakeExchange(legs=[leg1, leg2], quotes=quotes, index_price=123.45)
    strategy = DummyStrategy(exchange)
    manager = BotManager(strategy)

    summary = await manager.collect_summary()
    assert summary["index_price"] == 123.45
    assert "unrealized_pnl" in summary
    assert summary["position_count"] == 2
    assert isinstance(summary["positions"], list)

    status = manager.status()
    assert status["running"] in (True, False)
    assert "strategy_state" in status


def test_json_log_handler_emit_and_latest():
    handler = JsonLogHandler(max_records=5)
    logger = logging.getLogger("testlogger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("simple message")
    logger.info('{"structured": true, "value": 1}')

    latest = handler.latest()
    assert len(latest) >= 2
    # ensure parsed JSON was attached for structured message
    assert any(isinstance(r.get("data"), dict) for r in latest)
