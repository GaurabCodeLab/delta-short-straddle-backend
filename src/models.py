from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class OptionLeg:
    product_id: int
    symbol: str
    option_type: str  # call | put
    strike: float
    expiry: datetime
    size: float  # positive long, negative short
    entry_price: float
    contract_value: float = 1.0


@dataclass(slots=True)
class Quote:
    best_bid: float
    best_ask: float


@dataclass(slots=True)
class RatioSpread:
    long_leg: OptionLeg
    short_leg: OptionLeg
    short_leg_each_qty: float


@dataclass(slots=True)
class ShortStraddle:
    put_leg: OptionLeg
    call_leg: OptionLeg


@dataclass(slots=True)
class ShortStraddleBreakeven:
    strike: float
    total_premium_received: float
    lower_breakeven: float
    upper_breakeven: float


@dataclass(slots=True)
class MarketSnapshot:
    index_price: float
    short_leg_quote: Quote
    long_leg_quote: Quote
    opposite_leg_quote: Optional[Quote] = None
