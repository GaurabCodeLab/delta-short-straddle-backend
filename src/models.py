from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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
class ShortStraddle:
    put_leg: OptionLeg
    call_leg: OptionLeg


@dataclass(slots=True)
class ShortStrangle:
    put_leg: OptionLeg
    call_leg: OptionLeg
