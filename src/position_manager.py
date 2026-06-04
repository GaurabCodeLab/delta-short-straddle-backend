from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from src.exchange_client import DeltaExchangeClient
from src.models import OptionLeg, RatioSpread, ShortStraddle, ShortStraddleBreakeven


@dataclass(slots=True)
class PositionSnapshot:
    ratio: RatioSpread
    total_unrealized_pnl: float


class PositionManager:
    def __init__(self, exchange: DeltaExchangeClient) -> None:
        self.exchange = exchange
        self._option_strikes_by_expiry: dict[str, dict[str, List[float]]] = {}
        self._option_products_by_expiry: dict[str, dict[str, dict[float, dict]]] = {}

    @staticmethod
    def _to_dt_safe(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_option_type(contract_type: object) -> str:
        ct = str(contract_type or "").lower()
        if "call" in ct:
            return "call"
        if "put" in ct:
            return "put"
        return ""

    async def _ensure_option_strike_cache(self) -> None:
        if self._option_strikes_by_expiry and self._option_products_by_expiry:
            return

        products = await self.exchange.get_products_raw()
        by_expiry: dict[str, dict[str, set[float]]] = {}
        products_by_expiry: dict[str, dict[str, dict[float, dict]]] = {}

        for p in products:
            try:
                contract_type = str(p.get("contract_type") or "").lower()
                if "option" not in contract_type:
                    continue

                strike = float(p.get("strike_price") or 0)
                expiry_raw = p.get("settlement_time")
                expiry_dt = self._to_dt_safe(expiry_raw)
                option_type = self._normalize_option_type(contract_type)

                if strike <= 0 or expiry_dt is None or option_type not in {"put", "call"}:
                    continue

                expiry_key = expiry_dt.date().isoformat()
                expiry_bucket = by_expiry.setdefault(expiry_key, {"put": set(), "call": set()})
                expiry_bucket[option_type].add(strike)
                product_bucket = products_by_expiry.setdefault(expiry_key, {"put": {}, "call": {}})
                product_bucket[option_type][strike] = p
            except Exception:
                continue

        self._option_strikes_by_expiry = {
            exp: {
                "put": sorted(list(v.get("put", set()))),
                "call": sorted(list(v.get("call", set()))),
            }
            for exp, v in by_expiry.items()
        }
        self._option_products_by_expiry = products_by_expiry

    async def get_option_product_for_strike(
        self,
        option_type: str,
        strike: float,
        expiry: datetime,
    ) -> dict | None:
        await self._ensure_option_strike_cache()

        expiry_key = expiry.date().isoformat()
        expiry_products = self._option_products_by_expiry.get(expiry_key, {})
        typed_products = expiry_products.get(option_type.lower(), {})
        if not typed_products:
            return None

        direct = typed_products.get(float(strike))
        if direct is not None:
            return direct

        for product_strike, product in typed_products.items():
            if math.isclose(float(product_strike), float(strike), rel_tol=0, abs_tol=1e-8):
                return product

        return None

    async def find_open_option_leg(
        self,
        option_type: str,
        strike: float,
        expiry: datetime,
        side: str,
    ) -> OptionLeg | None:
        legs = await self.exchange.parse_option_positions()
        desired_side_positive = side.lower() == "long"

        for leg in legs:
            same_type = leg.option_type == option_type.lower()
            same_strike = math.isclose(leg.strike, strike, rel_tol=0, abs_tol=1e-8)
            same_expiry = leg.expiry == expiry or leg.expiry.date() == expiry.date()
            same_side = leg.size > 0 if desired_side_positive else leg.size < 0
            if same_type and same_strike and same_expiry and same_side:
                return leg

        return None

    async def nearest_strikes_for_breakeven_points(
        self,
        straddle: ShortStraddle,
        lower_breakeven: float,
        upper_breakeven: float,
    ) -> tuple[float, float]:
        await self._ensure_option_strike_cache()

        expiry_key = straddle.put_leg.expiry.date().isoformat()
        strikes = self._option_strikes_by_expiry.get(expiry_key)
        if not strikes:
            return straddle.put_leg.strike, straddle.call_leg.strike

        put_strikes = strikes.get("put", [])
        call_strikes = strikes.get("call", [])

        if put_strikes:
            nearest_put = min(put_strikes, key=lambda s: abs(s - lower_breakeven))
        else:
            nearest_put = straddle.put_leg.strike

        if call_strikes:
            nearest_call = min(call_strikes, key=lambda s: abs(s - upper_breakeven))
        else:
            nearest_call = straddle.call_leg.strike

        return float(nearest_put), float(nearest_call)

    async def detect_ratio_spread(self) -> RatioSpread:
        legs = await self.exchange.parse_option_positions()
        if not legs:
            raise RuntimeError("No open option positions found")

        # Expected structure: Nx long + 2Nx short, same expiry and option type.
        # Long strike can differ from short strike.
        longs = [x for x in legs if x.size > 0]
        shorts = [x for x in legs if x.size < 0]

        for long_leg in longs:
            for short_leg in shorts:
                same_expiry = (
                    long_leg.expiry == short_leg.expiry
                    or long_leg.expiry.date() == short_leg.expiry.date()
                )
                same_type = long_leg.option_type == short_leg.option_type
                long_qty = abs(long_leg.size)
                short_qty = abs(short_leg.size)
                if long_qty <= 0:
                    continue

                # Accept any quantity scale as long as it respects 1:2 principle.
                # Example: 1:2, 3:6, 7.5:15, etc.
                ratio_match = math.isclose(
                    short_qty / long_qty,
                    2.0,
                    rel_tol=2e-2,
                    abs_tol=1e-8,
                )
                if same_expiry and same_type and ratio_match:
                    return RatioSpread(
                        long_leg=long_leg,
                        short_leg=short_leg,
                        short_leg_each_qty=long_qty,
                    )

        leg_summary = ", ".join(
            f"pid={x.product_id}|{x.option_type}|strike={x.strike}|exp={x.expiry.isoformat()}|size={x.size}"
            for x in legs
        )
        raise RuntimeError(
            "No valid Nx long / 2Nx short ratio spread found. Parsed legs: " + leg_summary
        )

    async def compute_unrealized_pnl(self, ratio: RatioSpread) -> float:
        # Calculate PnL based on entry price and actual liquidation prices.
        # Long positions close at best_bid (conservative: worst price when selling)
        # Short positions close at best_ask (conservative: worst price when buying back)
        long_q = await self.exchange.get_best_quote(ratio.long_leg.product_id)
        short_q = await self.exchange.get_best_quote(ratio.short_leg.product_id)

        long_qty = abs(ratio.long_leg.size)
        short_qty = abs(ratio.short_leg.size)

        # Long leg PnL: (best_bid - entry_price) * qty * contract_value
        # (position closed by selling at bid price)
        long_leg_pnl = (
            long_qty
            * ratio.long_leg.contract_value
            * (long_q.best_bid - ratio.long_leg.entry_price)
        )

        # Short leg PnL: (entry_price - best_ask) * qty * contract_value
        # (position closed by buying back at ask price)
        short_leg_pnl = (
            short_qty
            * ratio.short_leg.contract_value
            * (ratio.short_leg.entry_price - short_q.best_ask)
        )

        return long_leg_pnl + short_leg_pnl

    async def detect_short_straddle(self) -> ShortStraddle:
        legs = await self.exchange.parse_option_positions()
        shorts = [x for x in legs if x.size < 0]
        puts = [x for x in shorts if x.option_type == "put"]
        calls = [x for x in shorts if x.option_type == "call"]

        for put_leg in puts:
            for call_leg in calls:
                same_strike = math.isclose(put_leg.strike, call_leg.strike, rel_tol=1e-3)
                same_expiry = put_leg.expiry.date() == call_leg.expiry.date()
                if same_strike and same_expiry:
                    return ShortStraddle(put_leg=put_leg, call_leg=call_leg)

        leg_summary = ", ".join(
            f"pid={x.product_id}|{x.option_type}|strike={x.strike}|size={x.size}"
            for x in legs
        )
        raise RuntimeError("No short straddle (short put + short call same strike) found. Legs: " + leg_summary)

    async def compute_straddle_pnl(self, straddle: ShortStraddle) -> float:
        put_q = await self.exchange.get_best_quote(straddle.put_leg.product_id)
        call_q = await self.exchange.get_best_quote(straddle.call_leg.product_id)

        put_qty = abs(straddle.put_leg.size)
        call_qty = abs(straddle.call_leg.size)

        # Short positions: PnL = (entry_price - best_ask) * qty * contract_value
        put_pnl = put_qty * straddle.put_leg.contract_value * (straddle.put_leg.entry_price - put_q.best_ask)
        call_pnl = call_qty * straddle.call_leg.contract_value * (straddle.call_leg.entry_price - call_q.best_ask)
        return put_pnl + call_pnl

    async def compute_iron_fly_pnl(
        self,
        straddle: ShortStraddle,
        wing_put_strike: float,
        wing_call_strike: float,
    ) -> float:
        total_pnl = await self.compute_straddle_pnl(straddle)

        put_wing = await self.find_open_option_leg(
            option_type="put",
            strike=wing_put_strike,
            expiry=straddle.put_leg.expiry,
            side="long",
        )
        call_wing = await self.find_open_option_leg(
            option_type="call",
            strike=wing_call_strike,
            expiry=straddle.call_leg.expiry,
            side="long",
        )

        for wing_leg in (put_wing, call_wing):
            if wing_leg is None:
                continue
            wing_quote = await self.exchange.get_best_quote(wing_leg.product_id)
            wing_qty = abs(wing_leg.size)
            total_pnl += wing_qty * wing_leg.contract_value * (wing_quote.best_bid - wing_leg.entry_price)

        return total_pnl

    def compute_straddle_breakevens(self, straddle: ShortStraddle) -> ShortStraddleBreakeven:
        put_exposure = abs(straddle.put_leg.size) * straddle.put_leg.contract_value
        call_exposure = abs(straddle.call_leg.size) * straddle.call_leg.contract_value
        if put_exposure <= 0 or call_exposure <= 0:
            raise RuntimeError("Invalid straddle exposure while calculating breakevens")

        put_premium = put_exposure * straddle.put_leg.entry_price
        call_premium = call_exposure * straddle.call_leg.entry_price
        total_premium_received = put_premium + call_premium

        lower_breakeven = straddle.put_leg.strike - (total_premium_received / put_exposure)
        upper_breakeven = straddle.call_leg.strike + (total_premium_received / call_exposure)

        return ShortStraddleBreakeven(
            strike=straddle.put_leg.strike,
            total_premium_received=total_premium_received,
            lower_breakeven=lower_breakeven,
            upper_breakeven=upper_breakeven,
        )

    async def compute_iron_fly_breakevens(
        self,
        straddle: ShortStraddle,
        wing_put_strike: float,
        wing_call_strike: float,
    ) -> ShortStraddleBreakeven:
        straddle_be = self.compute_straddle_breakevens(straddle)
        
        put_wing = await self.find_open_option_leg(
            option_type="put",
            strike=wing_put_strike,
            expiry=straddle.put_leg.expiry,
            side="long",
        )
        call_wing = await self.find_open_option_leg(
            option_type="call",
            strike=wing_call_strike,
            expiry=straddle.call_leg.expiry,
            side="long",
        )
        
        wing_cost = 0.0
        if put_wing:
            wing_cost += put_wing.entry_price * abs(put_wing.size) * put_wing.contract_value
        if call_wing:
            wing_cost += call_wing.entry_price * abs(call_wing.size) * call_wing.contract_value
        
        net_premium_received = straddle_be.total_premium_received - wing_cost
        
        put_exposure = abs(straddle.put_leg.size) * straddle.put_leg.contract_value
        call_exposure = abs(straddle.call_leg.size) * straddle.call_leg.contract_value
        
        lower_breakeven = straddle.put_leg.strike - (net_premium_received / put_exposure)
        upper_breakeven = straddle.call_leg.strike + (net_premium_received / call_exposure)
        
        return ShortStraddleBreakeven(
            strike=straddle.put_leg.strike,
            total_premium_received=net_premium_received,
            lower_breakeven=lower_breakeven,
            upper_breakeven=upper_breakeven,
        )
