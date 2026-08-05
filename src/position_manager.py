from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List

from src.exchange_client import DeltaExchangeClient
from src.models import OptionLeg, ShortStraddle, ShortStrangle


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

        target_strike = float(strike)
        direct = typed_products.get(target_strike)
        if direct is not None:
            return direct

        closest_product = None
        closest_distance = float("inf")
        for product_strike, product in typed_products.items():
            strike_value = float(product_strike)
            if math.isclose(strike_value, target_strike, rel_tol=0, abs_tol=1e-8):
                return product

            distance = abs(strike_value - target_strike)
            if distance < closest_distance:
                closest_distance = distance
                closest_product = product

        return closest_product

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

    

    async def detect_short_straddle(self) -> ShortStraddle:
        legs = await self.exchange.parse_option_positions()
        shorts = [x for x in legs if x.size < 0]
        puts = [x for x in shorts if x.option_type == "put"]
        calls = [x for x in shorts if x.option_type == "call"]

        for put_leg in puts:
            for call_leg in calls:
                same_strike = math.isclose(put_leg.strike, call_leg.strike, rel_tol=1e-3)
                same_expiry = put_leg.expiry.date() == call_leg.expiry.date()
                same_qty = math.isclose(abs(put_leg.size), abs(call_leg.size), rel_tol=0, abs_tol=1e-8)
                if same_strike and same_expiry and same_qty:
                    return ShortStraddle(put_leg=put_leg, call_leg=call_leg)

        leg_summary = ", ".join(
            f"pid={x.product_id}|{x.option_type}|strike={x.strike}|size={x.size}"
            for x in legs
        )
        raise RuntimeError("No short straddle (short put + short call same strike/expiry/quantity) found. Legs: " + leg_summary)

    async def detect_short_strangle(self, require_otm: bool = True) -> ShortStrangle:
        legs = await self.exchange.parse_option_positions()
        if not legs:
            raise RuntimeError("No open option positions found")

        shorts = [x for x in legs if x.size < 0]
        puts = [x for x in shorts if x.option_type == "put"]
        calls = [x for x in shorts if x.option_type == "call"]

        # Need current index price to determine OTM status only when requested
        try:
            index_price = await self.exchange.get_index_price("BTCUSDT")
        except Exception:
            index_price = None

        for put_leg in puts:
            for call_leg in calls:
                same_expiry = put_leg.expiry.date() == call_leg.expiry.date()
                same_qty = math.isclose(abs(put_leg.size), abs(call_leg.size), rel_tol=0, abs_tol=1e-8)
                if not same_expiry or not same_qty:
                    continue

                if require_otm and index_price is not None:
                    is_put_otm = put_leg.strike < index_price
                    is_call_otm = call_leg.strike > index_price
                    if not (is_put_otm and is_call_otm):
                        continue

                return ShortStrangle(put_leg=put_leg, call_leg=call_leg)

        leg_summary = ", ".join(
            f"pid={x.product_id}|{x.option_type}|strike={x.strike}|size={x.size}"
            for x in legs
        )
        raise RuntimeError("No short strangle (short put + short call OTM same expiry) found. Legs: " + leg_summary)
