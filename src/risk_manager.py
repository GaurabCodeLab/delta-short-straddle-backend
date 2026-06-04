from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskManager:
    max_single_order_qty: float
    max_total_option_notional: float

    def validate_order_qty(self, qty: float) -> None:
        if qty <= 0:
            raise ValueError("Order quantity must be positive")
        if qty > self.max_single_order_qty:
            raise ValueError(f"Order quantity {qty} exceeds max {self.max_single_order_qty}")

    def validate_total_notional(self, strike: float, total_contracts: float) -> None:
        notion = strike * total_contracts
        if notion > self.max_total_option_notional:
            raise ValueError(
                f"Total option notional {notion} exceeds max {self.max_total_option_notional}"
            )
