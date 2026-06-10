from __future__ import annotations

import logging
from typing import Any, Dict

from src.exchange_client import DeltaExchangeClient

LOGGER = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, exchange: DeltaExchangeClient) -> None:
        self.exchange = exchange

    @staticmethod
    def _extract_order_id(place_resp: Dict[str, Any]) -> str:
        root = place_resp.get("result", place_resp)
        order_id = root.get("id")
        if not order_id:
            raise RuntimeError(f"Cannot extract order id from response: {place_resp}")
        return str(order_id)

    async def execute_market_single_submission_with_fill_confirmation(
        self,
        product_id: int,
        side: str,
        size: float,
        reduce_only: bool,
    ) -> float:
        """
        Submit one full-size market IOC order and reconcile its fill once.
        This intentionally does not split by max_single_order_qty.
        """
        if size <= 0:
            raise ValueError("Order quantity must be positive")

        LOGGER.info(
            "Single order submission product_id=%s side=%s size=%s reduce_only=%s",
            product_id,
            side,
            size,
            reduce_only,
        )
        pre_size = await self.exchange.get_position_size(product_id)

        try:
            place_resp = await self.exchange.place_market_order(
                product_id=product_id,
                side=side,
                size=size,
                reduce_only=reduce_only,
            )
        except Exception as exc:
            if reduce_only and "no_position_for_reduce_only" in str(exc):
                LOGGER.info(
                    "Single order product_id=%s already flat, treating size=%s as filled",
                    product_id,
                    size,
                )
                return float(size)
            raise

        order_id = self._extract_order_id(place_resp)
        confirmed = 0.0
        post_size = await self.exchange.get_position_size(product_id)
        dir_sign = 1.0 if side.lower() == "buy" else -1.0
        inferred = max(0.0, dir_sign * (post_size - pre_size))
        confirmed = min(size, inferred)

        LOGGER.info(
            "Single order reconciliation order_id=%s confirmed=%s requested=%s pre_size=%s",
            order_id,
            confirmed,
            size,
            pre_size,
        )
        return confirmed