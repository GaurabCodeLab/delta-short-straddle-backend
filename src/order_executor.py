from __future__ import annotations

import logging
from typing import Any, Dict

from src.exchange_client import DeltaExchangeClient

LOGGER = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, exchange: DeltaExchangeClient) -> None:
        self.exchange = exchange
        self.last_fill_details: Dict[str, Any] | None = None

    @staticmethod
    def _extract_order_id(place_resp: Dict[str, Any]) -> str:
        root = place_resp.get("result", place_resp)
        order_id = root.get("id")
        if not order_id:
            raise RuntimeError(f"Cannot extract order id from response: {place_resp}")
        return str(order_id)

    @staticmethod
    def _extract_fill_price_from_payload(payload: Any) -> float | None:
        if payload is None:
            return None

        if isinstance(payload, list):
            for item in payload:
                price = OrderExecutor._extract_fill_price_from_payload(item)
                if price is not None:
                    return price
            return None

        if not isinstance(payload, dict):
            return None

        candidates: list[Any] = []
        for key in (
            "avg_fill_price",
            "average_fill_price",
            "filled_avg_price",
            "execution_price",
            "fill_price",
            "last_fill_price",
            "avg_price",
            "average_price",
            "price",
            "last_price",
        ):
            if key in payload:
                candidates.append(payload.get(key))

        for nested_key in ("result", "data", "order", "fill", "fills", "details", "meta_data"):
            if nested_key in payload:
                candidates.append(payload.get(nested_key))

        for value in candidates:
            if value is None:
                continue
            if isinstance(value, list):
                for item in value:
                    price = OrderExecutor._extract_fill_price_from_payload(item)
                    if price is not None:
                        return price
                continue
            if isinstance(value, dict):
                price = OrderExecutor._extract_fill_price_from_payload(value)
                if price is not None:
                    return price
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_exit_price_from_payload(payload: Any) -> float | None:
        if payload is None:
            return None

        if isinstance(payload, list):
            for item in payload:
                price = OrderExecutor._extract_exit_price_from_payload(item)
                if price is not None:
                    return price
            return None

        if not isinstance(payload, dict):
            return None

        meta = payload.get("meta_data")
        if isinstance(meta, dict):
            for key in ("avg_exit_price", "average_exit_price", "exit_price"):
                if key in meta:
                    try:
                        return float(meta[key])
                    except (TypeError, ValueError):
                        continue

        candidates: list[Any] = []
        for key in ("avg_exit_price", "average_exit_price", "exit_price"):
            if key in payload:
                candidates.append(payload.get(key))

        for nested_key in ("result", "data", "order", "fill", "fills", "details", "meta_data"):
            if nested_key in payload:
                candidates.append(payload.get(nested_key))

        for value in candidates:
            if value is None:
                continue
            if isinstance(value, list):
                for item in value:
                    price = OrderExecutor._extract_exit_price_from_payload(item)
                    if price is not None:
                        return price
                continue
            if isinstance(value, dict):
                price = OrderExecutor._extract_exit_price_from_payload(value)
                if price is not None:
                    return price
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_price_from_payload(payload: Any) -> float | None:
        exit_price = OrderExecutor._extract_exit_price_from_payload(payload)
        if exit_price is not None:
            return exit_price
        return OrderExecutor._extract_fill_price_from_payload(payload)

    @staticmethod
    def _extract_realized_pnl_from_payload(payload: Any) -> float | None:
        if payload is None:
            return None

        if isinstance(payload, list):
            for item in payload:
                pnl = OrderExecutor._extract_realized_pnl_from_payload(item)
                if pnl is not None:
                    return pnl
            return None

        if not isinstance(payload, dict):
            return None

        if "pnl" in payload:
            try:
                return float(payload["pnl"])
            except (TypeError, ValueError):
                pass

        meta = payload.get("meta_data")
        if isinstance(meta, dict) and "pnl" in meta:
            try:
                return float(meta["pnl"])
            except (TypeError, ValueError):
                pass

        for nested_key in ("result", "data", "order", "fill", "fills", "details"):
            if nested_key in payload:
                pnl = OrderExecutor._extract_realized_pnl_from_payload(payload[nested_key])
                if pnl is not None:
                    return pnl
        return None

    async def execute_market_single_submission_with_fill_confirmation(
        self,
        product_id: int,
        side: str,
        size: float,
        reduce_only: bool,
    ) -> float:
        """
        Submit one full-size market IOC order and reconcile its fill once.
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
        self.last_fill_details = None

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
                self.last_fill_details = {
                    "product_id": product_id,
                    "filled_qty": float(size),
                    "avg_fill_price": None,
                }
                return float(size)
            raise

        order_id = self._extract_order_id(place_resp)
        confirmed = 0.0
        post_size = await self.exchange.get_position_size(product_id)
        dir_sign = 1.0 if side.lower() == "buy" else -1.0
        inferred = max(0.0, dir_sign * (post_size - pre_size))
        confirmed = min(size, inferred)

        avg_fill_price = self._extract_fill_price_from_payload(place_resp)
        exit_price = self._extract_exit_price_from_payload(place_resp)
        api_realized_pnl = self._extract_realized_pnl_from_payload(place_resp)
        if hasattr(self.exchange, "get_order"):
            try:
                order_info = await self.exchange.get_order(order_id, product_id=product_id)
                if avg_fill_price is None:
                    avg_fill_price = self._extract_fill_price_from_payload(order_info)
                if exit_price is None:
                    exit_price = self._extract_exit_price_from_payload(order_info)
                if api_realized_pnl is None:
                    api_realized_pnl = self._extract_realized_pnl_from_payload(order_info)
                if exit_price is None and avg_fill_price is not None:
                    exit_price = avg_fill_price
            except Exception:
                pass

        if exit_price is None:
            exit_price = avg_fill_price

        if (api_realized_pnl is not None or exit_price is not None or avg_fill_price is not None) and size > 0:
            confirmed = float(size)

        self.last_fill_details = {
            "product_id": product_id,
            "filled_qty": float(confirmed),
            "avg_fill_price": float(avg_fill_price) if avg_fill_price is not None else None,
            "realized_pnl": float(api_realized_pnl) if api_realized_pnl is not None else None,
            "exit_price": float(exit_price) if exit_price is not None else None,
        }

        LOGGER.info(
            "Single order reconciliation order_id=%s confirmed=%s requested=%s pre_size=%s avg_fill_price=%s",
            order_id,
            confirmed,
            size,
            pre_size,
            avg_fill_price,
        )
        return confirmed