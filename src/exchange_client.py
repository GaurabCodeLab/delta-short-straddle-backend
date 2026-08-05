from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import urllib3
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.models import OptionLeg, Quote

LOGGER = logging.getLogger(__name__)


class ExchangeClientError(RuntimeError):
    pass


class DeltaExchangeClient:
    """
    Thin async wrapper around delta-rest-client sync methods.
    This class intentionally keeps all API-shape assumptions in one place.
    If Delta field names differ for your account, adjust mapping here only.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: str, ssl_verify: bool = True) -> None:
        try:
            from delta_rest_client import DeltaRestClient, OrderType, TimeInForce  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ExchangeClientError(
                "delta-rest-client import failed. Install with: pip install delta-rest-client"
            ) from exc

        self._client = DeltaRestClient(
            base_url=base_url,
            api_key=api_key,
            api_secret=api_secret,
        )
        self._client.session.verify = ssl_verify
        if not ssl_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self._order_type_market = OrderType.MARKET
        self._tif_ioc = TimeInForce.IOC

    @staticmethod
    def _to_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            # Supports ISO strings ending in Z.
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ExchangeClientError(f"Unsupported datetime value: {value}") from exc
        raise ExchangeClientError(f"Unsupported datetime value: {value}")

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call(self, method_name: str, **kwargs: Any) -> Any:
        method = getattr(self._client, method_name, None)
        if method is None:
            raise ExchangeClientError(f"delta-rest-client missing method: {method_name}")

        try:
            return await asyncio.to_thread(method, **kwargs)
        except TypeError:
            # Some client methods may not accept kwargs; try positionalless call.
            if kwargs:
                return await asyncio.to_thread(method)
            raise

    async def get_index_price(self, symbol: str = "BTCUSD") -> float:
        try:
            data = await self._call("get_ticker", **{"identifier": "BTCUSD", "auth": True})
            price = data.get("spot_price")
            if price is not None:
                return float(price)
        except Exception:   
            raise ExchangeClientError("Unable to fetch BTC index price from client methods.")

    async def get_open_positions_raw(self) -> List[Dict[str, Any]]:
        last_error = None     
        try:
            resp = await self._call("request", method="GET", path="/v2/positions/margined", auth=True)
            payload = await asyncio.to_thread(resp.json)
            LOGGER.debug(f"Margined positions response: {payload}")
            success = payload.get("success")
            if success is False:
                last_error = payload.get("error") or payload.get("message")
                raise ExchangeClientError(f"Unable to fetch open positions. Last error: {last_error}")
            else:
                result = payload.get("result") or []
                return result
        except Exception as exc:
            LOGGER.debug(f"Margined positions endpoint failed: {str(exc)}")
            last_error = str(exc)
            raise ExchangeClientError(f"Unable to fetch open positions. Last error: {last_error}")

    async def get_position_size(self, product_id: int) -> float:
        """Return signed net position size for a product (positive long, negative short)."""
        positions = await self.get_open_positions_raw()
        for pos in positions:
            try:              
                product = pos.get("product") if isinstance(pos.get("product"), dict) else None
                if product is not None:
                    pid = int(pos.get("product_id"))
                if pid != product_id:
                    continue
                return float(pos.get("size"))
            except Exception:
                continue
        return 0.0

    async def get_products_raw(self) -> List[Dict[str, Any]]:
        # delta-rest-client 1.0.13 does not expose get_products.
        # This endpoint can be paginated; collect multiple pages so strikes are not missed.
        collected: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()

        for page in range(1, 31):
            try:
                resp = await self._call(
                    "request",
                    method="GET",
                    path="/v2/products",
                    query={"page_size": 1000, "page_number": page},
                    auth=True,
                )
            except TypeError:
                # Fallback for clients that don't support query kwargs here.
                if page > 1:
                    break
                resp = await self._call("request", method="GET", path="/v2/products", auth=True)

            payload = await asyncio.to_thread(resp.json)
            if not payload.get("success"):
                break

            result = payload.get("result") or []
            if not isinstance(result, list) or not result:
                break

            new_count = 0
            for p in result:
                try:
                    pid = int(p.get("id")) if p.get("id") is not None else None
                except Exception:
                    pid = None

                if pid is not None:
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                collected.append(p)
                new_count += 1

            # Stop if this page had no new products after de-dup.
            if new_count == 0:
                break

        if collected:
            return collected
        raise ExchangeClientError("Unexpected products response shape")

    async def get_product_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        target = (symbol or "").strip()
        if not target:
            return None

        # Attempt direct filtered request first.
        try:
            resp = await self._call(
                "request",
                method="GET",
                path="/v2/products",
                query={"symbol": target},
                auth=True,
            )
            payload = await asyncio.to_thread(resp.json)
            if payload.get("success"):
                result = payload.get("result") or []
                if isinstance(result, list):
                    for p in result:
                        sym = str(p.get("symbol"))
                        if sym == target:
                            return p
                    if len(result) == 1:
                        return result[0]
        except Exception:
            pass

        # Fallback: scan cached/paginated product list.
        try:
            products = await self.get_products_raw()
            for p in products:
                sym = str(p.get("symbol"))
                if sym == target:
                    return p
        except Exception:
            return None

        return None

    async def get_best_quote(self, product_id: int) -> Quote:
        errors = []
        try:
            data = await self._call("get_ticker", identifier=product_id, auth=True)
            LOGGER.debug(f"Quote method get_ticker for product {product_id}: {data}")
            root = data.get("quotes", data) if isinstance(data, dict) else {}
            best_bid = root.get("best_bid")
            best_ask = root.get("best_ask")
            if best_bid is not None and best_ask is not None:
                return Quote(best_bid=float(best_bid), best_ask=float(best_ask))              
        except Exception as exc:
            error_msg = f"METHOD failed: {str(exc)}"
            LOGGER.debug(f"Quote fetch attempt for product {product_id}: {error_msg}")
            errors.append(error_msg)
            error_summary = "; ".join(errors)
            raise ExchangeClientError(f"Unable to fetch quote for product_id={product_id}. Tried: {error_summary}")

    async def place_market_order(
        self,
        product_id: int,
        side: str,
        size: float,
        reduce_only: bool,
    ) -> Dict[str, Any]:
        params = {
            "product_id": product_id,
            "size": size,
            "side": side,
            "order_type": self._order_type_market,
            "time_in_force": self._tif_ioc,
            "reduce_only": "true" if reduce_only else "false",
        }
        LOGGER.info(
            "Placing market order product_id=%s side=%s size=%s reduce_only=%s",
            product_id,
            side,
            size,
            reduce_only,
        )
        resp = await self._call("place_order", **params)
        if not isinstance(resp, dict):
            raise ExchangeClientError("Unexpected place_order response")
        return resp

    async def get_order(self, order_id: str, product_id: Optional[int] = None) -> Dict[str, Any]:
        query: Dict[str, Any] = {"id": order_id}
        if product_id is not None:
            query["product_id"] = product_id
        resp = await self._call("order_history", query=query, page_size=1)

        if isinstance(resp, dict):
            result = resp.get("result")
            if isinstance(result, list) and result:
                return result[0]
        return {}

    async def parse_option_positions(self) -> List[OptionLeg]:
        positions = await self.get_open_positions_raw()
        legs: List[OptionLeg] = []
        LOGGER.info(f"Parsing {len(positions)} raw positions")
        
        for idx, pos in enumerate(positions):
            size = float(pos.get("size"))
            if abs(size) <= 0:
                LOGGER.debug(f"Position {idx}: skipped (zero size)")
                continue

            product_id = int(pos.get("product_id"))
            product = pos.get("product") if isinstance(pos.get("product"), dict) else None
            if not product:
                LOGGER.debug(f"Position {idx}: skipped (no product found for id={product_id})")
                continue

            contract_type = str(product.get("contract_type") or "").lower()
            if "option" not in contract_type:
                LOGGER.debug(f"Position {idx}: skipped (contract_type={contract_type}, not option)")
                continue
            elif "call" in contract_type:
                option_type = "call"
            elif "put" in contract_type:
                option_type = "put"
            else:
                LOGGER.debug(f"Position {idx}: skipped (unable to determine option_type)")
                continue

            strike = float(product.get("strike_price"))
            expiry_raw = product.get("settlement_time")
            if strike <= 0 or not expiry_raw:
                LOGGER.debug(f"Position {idx}: skipped (strike={strike}, expiry={expiry_raw})")
                continue

            entry_price = float(pos.get("entry_price"))
            contract_value = float(product.get("contract_value"))

            leg = OptionLeg(
                product_id=product_id,
                symbol=str(product.get("symbol") or product_id),
                option_type=option_type,
                strike=strike,
                expiry=self._to_dt(expiry_raw),
                size=size,
                entry_price=entry_price,
                contract_value=contract_value,
            )
            legs.append(leg)
            LOGGER.info(f"Position {idx}: parsed as {leg.symbol} {option_type} strike={strike} size={size}")
        
        LOGGER.info(f"Parsed {len(legs)} option positions from {len(positions)} raw positions")
        return legs
