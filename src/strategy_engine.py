from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional

from src.config import Settings
from src.exchange_client import DeltaExchangeClient
from src.models import ShortStraddle, ShortStrangle
from src.order_executor import OrderExecutor
from src.position_manager import PositionManager

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyState:
    triggered: bool = False
    last_index_price: Optional[float] = None
    status_message: str = "starting"
    action: str = "waiting for short strangle"
    status: str = "initializing"
    trigger_price: Optional[str] = None
    trigger_pnl: Optional[float] = None


class StrategyEngine:
    def __init__(
        self,
        exchange: DeltaExchangeClient,
        positions: PositionManager,
        executor: OrderExecutor,
        settings: Settings,
    ) -> None:
        self.exchange = exchange
        self.positions = positions
        self.executor = executor
        self.settings = settings
        self.state = StrategyState()
        self._adjustment_lock = asyncio.Lock()
        self._exit_lock = asyncio.Lock()
        self._atm_reference_strike: Optional[float] = None
        self._current_structure: str = "unknown"
        self._realized_pnl = 0.0

    def _set_status(self, message: str) -> None:
        self.state.status_message = message

    def _set_strategy_state(
        self,
        action: str,
        status: str,
        status_message: str | None = None,
        trigger_price: str | None = None,
        trigger_pnl: float | None = None,
    ) -> None:
        self.state.action = action
        self.state.status = status
        if status_message is not None:
            self.state.status_message = status_message
        self.state.trigger_price = trigger_price
        self.state.trigger_pnl = trigger_pnl

    @staticmethod
    def _is_valid_short_straddle(straddle: ShortStraddle | None) -> bool:
        if straddle is None:
            return False
        if straddle.put_leg is None or straddle.call_leg is None:
            return False
        if straddle.put_leg.option_type.lower() != "put" or straddle.call_leg.option_type.lower() != "call":
            return False
        if straddle.put_leg.expiry.date() != straddle.call_leg.expiry.date():
            return False
        if straddle.put_leg.size >= 0 or straddle.call_leg.size >= 0:
            return False
        if not math.isclose(abs(straddle.put_leg.size), abs(straddle.call_leg.size), rel_tol=0, abs_tol=1e-8):
            return False
        return math.isclose(straddle.put_leg.strike, straddle.call_leg.strike, rel_tol=0, abs_tol=1e-8)

    @staticmethod
    def _is_valid_short_strangle(strangle: ShortStrangle | None) -> bool:
        if strangle is None:
            return False
        if strangle.put_leg is None or strangle.call_leg is None:
            return False
        if strangle.put_leg.option_type.lower() != "put" or strangle.call_leg.option_type.lower() != "call":
            return False
        if strangle.put_leg.expiry.date() != strangle.call_leg.expiry.date():
            return False
        if strangle.put_leg.size >= 0 or strangle.call_leg.size >= 0:
            return False
        if math.isclose(strangle.put_leg.strike, strangle.call_leg.strike, rel_tol=0, abs_tol=1e-8):
            return False
        return True

    @staticmethod
    def _compute_leg_market_value(leg, quote) -> float:
        return abs(leg.size) * leg.contract_value * quote.best_ask

    @staticmethod
    def _calculate_realized_pnl_from_close(leg, quote) -> float:
        if leg.size > 0:
            return abs(leg.size) * leg.contract_value * (quote.best_bid - leg.entry_price)
        return abs(leg.size) * leg.contract_value * (leg.entry_price - quote.best_ask)

    async def _cancel_pending_orders(self) -> None:
        cancel_orders = getattr(self.exchange, "cancel_all_orders", None)
        if callable(cancel_orders):
            try:
                await cancel_orders()
            except Exception as exc:
                LOGGER.warning("Unable to cancel pending orders: %s", exc)
            return
        LOGGER.info("No pending-order cancellation support available in exchange client")

    async def _close_all_open_option_positions(self) -> None:
        legs = await self.exchange.parse_option_positions()
        if not legs:
            LOGGER.info("No open option positions found to close")
            return

        for leg in legs:
            size = abs(leg.size)
            if size <= 0:
                continue
            side = "buy" if leg.size < 0 else "sell"
            try:
                quote = await self.exchange.get_best_quote(leg.product_id)
                self._realized_pnl += self._calculate_realized_pnl_from_close(leg, quote)
            except Exception:
                LOGGER.debug("Unable to compute realized P&L for closing leg %s", leg.product_id)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=leg.product_id,
                side=side,
                size=size,
                reduce_only=True,
            )

    async def _close_short_straddle(self, straddle: ShortStraddle) -> None:
        async with self._exit_lock:
            submitted_orders = []
            for leg in (straddle.put_leg, straddle.call_leg):
                current_leg = await self.positions.find_open_option_leg(
                    option_type=leg.option_type,
                    strike=leg.strike,
                    expiry=leg.expiry,
                    side="short",
                )
                if current_leg is None:
                    continue

                size = abs(current_leg.size)
                try:
                    quote = await self.exchange.get_best_quote(current_leg.product_id)
                    self._realized_pnl += self._calculate_realized_pnl_from_close(current_leg, quote)
                except Exception:
                    LOGGER.debug("Unable to compute realized P&L for closing leg %s", current_leg.product_id)
                filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                    product_id=current_leg.product_id,
                    side="buy",
                    size=size,
                    reduce_only=True,
                )
                submitted_orders.append(
                    {
                        "product_id": current_leg.product_id,
                        "filled_size": filled,
                        "requested_size": size,
                    }
                )

            await asyncio.sleep(0.1)
            LOGGER.info("Short straddle exit summary | submitted_orders=%s", submitted_orders)

    async def _monitor_short_straddle(self, straddle: ShortStraddle) -> None:
        profit_threshold = self.settings.profit_target
        loss_threshold = -self.settings.stop_loss

        while True:
            current_put = await self.positions.find_open_option_leg(
                option_type="put",
                strike=straddle.put_leg.strike,
                expiry=straddle.put_leg.expiry,
                side="short",
            )
            current_call = await self.positions.find_open_option_leg(
                option_type="call",
                strike=straddle.call_leg.strike,
                expiry=straddle.call_leg.expiry,
                side="short",
            )
            if current_put is None and current_call is None:
                return

            if current_put is None or current_call is None:
                # Do not perform a partial (single-leg) close here. Instead,
                # perform a full strategy exit to ensure we only ever close
                # all positions on overall strategy exit conditions.
                await self._exit_strategy("stop_loss")
                return

            total_pnl = 0.0
            missing_prices = False
            for leg in (current_put, current_call):
                try:
                    quote = await self.exchange.get_best_quote(leg.product_id)
                except Exception:
                    missing_prices = True
                    continue

                pnl = abs(leg.size) * leg.contract_value * (leg.entry_price - quote.best_ask)
                total_pnl += pnl

            if missing_prices:
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue

            self._set_strategy_state(
                action="monitoring short straddle",
                status="waiting for short straddle exit",
                status_message=(
                    f"short straddle monitoring pnl={total_pnl:.4f} "
                    f"profit_threshold={profit_threshold:.4f} loss_threshold={loss_threshold:.4f}"
                ),
                trigger_price=None,
                trigger_pnl=total_pnl,
            )

            if total_pnl >= profit_threshold or total_pnl <= loss_threshold:
                await self._close_short_straddle(straddle)
                return

            await asyncio.sleep(self.settings.poll_interval_seconds)

    @staticmethod
    def _crossed_or_touched(
        prev_price: Optional[float],
        curr_price: float,
        strike: float,
        option_type: str = "",
    ) -> bool:
        if curr_price == strike:
            return True
        if prev_price is None:
            # Market may already be past the strike when position is first detected.
            # Trigger immediately if price is already on the "in-the-money" side.
            opt = option_type.lower()
            if opt == "put":
                return curr_price <= strike
            if opt == "call":
                return curr_price >= strike
            return False

        opt = option_type.lower()
        if opt == "put":
            return (prev_price>strike and curr_price<=strike) or (curr_price <= strike)
        if opt == "call":
            return (prev_price<strike and curr_price>=strike) or (curr_price >= strike)
        return False

    async def _convert_straddle_to_strangle(self, straddle: ShortStraddle, current_index_price: float) -> float:
        if self._atm_reference_strike is None:
            self._atm_reference_strike = float(straddle.put_leg.strike)

        if current_index_price >= self._atm_reference_strike + self.settings.strike_adjustment_threshold:
            qty = abs(straddle.call_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=straddle.call_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            new_strike = current_index_price + self.settings.strike_adjustment_threshold
            new_product = await self.positions.get_option_product_for_strike(
                option_type="call",
                strike=new_strike,
                expiry=straddle.call_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find call product for straddle->strangle conversion at strike {new_strike}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._current_structure = "strangle"
            return new_strike

        if current_index_price <= self._atm_reference_strike - self.settings.strike_adjustment_threshold:
            qty = abs(straddle.put_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=straddle.put_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            new_strike = current_index_price - self.settings.strike_adjustment_threshold
            new_product = await self.positions.get_option_product_for_strike(
                option_type="put",
                strike=new_strike,
                expiry=straddle.put_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find put product for straddle->strangle conversion at strike {new_strike}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._current_structure = "strangle"
            return new_strike

        raise RuntimeError("No straddle-to-strangle adjustment required")

    async def _convert_strangle_to_straddle(self, strangle: ShortStrangle, current_index_price: float) -> float:
        if current_index_price >= strangle.call_leg.strike:
            qty = abs(strangle.put_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=strangle.put_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            new_product = await self.positions.get_option_product_for_strike(
                option_type="put",
                strike=current_index_price,
                expiry=strangle.put_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find put product for strangle->straddle conversion at strike {current_index_price}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._atm_reference_strike = float(current_index_price)
            self._current_structure = "straddle"
            return float(current_index_price)

        if current_index_price <= strangle.put_leg.strike:
            qty = abs(strangle.call_leg.size)
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=strangle.call_leg.product_id,
                side="buy",
                size=qty,
                reduce_only=True,
            )
            new_product = await self.positions.get_option_product_for_strike(
                option_type="call",
                strike=current_index_price,
                expiry=strangle.call_leg.expiry,
            )
            if new_product is None:
                raise RuntimeError(f"Could not find call product for strangle->straddle conversion at strike {current_index_price}")
            await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=int(new_product["id"]),
                side="sell",
                size=qty,
                reduce_only=False,
            )
            self._atm_reference_strike = float(current_index_price)
            self._current_structure = "straddle"
            return float(current_index_price)

        raise RuntimeError("No strangle-to-straddle adjustment required")

    async def _calculate_strategy_pnl(self) -> tuple[float, float, float]:
        realized_pnl = self._realized_pnl
        try:
            open_legs = await self.exchange.parse_option_positions()
        except Exception:
            open_legs = []

        unrealized_pnl = 0.0
        for leg in open_legs:
            try:
                quote = await self.exchange.get_best_quote(leg.product_id)
            except Exception:
                continue
            unrealized_pnl += self._calculate_realized_pnl_from_close(leg, quote)

        return realized_pnl, unrealized_pnl, realized_pnl + unrealized_pnl

    async def _exit_strategy(self, reason: str) -> None:
        await self._close_all_open_option_positions()
        await self._cancel_pending_orders()
        self.state.triggered = True
        if reason == "profit":
            self._set_strategy_state(
                action="completed",
                status="Completed - Profit Target Reached",
                status_message="Overall strategy P&L reached the profit target and all positions were closed",
                trigger_price=None,
                trigger_pnl=self.state.trigger_pnl,
            )
        else:
            self._set_strategy_state(
                action="completed",
                status="Completed - Stop Loss Hit",
                status_message="Overall strategy P&L reached the stop loss and all positions were closed",
                trigger_price=None,
                trigger_pnl=self.state.trigger_pnl,
            )

    async def _check_strategy_exit_conditions(self) -> bool:
        _, _, combined_pnl = await self._calculate_strategy_pnl()
        self.state.trigger_pnl = combined_pnl
        if combined_pnl >= self.settings.profit_target:
            await self._exit_strategy("profit")
            return True
        if combined_pnl <= -self.settings.stop_loss:
            await self._exit_strategy("stop_loss")
            return True
        return False

    async def _monitor_short_strangle(self, strangle: ShortStrangle) -> bool:
        if not self._is_valid_short_strangle(strangle):
            return False

        profit_threshold = self.settings.profit_target
        loss_threshold = -self.settings.stop_loss

        total_pnl = 0.0
        for leg in (strangle.put_leg, strangle.call_leg):
            current_leg = await self.positions.find_open_option_leg(
                option_type=leg.option_type,
                strike=leg.strike,
                expiry=leg.expiry,
                side="short",
            )
            if current_leg is None:
                continue
            try:
                quote = await self.exchange.get_best_quote(current_leg.product_id)
            except Exception:
                return False

            total_pnl += abs(current_leg.size) * current_leg.contract_value * (
                current_leg.entry_price - quote.best_ask
            )

        if total_pnl >= profit_threshold or total_pnl <= loss_threshold:
            await self._close_short_straddle(strangle)
            self._set_strategy_state(
                action="closing short strangle",
                status="short strangle closed",
                status_message="closed the short strangle",
                trigger_price=None,
                trigger_pnl=None,
            )
            return True

        return False

    async def _monitor_dynamic_straddle_strangle(self) -> bool:
        self._current_structure = "unknown"
        while True:
            if await self._check_strategy_exit_conditions():
                return True

            try:
                index_price = await self.exchange.get_index_price("BTCUSDT")
            except Exception as exc:
                LOGGER.warning("Unable to fetch BTC index price during dynamic adjustment monitoring: %s", exc)
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue

            self.state.last_index_price = index_price

            try:
                straddle = await self.positions.detect_short_straddle()
            except Exception:
                straddle = None

            if self._is_valid_short_straddle(straddle):
                if self._atm_reference_strike is None:
                    self._atm_reference_strike = float(straddle.put_leg.strike)
                self._current_structure = "straddle"
                if index_price >= self._atm_reference_strike + self.settings.strike_adjustment_threshold:
                    LOGGER.info(
                        "Dynamic straddle->strangle adjustment triggered at index %.2f using reference %.2f",
                        index_price,
                        self._atm_reference_strike,
                    )
                    await self._convert_straddle_to_strangle(straddle, index_price)
                    self._set_strategy_state(
                        action="adjusting short straddle",
                        status="strangle formed",
                        status_message="converted short straddle to short strangle",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                if index_price <= self._atm_reference_strike - self.settings.strike_adjustment_threshold:
                    LOGGER.info(
                        "Dynamic straddle->strangle adjustment triggered at index %.2f using reference %.2f",
                        index_price,
                        self._atm_reference_strike,
                    )
                    await self._convert_straddle_to_strangle(straddle, index_price)
                    self._set_strategy_state(
                        action="adjusting short straddle",
                        status="strangle formed",
                        status_message="converted short straddle to short strangle",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                self._set_strategy_state(
                    action="monitoring short straddle",
                    status="waiting for reference drift",
                    status_message="monitoring short straddle for dynamic adjustment",
                    trigger_price=None,
                    trigger_pnl=None,
                )
                await self._monitor_short_straddle(straddle)
                return True

            try:
                strangle = await self.positions.detect_short_strangle(require_otm=False)
            except Exception:
                strangle = None
            if self._is_valid_short_strangle(strangle):
                self._current_structure = "strangle"

                put_exists = await self.positions.find_open_option_leg(
                    option_type=strangle.put_leg.option_type,
                    strike=strangle.put_leg.strike,
                    expiry=strangle.put_leg.expiry,
                    side="short",
                )
                call_exists = await self.positions.find_open_option_leg(
                    option_type=strangle.call_leg.option_type,
                    strike=strangle.call_leg.strike,
                    expiry=strangle.call_leg.expiry,
                    side="short",
                )
                if put_exists is None and call_exists is None:
                    self._set_strategy_state(
                        action="monitoring dynamic adjustment",
                        status="no short strangle found",
                        status_message="no short strangle found",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    return True

                triggered_put = self._crossed_or_touched(
                    self.state.last_index_price,
                    index_price,
                    strangle.put_leg.strike,
                    "put",
                )
                triggered_call = self._crossed_or_touched(
                    self.state.last_index_price,
                    index_price,
                    strangle.call_leg.strike,
                    "call",
                )
                if triggered_put or triggered_call:
                    LOGGER.info(
                        "Dynamic strangle->straddle adjustment triggered at index %.2f put_strike=%.2f call_strike=%.2f",
                        index_price,
                        strangle.put_leg.strike,
                        strangle.call_leg.strike,
                    )
                    await self._convert_strangle_to_straddle(strangle, index_price)
                    self._set_strategy_state(
                        action="adjusting short strangle",
                        status="straddle formed",
                        status_message="converted short strangle to short straddle",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                self._set_strategy_state(
                    action="monitoring short strangle",
                    status="waiting for strike touch",
                    status_message="monitoring short strangle for dynamic adjustment",
                    trigger_price=None,
                    trigger_pnl=None,
                )
                if await self._monitor_short_strangle(strangle):
                    return True
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue

            LOGGER.warning(
                "Required initial short straddle position is missing; waiting for a valid short straddle or short strangle"
            )
            self._set_strategy_state(
                action="waiting for short straddle",
                status="initial position missing",
                status_message="required initial short straddle position is missing",
                trigger_price=None,
                trigger_pnl=None,
            )
            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def run(self) -> None:
        try:
            while True:
                self._set_strategy_state(
                    action="monitoring dynamic adjustment",
                    status="searching for positions",
                    status_message="monitoring short straddle/strangle transitions",
                    trigger_price=None,
                    trigger_pnl=None,
                )
                if await self._check_strategy_exit_conditions():
                    return
                # _monitor_dynamic_straddle_strangle returns True when it completed
                # a transition/monitoring cycle and the run loop should return.
                completed = await self._monitor_dynamic_straddle_strangle()
                if completed:
                    return
                await asyncio.sleep(self.settings.poll_interval_seconds)
        finally:
            LOGGER.info("Dynamic straddle/strangle monitoring loop stopped")
