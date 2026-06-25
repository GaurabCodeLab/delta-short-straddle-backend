from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

from src.config import Settings
from src.exchange_client import DeltaExchangeClient
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
                await self.executor.execute_market_single_submission_with_fill_confirmation(
                    product_id=leg.product_id,
                    side=side,
                    size=size,
                    reduce_only=True,
                )
            except Exception:
                LOGGER.exception(
                    "Failed to close option leg product_id=%s size=%s side=%s",
                    leg.product_id,
                    size,
                    side,
                )

    async def _convert_short_straddle_to_iron_fly(self, straddle, breakevens) -> tuple[float, float]:
        wing_put_strike, wing_call_strike = await self.positions.nearest_strikes_for_breakeven_points(
            straddle,
            breakevens.lower_breakeven,
            breakevens.upper_breakeven,
        )

        put_product = await self.positions.get_option_product_for_strike(
            option_type="put",
            strike=wing_put_strike,
            expiry=straddle.put_leg.expiry,
        )
        call_product = await self.positions.get_option_product_for_strike(
            option_type="call",
            strike=wing_call_strike,
            expiry=straddle.call_leg.expiry,
        )
        if put_product is None or call_product is None:
            raise RuntimeError(
                f"Could not find iron-fly wing contracts put={wing_put_strike} call={wing_call_strike}"
            )

        put_qty = abs(straddle.put_leg.size)
        call_qty = abs(straddle.call_leg.size)

        LOGGER.info(
            "Converting short straddle to iron fly center=%.2f wing_put=%.2f wing_call=%.2f qty_put=%s qty_call=%s",
            straddle.put_leg.strike,
            wing_put_strike,
            wing_call_strike,
            put_qty,
            call_qty,
        )

        put_filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
            product_id=int(put_product["id"]),
            side="buy",
            size=put_qty,
            reduce_only=False,
        )
        if put_filled < put_qty:
            raise RuntimeError(f"Put wing open partial after retries: {put_filled}/{put_qty}")

        # Small delay to ensure position updates propagate
        await asyncio.sleep(0.5)

        call_filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
            product_id=int(call_product["id"]),
            side="buy",
            size=call_qty,
            reduce_only=False,
        )
        if call_filled < call_qty:
            raise RuntimeError(f"Call wing open partial after retries: {call_filled}/{call_qty}")

        # Additional delay to ensure all position updates are propagated
        await asyncio.sleep(1.0)

        LOGGER.info(
            "Iron fly conversion complete center=%.2f put_wing=%.2f call_wing=%.2f",
            straddle.put_leg.strike,
            wing_put_strike,
            wing_call_strike,
        )
        return wing_put_strike, wing_call_strike

    async def _convert_strangle_to_straddle_and_iron_fly(self, strangle) -> None:
        """
        Atomic flow for strangle adjustment:
        1) Square off the opposite short leg (the one not breached)
        2) Sell the opposite option at the same strike to form a short straddle
        3) Convert the short straddle to an iron fly (buy wings)
        """
        async with self._adjustment_lock:
            remaining_put = await self.positions.find_open_option_leg(
                option_type="put",
                strike=strangle.put_leg.strike,
                expiry=strangle.put_leg.expiry,
                side="short",
            )
            remaining_call = await self.positions.find_open_option_leg(
                option_type="call",
                strike=strangle.call_leg.strike,
                expiry=strangle.call_leg.expiry,
                side="short",
            )

            # If put remains, sell call at same strike; if call remains, sell put at same strike
            if remaining_put is not None and remaining_call is None:
                base_leg = remaining_put
                opposite_symbol = base_leg.symbol
                if opposite_symbol.startswith("P-"):
                    opposite_symbol = "C-" + opposite_symbol[2:]
                elif opposite_symbol.startswith("C-"):
                    opposite_symbol = "P-" + opposite_symbol[2:]
                direct = await self.exchange.get_product_by_symbol(opposite_symbol)
                if not isinstance(direct, dict):
                    raise RuntimeError("Could not find opposite option contract for strangle->straddle conversion")
                opposite_product_id = int(direct["id"])
                sell_qty = abs(base_leg.size)
                sold = await self.executor.execute_market_single_submission_with_fill_confirmation(
                    product_id=opposite_product_id,
                    side="sell",
                    size=sell_qty,
                    reduce_only=False,
                )
                if sold < sell_qty:
                    raise RuntimeError(f"Opposite short open partial after retries: {sold}/{sell_qty}")

            elif remaining_call is not None and remaining_put is None:
                base_leg = remaining_call
                opposite_symbol = base_leg.symbol
                if opposite_symbol.startswith("P-"):
                    opposite_symbol = "C-" + opposite_symbol[2:]
                elif opposite_symbol.startswith("C-"):
                    opposite_symbol = "P-" + opposite_symbol[2:]
                direct = await self.exchange.get_product_by_symbol(opposite_symbol)
                if not isinstance(direct, dict):
                    raise RuntimeError("Could not find opposite option contract for strangle->straddle conversion")
                opposite_product_id = int(direct["id"])
                sell_qty = abs(base_leg.size)
                sold = await self.executor.execute_market_single_submission_with_fill_confirmation(
                    product_id=opposite_product_id,
                    side="sell",
                    size=sell_qty,
                    reduce_only=False,
                )
                if sold < sell_qty:
                    raise RuntimeError(f"Opposite short open partial after retries: {sold}/{sell_qty}")

            else:
                # Nothing to convert (both legs missing or both present) — bail out
                raise RuntimeError("Unexpected position state during strangle->straddle conversion")

            # Small delay for positions to settle
            await asyncio.sleep(0.5)

            # Now detect the short straddle and convert to iron fly using existing helper
            try:
                straddle = await self.positions.detect_short_straddle()
            except Exception as exc:
                raise RuntimeError(f"Failed to detect short straddle after conversion: {exc}") from exc

            straddle_be = self.positions.compute_straddle_breakevens(straddle)
            wing_put_strike, wing_call_strike = await self._convert_short_straddle_to_iron_fly(
                straddle, straddle_be
            )
            # Finalize iron fly breakevens (positions monitoring loop will pick this up)
            return (wing_put_strike, wing_call_strike)

    async def run(self) -> None:
        strangle = None
        wait_cycles = 0
        strangle_scan_task = None
        total_premium_at_entry: float | None = None

        try:
            while True:
                if strangle is None:
                    self._set_strategy_state(
                        action="waiting for short strangle",
                        status="no short strangle found",
                        status_message="no short strangle found",
                        trigger_price=None,
                        trigger_pnl=None,
                    )

                    if strangle_scan_task is None:
                        strangle_scan_task = asyncio.create_task(self.positions.detect_short_strangle())

                    if strangle_scan_task.done():
                        try:
                            strangle = strangle_scan_task.result()
                            # Capture the total premium at detection time so leg-wise
                            # thresholds remain stable even if one leg is later closed.
                            try:
                                total_premium_at_entry = self.positions.compute_total_premium_received(strangle)
                            except Exception:
                                total_premium_at_entry = None
                            strangle_scan_task = None
                            self._set_strategy_state(
                                action="monitoring short-strangle",
                                status="waiting for trigger",
                                status_message="short strangle detected",
                                trigger_price=(f"put-{strangle.put_leg.strike},call-{strangle.call_leg.strike}"),
                                trigger_pnl=None,
                            )
                            LOGGER.info(
                                "Detected initial short strangle put=%s@%s call=%s@%s",
                                strangle.put_leg.symbol,
                                strangle.put_leg.strike,
                                strangle.call_leg.symbol,
                                strangle.call_leg.strike,
                            )
                        except Exception as exc:
                            strangle_scan_task = None
                            wait_cycles += 1
                            try:
                                index_price = await self.exchange.get_index_price("BTCUSDT")
                                self.state.last_index_price = index_price
                                LOGGER.info(
                                    "no short strangle found | BTC index=%.2f | attempt=%s | detail=%s",
                                    index_price,
                                    wait_cycles,
                                    str(exc),
                                )
                            except Exception:
                                LOGGER.info(
                                    "no short strangle found | BTC index=unavailable | attempt=%s | detail=%s",
                                    wait_cycles,
                                    str(exc),
                                )
                            await asyncio.sleep(self.settings.poll_interval_seconds)
                            continue
                    else:
                        wait_cycles += 1
                        try:
                            index_price = await self.exchange.get_index_price("BTCUSDT")
                            self.state.last_index_price = index_price
                            LOGGER.info(
                                "no short strangle found | BTC index=%.2f | attempt=%s | detail=position scan in progress",
                                index_price,
                                wait_cycles,
                            )
                        except Exception:
                            LOGGER.info(
                                "no short strangle found | BTC index=unavailable | attempt=%s | detail=position scan in progress",
                                wait_cycles,
                            )
                        await asyncio.sleep(self.settings.poll_interval_seconds)
                        continue

                self._set_strategy_state(
                    action="monitoring short-strangle",
                    status="waiting for trigger",
                    status_message="monitoring index and unrealized pnl",
                    trigger_price=(f"put-{strangle.put_leg.strike},call-{strangle.call_leg.strike}"),
                    trigger_pnl=None,
                )

                # Re-verify that the strangle still exists on the exchange.
                try:
                    await self.positions.detect_short_strangle(require_otm=False)
                except Exception:
                    LOGGER.info("no short strangle found | positions closed externally, resetting monitor")
                    strangle = None
                    self._set_strategy_state(
                        action="waiting for short strangle",
                        status="no short strangle found",
                        status_message="no short strangle found",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                index_price = await self.exchange.get_index_price("BTCUSDT")

                # Determine which legs are currently open (live state)
                remaining_put = await self.positions.find_open_option_leg(
                    option_type="put",
                    strike=strangle.put_leg.strike,
                    expiry=strangle.put_leg.expiry,
                    side="short",
                )
                remaining_call = await self.positions.find_open_option_leg(
                    option_type="call",
                    strike=strangle.call_leg.strike,
                    expiry=strangle.call_leg.expiry,
                    side="short",
                )

                put_q = None
                call_q = None
                if remaining_put is not None:
                    put_q = await self.exchange.get_best_quote(remaining_put.product_id)
                if remaining_call is not None:
                    call_q = await self.exchange.get_best_quote(remaining_call.product_id)

                # Compute PnL for current live legs
                if remaining_put is not None and remaining_call is not None:
                    from src.models import ShortStraddle

                    live_straddle = ShortStraddle(put_leg=remaining_put, call_leg=remaining_call)
                    pnl = await self.positions.compute_straddle_pnl(live_straddle)
                else:
                    pnl = 0.0
                    if remaining_put is not None and put_q is not None:
                        put_qty = abs(remaining_put.size)
                        pnl += put_qty * remaining_put.contract_value * (remaining_put.entry_price - put_q.best_ask)
                    if remaining_call is not None and call_q is not None:
                        call_qty = abs(remaining_call.size)
                        pnl += call_qty * remaining_call.contract_value * (remaining_call.entry_price - call_q.best_ask)

                # Use captured entry premium if available to keep thresholds stable
                total_premium_received = (
                    total_premium_at_entry
                    if total_premium_at_entry is not None
                    else self.positions.compute_total_premium_received(strangle)
                )
                profit_threshold = total_premium_received * self.settings.profit_capture_ratio

                LOGGER.info(
                    "Live monitor index=%.2f put_strike=%.2f call_strike=%.2f unrealized_pnl=%.4f premium_received=%.4f threshold=%.4f ratio=%.2f",
                    index_price,
                    strangle.put_leg.strike,
                    strangle.call_leg.strike,
                    pnl,
                    total_premium_received,
                    profit_threshold,
                    self.settings.profit_capture_ratio,
                )

                if pnl >= profit_threshold:
                    self._set_strategy_state(
                        action="closing whole position",
                        status="profit target reached",
                        status_message=(
                            f"profit target reached: pnl={pnl:.4f}, "
                            f"threshold={profit_threshold:.4f}"
                        ),
                        trigger_price=None,
                        trigger_pnl=pnl,
                    )
                    LOGGER.info(
                        "Profit capture condition met: pnl %.4f >= %.1f%% premium threshold %.4f; closing all positions",
                        pnl,
                        self.settings.profit_capture_ratio * 100,
                        profit_threshold,
                    )
                    await self._close_all_open_option_positions()
                    return
                leg_snapshot = {
                    "event": "strangle_monitor_snapshot",
                    "index_price": round(index_price, 4),
                    "unrealized_pnl": round(pnl, 8),
                    "total_premium_received": round(total_premium_received, 8),
                    "profit_threshold": round(profit_threshold, 8),
                    "put_leg": {
                        "symbol": strangle.put_leg.symbol,
                        "type": strangle.put_leg.option_type,
                        "strike": strangle.put_leg.strike,
                        "quantity": abs(strangle.put_leg.size),
                        "entry_price": strangle.put_leg.entry_price,
                        "best_bid": put_q.best_bid,
                        "best_ask": put_q.best_ask,
                    },
                    "call_leg": {
                        "symbol": strangle.call_leg.symbol,
                        "type": strangle.call_leg.option_type,
                        "strike": strangle.call_leg.strike,
                        "quantity": abs(strangle.call_leg.size),
                        "entry_price": strangle.call_leg.entry_price,
                        "best_bid": call_q.best_bid,
                        "best_ask": call_q.best_ask,
                    },
                }
                LOGGER.info("Snapshot: %s", json.dumps(leg_snapshot, separators=(",", ":")))

                # Leg-wise stop condition: threshold = total premium received + configurable buffer
                try:
                    leg_exit_threshold = total_premium_received + float(self.settings.leg_exit_buffer)
                    async with self._adjustment_lock:
                        cur_put = await self.positions.find_open_option_leg(
                            option_type="put",
                            strike=strangle.put_leg.strike,
                            expiry=strangle.put_leg.expiry,
                            side="short",
                        )
                        cur_call = await self.positions.find_open_option_leg(
                            option_type="call",
                            strike=strangle.call_leg.strike,
                            expiry=strangle.call_leg.expiry,
                            side="short",
                        )

                        if cur_put is not None and put_q is not None and put_q.best_ask >= leg_exit_threshold:
                            qty = abs(cur_put.size)
                            LOGGER.info(
                                "Put leg ask %.4f >= leg-exit-threshold %.4f; exiting put leg pid=%s qty=%s",
                                put_q.best_ask,
                                leg_exit_threshold,
                                cur_put.product_id,
                                qty,
                            )
                            filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                                product_id=cur_put.product_id,
                                side="buy",
                                size=qty,
                                reduce_only=True,
                            )
                            if filled < qty:
                                LOGGER.warning("Partial fill closing put leg: %s/%s", filled, qty)

                        if cur_call is not None and call_q is not None and call_q.best_ask >= leg_exit_threshold:
                            qty = abs(cur_call.size)
                            LOGGER.info(
                                "Call leg ask %.4f >= leg-exit-threshold %.4f; exiting call leg pid=%s qty=%s",
                                call_q.best_ask,
                                leg_exit_threshold,
                                cur_call.product_id,
                                qty,
                            )
                            filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                                product_id=cur_call.product_id,
                                side="buy",
                                size=qty,
                                reduce_only=True,
                            )
                            if filled < qty:
                                LOGGER.warning("Partial fill closing call leg: %s/%s", filled, qty)

                        # Check live state after attempting exits
                        cur_put_after = await self.positions.find_open_option_leg(
                            option_type="put",
                            strike=strangle.put_leg.strike,
                            expiry=strangle.put_leg.expiry,
                            side="short",
                        )
                        cur_call_after = await self.positions.find_open_option_leg(
                            option_type="call",
                            strike=strangle.call_leg.strike,
                            expiry=strangle.call_leg.expiry,
                            side="short",
                        )

                        if cur_put_after is None and cur_call_after is None:
                            LOGGER.info("Both legs closed via leg-wise stops or externally; resetting monitor")
                            strangle = None
                            total_premium_at_entry = None
                            await asyncio.sleep(self.settings.poll_interval_seconds)
                            continue
                except Exception:
                    LOGGER.exception("Error while processing leg-wise stop condition")

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
                self.state.last_index_price = index_price

                triggered_now = triggered_put or triggered_call

                if not self.state.triggered and triggered_now:
                    self.state.triggered = True
                    total_premium_received = self.positions.compute_total_premium_received(strangle)
                    pnl_threshold = total_premium_received * self.settings.profit_capture_ratio
                    self._set_strategy_state(
                        action="closing whole position" if pnl >= pnl_threshold else "adjusting short strangle",
                        status="waiting for closing positions" if pnl >= pnl_threshold else "waiting for adjustment",
                        status_message="trigger hit, evaluating pnl decision",
                        trigger_price=None,
                        trigger_pnl=pnl,
                    )
                    LOGGER.info(
                        "Trigger hit at index=%.2f put=%s call=%s premium_received=%.4f threshold=%.4f",
                        index_price,
                        strangle.put_leg.strike,
                        strangle.call_leg.strike,
                        total_premium_received,
                        pnl_threshold,
                    )

                    if pnl >= pnl_threshold:
                        self._set_status("closing all positions and exiting")
                        LOGGER.info(
                            "PnL %.4f >= 50%% premium threshold %.4f; closing all positions and exiting",
                            pnl,
                            pnl_threshold,
                        )
                        await self._close_all_open_option_positions()
                        return

                    # Determine which strike was breached. If both, fallback to closing all.
                    if triggered_put and not triggered_call:
                        breached = "put"
                    elif triggered_call and not triggered_put:
                        breached = "call"
                    else:
                        # both breached -> emergency close
                        LOGGER.info("Both strikes breached simultaneously; flattening positions")
                        await self._close_all_open_option_positions()
                        return

                    # Square off the opposite short leg
                    try:
                        if breached == "put":
                            # buy back the short call
                            await self.executor.execute_market_single_submission_with_fill_confirmation(
                                product_id=strangle.call_leg.product_id,
                                side="buy",
                                size=abs(strangle.call_leg.size),
                                reduce_only=True,
                            )
                        else:
                            # buy back the short put
                            await self.executor.execute_market_single_submission_with_fill_confirmation(
                                product_id=strangle.put_leg.product_id,
                                side="buy",
                                size=abs(strangle.put_leg.size),
                                reduce_only=True,
                            )
                    except Exception:
                        LOGGER.exception("Failed to square off opposite short during adjustment; attempting emergency flatten")
                        await self._close_all_open_option_positions()
                        return

                    # Convert remaining short into short straddle and then to iron fly
                    try:
                        wing_put_strike, wing_call_strike = await self._convert_strangle_to_straddle_and_iron_fly(strangle)
                    except Exception:
                        LOGGER.exception("Adjustment failed. Emergency flattening remaining legs.")
                        await self._close_all_open_option_positions()
                        strangle = None
                        self._set_strategy_state(
                            action="waiting for short strangle",
                            status="no short strangle found",
                            status_message="no short strangle found",
                            trigger_price=None,
                            trigger_pnl=None,
                        )
                        await asyncio.sleep(self.settings.poll_interval_seconds)
                        continue

                    # Conversion succeeded — enter iron fly monitoring (reuse existing block)
                    self._set_strategy_state(
                        action="converting short straddle to iron fly",
                        status="waiting for conversion",
                        status_message="converting short straddle to iron fly",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    straddle = None
                    breakevens_logged = False
                    iron_fly_be = None
                    while True:
                        try:
                            straddle = await self.positions.detect_short_straddle()
                        except Exception:
                            LOGGER.info(
                                "Straddle positions closed | exiting straddle monitor"
                            )
                            self._set_status("straddle closed")
                            return

                        if iron_fly_be is None:
                            # compute iron fly breakevens
                            iron_fly_be = await self.positions.compute_iron_fly_breakevens(
                                straddle, wing_put_strike, wing_call_strike
                            )
                            position_qty = abs(straddle.put_leg.size)
                            pnl_threshold = position_qty * 0.1
                            self._set_strategy_state(
                                action="monitoring iron fly",
                                status="waiting for trigger",
                                status_message="monitoring iron fly",
                                trigger_price=(
                                    f"upside-{iron_fly_be.upper_breakeven:.2f}, "
                                    f"downside-{iron_fly_be.lower_breakeven:.2f}"
                                ),
                                trigger_pnl=pnl_threshold,
                            )

                            LOGGER.info(
                                "Iron fly created! Breakevens: strike=%.2f net_premium=%.6f lower=%.2f upper=%.2f wing_put=%.2f wing_call=%.2f",
                                iron_fly_be.strike,
                                iron_fly_be.total_premium_received,
                                iron_fly_be.lower_breakeven,
                                iron_fly_be.upper_breakeven,
                                wing_put_strike,
                                wing_call_strike,
                            )

                        if not breakevens_logged:
                            breakevens_logged = True

                        iron_fly_pnl = await self.positions.compute_iron_fly_pnl(
                            straddle,
                            wing_put_strike,
                            wing_call_strike,
                        )
                        put_wing_leg = await self.positions.find_open_option_leg(
                            option_type="put",
                            strike=wing_put_strike,
                            expiry=straddle.put_leg.expiry,
                            side="long",
                        )
                        call_wing_leg = await self.positions.find_open_option_leg(
                            option_type="call",
                            strike=wing_call_strike,
                            expiry=straddle.call_leg.expiry,
                            side="long",
                        )
                        put_q = await self.exchange.get_best_quote(straddle.put_leg.product_id)
                        call_q = await self.exchange.get_best_quote(straddle.call_leg.product_id)
                        index_price_s = await self.exchange.get_index_price("BTCUSDT")
                        nearest_put_strike, nearest_call_strike = await self.positions.nearest_strikes_for_breakeven_points(
                            straddle,
                            iron_fly_be.lower_breakeven,
                            iron_fly_be.upper_breakeven,
                        )
                        straddle_snapshot = {
                            "event": "iron_fly_monitor_snapshot",
                            "index_price": round(index_price_s, 4),
                            "unrealized_pnl": round(iron_fly_pnl, 8),
                            "breakeven": {
                                "strike": round(iron_fly_be.strike, 4),
                                "total_premium_received": round(iron_fly_be.total_premium_received, 8),
                                "lower": round(iron_fly_be.lower_breakeven, 4),
                                "upper": round(iron_fly_be.upper_breakeven, 4),
                                "nearest_put_strike": round(nearest_put_strike, 4),
                                "nearest_call_strike": round(nearest_call_strike, 4),
                                "wing_put_strike": round(wing_put_strike, 4),
                                "wing_call_strike": round(wing_call_strike, 4),
                            },
                            "put_leg": {
                                "symbol": straddle.put_leg.symbol,
                                "type": straddle.put_leg.option_type,
                                "strike": straddle.put_leg.strike,
                                "quantity": abs(straddle.put_leg.size),
                                "entry_price": straddle.put_leg.entry_price,
                                "best_bid": put_q.best_bid,
                                "best_ask": put_q.best_ask,
                            },
                            "call_leg": {
                                "symbol": straddle.call_leg.symbol,
                                "type": straddle.call_leg.option_type,
                                "strike": straddle.call_leg.strike,
                                "quantity": abs(straddle.call_leg.size),
                                "entry_price": straddle.call_leg.entry_price,
                                "best_bid": call_q.best_bid,
                                "best_ask": call_q.best_ask,
                            },
                        }
                        if put_wing_leg is not None:
                            put_wing_q = await self.exchange.get_best_quote(put_wing_leg.product_id)
                            straddle_snapshot["put_wing"] = {
                                "symbol": put_wing_leg.symbol,
                                "type": put_wing_leg.option_type,
                                "strike": put_wing_leg.strike,
                                "quantity": abs(put_wing_leg.size),
                                "entry_price": put_wing_leg.entry_price,
                                "best_bid": put_wing_q.best_bid,
                                "best_ask": put_wing_q.best_ask,
                            }
                        if call_wing_leg is not None:
                            call_wing_q = await self.exchange.get_best_quote(call_wing_leg.product_id)
                            straddle_snapshot["call_wing"] = {
                                "symbol": call_wing_leg.symbol,
                                "type": call_wing_leg.option_type,
                                "strike": call_wing_leg.strike,
                                "quantity": abs(call_wing_leg.size),
                                "entry_price": call_wing_leg.entry_price,
                                "best_bid": call_wing_q.best_bid,
                                "best_ask": call_wing_q.best_ask,
                            }
                        LOGGER.info("Snapshot: %s", json.dumps(straddle_snapshot, separators=(",", ":")))
                        
                        # Log iron fly breakevens prominently
                        LOGGER.info(
                            "Iron Fly Breakevens | Strike: %.2f | Net Premium: %.6f | Lower: %.2f | Upper: %.2f | Wings: Put@%.2f Call@%.2f",
                            iron_fly_be.strike,
                            iron_fly_be.total_premium_received,
                            iron_fly_be.lower_breakeven,
                            iron_fly_be.upper_breakeven,
                            wing_put_strike,
                            wing_call_strike,
                        )

                        position_qty = abs(straddle.put_leg.size)
                        pnl_threshold = position_qty * 0.1
                        reached_lower = index_price_s <= iron_fly_be.lower_breakeven
                        reached_upper = index_price_s >= iron_fly_be.upper_breakeven
                        pnl_trigger = iron_fly_pnl >= pnl_threshold

                        if pnl_trigger or reached_lower or reached_upper:
                            self._set_strategy_state(
                                action="closing whole position",
                                status="waiting for closing positions",
                                status_message="closing all positions and exiting",
                                trigger_price=None,
                                trigger_pnl=None,
                            )
                            reason_parts = []
                            if pnl_trigger:
                                reason_parts.append(
                                    f"unrealized_pnl {iron_fly_pnl:.4f} >= 10% qty threshold {pnl_threshold:.4f}"
                                )
                            if reached_lower:
                                reason_parts.append(
                                    f"index_price {index_price_s:.4f} reached lower breakeven {iron_fly_be.lower_breakeven:.4f}"
                                )
                            if reached_upper:
                                reason_parts.append(
                                    f"index_price {index_price_s:.4f} reached upper breakeven {iron_fly_be.upper_breakeven:.4f}"
                                )

                            LOGGER.info(
                                "Iron fly exit triggered: %s",
                                "; ".join(reason_parts),
                            )
                            await self._close_all_open_option_positions()
                            return

                        await asyncio.sleep(self.settings.poll_interval_seconds)

                await asyncio.sleep(self.settings.poll_interval_seconds)
        finally:
            if strangle_scan_task is not None and not strangle_scan_task.done():
                strangle_scan_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await strangle_scan_task
