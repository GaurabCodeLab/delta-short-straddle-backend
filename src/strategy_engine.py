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
from src.risk_manager import RiskManager

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyState:
    triggered: bool = False
    last_index_price: Optional[float] = None
    status_message: str = "starting"
    action: str = "waiting for ratio spread"
    status: str = "initializing"
    trigger_price: Optional[str] = None
    trigger_pnl: Optional[float] = None


class StrategyEngine:
    def __init__(
        self,
        exchange: DeltaExchangeClient,
        positions: PositionManager,
        executor: OrderExecutor,
        risk: RiskManager,
        settings: Settings,
    ) -> None:
        self.exchange = exchange
        self.positions = positions
        self.executor = executor
        self.risk = risk
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

    async def _close_entire_ratio(self, ratio) -> None:
        long_qty = abs(ratio.long_leg.size)
        short_qty = abs(ratio.short_leg.size)

        # Long close = sell. Short close = buy.
        filled_long = await self.executor.execute_market_single_submission_with_fill_confirmation(
            product_id=ratio.long_leg.product_id,
            side="sell",
            size=long_qty,
            reduce_only=True,
        )
        filled_short = await self.executor.execute_market_single_submission_with_fill_confirmation(
            product_id=ratio.short_leg.product_id,
            side="buy",
            size=short_qty,
            reduce_only=True,
        )
        LOGGER.info("All positions close requested long_filled=%s short_filled=%s", filled_long, filled_short)

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

    async def _convert_to_short_straddle(self, ratio) -> None:
        """
        Atomic intent:
        1) Close long leg
        2) Buy back half short leg
        3) Open opposite-type short with same qty at same strike/expiry
        """
        async with self._adjustment_lock:
            qty = ratio.short_leg_each_qty
            if qty <= 0:
                raise ValueError(f"Invalid conversion quantity: {qty}")
            # Notional check is intentionally skipped here: straddle conversion is a
            # risk-reducing hedge on an existing position, not a new speculative bet.

            close_long_side = "sell"
            close_short_half_side = "buy"
            open_opposite_short_side = "sell"

            opposite_type = "put" if ratio.short_leg.option_type == "call" else "call"
            opposite_symbol = ratio.short_leg.symbol
            if opposite_symbol.startswith("P-"):
                opposite_symbol = "C-" + opposite_symbol[2:]
            elif opposite_symbol.startswith("C-"):
                opposite_symbol = "P-" + opposite_symbol[2:]

            opp_candidates = []

            if not opp_candidates and opposite_symbol:
                direct = await self.exchange.get_product_by_symbol(opposite_symbol)
                if isinstance(direct, dict):
                    opp_candidates.append(direct)
            if not opp_candidates:
                raise RuntimeError("Could not find opposite option contract for straddle conversion")

            opposite_product_id = int(opp_candidates[0]["id"])

            LOGGER.info(
                "Converting ratio spread to short straddle strike=%s qty=%s opposite_type=%s",
                ratio.short_leg.strike,
                qty,
                opposite_type,
            )

            # Step 1: close long
            long_filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=ratio.long_leg.product_id,
                side=close_long_side,
                size=qty,
                reduce_only=True,
            )
            if long_filled < qty:
                raise RuntimeError(f"Long close partial after retries: {long_filled}/{qty}")

            # Step 2: close half short
            short_half_filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=ratio.short_leg.product_id,
                side=close_short_half_side,
                size=qty,
                reduce_only=True,
            )
            if short_half_filled < qty:
                raise RuntimeError(f"Short half close partial after retries: {short_half_filled}/{qty}")

            # Step 3: open opposite short
            opp_filled = await self.executor.execute_market_single_submission_with_fill_confirmation(
                product_id=opposite_product_id,
                side=open_opposite_short_side,
                size=qty,
                reduce_only=False,
            )
            if opp_filled < qty:
                raise RuntimeError(f"Opposite short open partial after retries: {opp_filled}/{qty}")

            LOGGER.info(
                "Conversion complete: short original type + short opposite type at strike=%s",
                ratio.short_leg.strike,
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

    async def run(self) -> None:
        ratio = None
        wait_cycles = 0
        ratio_scan_task = None

        try:
            while True:
                if ratio is None:
                    self._set_strategy_state(
                        action="waiting for ratio spread",
                        status="no ratio spread found",
                        status_message="no ratio spread found",
                        trigger_price=None,
                        trigger_pnl=None,
                    )

                    if ratio_scan_task is None:
                        ratio_scan_task = asyncio.create_task(self.positions.detect_ratio_spread())

                    if ratio_scan_task.done():
                        try:
                            ratio = ratio_scan_task.result()
                            ratio_scan_task = None
                            self._set_strategy_state(
                                action="monitoring ratio-spread",
                                status="waiting for trigger",
                                status_message="ratio spread detected",
                                trigger_price=ratio.short_leg.strike,
                                trigger_pnl=None,
                            )
                            LOGGER.info(
                                "Detected initial ratio spread long=%s %s@%s short=%s %s@%s",
                                ratio.long_leg.option_type,
                                abs(ratio.long_leg.size),
                                ratio.long_leg.strike,
                                ratio.short_leg.option_type,
                                abs(ratio.short_leg.size),
                                ratio.short_leg.strike,
                            )
                        except Exception as exc:
                            ratio_scan_task = None
                            wait_cycles += 1
                            try:
                                index_price = await self.exchange.get_index_price("BTCUSDT")
                                self.state.last_index_price = index_price
                                LOGGER.info(
                                    "no ratio spread found | BTC index=%.2f | attempt=%s | detail=%s",
                                    index_price,
                                    wait_cycles,
                                    str(exc),
                                )
                            except Exception:
                                LOGGER.info(
                                    "no ratio spread found | BTC index=unavailable | attempt=%s | detail=%s",
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
                                "no ratio spread found | BTC index=%.2f | attempt=%s | detail=position scan in progress",
                                index_price,
                                wait_cycles,
                            )
                        except Exception:
                            LOGGER.info(
                                "no ratio spread found | BTC index=unavailable | attempt=%s | detail=position scan in progress",
                                wait_cycles,
                            )
                        await asyncio.sleep(self.settings.poll_interval_seconds)
                        continue

                self._set_strategy_state(
                    action="monitoring ratio-spread",
                    status="waiting for trigger",
                    status_message="monitoring index and unrealized pnl",
                    trigger_price=ratio.short_leg.strike,
                    trigger_pnl=None,
                )

                # Re-verify that the ratio spread still exists on the exchange.
                # If positions were closed externally, reset ratio and loop back.
                try:
                    await self.positions.detect_ratio_spread()
                except Exception:
                    LOGGER.info("no ratio spread found | positions closed externally, resetting monitor")
                    ratio = None
                    self._set_strategy_state(
                        action="waiting for ratio spread",
                        status="no ratio spread found",
                        status_message="no ratio spread found",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue

                index_price = await self.exchange.get_index_price("BTCUSDT")
                pnl = await self.positions.compute_unrealized_pnl(ratio)
                long_q = await self.exchange.get_best_quote(ratio.long_leg.product_id)
                short_q = await self.exchange.get_best_quote(ratio.short_leg.product_id)
                LOGGER.info(
                    "Live monitor index=%.2f strike=%.2f unrealized_pnl=%.4f",
                    index_price,
                    ratio.short_leg.strike,
                    pnl,
                )
                leg_snapshot = {
                    "event": "ratio_monitor_snapshot",
                    "index_price": round(index_price, 4),
                    "unrealized_pnl": round(pnl, 8),
                    "long_leg": {
                        "symbol": ratio.long_leg.symbol,
                        "type": ratio.long_leg.option_type,
                        "strike": ratio.long_leg.strike,
                        "quantity": abs(ratio.long_leg.size),
                        "entry_price": ratio.long_leg.entry_price,
                        "best_bid": long_q.best_bid,
                        "best_ask": long_q.best_ask,
                    },
                    "short_leg": {
                        "symbol": ratio.short_leg.symbol,
                        "type": ratio.short_leg.option_type,
                        "strike": ratio.short_leg.strike,
                        "quantity": abs(ratio.short_leg.size),
                        "entry_price": ratio.short_leg.entry_price,
                        "best_bid": short_q.best_bid,
                        "best_ask": short_q.best_ask,
                    },
                }
                LOGGER.info("Snapshot: %s", json.dumps(leg_snapshot, separators=(",", ":")))

                triggered_now = self._crossed_or_touched(
                    self.state.last_index_price,
                    index_price,
                    ratio.short_leg.strike,
                    ratio.short_leg.option_type,
                )
                self.state.last_index_price = index_price

                if not self.state.triggered and triggered_now:
                    self.state.triggered = True
                    self._set_strategy_state(
                        action="closing whole position" if pnl >= abs(ratio.short_leg_each_qty)*0.1 else "converting ratio spread to short straddle",
                        status="waiting for closing positions" if pnl >= abs(ratio.short_leg_each_qty)*0.1 else "waiting for conversion",
                        status_message="trigger hit, evaluating pnl decision",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    LOGGER.info("Trigger hit at index=%.2f strike=%.2f", index_price, ratio.short_leg.strike)

                    if pnl >= abs(ratio.short_leg_each_qty)*0.1:
                        self._set_status("closing all positions and exiting")
                        LOGGER.info(
                            "PnL %.4f >= threshold %.2f; closing all positions and exiting",
                            pnl,
                            abs(ratio.short_leg_each_qty)*0.1,
                        )
                        await self._close_entire_ratio(ratio)
                        return

                    self._set_strategy_state(
                        action="converting ratio spread to short straddle",
                        status="waiting for conversion",
                        status_message="converting ratio spread to short straddle",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    LOGGER.info(
                        "PnL %.4f < threshold %.2f; converting to short straddle",
                        pnl,
                        abs(ratio.short_leg_each_qty)*0.1,
                    )

                    try:
                        await self._convert_to_short_straddle(ratio)
                    except Exception:
                        self._set_status("adjustment failed, emergency flatten")
                        LOGGER.exception(
                            "Adjustment failed. Emergency flattening remaining initial ratio legs."
                        )
                        await self._close_entire_ratio(ratio)
                        ratio = None
                        self._set_strategy_state(
                            action="waiting for ratio spread",
                            status="no ratio spread found",
                            status_message="no ratio spread found",
                            trigger_price=None,
                            trigger_pnl=None,
                        )
                        await asyncio.sleep(self.settings.poll_interval_seconds)
                        continue

                    # Conversion succeeded — switch to straddle monitoring mode.
                    self._set_strategy_state(
                        action="converting short straddle to iron fly",
                        status="waiting for conversion",
                        status_message="converting short straddle to iron fly",
                        trigger_price=None,
                        trigger_pnl=None,
                    )
                    straddle = None
                    breakevens_logged = False
                    wing_put_strike = None
                    wing_call_strike = None
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

                        straddle_be = self.positions.compute_straddle_breakevens(straddle)
                        if wing_put_strike is None or wing_call_strike is None:
                            wing_put_strike, wing_call_strike = await self._convert_short_straddle_to_iron_fly(
                                straddle,
                                straddle_be,
                            )
                            # Now compute iron fly breakevens
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
                            
                            # Log iron fly breakevens immediately after creation
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
            if ratio_scan_task is not None and not ratio_scan_task.done():
                ratio_scan_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ratio_scan_task
