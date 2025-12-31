"""
Smart Execution Model - Institutional Grade
Handles order execution with edge verification, TCA, and adaptive routing.

Author: OMA Strategy Team
Version: 2.0 - Enhanced with min edge checks, TCA logging, cancel/replace
"""

from AlgorithmImports import *
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from utils.logger import LogLevel


class OrderStatus(Enum):
    """Custom order status for tracking."""

    PENDING = "PENDING"
    WORKING = "WORKING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REPLACED = "REPLACED"


@dataclass
class ExecutionConfig:
    """
    Configuration for smart execution.
    """

    # Latency
    simulated_latency_ms: int = 50

    # Limit order placement
    limit_order_offset_bps: float = 10  # Basis points inside spread
    min_absolute_tick_offset: float = 0.05  # Minimum $0.05 offset

    # Edge requirements (in basis points)
    min_edge_after_costs_bps: float = 20  # 20 bps minimum edge to trade
    spread_cost_multiplier: float = 0.5  # Assume cross half the spread
    impact_buffer_bps: float = 5  # Buffer for market impact

    # Participation limits
    max_participation_pct: float = 0.20  # Max 20% of displayed liquidity
    illiquid_participation_pct: float = 0.05  # 5% for illiquid contracts
    illiquid_size_threshold: int = 10  # Size < 10 = illiquid

    # Cancel/replace logic
    unfilled_timeout_seconds: int = 60  # Cancel/replace after 60s
    max_replace_attempts: int = 3  # Max replacements per order
    replace_price_improvement_bps: float = 5  # Move 5 bps toward mid on replace

    # Spread thresholds
    wide_spread_threshold_pct: float = 0.005  # 0.5% = wide spread
    use_limit_for_wide_spreads: bool = True
    use_market_for_urgent: bool = True

    # TCA logging
    enable_tca_logging: bool = True
    tca_log_frequency: int = 100  # Log summary every 100 fills


@dataclass
class TCARecord:
    """
    Transaction Cost Analysis record for a single execution.
    """

    timestamp: datetime
    symbol: Symbol
    order_type: str
    side: str  # BUY or SELL
    quantity: int

    # Prices
    mid_at_submission: float
    bid_at_submission: float
    ask_at_submission: float
    spread_at_submission: float
    execution_price: float

    # Slippage metrics
    slippage_vs_mid_bps: float
    slippage_vs_touch_bps: float
    realized_spread_capture_bps: float

    # Theoretical edge from alpha (if available)
    theoretical_edge_bps: Optional[float] = None
    net_edge_after_costs_bps: Optional[float] = None

    # Timing
    time_to_fill_seconds: float = 0
    was_replaced: bool = False
    replace_count: int = 0


@dataclass
class OrderTracker:
    """
    Tracks an order through its lifecycle.
    """

    order_id: int
    symbol: Symbol
    quantity: int
    original_quantity: int
    order_type: str
    submission_time: datetime
    limit_price: Optional[float]

    # Market data at submission
    bid_at_submission: float
    ask_at_submission: float
    mid_at_submission: float

    # Theoretical edge
    theoretical_edge_bps: Optional[float] = None

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0

    # Replace tracking
    replace_count: int = 0
    last_replace_time: Optional[datetime] = None


class SmartExecutionModel(ExecutionModel):
    """
    Institutional-grade execution model with:

    1. Minimum edge verification before trading
    2. Adaptive limit order placement (relative + absolute offset)
    3. Cancel/replace logic for stale orders
    4. Participation limits based on displayed liquidity
    5. Comprehensive TCA logging

    Key improvements over V1:
    - Edge verification: only trade when edge > spread + impact
    - Absolute tick offset for low-priced options
    - Cancel/replace after timeout
    - TCA tracking with slippage analysis
    """

    def __init__(
        self,
        latency_ms: int = 50,
        limit_order_offset_bps: float = 10,
        use_adaptive_sizing: bool = True,
        logger=None,
        config: ExecutionConfig = None,
    ):
        """
        Initialize the smart execution model.

        Args:
            latency_ms: Simulated latency in milliseconds
            limit_order_offset_bps: Basis points inside spread for limits
            use_adaptive_sizing: Whether to adapt size based on liquidity
            logger: StrategyLogger instance
            config: Full ExecutionConfig (overrides individual params)
        """
        if config:
            self.config = config
        else:
            self.config = ExecutionConfig(
                simulated_latency_ms=latency_ms,
                limit_order_offset_bps=limit_order_offset_bps,
            )

        self.use_adaptive_sizing = use_adaptive_sizing
        self.logger = logger

        # Order tracking
        self.pending_orders: Dict[int, OrderTracker] = {}
        self.working_orders: Dict[int, OrderTracker] = {}

        # TCA records
        self.tca_records: deque = deque(maxlen=10000)
        self.tca_summary: Dict = {
            "total_fills": 0,
            "limit_fills": 0,
            "market_fills": 0,
            "cancelled_orders": 0,
            "replaced_orders": 0,
            "blocked_for_edge": 0,
            "total_slippage_bps": 0,
            "total_spread_capture_bps": 0,
        }

        # Execution metrics
        self.execution_metrics = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "replaced_orders": 0,
            "avg_fill_time": 0,
            "avg_slippage_bps": 0,
            "edge_blocked_count": 0,
        }

    def Execute(self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]) -> None:
        """
        Execute portfolio targets with smart routing.

        Args:
            algorithm: QuantConnect algorithm instance
            targets: List of portfolio targets to execute
        """
        if not targets:
            # Check for stale orders to replace/cancel
            self._manage_working_orders(algorithm)
            return

        if self.logger:
            self.logger.log(
                f"Executing {len(targets)} portfolio targets", LogLevel.INFO
            )

        # Manage existing orders first
        self._manage_working_orders(algorithm)

        # Process each target
        for target in targets:
            self._execute_target(algorithm, target)

    def _execute_target(self, algorithm: QCAlgorithm, target: PortfolioTarget) -> None:
        """
        Execute individual portfolio target with edge verification.

        Args:
            algorithm: Algorithm instance
            target: Portfolio target to execute
        """
        try:
            # Get current holdings
            current_quantity = algorithm.Portfolio[target.Symbol].Quantity
            order_quantity = target.Quantity - current_quantity

            # Skip if no change needed
            if abs(order_quantity) < 1:
                return

            # Get security
            security = algorithm.Securities.get(target.Symbol)
            if not security or not security.HasData:
                if self.logger:
                    self.logger.log(
                        f"No market data for {target.Symbol}", LogLevel.WARNING
                    )
                return

            # Route based on security type
            if target.Symbol.SecurityType == SecurityType.Option:
                self._execute_option_order(algorithm, security, order_quantity, target)
            else:
                self._execute_equity_order(algorithm, security, order_quantity, target)

        except Exception as e:
            if self.logger:
                self.logger.log(
                    f"Execution error for {target.Symbol}: {e}", LogLevel.ERROR
                )

    def _execute_option_order(
        self, algorithm: QCAlgorithm, security, quantity: int, target: PortfolioTarget
    ) -> None:
        """
        Execute option order with edge verification and smart routing.

        Args:
            algorithm: Algorithm instance
            security: Option security
            quantity: Order quantity (positive = buy, negative = sell)
            target: Portfolio target
        """
        # Get market data
        bid = security.BidPrice
        ask = security.AskPrice

        if bid <= 0 or ask <= 0 or ask <= bid:
            if self.logger:
                self.logger.log(
                    f"Invalid quote for {security.Symbol}: bid={bid}, ask={ask}",
                    LogLevel.WARNING,
                )
            return

        spread = ask - bid
        mid = (bid + ask) / 2
        spread_pct = spread / mid if mid > 0 else 0

        # Get theoretical edge from insight if available
        theoretical_edge_bps = self._get_theoretical_edge(target)

        # Calculate estimated costs
        spread_cost_bps = (spread_pct * 10000) * self.config.spread_cost_multiplier
        impact_buffer_bps = self.config.impact_buffer_bps
        total_cost_bps = spread_cost_bps + impact_buffer_bps

        # Check minimum edge
        if theoretical_edge_bps is not None:
            net_edge = theoretical_edge_bps - total_cost_bps

            if net_edge < self.config.min_edge_after_costs_bps:
                self.execution_metrics["edge_blocked_count"] += 1
                self.tca_summary["blocked_for_edge"] += 1

                if self.logger:
                    self.logger.log(
                        f"Order blocked - insufficient edge: {security.Symbol} | "
                        f"Edge: {theoretical_edge_bps:.0f}bps | "
                        f"Cost: {total_cost_bps:.0f}bps | "
                        f"Net: {net_edge:.0f}bps < min {self.config.min_edge_after_costs_bps}bps",
                        LogLevel.DEBUG,
                    )
                return

        # Adaptive sizing
        if self.use_adaptive_sizing:
            quantity = self._adapt_order_size(security, quantity)

        if abs(quantity) < 1:
            return

        # Determine order type
        use_limit = self._should_use_limit_order(target, spread_pct)

        # Create order tracker
        tracker = OrderTracker(
            order_id=0,  # Will be set after order submission
            symbol=security.Symbol,
            quantity=quantity,
            original_quantity=quantity,
            order_type="LIMIT" if use_limit else "MARKET",
            submission_time=algorithm.Time,
            limit_price=None,
            bid_at_submission=bid,
            ask_at_submission=ask,
            mid_at_submission=mid,
            theoretical_edge_bps=theoretical_edge_bps,
        )

        # Execute order
        qty = int(quantity)
        if use_limit:
            limit_price = self._calculate_limit_price(bid, ask, quantity, mid)
            tracker.limit_price = limit_price

            order = algorithm.LimitOrder(security.Symbol, qty, limit_price)

            if self.logger:
                self.logger.log(
                    f"LIMIT {qty:+d} {security.Symbol} @ ${limit_price:.2f} | "
                    f"Spread: {spread_pct:.2%} | Mid: ${mid:.2f}",
                    LogLevel.INFO,
                )
        else:
            order = algorithm.MarketOrder(security.Symbol, qty)

            if self.logger:
                self.logger.log(
                    f"MARKET {qty:+d} {security.Symbol} @ ~${mid:.2f}",
                    LogLevel.INFO,
                )

        # Track order
        if order:
            tracker.order_id = order.OrderId
            tracker.status = OrderStatus.WORKING
            self.working_orders[order.OrderId] = tracker
            self.execution_metrics["total_orders"] += 1

    def _execute_equity_order(
        self, algorithm: QCAlgorithm, security, quantity: int, target: PortfolioTarget
    ) -> None:
        """
        Execute equity order (typically for hedging).

        Uses market orders for immediacy on hedges.
        """
        # Market orders for equity hedges
        qty = int(quantity)
        order = algorithm.MarketOrder(security.Symbol, qty)

        if order and self.logger:
            self.logger.log(
                f"HEDGE MARKET {qty:+d} {security.Symbol} @ ${security.Price:.2f}",
                LogLevel.INFO,
            )

        if order:
            self.execution_metrics["total_orders"] += 1

    def _get_theoretical_edge(self, target: PortfolioTarget) -> Optional[float]:
        """
        Extract theoretical edge from insight metadata.

        Args:
            target: Portfolio target

        Returns:
            Edge in basis points or None
        """
        # Try to get edge from insight
        if hasattr(target, "Insight") and target.Insight:
            insight = target.Insight

            # Check for stored edge metrics
            iv_rv_signal = insight.Properties.get("IVRVSignal", 0)
            spread_signal = insight.Properties.get("SpreadSignal", 0)

            # Convert signal strength to approximate edge
            # Assumes 1.0 signal ≈ 100 bps edge (rough calibration)
            total_signal = abs(iv_rv_signal) + abs(spread_signal) * 0.5
            edge_bps = total_signal * 100

            return edge_bps

        return None

    def _should_use_limit_order(
        self, target: PortfolioTarget, spread_pct: float
    ) -> bool:
        """
        Determine whether to use limit order.

        Args:
            target: Portfolio target
            spread_pct: Current spread as percentage

        Returns:
            True if should use limit order
        """
        # Check for spread capture signal
        if hasattr(target, "Insight") and target.Insight:
            signal_type = target.Insight.Properties.get("SignalType", "")
            spread_signal = target.Insight.Properties.get("SpreadSignal", 0)

            if signal_type == "Spread" or spread_signal > 0.5:
                return True

        # Wide spreads -> limit orders
        if spread_pct > self.config.wide_spread_threshold_pct:
            return self.config.use_limit_for_wide_spreads

        return True  # Default to limit orders for options

    def _calculate_limit_price(
        self, bid: float, ask: float, quantity: int, mid: float
    ) -> float:
        """
        Calculate optimal limit price with both relative and absolute offsets.

        Args:
            bid: Current bid
            ask: Current ask
            quantity: Order quantity (positive = buy)
            mid: Mid price

        Returns:
            Limit price
        """
        spread = ask - bid

        # Relative offset (basis points of mid)
        relative_offset = mid * self.config.limit_order_offset_bps / 10000

        # Absolute offset (minimum tick improvement)
        absolute_offset = self.config.min_absolute_tick_offset

        # Use larger of the two
        offset = max(relative_offset, absolute_offset)

        if quantity > 0:
            # Buying - bid side, offset up from bid
            limit_price = bid + offset
            # Don't exceed mid
            limit_price = min(limit_price, mid)
        else:
            # Selling - ask side, offset down from ask
            limit_price = ask - offset
            # Don't go below mid
            limit_price = max(limit_price, mid)

        # Round to tick size (assume $0.01)
        limit_price = round(limit_price, 2)

        return limit_price

    def _adapt_order_size(self, security, quantity: int) -> int:
        """
        Adapt order size based on displayed liquidity.

        Args:
            security: Security object
            quantity: Desired quantity

        Returns:
            Adapted quantity
        """
        bid_size = security.BidSize if security.BidSize > 0 else 1
        ask_size = security.AskSize if security.AskSize > 0 else 1

        # Determine relevant liquidity
        if quantity > 0:
            available = ask_size
        else:
            available = bid_size

        # Determine participation rate
        if available < self.config.illiquid_size_threshold:
            max_participation = self.config.illiquid_participation_pct
        else:
            max_participation = self.config.max_participation_pct

        max_size = int(available * max_participation)
        max_size = max(1, max_size)  # At least 1 contract

        # Apply limit
        if abs(quantity) > max_size:
            adapted = max_size if quantity > 0 else -max_size

            if self.logger:
                self.logger.log(
                    f"Adapted size {quantity} -> {adapted} | "
                    f"Available: {available} | Max participation: {max_participation:.0%}",
                    LogLevel.DEBUG,
                )

            return adapted

        return quantity

    def _manage_working_orders(self, algorithm: QCAlgorithm) -> None:
        """
        Manage working orders - cancel/replace stale orders.

        Args:
            algorithm: Algorithm instance
        """
        current_time = algorithm.Time
        orders_to_remove = []

        for order_id, tracker in self.working_orders.items():
            # Check if order is still working
            order = algorithm.Transactions.GetOrderById(order_id)
            if order is None:
                orders_to_remove.append(order_id)
                continue

            if order.Status in [
                OrderStatus.Filled,
                OrderStatus.Canceled,
                OrderStatus.Invalid,
            ]:
                orders_to_remove.append(order_id)
                continue

            # Check for timeout
            elapsed = (current_time - tracker.submission_time).total_seconds()
            if elapsed > self.config.unfilled_timeout_seconds:
                # Check if we can replace
                if tracker.replace_count < self.config.max_replace_attempts:
                    self._replace_order(algorithm, order_id, tracker)
                else:
                    # Max replacements reached - cancel
                    algorithm.Transactions.CancelOrder(order_id)
                    tracker.status = OrderStatus.CANCELLED
                    self.execution_metrics["cancelled_orders"] += 1
                    self.tca_summary["cancelled_orders"] += 1

                    if self.logger:
                        self.logger.log(
                            f"Cancelled stale order after {tracker.replace_count} replacements: "
                            f"{tracker.symbol}",
                            LogLevel.INFO,
                        )

        # Clean up
        for order_id in orders_to_remove:
            del self.working_orders[order_id]

    def _replace_order(
        self, algorithm: QCAlgorithm, order_id: int, tracker: OrderTracker
    ) -> None:
        """
        Replace a stale order with improved price.

        Args:
            algorithm: Algorithm instance
            order_id: Order ID to replace
            tracker: Order tracker
        """
        # Cancel existing order
        algorithm.Transactions.CancelOrder(order_id)

        # Get current market data
        security = algorithm.Securities.get(tracker.symbol)
        if not security or not security.HasData:
            return

        bid = security.BidPrice
        ask = security.AskPrice

        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2

        # Check if edge is still valid
        theoretical_edge = tracker.theoretical_edge_bps
        if theoretical_edge is not None:
            spread_pct = (ask - bid) / mid
            total_cost = (
                spread_pct * 10000 * self.config.spread_cost_multiplier
                + self.config.impact_buffer_bps
            )

            if theoretical_edge - total_cost < self.config.min_edge_after_costs_bps:
                # Edge gone - don't replace
                tracker.status = OrderStatus.CANCELLED
                if self.logger:
                    self.logger.log(
                        f"Edge gone - not replacing: {tracker.symbol}", LogLevel.DEBUG
                    )
                return

        # Calculate new limit price (more aggressive toward mid)
        improvement = mid * self.config.replace_price_improvement_bps / 10000

        if tracker.quantity > 0:
            # Buying - move bid up
            new_limit = min(tracker.limit_price + improvement, mid)
        else:
            # Selling - move ask down
            new_limit = max(tracker.limit_price - improvement, mid)

        new_limit = round(new_limit, 2)

        # Submit replacement order
        new_order = algorithm.LimitOrder(tracker.symbol, tracker.quantity, new_limit)

        if new_order:
            # Update tracker
            tracker.order_id = new_order.OrderId
            tracker.limit_price = new_limit
            tracker.replace_count += 1
            tracker.last_replace_time = algorithm.Time
            tracker.submission_time = algorithm.Time

            # Re-add to working orders
            self.working_orders[new_order.OrderId] = tracker
            self.execution_metrics["replaced_orders"] += 1
            self.tca_summary["replaced_orders"] += 1

            if self.logger:
                self.logger.log(
                    f"Replaced order #{order_id} -> #{new_order.OrderId} | "
                    f"{tracker.symbol} @ ${new_limit:.2f} (attempt {tracker.replace_count})",
                    LogLevel.INFO,
                )

    def OnOrderEvent(self, algorithm: QCAlgorithm, orderEvent) -> None:
        """
        Handle order events for TCA and metrics tracking.

        Args:
            algorithm: Algorithm instance
            orderEvent: Order event from Lean
        """
        order_id = orderEvent.OrderId

        if orderEvent.Status == OrderStatus.Filled:
            self._handle_fill(algorithm, orderEvent)
        elif orderEvent.Status == OrderStatus.Canceled:
            self._handle_cancel(algorithm, orderEvent)

    def _handle_fill(self, algorithm: QCAlgorithm, orderEvent) -> None:
        """
        Handle order fill - record TCA metrics.

        Args:
            algorithm: Algorithm instance
            orderEvent: Fill event
        """
        order_id = orderEvent.OrderId
        tracker = self.working_orders.get(order_id) or self.pending_orders.get(order_id)

        self.execution_metrics["filled_orders"] += 1

        # Create TCA record
        if tracker:
            fill_price = orderEvent.FillPrice
            fill_quantity = orderEvent.FillQuantity

            # Calculate slippage
            mid = tracker.mid_at_submission
            slippage_vs_mid = fill_price - mid

            if tracker.quantity > 0:
                # Bought - slippage is price above mid
                touch_price = tracker.ask_at_submission
                slippage_vs_touch = fill_price - touch_price
            else:
                # Sold - slippage is price below mid (invert)
                slippage_vs_mid = -slippage_vs_mid
                touch_price = tracker.bid_at_submission
                slippage_vs_touch = touch_price - fill_price

            # Convert to bps
            slippage_vs_mid_bps = (slippage_vs_mid / mid) * 10000 if mid > 0 else 0
            slippage_vs_touch_bps = (slippage_vs_touch / mid) * 10000 if mid > 0 else 0

            # Spread capture = how much of spread we captured
            spread = tracker.ask_at_submission - tracker.bid_at_submission
            if tracker.quantity > 0:
                spread_capture = tracker.ask_at_submission - fill_price
            else:
                spread_capture = fill_price - tracker.bid_at_submission

            spread_capture_bps = (spread_capture / mid) * 10000 if mid > 0 else 0

            # Fill time
            fill_time = (algorithm.Time - tracker.submission_time).total_seconds()

            # Net edge
            net_edge = None
            if tracker.theoretical_edge_bps is not None:
                net_edge = tracker.theoretical_edge_bps - slippage_vs_mid_bps

            # Create TCA record
            tca = TCARecord(
                timestamp=algorithm.Time,
                symbol=tracker.symbol,
                order_type=tracker.order_type,
                side="BUY" if tracker.quantity > 0 else "SELL",
                quantity=abs(fill_quantity),
                mid_at_submission=mid,
                bid_at_submission=tracker.bid_at_submission,
                ask_at_submission=tracker.ask_at_submission,
                spread_at_submission=spread,
                execution_price=fill_price,
                slippage_vs_mid_bps=slippage_vs_mid_bps,
                slippage_vs_touch_bps=slippage_vs_touch_bps,
                realized_spread_capture_bps=spread_capture_bps,
                theoretical_edge_bps=tracker.theoretical_edge_bps,
                net_edge_after_costs_bps=net_edge,
                time_to_fill_seconds=fill_time,
                was_replaced=tracker.replace_count > 0,
                replace_count=tracker.replace_count,
            )

            self.tca_records.append(tca)

            # Update summary
            self.tca_summary["total_fills"] += 1
            if tracker.order_type == "LIMIT":
                self.tca_summary["limit_fills"] += 1
            else:
                self.tca_summary["market_fills"] += 1

            self.tca_summary["total_slippage_bps"] += slippage_vs_mid_bps
            self.tca_summary["total_spread_capture_bps"] += spread_capture_bps

            # Update execution metrics
            n = self.execution_metrics["filled_orders"]
            avg_slippage = self.execution_metrics["avg_slippage_bps"]
            self.execution_metrics["avg_slippage_bps"] = (
                avg_slippage * (n - 1) + slippage_vs_mid_bps
            ) / n

            avg_fill_time = self.execution_metrics["avg_fill_time"]
            self.execution_metrics["avg_fill_time"] = (
                avg_fill_time * (n - 1) + fill_time
            ) / n

            # Log TCA
            if self.config.enable_tca_logging and self.logger:
                self.logger.log(
                    f"TCA: {tca.side} {tca.quantity} {tca.symbol} | "
                    f"Fill: ${fill_price:.2f} vs Mid: ${mid:.2f} | "
                    f"Slippage: {slippage_vs_mid_bps:.1f}bps | "
                    f"Spread capture: {spread_capture_bps:.1f}bps | "
                    f"Time: {fill_time:.1f}s",
                    LogLevel.DEBUG,
                )

            # Periodic summary
            if self.tca_summary["total_fills"] % self.config.tca_log_frequency == 0:
                self._log_tca_summary()

            # Clean up
            if order_id in self.working_orders:
                del self.working_orders[order_id]

    def _handle_cancel(self, algorithm: QCAlgorithm, orderEvent) -> None:
        """Handle order cancellation."""
        order_id = orderEvent.OrderId

        if order_id in self.working_orders:
            del self.working_orders[order_id]

        # Metrics updated in _manage_working_orders

    def _log_tca_summary(self) -> None:
        """Log TCA summary statistics."""
        if not self.logger:
            return

        total = self.tca_summary["total_fills"]
        if total == 0:
            return

        avg_slippage = self.tca_summary["total_slippage_bps"] / total
        avg_spread_capture = self.tca_summary["total_spread_capture_bps"] / total
        limit_pct = self.tca_summary["limit_fills"] / total * 100

        self.logger.log(
            f"TCA Summary ({total} fills) | "
            f"Avg slippage: {avg_slippage:.1f}bps | "
            f"Avg spread capture: {avg_spread_capture:.1f}bps | "
            f"Limit orders: {limit_pct:.0f}% | "
            f"Cancelled: {self.tca_summary['cancelled_orders']} | "
            f"Edge-blocked: {self.tca_summary['blocked_for_edge']}",
            LogLevel.INFO,
        )

    def GetMetrics(self) -> Dict:
        """
        Get execution metrics.

        Returns:
            Dictionary of execution metrics
        """
        return self.execution_metrics.copy()

    def get_tca_summary(self) -> Dict:
        """
        Get TCA summary statistics.

        Returns:
            Dictionary with TCA statistics
        """
        summary = self.tca_summary.copy()

        if summary["total_fills"] > 0:
            summary["avg_slippage_bps"] = (
                summary["total_slippage_bps"] / summary["total_fills"]
            )
            summary["avg_spread_capture_bps"] = (
                summary["total_spread_capture_bps"] / summary["total_fills"]
            )
            summary["limit_fill_pct"] = summary["limit_fills"] / summary["total_fills"]
            summary["fill_rate"] = (
                summary["total_fills"]
                / (summary["total_fills"] + summary["cancelled_orders"])
                if summary["total_fills"] + summary["cancelled_orders"] > 0
                else 0
            )

        return summary

    def get_recent_tca_records(self, n: int = 100) -> List[TCARecord]:
        """
        Get recent TCA records.

        Args:
            n: Number of records to return

        Returns:
            List of TCARecord objects
        """
        return list(self.tca_records)[-n:]

    def get_tca_by_symbol(self, symbol: Symbol = None) -> Dict:
        """
        Get TCA statistics aggregated by symbol.

        Args:
            symbol: Optional filter by symbol

        Returns:
            Dictionary mapping symbol to TCA stats
        """
        by_symbol = {}

        for record in self.tca_records:
            if symbol and record.symbol != symbol:
                continue

            sym_str = str(record.symbol)
            if sym_str not in by_symbol:
                by_symbol[sym_str] = {
                    "fills": 0,
                    "total_slippage_bps": 0,
                    "total_spread_capture_bps": 0,
                    "avg_fill_time": 0,
                }

            stats = by_symbol[sym_str]
            stats["fills"] += 1
            stats["total_slippage_bps"] += record.slippage_vs_mid_bps
            stats["total_spread_capture_bps"] += record.realized_spread_capture_bps
            stats["avg_fill_time"] += record.time_to_fill_seconds

        # Calculate averages
        for sym_str, stats in by_symbol.items():
            if stats["fills"] > 0:
                stats["avg_slippage_bps"] = stats["total_slippage_bps"] / stats["fills"]
                stats["avg_spread_capture_bps"] = (
                    stats["total_spread_capture_bps"] / stats["fills"]
                )
                stats["avg_fill_time"] = stats["avg_fill_time"] / stats["fills"]

        return by_symbol
