"""
Smart Execution Model
Handles order execution with latency simulation and intelligent routing.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from typing import List, Dict
import numpy as np
from datetime import datetime, timedelta
import time


class SmartExecutionModel(ExecutionModel):
    """
    Smart execution model that implements:
    1. Latency simulation (50ms default)
    2. Limit order placement for spread capture
    3. Market order fallback for urgent executions
    4. Order type selection based on signal characteristics
    """
    
    def __init__(self, 
                 latency_ms: int = 50,
                 limit_order_offset_bps: float = 10,
                 use_adaptive_sizing: bool = True,
                 logger = None):
        """
        Initialize the smart execution model.
        
        Args:
            latency_ms: Simulated latency in milliseconds
            limit_order_offset_bps: Basis points inside spread for limit orders
            use_adaptive_sizing: Whether to adapt order size based on liquidity
            logger: Strategy logger instance
        """
        self.latency_ms = latency_ms
        self.limit_order_offset_bps = limit_order_offset_bps
        self.use_adaptive_sizing = use_adaptive_sizing
        self.logger = logger
        
        # Track pending orders
        self.pending_orders = {}
        self.order_tracker = {}
        
        # Execution metrics
        self.execution_metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'avg_fill_time': 0
        }
    
    def Execute(self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]) -> None:
        """
        Execute the portfolio targets with smart routing.
        
        Args:
            algorithm: Algorithm instance
            targets: List of portfolio targets to execute
        """
        if not targets:
            return
        
        # Log execution batch
        if self.logger:
            self.logger.log(f"Executing {len(targets)} portfolio targets", LogLevel.INFO)
        
        # Simulate latency
        self._simulate_latency()
        
        # Process each target
        for target in targets:
            self._execute_target(algorithm, target)
    
    def _simulate_latency(self) -> None:
        """Simulate network/processing latency."""
        if self.latency_ms > 0:
            # In backtesting, we can't actually sleep, but we track the latency
            # Real QC execution would handle this differently
            pass
    
    def _execute_target(self, algorithm: QCAlgorithm, target: PortfolioTarget) -> None:
        """
        Execute individual portfolio target.
        
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
            
            # Get security and market data
            security = algorithm.Securities[target.Symbol]
            
            # Skip if no market data
            if not security.HasData:
                if self.logger:
                    self.logger.log(f"No market data for {target.Symbol}", LogLevel.WARNING)
                return
            
            # Determine order type and execute
            if target.Symbol.SecurityType == SecurityType.Option:
                self._execute_option_order(algorithm, security, order_quantity, target)
            else:
                self._execute_equity_order(algorithm, security, order_quantity, target)
                
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error executing target for {target.Symbol}: {e}", LogLevel.ERROR)
    
    def _execute_option_order(self, 
                             algorithm: QCAlgorithm, 
                             security, 
                             quantity: float,
                             target: PortfolioTarget) -> None:
        """
        Execute option order with appropriate order type.
        
        Args:
            algorithm: Algorithm instance
            security: Option security
            quantity: Order quantity (positive = buy, negative = sell)
            target: Original portfolio target
        """
        # Get current bid/ask
        bid = security.BidPrice
        ask = security.AskPrice
        
        # Validate prices
        if bid <= 0 or ask <= 0:
            if self.logger:
                self.logger.log(f"Invalid bid/ask for {security.Symbol}: Bid={bid}, Ask={ask}", LogLevel.WARNING)
            return
        
        # Calculate spread metrics
        spread = ask - bid
        mark_price = (bid + ask) / 2
        spread_pct = spread / mark_price if mark_price > 0 else 0
        
        # Adapt size based on liquidity if enabled
        if self.use_adaptive_sizing:
            quantity = self._adapt_order_size(security, quantity)
        
        # Determine order type based on signal and spread
        use_limit_order = self._should_use_limit_order(target, spread_pct)
        
        if use_limit_order:
            # Place limit order
            limit_price = self._calculate_limit_price(bid, ask, quantity)
            order = algorithm.LimitOrder(security.Symbol, int(quantity), limit_price)
            
            if self.logger:
                self.logger.log(
                    f"Placed limit order: {security.Symbol} "
                    f"Qty={quantity} Price={limit_price:.2f} "
                    f"Spread={spread_pct:.2%}",
                    LogLevel.INFO
                )
        else:
            # Place market order for urgent execution
            order = algorithm.MarketOrder(security.Symbol, int(quantity))
            
            if self.logger:
                self.logger.log(
                    f"Placed market order: {security.Symbol} "
                    f"Qty={quantity} Mark={mark_price:.2f}",
                    LogLevel.INFO
                )
        
        # Track order
        if order:
            self.order_tracker[order.OrderId] = {
                'symbol': security.Symbol,
                'quantity': quantity,
                'time': algorithm.Time,
                'type': 'limit' if use_limit_order else 'market'
            }
            self.execution_metrics['total_orders'] += 1
    
    def _execute_equity_order(self,
                             algorithm: QCAlgorithm,
                             security,
                             quantity: float,
                             target: PortfolioTarget) -> None:
        """
        Execute equity order (for hedging).
        
        Args:
            algorithm: Algorithm instance
            security: Equity security
            quantity: Order quantity
            target: Portfolio target
        """
        # Use market orders for equity hedges
        order = algorithm.MarketOrder(security.Symbol, int(quantity))
        
        if order and self.logger:
            self.logger.log(
                f"Placed equity hedge order: {security.Symbol} Qty={quantity}",
                LogLevel.INFO
            )
    
    def _should_use_limit_order(self, target: PortfolioTarget, spread_pct: float) -> bool:
        """
        Determine whether to use limit order based on signal type and spread.
        
        Args:
            target: Portfolio target with insight metadata
            spread_pct: Current bid/ask spread as percentage
            
        Returns:
            True if should use limit order
        """
        # Check if this is a spread capture signal
        if hasattr(target, 'Insight') and target.Insight:
            signal_type = target.Insight.Properties.get('SignalType', '')
            spread_signal = target.Insight.Properties.get('SpreadSignal', 0)
            
            # Use limit orders for spread capture signals
            if signal_type == 'Spread' or spread_signal > 0.5:
                return True
        
        # Use limit orders for wide spreads
        return spread_pct > 0.005  # 0.5%
    
    def _calculate_limit_price(self, bid: float, ask: float, quantity: float) -> float:
        """
        Calculate optimal limit price.
        
        Args:
            bid: Current bid price
            ask: Current ask price
            quantity: Order quantity (positive = buy)
            
        Returns:
            Limit price
        """
        spread = ask - bid
        offset = spread * self.limit_order_offset_bps / 10000
        
        if quantity > 0:
            # Buying - place order slightly above bid
            return bid + offset
        else:
            # Selling - place order slightly below ask
            return ask - offset
    
    def _adapt_order_size(self, security, quantity: float) -> float:
        """
        Adapt order size based on available liquidity.
        
        Args:
            security: Security object
            quantity: Desired quantity
            
        Returns:
            Adapted quantity
        """
        # Get market depth
        bid_size = security.BidSize
        ask_size = security.AskSize
        
        # Limit to available liquidity
        if quantity > 0:
            # Buying - limited by ask size
            max_size = ask_size * 0.2  # Take max 20% of displayed liquidity
            return min(quantity, max_size) if max_size > 0 else quantity
        else:
            # Selling - limited by bid size
            max_size = bid_size * 0.2
            return max(quantity, -max_size) if max_size > 0 else quantity
    
    def OnOrderEvent(self, algorithm: QCAlgorithm, orderEvent) -> None:
        """
        Handle order events for tracking and metrics.
        
        Args:
            algorithm: Algorithm instance
            orderEvent: Order event
        """
        if orderEvent.OrderId not in self.order_tracker:
            return
        
        # Update metrics based on order status
        if orderEvent.Status == OrderStatus.Filled:
            self.execution_metrics['filled_orders'] += 1
            
            # Calculate fill time
            order_info = self.order_tracker[orderEvent.OrderId]
            fill_time = (algorithm.Time - order_info['time']).total_seconds()
            
            # Update average fill time
            n = self.execution_metrics['filled_orders']
            avg = self.execution_metrics['avg_fill_time']
            self.execution_metrics['avg_fill_time'] = (avg * (n-1) + fill_time) / n
            
            if self.logger:
                self.logger.log(
                    f"Order filled: {order_info['symbol']} "
                    f"Fill time: {fill_time:.1f}s",
                    LogLevel.DEBUG
                )
                
        elif orderEvent.Status == OrderStatus.Canceled:
            self.execution_metrics['cancelled_orders'] += 1
    
    def GetMetrics(self) -> Dict:
        """
        Get execution metrics.
        
        Returns:
            Dictionary of execution metrics
        """
        return self.execution_metrics.copy()