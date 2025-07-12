"""
IV vs RV Spread Alpha Model
Generates trading signals based on implied vs realized volatility spreads and bid/ask spreads.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime, timedelta


class IVRVSpreadAlphaModel(AlphaModel):
    """
    Alpha model that generates signals based on:
    1. IV vs RV spread arbitrage opportunities
    2. Bid/ask spread capture opportunities
    
    Signals:
    - Short premium when IV > RV * threshold
    - Place passive limit orders when bid/ask spread > threshold
    """
    
    def __init__(self, 
                 iv_rv_threshold: float = 1.2,
                 spread_threshold: float = 0.005,
                 rv_calculator = None,
                 logger = None):
        """
        Initialize the IV/RV spread alpha model.
        
        Args:
            iv_rv_threshold: Threshold for IV/RV ratio to trigger signal (default 1.2 = 20% premium)
            spread_threshold: Minimum bid/ask spread as % of mark price (default 0.5%)
            rv_calculator: Realized volatility calculator instance
            logger: Strategy logger instance
        """
        self.iv_rv_threshold = iv_rv_threshold
        self.spread_threshold = spread_threshold
        self.rv_calculator = rv_calculator
        self.logger = logger
        
        # Track active insights
        self.active_insights = {}
        
        # Greeks calculator
        self.greeks_cache = {}
        
    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        """
        Generate alpha signals based on current market data.
        
        Args:
            algorithm: Algorithm instance
            data: Current data slice
            
        Returns:
            List of insights for identified opportunities
        """
        insights = []
        
        # Skip if warming up
        if algorithm.IsWarmingUp:
            return insights
        
        # Process each option chain
        for kvp in data.OptionChains:
            chain = kvp.Value
            if len(chain) == 0:
                continue
                
            underlying_symbol = chain.Underlying.Symbol if chain.Underlying else chain.Symbol
            
            # Get realized volatility
            rv = self.rv_calculator.get_realized_vol(underlying_symbol)
            if rv is None or rv <= 0:
                continue
            
            # Analyze each contract
            for contract in chain:
                # Skip illiquid contracts
                if not self._is_liquid_contract(contract):
                    continue
                
                # Calculate signals
                iv_rv_signal = self._calculate_iv_rv_signal(contract, rv, algorithm)
                spread_signal = self._calculate_spread_signal(contract)
                
                # Generate insights based on signals
                if iv_rv_signal != 0 or spread_signal != 0:
                    insight = self._create_insight(
                        contract, 
                        iv_rv_signal, 
                        spread_signal,
                        rv,
                        algorithm
                    )
                    if insight:
                        insights.append(insight)
                        
        # Log insights generated
        if insights and self.logger:
            self.logger.log(f"Generated {len(insights)} alpha insights", LogLevel.DEBUG)
            
        return insights
    
    def _is_liquid_contract(self, contract) -> bool:
        """
        Check if contract meets liquidity requirements.
        
        Args:
            contract: Option contract
            
        Returns:
            True if contract is liquid enough to trade
        """
        return (contract.OpenInterest > 1000 and 
                contract.Volume > 1000 and
                contract.BidSize > 0 and 
                contract.AskSize > 0 and
                contract.Bid > 0 and 
                contract.Ask > 0)
    
    def _calculate_iv_rv_signal(self, contract, rv: float, algorithm: QCAlgorithm) -> float:
        """
        Calculate IV/RV arbitrage signal.
        
        Args:
            contract: Option contract
            rv: Realized volatility
            algorithm: Algorithm instance
            
        Returns:
            Signal strength (-1 to 1, negative = short premium)
        """
        try:
            # Get implied volatility
            iv = contract.ImpliedVolatility
            
            # Skip if IV not available or invalid
            if iv <= 0 or iv > 5:  # Cap at 500% IV
                return 0
            
            # Calculate IV/RV ratio
            iv_rv_ratio = iv / rv
            
            # Generate signal
            if iv_rv_ratio > self.iv_rv_threshold:
                # IV is expensive relative to RV - short premium
                signal_strength = min((iv_rv_ratio - self.iv_rv_threshold) / 0.5, 1.0)
                
                if self.logger:
                    self.logger.log(
                        f"IV/RV signal for {contract.Symbol}: "
                        f"IV={iv:.2%}, RV={rv:.2%}, Ratio={iv_rv_ratio:.2f}",
                        LogLevel.DEBUG
                    )
                
                return -signal_strength  # Negative for short
            
            return 0
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error calculating IV/RV signal: {e}", LogLevel.ERROR)
            return 0
    
    def _calculate_spread_signal(self, contract) -> float:
        """
        Calculate bid/ask spread capture signal.
        
        Args:
            contract: Option contract
            
        Returns:
            Signal strength (0 to 1)
        """
        try:
            # Calculate mark price and spread
            mark_price = (contract.Bid + contract.Ask) / 2
            spread = contract.Ask - contract.Bid
            
            # Avoid division by zero
            if mark_price <= 0:
                return 0
            
            # Calculate spread as percentage
            spread_pct = spread / mark_price
            
            # Generate signal if spread exceeds threshold
            if spread_pct > self.spread_threshold:
                signal_strength = min((spread_pct - self.spread_threshold) / self.spread_threshold, 1.0)
                
                if self.logger:
                    self.logger.log(
                        f"Spread signal for {contract.Symbol}: "
                        f"Spread={spread_pct:.2%}, Signal={signal_strength:.2f}",
                        LogLevel.DEBUG
                    )
                
                return signal_strength
            
            return 0
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error calculating spread signal: {e}", LogLevel.ERROR)
            return 0
    
    def _create_insight(self, 
                       contract, 
                       iv_rv_signal: float, 
                       spread_signal: float,
                       rv: float,
                       algorithm: QCAlgorithm) -> Insight:
        """
        Create insight from signals.
        
        Args:
            contract: Option contract
            iv_rv_signal: IV/RV arbitrage signal
            spread_signal: Bid/ask spread signal
            rv: Realized volatility
            algorithm: Algorithm instance
            
        Returns:
            Insight object or None
        """
        # Combine signals
        total_signal = iv_rv_signal + spread_signal * 0.5  # Weight spread signal less
        
        # Skip if signal too weak
        if abs(total_signal) < 0.1:
            return None
        
        # Determine direction
        if total_signal < 0:
            # Short premium (sell options)
            direction = InsightDirection.Down if contract.Right == OptionRight.Call else InsightDirection.Up
        else:
            # Long premium (buy options) - primarily for spread capture
            direction = InsightDirection.Up if contract.Right == OptionRight.Call else InsightDirection.Down
        
        # Calculate confidence and period
        confidence = min(abs(total_signal), 1.0)
        period = timedelta(hours=4)  # Hold for 4 hours by default
        
        # Create insight with metadata
        insight = Insight(
            contract.Symbol,
            period,
            InsightType.Price,
            direction,
            confidence,
            sourceModel="IVRVSpreadAlpha"
        )
        
        # Add metadata
        insight.Properties["SignalType"] = "IV_RV" if abs(iv_rv_signal) > abs(spread_signal) else "Spread"
        insight.Properties["IVRVSignal"] = iv_rv_signal
        insight.Properties["SpreadSignal"] = spread_signal
        insight.Properties["IV"] = contract.ImpliedVolatility
        insight.Properties["RV"] = rv
        insight.Properties["Strike"] = contract.Strike
        insight.Properties["Expiry"] = contract.Expiry
        insight.Properties["Right"] = contract.Right
        
        return insight
    
    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        """
        Handle security changes in the universe.
        
        Args:
            algorithm: Algorithm instance
            changes: Security changes
        """
        # Clean up removed securities
        for security in changes.RemovedSecurities:
            if security.Symbol in self.active_insights:
                del self.active_insights[security.Symbol]
            if security.Symbol in self.greeks_cache:
                del self.greeks_cache[security.Symbol]