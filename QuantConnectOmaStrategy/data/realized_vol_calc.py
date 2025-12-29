"""
Realized Volatility Calculator
Calculates historical realized volatility for underlying assets.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque


class RealizedVolatilityCalculator:
    """
    Calculates realized (historical) volatility using various methods:
    1. Close-to-close volatility
    2. Parkinson's high-low volatility
    3. Garman-Klass volatility (using OHLC)
    """
    
    def __init__(self, 
                 algorithm: QCAlgorithm,
                 lookback_days: int = 10,
                 annualization_factor: int = 252,
                 use_log_returns: bool = True):
        """
        Initialize the realized volatility calculator.
        
        Args:
            algorithm: Algorithm instance
            lookback_days: Number of days for volatility calculation
            annualization_factor: Trading days per year (252 for equities)
            use_log_returns: Whether to use log returns (recommended)
        """
        self.algorithm = algorithm
        self.lookback_days = lookback_days
        self.annualization_factor = annualization_factor
        self.use_log_returns = use_log_returns
        
        # Store price history for each symbol
        self.price_history = {}
        self.bar_history = {}
        
        # Cache for calculated volatilities
        self.volatility_cache = {}
        self.last_update_time = {}
    
    def update(self, symbol: Symbol, bar) -> None:
        """
        Update price history with new bar data.
        
        Args:
            symbol: Symbol to update
            bar: TradeBar or QuoteBar data
        """
        # Initialize storage if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_days + 1)
            self.bar_history[symbol] = deque(maxlen=self.lookback_days + 1)
        
        # Store close price
        self.price_history[symbol].append(bar.Close)
        
        # Store full bar for advanced calculations
        self.bar_history[symbol].append({
            'time': bar.Time,
            'open': bar.Open,
            'high': bar.High,
            'low': bar.Low,
            'close': bar.Close,
            'volume': bar.Volume
        })
        
        # Mark cache as stale
        self.last_update_time[symbol] = self.algorithm.Time
    
    def get_realized_vol(self, symbol: Symbol, method: str = 'close_to_close') -> float:
        """
        Get realized volatility for a symbol.
        
        Args:
            symbol: Symbol to calculate volatility for
            method: Calculation method ('close_to_close', 'parkinson', 'garman_klass')
            
        Returns:
            Annualized realized volatility (0-1 scale)
        """
        # Check cache first
        cache_key = f"{symbol}_{method}"
        if cache_key in self.volatility_cache:
            if symbol in self.last_update_time:
                # Return cached value if calculated after last update
                cache_time = self.volatility_cache[cache_key].get('time')
                if cache_time and cache_time >= self.last_update_time[symbol]:
                    return self.volatility_cache[cache_key]['value']
        
        # Calculate fresh volatility
        vol = self._calculate_volatility(symbol, method)
        
        # Cache result
        if vol is not None:
            self.volatility_cache[cache_key] = {
                'value': vol,
                'time': self.algorithm.Time
            }
        
        return vol
    
    def _calculate_volatility(self, symbol: Symbol, method: str) -> float:
        """
        Calculate volatility using specified method.
        
        Args:
            symbol: Symbol to calculate for
            method: Calculation method
            
        Returns:
            Annualized volatility or None
        """
        if method == 'close_to_close':
            return self._close_to_close_vol(symbol)
        elif method == 'parkinson':
            return self._parkinson_vol(symbol)
        elif method == 'garman_klass':
            return self._garman_klass_vol(symbol)
        else:
            return self._close_to_close_vol(symbol)  # Default
    
    def _close_to_close_vol(self, symbol: Symbol) -> float:
        """
        Calculate close-to-close volatility.
        
        Args:
            symbol: Symbol to calculate for
            
        Returns:
            Annualized volatility or None
        """
        if symbol not in self.price_history:
            return None
        
        prices = list(self.price_history[symbol])
        
        if len(prices) < self.lookback_days:
            return None
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                if self.use_log_returns:
                    ret = np.log(prices[i] / prices[i-1])
                else:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return None
        
        # Calculate standard deviation
        std_dev = np.std(returns, ddof=1)
        
        # Annualize
        annualized_vol = std_dev * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def _parkinson_vol(self, symbol: Symbol) -> float:
        """
        Calculate Parkinson's high-low volatility estimator.
        More efficient than close-to-close as it uses high/low information.
        
        Args:
            symbol: Symbol to calculate for
            
        Returns:
            Annualized volatility or None
        """
        if symbol not in self.bar_history:
            return None
        
        bars = list(self.bar_history[symbol])
        
        if len(bars) < self.lookback_days:
            return None
        
        # Calculate high-low ratios
        hl_ratios = []
        for bar in bars:
            if bar['high'] > 0 and bar['low'] > 0:
                hl_ratio = np.log(bar['high'] / bar['low'])
                hl_ratios.append(hl_ratio ** 2)
        
        if len(hl_ratios) < 2:
            return None
        
        # Parkinson's constant
        constant = 1 / (4 * np.log(2))
        
        # Calculate volatility
        daily_vol = np.sqrt(constant * np.mean(hl_ratios))
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def _garman_klass_vol(self, symbol: Symbol) -> float:
        """
        Calculate Garman-Klass volatility estimator.
        Uses OHLC data for more accurate volatility estimation.
        
        Args:
            symbol: Symbol to calculate for
            
        Returns:
            Annualized volatility or None
        """
        if symbol not in self.bar_history:
            return None
        
        bars = list(self.bar_history[symbol])
        
        if len(bars) < self.lookback_days:
            return None
        
        # Calculate components
        gk_values = []
        for bar in bars:
            if all(bar[k] > 0 for k in ['open', 'high', 'low', 'close']):
                # High-Low component
                hl_component = (np.log(bar['high'] / bar['low'])) ** 2
                
                # Close-Open component
                co_component = (np.log(bar['close'] / bar['open'])) ** 2
                
                # Garman-Klass formula
                gk_value = 0.5 * hl_component - (2 * np.log(2) - 1) * co_component
                gk_values.append(gk_value)
        
        if len(gk_values) < 2:
            return None
        
        # Calculate volatility
        daily_vol = np.sqrt(np.mean(gk_values))
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def get_volatility_metrics(self, symbol: Symbol) -> Dict:
        """
        Get comprehensive volatility metrics.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary of volatility metrics
        """
        metrics = {}
        
        # Calculate different volatility measures
        metrics['close_to_close'] = self.get_realized_vol(symbol, 'close_to_close')
        metrics['parkinson'] = self.get_realized_vol(symbol, 'parkinson')
        metrics['garman_klass'] = self.get_realized_vol(symbol, 'garman_klass')
        
        # Calculate average
        valid_vols = [v for v in metrics.values() if v is not None]
        if valid_vols:
            metrics['average'] = np.mean(valid_vols)
            metrics['min'] = min(valid_vols)
            metrics['max'] = max(valid_vols)
        
        return metrics
    
    def get_volatility_term_structure(self, 
                                    symbol: Symbol, 
                                    periods: List[int] = [5, 10, 20, 30]) -> Dict:
        """
        Calculate volatility term structure across different periods.
        
        Args:
            symbol: Symbol to analyze
            periods: List of lookback periods in days
            
        Returns:
            Dictionary mapping period to volatility
        """
        term_structure = {}
        
        # Store original lookback
        original_lookback = self.lookback_days
        
        for period in periods:
            # Temporarily change lookback period
            self.lookback_days = period
            
            # Calculate volatility
            vol = self.get_realized_vol(symbol)
            if vol is not None:
                term_structure[period] = vol
        
        # Restore original lookback
        self.lookback_days = original_lookback
        
        return term_structure
    
    def get_volatility_percentile(self, symbol: Symbol, current_vol: float = None) -> float:
        """
        Get current volatility percentile relative to historical range.
        
        Args:
            symbol: Symbol to analyze
            current_vol: Current volatility (will calculate if None)
            
        Returns:
            Percentile (0-100) or None
        """
        if current_vol is None:
            current_vol = self.get_realized_vol(symbol)
            
        if current_vol is None:
            return None
        
        # Get historical volatilities (would need more history in practice)
        # For now, return a simple estimate
        # In production, you'd maintain a longer history
        
        # Placeholder - assumes current vol is at 50th percentile
        return 50.0
    
    def clear_cache(self, symbol: Symbol = None) -> None:
        """
        Clear volatility cache.
        
        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            keys_to_remove = [k for k in self.volatility_cache.keys() if str(symbol) in k]
            for key in keys_to_remove:
                del self.volatility_cache[key]
        else:
            self.volatility_cache.clear()