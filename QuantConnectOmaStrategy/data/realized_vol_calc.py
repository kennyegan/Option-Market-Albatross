"""
Realized Volatility Calculator - Institutional Grade
Calculates historical realized volatility using multiple estimators with ensemble methods.

Author: OMA Strategy Team
Version: 2.0 - Enhanced with EWMA, ensemble methods, uncertainty quantification
"""

from AlgorithmImports import *
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
from collections import deque


@dataclass
class RealizedVolSummary:
    """
    Comprehensive realized volatility summary across multiple estimators.
    
    Attributes:
        symbol: The underlying symbol
        timestamp: When this summary was computed
        close_to_close: Close-to-close volatility
        parkinson: Parkinson high-low volatility
        garman_klass: Garman-Klass OHLC volatility
        ewma: Exponentially weighted moving average volatility
        ensemble: Ensemble estimate (median of all estimators)
        uncertainty: Uncertainty proxy (IQR of estimators)
        min_vol: Minimum across estimators
        max_vol: Maximum across estimators
        lookback_days: Lookback period used
        data_points: Number of data points used in calculation
    """
    symbol: Symbol
    timestamp: datetime
    close_to_close: Optional[float] = None
    parkinson: Optional[float] = None
    garman_klass: Optional[float] = None
    ewma: Optional[float] = None
    ensemble: Optional[float] = None
    uncertainty: Optional[float] = None
    min_vol: Optional[float] = None
    max_vol: Optional[float] = None
    lookback_days: int = 10
    data_points: int = 0
    
    def is_valid(self) -> bool:
        """Check if summary has valid ensemble estimate."""
        return self.ensemble is not None and self.ensemble > 0
    
    def has_high_uncertainty(self, threshold: float = 0.1) -> bool:
        """Check if uncertainty is above threshold (as fraction of ensemble)."""
        if self.ensemble is None or self.ensemble <= 0:
            return True
        if self.uncertainty is None:
            return True
        return (self.uncertainty / self.ensemble) > threshold


@dataclass
class MultiWindowVolSummary:
    """
    Volatility summary across multiple lookback windows.
    
    Attributes:
        symbol: The underlying symbol
        timestamp: When computed
        windows: Dictionary mapping lookback days to RealizedVolSummary
        term_structure_slope: Slope of vol term structure (positive = contango)
    """
    symbol: Symbol
    timestamp: datetime
    windows: Dict[int, RealizedVolSummary] = field(default_factory=dict)
    term_structure_slope: Optional[float] = None


class RealizedVolatilityCalculator:
    """
    Institutional-grade realized volatility calculator with:
    
    1. Multiple estimators: Close-to-close, Parkinson, Garman-Klass, EWMA
    2. Ensemble methods: Median of estimators for robust estimate
    3. Uncertainty quantification: IQR across estimators
    4. Multi-window support: 5d, 10d, 20d, 60d lookbacks
    5. Efficient caching with staleness tracking
    
    Usage:
        rv_calc = RealizedVolatilityCalculator(algorithm, default_lookback=10)
        rv_calc.update(symbol, bar)
        
        # Simple API
        rv = rv_calc.get_realized_vol(symbol, lookback_days=10, method="ensemble")
        
        # Full summary
        summary = rv_calc.get_realized_vol_summary(symbol, lookback_days=10)
    """
    
    # Default lookback windows for multi-window analysis
    DEFAULT_WINDOWS = [5, 10, 20, 60]
    
    def __init__(self, 
                 algorithm: QCAlgorithm,
                 lookback_days: int = 10,
                 annualization_factor: int = 252,
                 use_log_returns: bool = True,
                 ewma_decay: float = 0.94,
                 max_history_bars: int = 120):
        """
        Initialize the realized volatility calculator.
        
        Args:
            algorithm: QuantConnect algorithm instance
            lookback_days: Default number of days for volatility calculation
            annualization_factor: Trading days per year (252 for equities)
            use_log_returns: Whether to use log returns (recommended for accuracy)
            ewma_decay: EWMA decay factor (lambda), typical values 0.94-0.97
            max_history_bars: Maximum bars to store (should be > max lookback)
        """
        self.algorithm = algorithm
        self.lookback_days = lookback_days
        self.annualization_factor = annualization_factor
        self.use_log_returns = use_log_returns
        self.ewma_decay = ewma_decay
        self.max_history_bars = max_history_bars
        
        # Store price history for each symbol
        self.price_history: Dict[Symbol, deque] = {}
        self.bar_history: Dict[Symbol, deque] = {}
        
        # EWMA variance state (per symbol)
        self.ewma_variance: Dict[Symbol, float] = {}
        self.ewma_last_return: Dict[Symbol, float] = {}
        
        # Cache for calculated volatilities
        self.volatility_cache: Dict[str, Dict] = {}
        self.summary_cache: Dict[str, RealizedVolSummary] = {}
        self.last_update_time: Dict[Symbol, datetime] = {}
        
        # Cache validity duration (in minutes)
        self.cache_validity_minutes = 1
    
    def update(self, symbol: Symbol, bar) -> None:
        """
        Update price history with new bar data.
        
        Args:
            symbol: Symbol to update
            bar: TradeBar or QuoteBar data
        """
        # Initialize storage if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.max_history_bars)
            self.bar_history[symbol] = deque(maxlen=self.max_history_bars)
        
        # Store close price
        self.price_history[symbol].append(bar.Close)
        
        # Store full bar for advanced calculations
        self.bar_history[symbol].append({
            'time': bar.Time if hasattr(bar, 'Time') else self.algorithm.Time,
            'open': bar.Open,
            'high': bar.High,
            'low': bar.Low,
            'close': bar.Close,
            'volume': bar.Volume if hasattr(bar, 'Volume') else 0
        })
        
        # Update EWMA state
        self._update_ewma_state(symbol, bar.Close)
        
        # Mark cache as stale
        self.last_update_time[symbol] = self.algorithm.Time
        self._invalidate_cache(symbol)
    
    def _update_ewma_state(self, symbol: Symbol, close_price: float) -> None:
        """
        Update EWMA variance state with new price.
        
        Args:
            symbol: Symbol to update
            close_price: Latest close price
        """
        if symbol not in self.ewma_last_return:
            # First observation - initialize
            self.ewma_last_return[symbol] = 0
            self.ewma_variance[symbol] = 0
            return
        
        # Get previous close
        prices = list(self.price_history[symbol])
        if len(prices) < 2:
            return
        
        prev_close = prices[-2]
        if prev_close <= 0 or close_price <= 0:
            return
        
        # Calculate return
        if self.use_log_returns:
            ret = np.log(close_price / prev_close)
        else:
            ret = (close_price - prev_close) / prev_close
        
        # Update EWMA variance: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
        prev_variance = self.ewma_variance[symbol]
        self.ewma_variance[symbol] = (
            self.ewma_decay * prev_variance + 
            (1 - self.ewma_decay) * ret ** 2
        )
        self.ewma_last_return[symbol] = ret
    
    def _invalidate_cache(self, symbol: Symbol) -> None:
        """Invalidate cache entries for a symbol."""
        keys_to_remove = [k for k in self.volatility_cache.keys() if str(symbol) in k]
        for key in keys_to_remove:
            del self.volatility_cache[key]
        
        summary_keys = [k for k in self.summary_cache.keys() if str(symbol) in k]
        for key in summary_keys:
            del self.summary_cache[key]
    
    def get_realized_vol(self, 
                        symbol: Symbol, 
                        lookback_days: int = None,
                        method: str = "ensemble") -> Optional[float]:
        """
        Get realized volatility for a symbol using specified method.
        
        Args:
            symbol: Symbol to calculate volatility for
            lookback_days: Lookback period (uses default if None)
            method: Calculation method:
                - 'close_to_close': Standard close-to-close
                - 'parkinson': High-low estimator
                - 'garman_klass': OHLC estimator
                - 'ewma': Exponentially weighted
                - 'ensemble': Median of all estimators (recommended)
            
        Returns:
            Annualized realized volatility (0-1 scale), or None if insufficient data
        """
        lookback = lookback_days or self.lookback_days
        
        # Check cache first
        cache_key = f"{symbol}_{method}_{lookback}"
        if cache_key in self.volatility_cache:
            cached = self.volatility_cache[cache_key]
            if self._is_cache_valid(cached, symbol):
                return cached['value']
        
        # Calculate fresh volatility
        if method == "ensemble":
            vol = self._calculate_ensemble_vol(symbol, lookback)
        elif method == "ewma":
            vol = self._ewma_vol(symbol)
        else:
            vol = self._calculate_volatility(symbol, lookback, method)
        
        # Cache result
        if vol is not None:
            self.volatility_cache[cache_key] = {
                'value': vol,
                'time': self.algorithm.Time
            }
        
        return vol
    
    def get_realized_vol_summary(self, 
                                 symbol: Symbol,
                                 lookback_days: int = None) -> RealizedVolSummary:
        """
        Get comprehensive volatility summary across all estimators.
        
        Args:
            symbol: Symbol to analyze
            lookback_days: Lookback period (uses default if None)
            
        Returns:
            RealizedVolSummary dataclass with all estimators and uncertainty
        """
        lookback = lookback_days or self.lookback_days
        
        # Check cache
        cache_key = f"{symbol}_summary_{lookback}"
        if cache_key in self.summary_cache:
            cached = self.summary_cache[cache_key]
            if cached.timestamp and (self.algorithm.Time - cached.timestamp).total_seconds() < 60:
                return cached
        
        # Calculate all estimators
        c2c = self._close_to_close_vol(symbol, lookback)
        park = self._parkinson_vol(symbol, lookback)
        gk = self._garman_klass_vol(symbol, lookback)
        ewma = self._ewma_vol(symbol)
        
        # Collect valid estimates
        estimates = [v for v in [c2c, park, gk, ewma] if v is not None and v > 0]
        
        # Calculate ensemble and uncertainty
        ensemble = None
        uncertainty = None
        min_vol = None
        max_vol = None
        
        if len(estimates) >= 2:
            ensemble = float(np.median(estimates))
            # IQR as uncertainty
            q75, q25 = np.percentile(estimates, [75, 25])
            uncertainty = q75 - q25
            min_vol = min(estimates)
            max_vol = max(estimates)
        elif len(estimates) == 1:
            ensemble = estimates[0]
            uncertainty = 0
            min_vol = max_vol = estimates[0]
        
        # Count data points
        data_points = len(self.price_history.get(symbol, []))
        
        summary = RealizedVolSummary(
            symbol=symbol,
            timestamp=self.algorithm.Time,
            close_to_close=c2c,
            parkinson=park,
            garman_klass=gk,
            ewma=ewma,
            ensemble=ensemble,
            uncertainty=uncertainty,
            min_vol=min_vol,
            max_vol=max_vol,
            lookback_days=lookback,
            data_points=data_points
        )
        
        # Cache
        self.summary_cache[cache_key] = summary
        
        return summary
    
    def get_multi_window_summary(self, 
                                  symbol: Symbol,
                                  windows: List[int] = None) -> MultiWindowVolSummary:
        """
        Get volatility summary across multiple lookback windows.
        
        Args:
            symbol: Symbol to analyze
            windows: List of lookback periods (uses DEFAULT_WINDOWS if None)
            
        Returns:
            MultiWindowVolSummary with vol estimates per window
        """
        windows = windows or self.DEFAULT_WINDOWS
        
        result = MultiWindowVolSummary(
            symbol=symbol,
            timestamp=self.algorithm.Time,
            windows={}
        )
        
        # Calculate for each window
        for window in windows:
            summary = self.get_realized_vol_summary(symbol, lookback_days=window)
            if summary.is_valid():
                result.windows[window] = summary
        
        # Calculate term structure slope if we have multiple windows
        if len(result.windows) >= 2:
            sorted_windows = sorted(result.windows.keys())
            vols = [result.windows[w].ensemble for w in sorted_windows]
            if all(v is not None for v in vols):
                # Simple slope: (long - short) / short
                short_vol = vols[0]
                long_vol = vols[-1]
                if short_vol > 0:
                    result.term_structure_slope = (long_vol - short_vol) / short_vol
        
        return result
    
    def _is_cache_valid(self, cached: Dict, symbol: Symbol) -> bool:
        """Check if cached value is still valid."""
        if 'time' not in cached:
            return False
        
        cache_time = cached['time']
        
        # Check if data was updated after cache
        if symbol in self.last_update_time:
            if cache_time < self.last_update_time[symbol]:
                return False
        
        # Check time-based validity
        elapsed = (self.algorithm.Time - cache_time).total_seconds() / 60
        return elapsed < self.cache_validity_minutes
    
    def _calculate_ensemble_vol(self, symbol: Symbol, lookback: int) -> Optional[float]:
        """Calculate ensemble volatility as median of all estimators."""
        estimates = []
        
        c2c = self._close_to_close_vol(symbol, lookback)
        if c2c is not None and c2c > 0:
            estimates.append(c2c)
        
        park = self._parkinson_vol(symbol, lookback)
        if park is not None and park > 0:
            estimates.append(park)
        
        gk = self._garman_klass_vol(symbol, lookback)
        if gk is not None and gk > 0:
            estimates.append(gk)
        
        ewma = self._ewma_vol(symbol)
        if ewma is not None and ewma > 0:
            estimates.append(ewma)
        
        if not estimates:
            return None
        
        return float(np.median(estimates))
    
    def _calculate_volatility(self, symbol: Symbol, lookback: int, method: str) -> Optional[float]:
        """
        Calculate volatility using specified method.
        
        Args:
            symbol: Symbol to calculate for
            lookback: Lookback period in days
            method: Calculation method
            
        Returns:
            Annualized volatility or None
        """
        if method == 'close_to_close':
            return self._close_to_close_vol(symbol, lookback)
        elif method == 'parkinson':
            return self._parkinson_vol(symbol, lookback)
        elif method == 'garman_klass':
            return self._garman_klass_vol(symbol, lookback)
        elif method == 'ewma':
            return self._ewma_vol(symbol)
        else:
            return self._close_to_close_vol(symbol, lookback)
    
    def _close_to_close_vol(self, symbol: Symbol, lookback: int) -> Optional[float]:
        """
        Calculate close-to-close volatility using standard deviation of returns.
        
        Args:
            symbol: Symbol to calculate for
            lookback: Lookback period in days
            
        Returns:
            Annualized volatility or None if insufficient data
        """
        if symbol not in self.price_history:
            return None
        
        prices = list(self.price_history[symbol])
        
        # Need lookback + 1 prices to get lookback returns
        if len(prices) < lookback + 1:
            return None
        
        # Use most recent prices
        recent_prices = prices[-(lookback + 1):]
        
        # Calculate returns
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0 and recent_prices[i] > 0:
                if self.use_log_returns:
                    ret = np.log(recent_prices[i] / recent_prices[i-1])
                else:
                    ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return None
        
        # Calculate standard deviation with Bessel's correction
        std_dev = np.std(returns, ddof=1)
        
        # Annualize
        annualized_vol = std_dev * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def _parkinson_vol(self, symbol: Symbol, lookback: int) -> Optional[float]:
        """
        Calculate Parkinson's high-low volatility estimator.
        
        More efficient than close-to-close as it captures intraday volatility.
        Assumes no drift and continuous trading (underestimates with gaps).
        
        Args:
            symbol: Symbol to calculate for
            lookback: Lookback period in days
            
        Returns:
            Annualized volatility or None if insufficient data
        """
        if symbol not in self.bar_history:
            return None
        
        bars = list(self.bar_history[symbol])
        
        if len(bars) < lookback:
            return None
        
        # Use most recent bars
        recent_bars = bars[-lookback:]
        
        # Calculate high-low ratios
        hl_squared = []
        for bar in recent_bars:
            if bar['high'] > 0 and bar['low'] > 0 and bar['high'] >= bar['low']:
                hl_ratio = np.log(bar['high'] / bar['low'])
                hl_squared.append(hl_ratio ** 2)
        
        if len(hl_squared) < 2:
            return None
        
        # Parkinson's constant: 1 / (4 * ln(2)) ≈ 0.361
        constant = 1.0 / (4.0 * np.log(2))
        
        # Calculate daily variance and then volatility
        daily_variance = constant * np.mean(hl_squared)
        daily_vol = np.sqrt(daily_variance)
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def _garman_klass_vol(self, symbol: Symbol, lookback: int) -> Optional[float]:
        """
        Calculate Garman-Klass volatility estimator.
        
        Uses OHLC data for more accurate volatility estimation.
        More efficient than close-to-close and handles opening gaps better than Parkinson.
        
        Args:
            symbol: Symbol to calculate for
            lookback: Lookback period in days
            
        Returns:
            Annualized volatility or None if insufficient data
        """
        if symbol not in self.bar_history:
            return None
        
        bars = list(self.bar_history[symbol])
        
        if len(bars) < lookback:
            return None
        
        # Use most recent bars
        recent_bars = bars[-lookback:]
        
        # Calculate Garman-Klass components
        gk_values = []
        for bar in recent_bars:
            # Validate OHLC
            if not all(bar[k] > 0 for k in ['open', 'high', 'low', 'close']):
                continue
            if bar['high'] < bar['low']:
                continue
            
            # High-Low component: 0.5 * (ln(H/L))²
            hl_component = 0.5 * (np.log(bar['high'] / bar['low'])) ** 2
            
            # Close-Open component: -(2*ln(2) - 1) * (ln(C/O))²
            co_component = (2 * np.log(2) - 1) * (np.log(bar['close'] / bar['open'])) ** 2
            
            # Garman-Klass daily variance estimate
            gk_value = hl_component - co_component
            
            # Ensure non-negative (can happen with unusual price patterns)
            if gk_value >= 0:
                gk_values.append(gk_value)
        
        if len(gk_values) < 2:
            return None
        
        # Calculate volatility
        daily_variance = np.mean(gk_values)
        daily_vol = np.sqrt(daily_variance)
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def _ewma_vol(self, symbol: Symbol) -> Optional[float]:
        """
        Get EWMA (Exponentially Weighted Moving Average) volatility.
        
        Uses RiskMetrics-style exponential weighting with configurable decay.
        More responsive to recent volatility changes than simple lookback.
        
        Args:
            symbol: Symbol to get EWMA volatility for
            
        Returns:
            Annualized EWMA volatility or None if not initialized
        """
        if symbol not in self.ewma_variance:
            return None
        
        variance = self.ewma_variance[symbol]
        
        if variance <= 0:
            return None
        
        # Daily volatility from variance
        daily_vol = np.sqrt(variance)
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(self.annualization_factor)
        
        return annualized_vol
    
    def get_volatility_metrics(self, symbol: Symbol) -> Dict:
        """
        Get comprehensive volatility metrics (legacy compatibility).
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary of volatility metrics
        """
        summary = self.get_realized_vol_summary(symbol)
        
        return {
            'close_to_close': summary.close_to_close,
            'parkinson': summary.parkinson,
            'garman_klass': summary.garman_klass,
            'ewma': summary.ewma,
            'ensemble': summary.ensemble,
            'average': summary.ensemble,  # Legacy alias
            'uncertainty': summary.uncertainty,
            'min': summary.min_vol,
            'max': summary.max_vol
        }
    
    def get_volatility_term_structure(self, 
                                      symbol: Symbol, 
                                      periods: List[int] = None) -> Dict[int, float]:
        """
        Calculate volatility term structure across different periods.
        
        Args:
            symbol: Symbol to analyze
            periods: List of lookback periods in days
            
        Returns:
            Dictionary mapping period to ensemble volatility
        """
        periods = periods or self.DEFAULT_WINDOWS
        multi_summary = self.get_multi_window_summary(symbol, periods)
        
        return {
            window: summary.ensemble 
            for window, summary in multi_summary.windows.items()
            if summary.ensemble is not None
        }
    
    def get_volatility_percentile(self, 
                                   symbol: Symbol, 
                                   current_vol: float = None,
                                   lookback_bars: int = 60) -> Optional[float]:
        """
        Get current volatility percentile relative to historical range.
        
        Args:
            symbol: Symbol to analyze
            current_vol: Current volatility (will calculate if None)
            lookback_bars: Number of historical bars for percentile calc
            
        Returns:
            Percentile (0-100) or None if insufficient history
        """
        if current_vol is None:
            current_vol = self.get_realized_vol(symbol, method="ensemble")
        
        if current_vol is None:
            return None
        
        # Get historical bars
        if symbol not in self.bar_history:
            return None
        
        bars = list(self.bar_history[symbol])
        if len(bars) < lookback_bars:
            return None
        
        # Calculate historical volatilities (using simple method for efficiency)
        historical_vols = []
        window = 10
        
        for i in range(window, len(bars)):
            window_bars = bars[i-window:i]
            returns = []
            for j in range(1, len(window_bars)):
                if window_bars[j-1]['close'] > 0 and window_bars[j]['close'] > 0:
                    ret = np.log(window_bars[j]['close'] / window_bars[j-1]['close'])
                    returns.append(ret)
            
            if len(returns) >= 2:
                vol = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)
                historical_vols.append(vol)
        
        if len(historical_vols) < 10:
            return None
        
        # Calculate percentile
        percentile = (np.sum(np.array(historical_vols) < current_vol) / len(historical_vols)) * 100
        
        return percentile
    
    def clear_cache(self, symbol: Symbol = None) -> None:
        """
        Clear volatility cache.
        
        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            self._invalidate_cache(symbol)
        else:
            self.volatility_cache.clear()
            self.summary_cache.clear()
    
    def get_data_status(self, symbol: Symbol) -> Dict:
        """
        Get data availability status for a symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Dictionary with data status information
        """
        return {
            'has_price_history': symbol in self.price_history,
            'price_count': len(self.price_history.get(symbol, [])),
            'bar_count': len(self.bar_history.get(symbol, [])),
            'has_ewma': symbol in self.ewma_variance,
            'last_update': self.last_update_time.get(symbol),
            'max_lookback_available': len(self.price_history.get(symbol, [])) - 1
        }
