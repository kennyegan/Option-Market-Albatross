"""
Volatility Regime Classifier
Classifies market volatility regime for regime-aware trading decisions.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np


class VolatilityRegime(Enum):
    """
    Market volatility regime classifications.
    
    CALM: Low volatility, stable markets
    NORMAL: Average volatility conditions
    STRESSED: Elevated volatility, increased uncertainty
    CRISIS: Extreme volatility, potential market dislocation
    """
    CALM = "CALM"
    NORMAL = "NORMAL"
    STRESSED = "STRESSED"
    CRISIS = "CRISIS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeConfig:
    """
    Configuration for regime classification thresholds.
    
    All VIX thresholds are in absolute points (e.g., 15 = VIX at 15).
    Change thresholds are in absolute points change.
    """
    # VIX level thresholds
    vix_calm_upper: float = 15.0        # Below this = CALM
    vix_normal_upper: float = 20.0      # Below this = NORMAL
    vix_stressed_upper: float = 30.0    # Below this = STRESSED, above = CRISIS
    
    # VIX change thresholds (1-day change)
    vix_spike_threshold: float = 3.0    # Points increase triggers stress
    vix_crash_threshold: float = 5.0    # Points increase triggers crisis
    
    # Realized vol thresholds (annualized)
    rv_calm_upper: float = 0.12         # 12% annualized
    rv_normal_upper: float = 0.18       # 18% annualized
    rv_stressed_upper: float = 0.30     # 30% annualized
    
    # SPX return thresholds (1-day, absolute)
    spx_stress_threshold: float = 0.02  # 2% move
    spx_crisis_threshold: float = 0.03  # 3% move
    
    # Regime persistence (minimum bars to confirm regime change)
    min_bars_for_change: int = 3
    
    # Weights for combining signals (must sum to 1.0)
    vix_weight: float = 0.50
    rv_weight: float = 0.25
    returns_weight: float = 0.25


@dataclass
class RegimeSnapshot:
    """
    Snapshot of current regime state and contributing factors.
    """
    regime: VolatilityRegime
    timestamp: datetime
    confidence: float  # 0-1 confidence in classification
    
    # Contributing metrics
    vix_level: Optional[float] = None
    vix_change_1d: Optional[float] = None
    vix_percentile: Optional[float] = None
    
    realized_vol: Optional[float] = None
    rv_percentile: Optional[float] = None
    
    spx_return_1d: Optional[float] = None
    spx_return_5d: Optional[float] = None
    
    # Regime scores (0-3, maps to CALM=0, NORMAL=1, STRESSED=2, CRISIS=3)
    vix_score: float = 1.0
    rv_score: float = 1.0
    returns_score: float = 1.0
    composite_score: float = 1.0
    
    # Regime history
    bars_in_current_regime: int = 0
    previous_regime: Optional[VolatilityRegime] = None


class VolatilityRegimeClassifier:
    """
    Classifies market volatility regime based on multiple indicators:
    
    1. VIX level and changes (or ATM IV proxy)
    2. Realized volatility of SPX/underlying
    3. Recent returns and moves
    
    Outputs regime classification for use in:
    - Alpha model signal filtering
    - Risk management position sizing
    - Execution urgency decisions
    
    Usage:
        classifier = VolatilityRegimeClassifier(algorithm, rv_calculator)
        classifier.update(vix_value, spx_price, realized_vol)
        
        snapshot = classifier.get_current_regime()
        if snapshot.regime == VolatilityRegime.CRISIS:
            # Reduce risk, disable new shorts, etc.
    """
    
    def __init__(self,
                 algorithm: QCAlgorithm,
                 rv_calculator = None,
                 config: RegimeConfig = None,
                 lookback_bars: int = 60):
        """
        Initialize the volatility regime classifier.
        
        Args:
            algorithm: QuantConnect algorithm instance
            rv_calculator: RealizedVolatilityCalculator instance (optional)
            config: RegimeConfig with classification thresholds
            lookback_bars: Number of bars to store for history
        """
        self.algorithm = algorithm
        self.rv_calculator = rv_calculator
        self.config = config or RegimeConfig()
        self.lookback_bars = lookback_bars
        
        # Current regime state
        self.current_regime = VolatilityRegime.UNKNOWN
        self.current_snapshot: Optional[RegimeSnapshot] = None
        self.bars_in_regime = 0
        self.previous_regime = VolatilityRegime.UNKNOWN
        
        # Historical data
        self.vix_history: List[Tuple[datetime, float]] = []
        self.spx_history: List[Tuple[datetime, float]] = []
        self.regime_history: List[RegimeSnapshot] = []
        
        # VIX symbol (if subscribed)
        self.vix_symbol: Optional[Symbol] = None
        
        # Regime change callbacks
        self.regime_change_callbacks: List[callable] = []
    
    def set_vix_symbol(self, vix_symbol: Symbol) -> None:
        """
        Set VIX symbol for automatic updates.
        
        Args:
            vix_symbol: Symbol for VIX data feed
        """
        self.vix_symbol = vix_symbol
    
    def update(self,
               vix_value: Optional[float] = None,
               spx_price: Optional[float] = None,
               realized_vol: Optional[float] = None,
               underlying_symbol: Symbol = None) -> RegimeSnapshot:
        """
        Update regime classification with new market data.
        
        Args:
            vix_value: Current VIX level (or ATM IV as proxy)
            spx_price: Current SPX price
            realized_vol: Current realized volatility (will use rv_calculator if None)
            underlying_symbol: Symbol for RV calculation if using rv_calculator
            
        Returns:
            Current RegimeSnapshot
        """
        timestamp = self.algorithm.Time
        
        # Store history
        if vix_value is not None:
            self._add_to_history(self.vix_history, timestamp, vix_value)
        
        if spx_price is not None:
            self._add_to_history(self.spx_history, timestamp, spx_price)
        
        # Get realized vol from calculator if not provided
        if realized_vol is None and self.rv_calculator is not None and underlying_symbol is not None:
            realized_vol = self.rv_calculator.get_realized_vol(underlying_symbol, method="ensemble")
        
        # Calculate regime scores
        vix_score = self._calculate_vix_score(vix_value)
        rv_score = self._calculate_rv_score(realized_vol)
        returns_score = self._calculate_returns_score()
        
        # Weighted composite score
        composite = (
            vix_score * self.config.vix_weight +
            rv_score * self.config.rv_weight +
            returns_score * self.config.returns_weight
        )
        
        # Map composite to regime
        new_regime = self._score_to_regime(composite)
        
        # Calculate confidence
        confidence = self._calculate_confidence(vix_score, rv_score, returns_score)
        
        # Check for regime change
        if new_regime != self.current_regime:
            self.bars_in_regime += 1
            
            # Confirm regime change after min_bars
            if self.bars_in_regime >= self.config.min_bars_for_change:
                self.previous_regime = self.current_regime
                self.current_regime = new_regime
                self.bars_in_regime = 0
                
                # Fire callbacks
                for callback in self.regime_change_callbacks:
                    try:
                        callback(self.previous_regime, self.current_regime)
                    except Exception:
                        pass
        else:
            self.bars_in_regime = 0
        
        # Calculate derived metrics
        vix_change_1d = self._get_vix_change(1)
        spx_return_1d = self._get_spx_return(1)
        spx_return_5d = self._get_spx_return(5)
        vix_percentile = self._get_vix_percentile(vix_value)
        
        # Create snapshot
        self.current_snapshot = RegimeSnapshot(
            regime=self.current_regime,
            timestamp=timestamp,
            confidence=confidence,
            vix_level=vix_value,
            vix_change_1d=vix_change_1d,
            vix_percentile=vix_percentile,
            realized_vol=realized_vol,
            rv_percentile=None,  # Would need longer history
            spx_return_1d=spx_return_1d,
            spx_return_5d=spx_return_5d,
            vix_score=vix_score,
            rv_score=rv_score,
            returns_score=returns_score,
            composite_score=composite,
            bars_in_current_regime=self.bars_in_regime,
            previous_regime=self.previous_regime
        )
        
        # Store in history
        self.regime_history.append(self.current_snapshot)
        if len(self.regime_history) > self.lookback_bars:
            self.regime_history.pop(0)
        
        return self.current_snapshot
    
    def get_current_regime(self) -> RegimeSnapshot:
        """
        Get current regime snapshot.
        
        Returns:
            Current RegimeSnapshot, or creates one with UNKNOWN if not updated
        """
        if self.current_snapshot is None:
            return RegimeSnapshot(
                regime=VolatilityRegime.UNKNOWN,
                timestamp=self.algorithm.Time,
                confidence=0.0
            )
        return self.current_snapshot
    
    def is_high_vol_regime(self) -> bool:
        """Check if current regime is STRESSED or CRISIS."""
        return self.current_regime in [VolatilityRegime.STRESSED, VolatilityRegime.CRISIS]
    
    def is_crisis(self) -> bool:
        """Check if current regime is CRISIS."""
        return self.current_regime == VolatilityRegime.CRISIS
    
    def get_regime_multiplier(self) -> float:
        """
        Get position size multiplier based on regime.
        
        Returns:
            Multiplier (0.0 to 1.0) for position sizing
            CALM: 1.0 (full size)
            NORMAL: 1.0 (full size)
            STRESSED: 0.5 (half size)
            CRISIS: 0.0 (no new positions)
        """
        multipliers = {
            VolatilityRegime.CALM: 1.0,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.STRESSED: 0.5,
            VolatilityRegime.CRISIS: 0.0,
            VolatilityRegime.UNKNOWN: 0.5
        }
        return multipliers.get(self.current_regime, 0.5)
    
    def get_greek_limit_multiplier(self) -> float:
        """
        Get Greek limit multiplier based on regime.
        
        Returns:
            Multiplier for Greek limits (lower in stress)
            CALM: 1.2 (can exceed normal limits slightly)
            NORMAL: 1.0 (standard limits)
            STRESSED: 0.7 (tighter limits)
            CRISIS: 0.3 (very tight limits)
        """
        multipliers = {
            VolatilityRegime.CALM: 1.2,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.STRESSED: 0.7,
            VolatilityRegime.CRISIS: 0.3,
            VolatilityRegime.UNKNOWN: 0.7
        }
        return multipliers.get(self.current_regime, 0.7)
    
    def should_disable_short_vol(self) -> bool:
        """
        Check if short volatility trades should be disabled.
        
        Returns:
            True if regime is STRESSED or CRISIS
        """
        return self.is_high_vol_regime()
    
    def on_regime_change(self, callback: callable) -> None:
        """
        Register callback for regime changes.
        
        Args:
            callback: Function(old_regime, new_regime) to call on changes
        """
        self.regime_change_callbacks.append(callback)
    
    def _add_to_history(self, history: List, timestamp: datetime, value: float) -> None:
        """Add value to history list, maintaining max size."""
        history.append((timestamp, value))
        if len(history) > self.lookback_bars:
            history.pop(0)
    
    def _calculate_vix_score(self, vix_value: Optional[float]) -> float:
        """
        Calculate regime score from VIX level and changes.
        
        Returns:
            Score from 0 (CALM) to 3 (CRISIS)
        """
        if vix_value is None:
            return 1.0  # Default to NORMAL
        
        # Level-based score
        if vix_value < self.config.vix_calm_upper:
            level_score = 0.0
        elif vix_value < self.config.vix_normal_upper:
            level_score = 1.0
        elif vix_value < self.config.vix_stressed_upper:
            level_score = 2.0
        else:
            level_score = 3.0
        
        # Change-based adjustment
        vix_change = self._get_vix_change(1)
        change_adjustment = 0.0
        
        if vix_change is not None:
            if vix_change >= self.config.vix_crash_threshold:
                change_adjustment = 1.0  # Bump up regime
            elif vix_change >= self.config.vix_spike_threshold:
                change_adjustment = 0.5
            elif vix_change <= -self.config.vix_spike_threshold:
                change_adjustment = -0.3  # VIX dropping fast
        
        return min(3.0, max(0.0, level_score + change_adjustment))
    
    def _calculate_rv_score(self, realized_vol: Optional[float]) -> float:
        """
        Calculate regime score from realized volatility.
        
        Returns:
            Score from 0 (CALM) to 3 (CRISIS)
        """
        if realized_vol is None:
            return 1.0  # Default to NORMAL
        
        if realized_vol < self.config.rv_calm_upper:
            return 0.0
        elif realized_vol < self.config.rv_normal_upper:
            return 1.0
        elif realized_vol < self.config.rv_stressed_upper:
            return 2.0
        else:
            return 3.0
    
    def _calculate_returns_score(self) -> float:
        """
        Calculate regime score from recent returns.
        
        Returns:
            Score from 0 (CALM) to 3 (CRISIS)
        """
        spx_return_1d = self._get_spx_return(1)
        
        if spx_return_1d is None:
            return 1.0  # Default to NORMAL
        
        abs_return = abs(spx_return_1d)
        
        if abs_return < self.config.spx_stress_threshold / 2:
            return 0.0
        elif abs_return < self.config.spx_stress_threshold:
            return 1.0
        elif abs_return < self.config.spx_crisis_threshold:
            return 2.0
        else:
            return 3.0
    
    def _score_to_regime(self, score: float) -> VolatilityRegime:
        """Map composite score to regime enum."""
        if score < 0.5:
            return VolatilityRegime.CALM
        elif score < 1.5:
            return VolatilityRegime.NORMAL
        elif score < 2.5:
            return VolatilityRegime.STRESSED
        else:
            return VolatilityRegime.CRISIS
    
    def _calculate_confidence(self, vix_score: float, rv_score: float, returns_score: float) -> float:
        """
        Calculate confidence in regime classification based on signal agreement.
        
        Returns:
            Confidence from 0 to 1
        """
        scores = [vix_score, rv_score, returns_score]
        
        # If all signals agree (within 0.5), high confidence
        score_range = max(scores) - min(scores)
        
        if score_range < 0.5:
            return 0.9
        elif score_range < 1.0:
            return 0.7
        elif score_range < 1.5:
            return 0.5
        else:
            return 0.3
    
    def _get_vix_change(self, periods: int) -> Optional[float]:
        """Get VIX change over N periods."""
        if len(self.vix_history) < periods + 1:
            return None
        
        current = self.vix_history[-1][1]
        previous = self.vix_history[-(periods + 1)][1]
        
        return current - previous
    
    def _get_spx_return(self, periods: int) -> Optional[float]:
        """Get SPX return over N periods."""
        if len(self.spx_history) < periods + 1:
            return None
        
        current = self.spx_history[-1][1]
        previous = self.spx_history[-(periods + 1)][1]
        
        if previous <= 0:
            return None
        
        return (current - previous) / previous
    
    def _get_vix_percentile(self, vix_value: Optional[float]) -> Optional[float]:
        """Get VIX percentile based on recent history."""
        if vix_value is None or len(self.vix_history) < 10:
            return None
        
        historical_vix = [v[1] for v in self.vix_history]
        percentile = (np.sum(np.array(historical_vix) < vix_value) / len(historical_vix)) * 100
        
        return percentile
    
    def get_regime_history(self, periods: int = None) -> List[RegimeSnapshot]:
        """
        Get historical regime snapshots.
        
        Args:
            periods: Number of periods to return (all if None)
            
        Returns:
            List of RegimeSnapshot objects
        """
        if periods is None:
            return self.regime_history.copy()
        return self.regime_history[-periods:]
    
    def get_regime_duration(self) -> timedelta:
        """Get duration in current regime."""
        if not self.regime_history:
            return timedelta(0)
        
        # Find when regime started
        current = self.current_regime
        duration_bars = 0
        
        for snapshot in reversed(self.regime_history):
            if snapshot.regime == current:
                duration_bars += 1
            else:
                break
        
        # Approximate duration (assumes 1 bar = 1 minute for options)
        return timedelta(minutes=duration_bars)
    
    def to_dict(self) -> Dict:
        """
        Export current state as dictionary.
        
        Returns:
            Dictionary with regime state
        """
        snapshot = self.get_current_regime()
        
        return {
            'regime': snapshot.regime.value,
            'confidence': snapshot.confidence,
            'vix_level': snapshot.vix_level,
            'vix_change_1d': snapshot.vix_change_1d,
            'realized_vol': snapshot.realized_vol,
            'composite_score': snapshot.composite_score,
            'position_multiplier': self.get_regime_multiplier(),
            'greek_limit_multiplier': self.get_greek_limit_multiplier(),
            'disable_short_vol': self.should_disable_short_vol()
        }

