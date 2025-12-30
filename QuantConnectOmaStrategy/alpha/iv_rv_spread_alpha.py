"""
IV vs RV Spread Alpha Model - Institutional Grade
Generates trading signals based on implied vs realized volatility spreads and bid/ask spreads.

Author: OMA Strategy Team
Version: 2.0 - Enhanced with regime awareness, ensemble RV, moneyness filtering
"""

from AlgorithmImports import *
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta


@dataclass
class AlphaConfig:
    """
    Configuration for IV/RV spread alpha model.

    All parameters are configurable via config files.
    """

    # IV/RV threshold parameters
    iv_rv_threshold: float = 1.2  # Base threshold: IV/RV ratio to trigger signal
    iv_rv_max_signal: float = 2.0  # Cap for signal strength calculation
    signal_strength_divisor: float = 0.5  # Divisor for signal strength calc

    # Moneyness filtering
    moneyness_min: float = 0.90  # Minimum K/S ratio
    moneyness_max: float = 1.10  # Maximum K/S ratio
    prefer_atm: bool = True  # Weight ATM options more heavily
    atm_range: float = 0.02  # ±2% from spot = ATM

    # Spread capture parameters
    spread_threshold: float = 0.005  # 0.5% minimum spread to flag
    spread_weight: float = 0.5  # Weight for spread signal vs IV/RV

    # Liquidity filters
    min_open_interest: int = 1000
    min_daily_volume: int = 1000
    min_bid: float = 0.05  # Minimum bid price

    # DTE preferences
    min_dte: int = 1  # Minimum days to expiry
    max_dte: int = 45  # Maximum days to expiry
    optimal_dte_min: int = 7  # Prefer DTEs in this range
    optimal_dte_max: int = 21

    # Regime-based adjustments
    reduce_signal_in_stress: bool = True  # Reduce signals in STRESSED regime
    disable_short_vol_in_crisis: bool = True  # No short vol in CRISIS
    regime_stress_multiplier: float = 0.5  # Multiply signal by this in STRESSED

    # Uncertainty-based adjustments
    max_rv_uncertainty_ratio: float = 0.2  # Max (uncertainty/RV) to trade
    reduce_on_high_uncertainty: bool = True

    # Signal output
    min_signal_strength: float = 0.1  # Minimum to emit insight
    default_insight_period_hours: int = 4  # Default holding period


class IVRVSpreadAlphaModel(AlphaModel):
    """
    Institutional-grade alpha model that generates signals based on:

    1. IV vs RV spread arbitrage opportunities (ensemble RV)
    2. Bid/ask spread capture opportunities
    3. Regime-aware signal generation
    4. Moneyness and DTE filtering

    Signals:
    - Short premium when IV > RV * threshold (adjusted for regime)
    - Place passive limit orders when bid/ask spread > threshold

    Key improvements over V1:
    - Uses ensemble RV from multiple estimators
    - Considers RV uncertainty in signal generation
    - Regime-aware: reduces/disables signals in stress
    - Moneyness filtering with ATM preference
    - Configurable via external config
    """

    def __init__(
        self,
        iv_rv_threshold: float = 1.2,
        spread_threshold: float = 0.005,
        rv_calculator=None,
        regime_classifier=None,
        logger=None,
        config: AlphaConfig = None,
    ):
        """
        Initialize the IV/RV spread alpha model.

        Args:
            iv_rv_threshold: Threshold for IV/RV ratio to trigger signal
            spread_threshold: Minimum bid/ask spread as % of mark price
            rv_calculator: RealizedVolatilityCalculator instance
            regime_classifier: VolatilityRegimeClassifier instance (optional)
            logger: StrategyLogger instance
            config: Full AlphaConfig (overrides individual params if provided)
        """
        # Use config if provided, otherwise construct from params
        if config:
            self.config = config
        else:
            self.config = AlphaConfig(
                iv_rv_threshold=iv_rv_threshold, spread_threshold=spread_threshold
            )

        self.rv_calculator = rv_calculator
        self.regime_classifier = regime_classifier
        self.logger = logger

        # Track active insights per symbol
        self.active_insights: Dict[Symbol, Insight] = {}

        # Cache for RV per underlying (avoid recalculating per contract)
        self.rv_cache: Dict[Symbol, Dict] = {}
        self.rv_cache_time: Dict[Symbol, datetime] = {}

        # Diagnostics tracking
        self.diagnostics = {
            "signals_generated": 0,
            "signals_filtered_regime": 0,
            "signals_filtered_moneyness": 0,
            "signals_filtered_liquidity": 0,
            "signals_filtered_uncertainty": 0,
            "iv_rv_ratios": [],
            "signal_regimes": [],
        }

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        """
        Generate alpha signals based on current market data.

        Args:
            algorithm: QuantConnect algorithm instance
            data: Current data slice with option chains

        Returns:
            List of Insight objects for identified opportunities
        """
        insights = []

        # Skip during warmup
        if algorithm.IsWarmingUp:
            return insights

        # Get current regime
        regime = self._get_current_regime()

        # Check if we should disable short vol entirely
        if regime and self.config.disable_short_vol_in_crisis:
            if self.regime_classifier and self.regime_classifier.is_crisis():
                if self.logger:
                    self.logger.log(
                        "Alpha model disabled - CRISIS regime", LogLevel.WARNING
                    )
                return insights

        # Process each option chain
        for kvp in data.OptionChains:
            chain = kvp.Value
            if len(chain) == 0:
                continue

            # Get underlying and its RV
            underlying_symbol = (
                chain.Underlying.Symbol if chain.Underlying else chain.Symbol
            )
            underlying_price = chain.Underlying.Price if chain.Underlying else None

            if underlying_price is None or underlying_price <= 0:
                continue

            # Get cached or fresh RV data for this underlying
            rv_data = self._get_rv_data(underlying_symbol)
            if rv_data is None or rv_data.get("ensemble") is None:
                continue

            # Analyze each contract
            for contract in chain:
                insight = self._analyze_contract(
                    contract, underlying_price, rv_data, regime, algorithm
                )

                if insight:
                    insights.append(insight)

        # Log summary
        if insights and self.logger:
            self.logger.log(
                f"Generated {len(insights)} alpha insights | Regime: {regime}",
                LogLevel.DEBUG,
            )
            self.diagnostics["signals_generated"] += len(insights)

        return insights

    def _get_rv_data(self, underlying_symbol: Symbol) -> Optional[Dict]:
        """
        Get realized volatility data for an underlying, using cache.

        Args:
            underlying_symbol: The underlying symbol

        Returns:
            Dictionary with RV data or None
        """
        if self.rv_calculator is None:
            return None

        # Check cache (valid for 1 bar)
        if underlying_symbol in self.rv_cache_time:
            # Cache is valid if updated recently (simple approach)
            cached = self.rv_cache.get(underlying_symbol)
            if cached:
                return cached

        # Get fresh RV summary
        summary = self.rv_calculator.get_realized_vol_summary(underlying_symbol)

        if not summary.is_valid():
            return None

        # Build RV data dict
        rv_data = {
            "ensemble": summary.ensemble,
            "uncertainty": summary.uncertainty or 0,
            "close_to_close": summary.close_to_close,
            "parkinson": summary.parkinson,
            "garman_klass": summary.garman_klass,
            "ewma": summary.ewma,
            "high_uncertainty": summary.has_high_uncertainty(
                self.config.max_rv_uncertainty_ratio
            ),
        }

        # Cache
        self.rv_cache[underlying_symbol] = rv_data
        self.rv_cache_time[underlying_symbol] = self.rv_calculator.algorithm.Time

        return rv_data

    def _get_current_regime(self) -> Optional[str]:
        """Get current volatility regime as string."""
        if self.regime_classifier is None:
            return None

        snapshot = self.regime_classifier.get_current_regime()
        return snapshot.regime.value if snapshot else None

    def _analyze_contract(
        self,
        contract,
        underlying_price: float,
        rv_data: Dict,
        regime: Optional[str],
        algorithm: QCAlgorithm,
    ) -> Optional[Insight]:
        """
        Analyze a single option contract for trading signal.

        Args:
            contract: Option contract from chain
            underlying_price: Current price of underlying
            rv_data: Realized volatility data dictionary
            regime: Current volatility regime
            algorithm: Algorithm instance

        Returns:
            Insight if signal generated, None otherwise
        """
        # 1. Liquidity filter
        if not self._passes_liquidity_filter(contract):
            return None

        # 2. Moneyness filter
        moneyness = contract.Strike / underlying_price if underlying_price > 0 else 0
        if not self._passes_moneyness_filter(moneyness):
            self.diagnostics["signals_filtered_moneyness"] += 1
            return None

        # 3. DTE filter
        dte = (contract.Expiry - algorithm.Time).days
        if not self._passes_dte_filter(dte):
            return None

        # 4. RV uncertainty check
        if self.config.reduce_on_high_uncertainty and rv_data.get(
            "high_uncertainty", False
        ):
            self.diagnostics["signals_filtered_uncertainty"] += 1
            # Don't return None, but will reduce signal strength

        # 5. Calculate IV/RV signal
        iv_rv_signal = self._calculate_iv_rv_signal(
            contract, rv_data["ensemble"], rv_data.get("high_uncertainty", False)
        )

        # 6. Calculate spread signal
        spread_signal = self._calculate_spread_signal(contract)

        # 7. Apply regime adjustments
        adjusted_iv_rv_signal = self._apply_regime_adjustment(iv_rv_signal, regime)

        if regime and regime in ["STRESSED", "CRISIS"]:
            self.diagnostics["signals_filtered_regime"] += 1

        # 8. Combine signals
        total_signal = adjusted_iv_rv_signal + spread_signal * self.config.spread_weight

        # 9. Check minimum threshold
        if abs(total_signal) < self.config.min_signal_strength:
            return None

        # 10. Apply moneyness weighting (prefer ATM)
        if self.config.prefer_atm:
            total_signal *= self._get_moneyness_weight(moneyness)

        # 11. Apply DTE weighting (prefer optimal DTE range)
        total_signal *= self._get_dte_weight(dte)

        # 12. Create insight
        insight = self._create_insight(
            contract,
            total_signal,
            iv_rv_signal,
            spread_signal,
            rv_data,
            moneyness,
            dte,
            regime,
            algorithm,
        )

        # Track diagnostics
        iv = contract.ImpliedVolatility if hasattr(contract, "ImpliedVolatility") else 0
        rv = rv_data["ensemble"]
        if rv > 0 and iv > 0:
            self.diagnostics["iv_rv_ratios"].append(iv / rv)
            self.diagnostics["signal_regimes"].append(regime or "UNKNOWN")

        return insight

    def _passes_liquidity_filter(self, contract) -> bool:
        """
        Check if contract meets liquidity requirements.

        Args:
            contract: Option contract

        Returns:
            True if contract is liquid enough to trade
        """
        if contract.OpenInterest < self.config.min_open_interest:
            self.diagnostics["signals_filtered_liquidity"] += 1
            return False

        if contract.Volume < self.config.min_daily_volume:
            self.diagnostics["signals_filtered_liquidity"] += 1
            return False

        if contract.BidSize <= 0 or contract.AskSize <= 0:
            return False

        if contract.Bid < self.config.min_bid:
            return False

        if contract.Ask <= contract.Bid:
            return False

        return True

    def _passes_moneyness_filter(self, moneyness: float) -> bool:
        """
        Check if moneyness is within acceptable range.

        Args:
            moneyness: Strike / Spot ratio

        Returns:
            True if within range
        """
        return self.config.moneyness_min <= moneyness <= self.config.moneyness_max

    def _passes_dte_filter(self, dte: int) -> bool:
        """
        Check if DTE is within acceptable range.

        Args:
            dte: Days to expiry

        Returns:
            True if within range
        """
        return self.config.min_dte <= dte <= self.config.max_dte

    def _calculate_iv_rv_signal(
        self, contract, rv: float, high_uncertainty: bool
    ) -> float:
        """
        Calculate IV/RV arbitrage signal.

        Uses ensemble RV and considers uncertainty.

        Args:
            contract: Option contract
            rv: Ensemble realized volatility
            high_uncertainty: Whether RV uncertainty is high

        Returns:
            Signal strength (-1 to 1, negative = short premium)
        """
        try:
            # Get implied volatility
            iv = contract.ImpliedVolatility

            # Validate IV
            if iv <= 0 or iv > 5:  # Cap at 500% IV
                return 0

            if rv <= 0:
                return 0

            # Calculate IV/RV ratio
            iv_rv_ratio = iv / rv

            # Check against threshold
            threshold = self.config.iv_rv_threshold

            if iv_rv_ratio > threshold:
                # IV is expensive relative to RV - short premium signal
                excess = iv_rv_ratio - threshold
                capped_excess = min(excess, self.config.iv_rv_max_signal - threshold)
                signal_strength = capped_excess / self.config.signal_strength_divisor
                signal_strength = min(signal_strength, 1.0)

                # Reduce signal if high uncertainty
                if high_uncertainty and self.config.reduce_on_high_uncertainty:
                    signal_strength *= 0.5

                return -signal_strength  # Negative for short premium

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

            if mark_price <= 0:
                return 0

            # Spread as percentage of mark
            spread_pct = spread / mark_price

            # Generate signal if spread exceeds threshold
            if spread_pct > self.config.spread_threshold:
                excess = spread_pct - self.config.spread_threshold
                signal_strength = min(excess / self.config.spread_threshold, 1.0)
                return signal_strength

            return 0

        except Exception:
            return 0

    def _apply_regime_adjustment(self, signal: float, regime: Optional[str]) -> float:
        """
        Apply regime-based adjustment to signal.

        Args:
            signal: Original signal strength
            regime: Current volatility regime

        Returns:
            Adjusted signal strength
        """
        if regime is None:
            return signal

        if not self.config.reduce_signal_in_stress:
            return signal

        if regime == "STRESSED":
            return signal * self.config.regime_stress_multiplier
        elif regime == "CRISIS":
            # In crisis, only allow long vol signals (positive)
            if signal < 0:
                return 0  # Disable short vol
            return signal

        return signal

    def _get_moneyness_weight(self, moneyness: float) -> float:
        """
        Get weight based on moneyness (prefer ATM).

        Args:
            moneyness: Strike / Spot ratio

        Returns:
            Weight multiplier (0.5 to 1.0)
        """
        distance_from_atm = abs(moneyness - 1.0)

        if distance_from_atm <= self.config.atm_range:
            return 1.0  # ATM - full weight
        elif distance_from_atm <= self.config.atm_range * 2:
            return 0.8  # Near ATM
        elif distance_from_atm <= self.config.atm_range * 4:
            return 0.6  # Moderate OTM/ITM
        else:
            return 0.5  # Far OTM/ITM

    def _get_dte_weight(self, dte: int) -> float:
        """
        Get weight based on DTE (prefer optimal range).

        Args:
            dte: Days to expiry

        Returns:
            Weight multiplier (0.5 to 1.0)
        """
        if self.config.optimal_dte_min <= dte <= self.config.optimal_dte_max:
            return 1.0  # Optimal range
        elif dte < self.config.optimal_dte_min:
            return 0.7  # Too short - gamma risk
        else:
            return 0.8  # Longer dated - less theta decay

    def _create_insight(
        self,
        contract,
        total_signal: float,
        iv_rv_signal: float,
        spread_signal: float,
        rv_data: Dict,
        moneyness: float,
        dte: int,
        regime: Optional[str],
        algorithm: QCAlgorithm,
    ) -> Insight:
        """
        Create Insight object from signal analysis.

        Args:
            contract: Option contract
            total_signal: Combined signal strength
            iv_rv_signal: IV/RV signal component
            spread_signal: Spread signal component
            rv_data: RV data dictionary
            moneyness: Strike/Spot ratio
            dte: Days to expiry
            regime: Current regime
            algorithm: Algorithm instance

        Returns:
            Insight object with metadata
        """
        # Determine direction based on signal
        if total_signal < 0:
            # Short premium (sell options)
            if contract.Right == OptionRight.Call:
                direction = InsightDirection.Down
            else:
                direction = InsightDirection.Up
        else:
            # Long premium (buy options)
            if contract.Right == OptionRight.Call:
                direction = InsightDirection.Up
            else:
                direction = InsightDirection.Down

        # Calculate confidence from signal strength
        confidence = min(abs(total_signal), 1.0)

        # Insight period based on DTE
        if dte <= 7:
            period = timedelta(hours=2)
        elif dte <= 14:
            period = timedelta(hours=4)
        else:
            period = timedelta(hours=self.config.default_insight_period_hours)

        # Create insight
        insight = Insight(
            contract.Symbol,
            period,
            InsightType.Price,
            direction,
            confidence,
            sourceModel="IVRVSpreadAlpha_v2",
        )

        # Add comprehensive metadata
        iv = contract.ImpliedVolatility if hasattr(contract, "ImpliedVolatility") else 0
        insight.Properties["SignalType"] = (
            "IV_RV" if abs(iv_rv_signal) > abs(spread_signal) else "Spread"
        )
        insight.Properties["IVRVSignal"] = iv_rv_signal
        insight.Properties["SpreadSignal"] = spread_signal
        insight.Properties["TotalSignal"] = total_signal
        insight.Properties["IV"] = iv
        insight.Properties["RV_Ensemble"] = rv_data.get("ensemble", 0)
        insight.Properties["RV_Uncertainty"] = rv_data.get("uncertainty", 0)
        insight.Properties["IVRVRatio"] = (
            iv / rv_data["ensemble"] if rv_data["ensemble"] > 0 else 0
        )
        insight.Properties["Strike"] = contract.Strike
        insight.Properties["Expiry"] = contract.Expiry
        insight.Properties["Right"] = str(contract.Right)
        insight.Properties["Moneyness"] = moneyness
        insight.Properties["DTE"] = dte
        insight.Properties["Regime"] = regime or "UNKNOWN"
        insight.Properties["Bid"] = contract.Bid
        insight.Properties["Ask"] = contract.Ask
        insight.Properties["Mark"] = (contract.Bid + contract.Ask) / 2
        insight.Properties["SpreadPct"] = (
            (contract.Ask - contract.Bid) / ((contract.Bid + contract.Ask) / 2)
            if contract.Bid + contract.Ask > 0
            else 0
        )

        # Log signal generation
        if self.logger:
            self.logger.log_signal(
                signal_type=insight.Properties["SignalType"],
                symbol=contract.Symbol,
                strength=total_signal,
                metadata={
                    "iv": iv,
                    "rv": rv_data["ensemble"],
                    "ratio": insight.Properties["IVRVRatio"],
                    "moneyness": moneyness,
                    "dte": dte,
                    "regime": regime,
                },
            )

        return insight

    def OnSecuritiesChanged(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
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

            # Clear underlying cache if needed
            underlying = (
                security.Symbol.Underlying
                if hasattr(security.Symbol, "Underlying")
                else None
            )
            if underlying and underlying in self.rv_cache:
                del self.rv_cache[underlying]
                if underlying in self.rv_cache_time:
                    del self.rv_cache_time[underlying]

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic statistics for the alpha model.

        Returns:
            Dictionary with diagnostic metrics
        """
        diag = self.diagnostics.copy()

        # Calculate additional stats
        if diag["iv_rv_ratios"]:
            diag["iv_rv_ratio_mean"] = np.mean(diag["iv_rv_ratios"])
            diag["iv_rv_ratio_median"] = np.median(diag["iv_rv_ratios"])
            diag["iv_rv_ratio_std"] = np.std(diag["iv_rv_ratios"])

        # Regime distribution
        if diag["signal_regimes"]:
            from collections import Counter

            diag["regime_distribution"] = dict(Counter(diag["signal_regimes"]))

        return diag

    def reset_diagnostics(self) -> None:
        """Reset diagnostic counters."""
        self.diagnostics = {
            "signals_generated": 0,
            "signals_filtered_regime": 0,
            "signals_filtered_moneyness": 0,
            "signals_filtered_liquidity": 0,
            "signals_filtered_uncertainty": 0,
            "iv_rv_ratios": [],
            "signal_regimes": [],
        }
