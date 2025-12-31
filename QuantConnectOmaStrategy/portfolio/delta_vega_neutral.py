"""Delta-Vega Neutral Portfolio Construction Model - Uses Lean Greeks, vol targeting, factor buckets."""

from AlgorithmImports import *
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
from utils.logger import LogLevel


class RiskBucket(Enum):
    """Risk factor bucket classifications."""

    INDEX = "INDEX"  # SPX, SPY, QQQ
    TECH = "TECH"  # AAPL, MSFT, NVDA, etc.
    SMALL_CAP = "SMALL_CAP"  # IWM, small caps
    SINGLE_NAME = "SINGLE_NAME"  # Individual stocks
    UNKNOWN = "UNKNOWN"


@dataclass
class PortfolioConfig:
    """Portfolio construction configuration."""

    # Position sizing
    max_position_size: float = 0.05  # 5% NAV per leg
    min_position_value: float = 1000  # Minimum position value

    # Greek limits
    vega_limit: float = 10000  # Portfolio-level vega cap
    delta_tolerance: float = 100  # Delta neutrality tolerance
    gamma_limit: float = 500  # Portfolio gamma limit

    # Volatility targeting
    target_daily_vol: float = 0.01  # 1% daily portfolio vol target
    vol_scaling_enabled: bool = True
    min_vol_scale: float = 0.2  # Don't scale below 20%
    max_vol_scale: float = 2.0  # Don't scale above 200%

    # Factor bucket limits (as fraction of portfolio vega limit)
    bucket_vega_limit_pct: Dict[str, float] = field(
        default_factory=lambda: {
            "INDEX": 0.6,  # 60% of total vega in indices
            "TECH": 0.3,  # 30% in tech
            "SMALL_CAP": 0.2,  # 20% in small cap
            "SINGLE_NAME": 0.15,  # 15% per single name
            "UNKNOWN": 0.1,
        }
    )

    bucket_gamma_limit_pct: Dict[str, float] = field(
        default_factory=lambda: {
            "INDEX": 0.6,
            "TECH": 0.3,
            "SMALL_CAP": 0.2,
            "SINGLE_NAME": 0.15,
            "UNKNOWN": 0.1,
        }
    )

    # Rebalancing
    rebalance_threshold: float = 0.1  # 10% of limits triggers rebalance

    # Hedging
    hedge_with_underlying: bool = True
    min_hedge_delta: float = 50  # Minimum delta deviation to hedge


@dataclass
class PortfolioRiskSnapshot:
    """Current portfolio risk metrics snapshot."""

    timestamp: datetime

    # Portfolio-level Greeks
    total_delta: float = 0
    total_vega: float = 0
    total_gamma: float = 0
    total_theta: float = 0

    # Per-symbol Greeks
    delta_by_symbol: Dict[Symbol, float] = field(default_factory=dict)
    vega_by_symbol: Dict[Symbol, float] = field(default_factory=dict)
    gamma_by_symbol: Dict[Symbol, float] = field(default_factory=dict)

    # Per-bucket Greeks
    delta_by_bucket: Dict[RiskBucket, float] = field(default_factory=dict)
    vega_by_bucket: Dict[RiskBucket, float] = field(default_factory=dict)
    gamma_by_bucket: Dict[RiskBucket, float] = field(default_factory=dict)

    # Risk metrics
    portfolio_value: float = 0
    estimated_daily_var: float = 0  # Dollar VaR
    estimated_daily_vol: float = 0  # Volatility
    vol_scale_factor: float = 1.0

    # Limit utilization (0 to 1)
    delta_utilization: float = 0
    vega_utilization: float = 0
    gamma_utilization: float = 0


class DeltaVegaNeutralPortfolioConstructionModel(PortfolioConstructionModel):
    """Portfolio construction: Lean Greeks, delta-neutral, vol targeting, factor buckets."""

    # Symbol to bucket mapping (can be extended)
    SYMBOL_BUCKET_MAP = {
        "SPX": RiskBucket.INDEX,
        "SPY": RiskBucket.INDEX,
        "QQQ": RiskBucket.INDEX,
        "IWM": RiskBucket.SMALL_CAP,
        "AAPL": RiskBucket.TECH,
        "MSFT": RiskBucket.TECH,
        "NVDA": RiskBucket.TECH,
        "GOOGL": RiskBucket.TECH,
        "AMZN": RiskBucket.TECH,
        "META": RiskBucket.TECH,
        "TSLA": RiskBucket.TECH,
    }

    def __init__(
        self,
        max_position_size: float = 0.05,
        vega_limit: float = 10000,
        delta_tolerance: float = 100,
        rebalance_threshold: float = 0.1,
        logger=None,
        rv_calculator=None,
        regime_classifier=None,
        config: PortfolioConfig = None,
    ):
        """Initialize portfolio construction model."""
        if config:
            self.config = config
        else:
            self.config = PortfolioConfig(
                max_position_size=max_position_size,
                vega_limit=vega_limit,
                delta_tolerance=delta_tolerance,
                rebalance_threshold=rebalance_threshold,
            )

        self.logger = logger
        self.rv_calculator = rv_calculator
        self.regime_classifier = regime_classifier

        # Current risk state
        self.risk_snapshot: Optional[PortfolioRiskSnapshot] = None

        # Active positions tracking
        self.active_positions: Dict[Symbol, float] = {}

        # Greeks cache (refresh each bar)
        self.greeks_cache: Dict[Symbol, Dict] = {}
        self.greeks_cache_time: Optional[datetime] = None

    def CreateTargets(
        self, algorithm: QCAlgorithm, insights: List[Insight]
    ) -> List[PortfolioTarget]:
        """Create portfolio targets from alpha insights."""
        targets = []

        # Update risk snapshot
        self._update_risk_snapshot(algorithm)

        if not insights:
            # No new insights - check if rebalancing needed
            if self._needs_rebalancing():
                targets = self._rebalance_portfolio(algorithm)
            return targets

        if self.logger:
            self.logger.log(
                f"Processing {len(insights)} insights | "
                f"Current delta: {self.risk_snapshot.total_delta:.0f} | "
                f"Vega: {self.risk_snapshot.total_vega:.0f}",
                LogLevel.INFO,
            )

        # Get volatility scale factor for position sizing
        vol_scale = self._calculate_vol_scale_factor(algorithm)

        # Get regime-based scale factor
        regime_scale = self._get_regime_scale_factor()

        # Combined scale
        total_scale = vol_scale * regime_scale

        # Group insights by underlying
        insights_by_underlying = self._group_insights_by_underlying(insights)

        # Process each underlying's insights
        for underlying, underlying_insights in insights_by_underlying.items():
            basket_targets = self._create_option_basket(
                algorithm, underlying, underlying_insights, total_scale
            )
            targets.extend(basket_targets)

        # Add hedging targets if needed
        hedge_targets = self._create_hedge_targets(algorithm, targets)
        targets.extend(hedge_targets)

        # Final validation against risk limits
        targets = self._validate_risk_limits(algorithm, targets)

        # Log risk snapshot after proposed changes
        if targets and self.logger:
            self.logger.log_greek_exposure(
                {
                    "delta": self.risk_snapshot.total_delta,
                    "vega": self.risk_snapshot.total_vega,
                    "gamma": self.risk_snapshot.total_gamma,
                    "theta": self.risk_snapshot.total_theta,
                    "vol_scale": total_scale,
                }
            )

        return targets

    def _update_risk_snapshot(self, algorithm: QCAlgorithm) -> None:
        """Update portfolio risk snapshot using Lean Greeks."""
        snapshot = PortfolioRiskSnapshot(timestamp=algorithm.Time)
        snapshot.portfolio_value = algorithm.Portfolio.TotalPortfolioValue

        # Initialize bucket accumulators
        for bucket in RiskBucket:
            snapshot.delta_by_bucket[bucket] = 0
            snapshot.vega_by_bucket[bucket] = 0
            snapshot.gamma_by_bucket[bucket] = 0

        # Clear Greeks cache if stale
        if self.greeks_cache_time != algorithm.Time:
            self.greeks_cache.clear()
            self.greeks_cache_time = algorithm.Time

        # Calculate Greeks for each position
        for holding in algorithm.Portfolio.Values:
            if holding.Quantity == 0:
                continue

            symbol = holding.Symbol

            if symbol.SecurityType == SecurityType.Option:
                # Get Greeks from Lean or calculate fallback
                greeks = self._get_option_greeks(algorithm, symbol)

                if greeks:
                    multiplier = 100  # Standard option multiplier
                    qty = holding.Quantity

                    # Per-symbol Greeks
                    symbol_delta = greeks["delta"] * qty * multiplier
                    symbol_vega = greeks["vega"] * qty * multiplier
                    symbol_gamma = greeks["gamma"] * qty * multiplier
                    symbol_theta = greeks["theta"] * qty * multiplier

                    snapshot.delta_by_symbol[symbol] = symbol_delta
                    snapshot.vega_by_symbol[symbol] = symbol_vega
                    snapshot.gamma_by_symbol[symbol] = symbol_gamma

                    # Aggregate to portfolio
                    snapshot.total_delta += symbol_delta
                    snapshot.total_vega += symbol_vega
                    snapshot.total_gamma += symbol_gamma
                    snapshot.total_theta += symbol_theta

                    # Aggregate to bucket
                    bucket = self._get_symbol_bucket(
                        symbol.Underlying if symbol.Underlying else symbol
                    )
                    snapshot.delta_by_bucket[bucket] += symbol_delta
                    snapshot.vega_by_bucket[bucket] += symbol_vega
                    snapshot.gamma_by_bucket[bucket] += symbol_gamma
            else:
                # Equity/Index - delta = quantity
                snapshot.total_delta += holding.Quantity
                snapshot.delta_by_symbol[symbol] = holding.Quantity

                bucket = self._get_symbol_bucket(symbol)
                snapshot.delta_by_bucket[bucket] += holding.Quantity

        # Calculate limit utilizations
        snapshot.delta_utilization = (
            abs(snapshot.total_delta) / self.config.delta_tolerance
            if self.config.delta_tolerance > 0
            else 0
        )
        snapshot.vega_utilization = (
            abs(snapshot.total_vega) / self.config.vega_limit
            if self.config.vega_limit > 0
            else 0
        )
        snapshot.gamma_utilization = (
            abs(snapshot.total_gamma) / self.config.gamma_limit
            if self.config.gamma_limit > 0
            else 0
        )

        # Estimate portfolio daily volatility
        snapshot.estimated_daily_vol = self._estimate_portfolio_vol(algorithm, snapshot)
        snapshot.vol_scale_factor = self._calculate_vol_scale_factor(
            algorithm, snapshot
        )

        self.risk_snapshot = snapshot

    def _get_option_greeks(
        self, algorithm: QCAlgorithm, option_symbol: Symbol
    ) -> Optional[Dict]:
        """Get option Greeks, preferring Lean's built-in Greeks."""
        # Check cache
        if option_symbol in self.greeks_cache:
            return self.greeks_cache[option_symbol]

        security = algorithm.Securities.get(option_symbol)
        if not security:
            return None

        greeks = {}

        try:
            # Try to get Lean's built-in Greeks
            if hasattr(security, "Greeks") and security.Greeks is not None:
                lean_greeks = security.Greeks

                # Use Lean Greeks if available
                if hasattr(lean_greeks, "Delta") and lean_greeks.Delta != 0:
                    greeks["delta"] = lean_greeks.Delta
                    greeks["vega"] = (
                        lean_greeks.Vega if hasattr(lean_greeks, "Vega") else 0
                    )
                    greeks["gamma"] = (
                        lean_greeks.Gamma if hasattr(lean_greeks, "Gamma") else 0
                    )
                    greeks["theta"] = (
                        lean_greeks.Theta if hasattr(lean_greeks, "Theta") else 0
                    )

                    # Cache and return
                    self.greeks_cache[option_symbol] = greeks
                    return greeks

            # Fallback: Calculate approximate Greeks
            greeks = self._calculate_fallback_greeks(algorithm, option_symbol, security)

        except Exception as e:
            if self.logger:
                self.logger.log(
                    f"Error getting Greeks for {option_symbol}: {e}", LogLevel.DEBUG
                )
            greeks = self._calculate_fallback_greeks(algorithm, option_symbol, security)

        if greeks:
            self.greeks_cache[option_symbol] = greeks

        return greeks

    def _calculate_fallback_greeks(
        self, algorithm: QCAlgorithm, option_symbol: Symbol, security
    ) -> Optional[Dict]:
        """Calculate approximate Greeks using simplified Black-Scholes."""
        try:
            # Get underlying symbol and security
            if (
                not hasattr(option_symbol, "Underlying")
                or option_symbol.Underlying is None
            ):
                return None

            underlying_symbol = option_symbol.Underlying
            if underlying_symbol not in algorithm.Securities:
                return None

            underlying = algorithm.Securities[underlying_symbol]
            if underlying is None or not hasattr(underlying, "Price"):
                return None

            underlying_price = underlying.Price
            if underlying_price <= 0:
                return None

            strike = option_symbol.ID.StrikePrice
            tte = (option_symbol.ID.Date - algorithm.Time).days / 365.25

            if tte <= 0:
                return None

            # Moneyness
            moneyness = underlying_price / strike

            # IV from security if available
            iv = 0.25
            if hasattr(security, "ImpliedVolatility") and security.ImpliedVolatility:
                if security.ImpliedVolatility > 0:
                    iv = security.ImpliedVolatility

            # Approximate delta using simple approximation (no recursion)
            is_call = option_symbol.ID.OptionRight == OptionRight.Call
            if moneyness > 1.1:
                call_delta = min(0.95, 0.5 + (moneyness - 1) * 3)
            elif moneyness < 0.9:
                call_delta = max(0.05, 0.5 - (1 - moneyness) * 3)
            else:
                # Near ATM - use normal approximation
                d1 = (np.log(moneyness) + 0.5 * iv * iv * tte) / (iv * np.sqrt(tte))
                call_delta = 0.5 * (1 + np.clip(d1 / 2, -1, 1))

            # Delta based on option type
            if is_call:
                delta = call_delta
            else:
                delta = call_delta - 1  # Put-Call parity

            # Vega - peaks at ATM, scales with sqrt(T)
            atm_factor = np.exp(-8 * (moneyness - 1) ** 2)
            vega = underlying_price * np.sqrt(tte) * atm_factor * 0.01 * 0.4

            # Gamma - peaks at ATM, inversely proportional to sqrt(T)
            gamma = (
                atm_factor / (underlying_price * iv * np.sqrt(tte)) * 0.01
                if tte > 0.01
                else 0
            )

            # Theta - simplified
            theta = -vega * iv / (2 * np.sqrt(tte)) if tte > 0.01 else 0

            return {"delta": delta, "vega": vega, "gamma": gamma, "theta": theta}

        except Exception as e:
            if self.logger:
                self.logger.log(f"Fallback Greeks calc error: {e}", LogLevel.DEBUG)
            return None

    def _get_symbol_bucket(self, symbol: Symbol) -> RiskBucket:
        """
        Get risk bucket for a symbol.

        Args:
            symbol: Symbol to classify

        Returns:
            RiskBucket enum value
        """
        # Extract ticker from symbol
        ticker = str(symbol.Value).upper()

        # Check mapping
        for key, bucket in self.SYMBOL_BUCKET_MAP.items():
            if key.upper() in ticker:
                return bucket

        # Default classification based on symbol type
        if symbol.SecurityType in [SecurityType.Index, SecurityType.IndexOption]:
            return RiskBucket.INDEX

        return RiskBucket.SINGLE_NAME

    def _estimate_portfolio_vol(
        self, algorithm: QCAlgorithm, snapshot: PortfolioRiskSnapshot
    ) -> float:
        """Estimate portfolio daily vol using Greeks and RV."""
        if self.rv_calculator is None:
            return 0.01  # Default 1% if no RV calculator

        portfolio_value = snapshot.portfolio_value
        if portfolio_value <= 0:
            return 0.01

        # Get representative underlying RV
        rv = None
        for symbol in snapshot.delta_by_symbol.keys():
            underlying = (
                symbol.Underlying
                if hasattr(symbol, "Underlying") and symbol.Underlying
                else symbol
            )
            rv = self.rv_calculator.get_realized_vol(underlying, method="ensemble")
            if rv:
                break

        if rv is None:
            rv = 0.15  # Default 15% annualized

        # Daily vol from annualized
        daily_rv = rv / np.sqrt(252)

        # Estimate portfolio var from delta exposure
        # Simplified: portfolio_var ≈ delta² * underlying_var + vega² * iv_var
        delta_var = (snapshot.total_delta**2) * (daily_rv**2)

        # Vega contribution (assume IV can move 1-2 vol points daily)
        iv_daily_vol = 0.02  # 2 vol points daily std
        vega_var = (snapshot.total_vega**2) * (iv_daily_vol**2)

        # Total variance
        total_var = delta_var + vega_var

        # Convert to portfolio vol
        portfolio_vol = (
            np.sqrt(total_var) / portfolio_value if portfolio_value > 0 else 0.01
        )

        return max(0.001, min(0.1, portfolio_vol))  # Clamp between 0.1% and 10%

    def _calculate_vol_scale_factor(
        self, algorithm: QCAlgorithm, snapshot: PortfolioRiskSnapshot = None
    ) -> float:
        """Calculate position scaling factor for target volatility."""
        if not self.config.vol_scaling_enabled:
            return 1.0

        if snapshot is None:
            snapshot = self.risk_snapshot

        if snapshot is None:
            return 1.0

        estimated_vol = snapshot.estimated_daily_vol
        if estimated_vol <= 0:
            return 1.0

        target_vol = self.config.target_daily_vol

        # Scale factor = target / current
        scale = target_vol / estimated_vol

        # Clamp to reasonable range
        scale = max(self.config.min_vol_scale, min(self.config.max_vol_scale, scale))

        return scale

    def _get_regime_scale_factor(self) -> float:
        """Get position scale factor based on volatility regime."""
        if self.regime_classifier is None:
            return 1.0

        return self.regime_classifier.get_regime_multiplier()

    def _group_insights_by_underlying(
        self, insights: List[Insight]
    ) -> Dict[Symbol, List[Insight]]:
        """Group insights by underlying asset."""
        grouped = {}

        for insight in insights:
            if insight.Symbol.SecurityType == SecurityType.Option:
                underlying = insight.Symbol.Underlying
            else:
                underlying = insight.Symbol

            if underlying not in grouped:
                grouped[underlying] = []
            grouped[underlying].append(insight)

        return grouped

    def _create_option_basket(
        self,
        algorithm: QCAlgorithm,
        underlying: Symbol,
        insights: List[Insight],
        scale_factor: float,
    ) -> List[PortfolioTarget]:
        """Create delta-neutral option basket from insights."""
        targets = []

        # Get bucket for underlying
        bucket = self._get_symbol_bucket(underlying)

        # Check bucket limits
        bucket_vega_limit = (
            self.config.vega_limit
            * self.config.bucket_vega_limit_pct.get(bucket.value, 0.1)
        )
        current_bucket_vega = (
            self.risk_snapshot.vega_by_bucket.get(bucket, 0)
            if self.risk_snapshot
            else 0
        )

        # Portfolio value for sizing
        portfolio_value = algorithm.Portfolio.TotalPortfolioValue
        max_position_value = (
            portfolio_value * self.config.max_position_size * scale_factor
        )

        # Sort insights by confidence
        sorted_insights = sorted(
            insights, key=lambda x: abs(x.Confidence), reverse=True
        )

        # Track basket Greeks
        basket_delta = 0
        basket_vega = 0

        for insight in sorted_insights:
            # Check remaining bucket capacity
            remaining_bucket_vega = bucket_vega_limit - abs(
                current_bucket_vega + basket_vega
            )
            if remaining_bucket_vega <= 0:
                break

            # Check portfolio vega limit
            total_vega = (
                self.risk_snapshot.total_vega if self.risk_snapshot else 0
            ) + basket_vega
            if abs(total_vega) >= self.config.vega_limit * 0.9:
                break

            # Get option security
            option_symbol = insight.Symbol
            security = algorithm.Securities.get(option_symbol)

            if not security or not security.HasData:
                continue

            # Get Greeks
            greeks = self._get_option_greeks(algorithm, option_symbol)
            if not greeks:
                continue

            # Calculate position size
            position_size = self._calculate_position_size(
                insight,
                greeks,
                max_position_value,
                remaining_bucket_vega,
                basket_vega,
                algorithm,
            )

            if position_size == 0:
                continue

            # Create target
            target = PortfolioTarget(option_symbol, position_size)
            targets.append(target)

            # Update basket Greeks
            multiplier = 100
            basket_delta += greeks["delta"] * position_size * multiplier
            basket_vega += greeks["vega"] * position_size * multiplier

            if self.logger:
                self.logger.log(
                    f"Basket add: {option_symbol} Size={position_size} "
                    f"Delta={greeks['delta']:.3f} Vega={greeks['vega']:.2f}",
                    LogLevel.DEBUG,
                )

        # Create delta hedge if needed
        if (
            self.config.hedge_with_underlying
            and abs(basket_delta) > self.config.min_hedge_delta
        ):
            hedge_target = self._create_delta_hedge(
                algorithm, underlying, -basket_delta
            )
            if hedge_target:
                targets.append(hedge_target)

        return targets

    def _calculate_position_size(
        self,
        insight: Insight,
        greeks: Dict,
        max_position_value: float,
        remaining_bucket_vega: float,
        current_basket_vega: float,
        algorithm: QCAlgorithm,
    ) -> int:
        """Calculate position size respecting all limits."""
        security = algorithm.Securities.get(insight.Symbol)
        if not security:
            return 0

        option_price = security.Price
        if option_price <= 0:
            return 0

        # Contract value
        contract_value = option_price * 100

        # Confidence factor
        confidence_factor = abs(insight.Confidence)

        # Max contracts by position value
        max_by_value = (
            int(max_position_value * confidence_factor / contract_value)
            if contract_value > 0
            else 0
        )

        # Max contracts by vega
        vega_per_contract = abs(greeks.get("vega", 0)) * 100
        if vega_per_contract > 0:
            max_by_vega = int(remaining_bucket_vega / vega_per_contract)
        else:
            max_by_vega = max_by_value

        # Max contracts by portfolio vega limit
        remaining_portfolio_vega = self.config.vega_limit - abs(
            (self.risk_snapshot.total_vega if self.risk_snapshot else 0)
            + current_basket_vega
        )
        if vega_per_contract > 0:
            max_by_portfolio_vega = int(remaining_portfolio_vega / vega_per_contract)
        else:
            max_by_portfolio_vega = max_by_value

        # Take minimum of all constraints
        position_size = max(1, min(max_by_value, max_by_vega, max_by_portfolio_vega))

        # Apply direction from insight
        if insight.Direction == InsightDirection.Down:
            position_size = -position_size

        return position_size

    def _create_delta_hedge(
        self, algorithm: QCAlgorithm, underlying: Symbol, delta_to_hedge: float
    ) -> Optional[PortfolioTarget]:
        """Create delta hedge using underlying shares."""
        shares = int(round(delta_to_hedge))

        if abs(shares) < 1:
            return None

        # Get current position
        current_shares = (
            algorithm.Portfolio[underlying].Quantity
            if underlying in algorithm.Portfolio
            else 0
        )
        target_shares = current_shares + shares

        if self.logger:
            self.logger.log(
                f"Delta hedge: {underlying} Current={current_shares} "
                f"Hedge={shares} Target={target_shares}",
                LogLevel.INFO,
            )

        return PortfolioTarget(underlying, target_shares)

    def _create_hedge_targets(
        self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]
    ) -> List[PortfolioTarget]:
        """Create hedging targets for portfolio delta neutrality."""
        # Calculate projected delta from targets
        projected_delta = self.risk_snapshot.total_delta if self.risk_snapshot else 0

        for target in targets:
            if target.Symbol.SecurityType == SecurityType.Option:
                greeks = self._get_option_greeks(algorithm, target.Symbol)
                if greeks:
                    projected_delta += greeks["delta"] * target.Quantity * 100
            else:
                # Adjust for underlying targets already in list
                projected_delta += target.Quantity

        hedge_targets = []

        # Check if overall delta exceeds tolerance
        if abs(projected_delta) > self.config.delta_tolerance:
            # Find underlyings to hedge with
            underlyings: Set[Symbol] = set()

            for target in targets:
                if target.Symbol.SecurityType == SecurityType.Option:
                    underlyings.add(target.Symbol.Underlying)

            # Also check existing positions
            for holding in algorithm.Portfolio.Values:
                if (
                    holding.Symbol.SecurityType == SecurityType.Option
                    and holding.Quantity != 0
                ):
                    underlyings.add(holding.Symbol.Underlying)

            if underlyings:
                # Split hedge across underlyings (could be improved)
                hedge_per_underlying = -projected_delta / len(underlyings)

                for underlying in underlyings:
                    if underlying in algorithm.Securities:
                        hedge_target = self._create_delta_hedge(
                            algorithm, underlying, hedge_per_underlying
                        )
                        if hedge_target:
                            hedge_targets.append(hedge_target)

        return hedge_targets

    def _validate_risk_limits(
        self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]
    ) -> List[PortfolioTarget]:
        """Validate targets against risk limits and scale if needed."""
        # Calculate projected Greeks
        projected_delta = self.risk_snapshot.total_delta if self.risk_snapshot else 0
        projected_vega = self.risk_snapshot.total_vega if self.risk_snapshot else 0
        projected_gamma = self.risk_snapshot.total_gamma if self.risk_snapshot else 0

        for target in targets:
            if target.Symbol.SecurityType == SecurityType.Option:
                greeks = self._get_option_greeks(algorithm, target.Symbol)
                if greeks:
                    delta_change = greeks["delta"] * target.Quantity * 100
                    vega_change = greeks["vega"] * target.Quantity * 100
                    gamma_change = greeks["gamma"] * target.Quantity * 100

                    projected_delta += delta_change
                    projected_vega += vega_change
                    projected_gamma += gamma_change
            else:
                projected_delta += target.Quantity

        # Check if scaling needed
        scale_factor = 1.0

        # Vega scaling
        if abs(projected_vega) > self.config.vega_limit:
            vega_scale = self.config.vega_limit * 0.95 / abs(projected_vega)
            scale_factor = min(scale_factor, vega_scale)

            if self.logger:
                self.logger.log(
                    f"Vega limit exceeded: {projected_vega:.0f} > {self.config.vega_limit}, "
                    f"scaling by {vega_scale:.2f}",
                    LogLevel.WARNING,
                )

        # Gamma scaling
        if abs(projected_gamma) > self.config.gamma_limit:
            gamma_scale = self.config.gamma_limit * 0.95 / abs(projected_gamma)
            scale_factor = min(scale_factor, gamma_scale)

        # Apply scaling if needed
        if scale_factor < 1.0:
            for target in targets:
                if target.Symbol.SecurityType == SecurityType.Option:
                    target.Quantity = int(target.Quantity * scale_factor)

        return targets

    def _needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing."""
        if self.risk_snapshot is None:
            return False

        # Check delta
        if self.risk_snapshot.delta_utilization > (1 + self.config.rebalance_threshold):
            return True

        # Check vega
        if self.risk_snapshot.vega_utilization > (1 + self.config.rebalance_threshold):
            return True

        return False

    def _rebalance_portfolio(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """Rebalance portfolio to maintain risk limits."""
        targets = []

        if self.risk_snapshot is None:
            return targets

        # Delta rebalancing via hedge
        if abs(self.risk_snapshot.total_delta) > self.config.delta_tolerance:
            underlyings: Set[Symbol] = set()

            for holding in algorithm.Portfolio.Values:
                if (
                    holding.Symbol.SecurityType == SecurityType.Option
                    and holding.Quantity != 0
                ):
                    underlyings.add(holding.Symbol.Underlying)

            if underlyings:
                # Use first underlying for simplicity
                underlying = list(underlyings)[0]
                hedge_target = self._create_delta_hedge(
                    algorithm, underlying, -self.risk_snapshot.total_delta
                )
                if hedge_target:
                    targets.append(hedge_target)

        return targets

    def get_risk_snapshot(self) -> Optional[PortfolioRiskSnapshot]:
        """Get current portfolio risk snapshot."""
        return self.risk_snapshot

    def OnSecuritiesChanged(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
        """Handle security universe changes."""
        # Clear Greeks cache for removed securities
        for security in changes.RemovedSecurities:
            if security.Symbol in self.greeks_cache:
                del self.greeks_cache[security.Symbol]
            if security.Symbol in self.active_positions:
                del self.active_positions[security.Symbol]
