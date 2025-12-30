"""
Exposure Limits Risk Management Model - Institutional Grade
Enforces risk limits with scenario-based risk checks and regime awareness.

Author: OMA Strategy Team
Version: 2.0 - Enhanced with scenarios, regime integration, hard enforcement
"""

from AlgorithmImports import *
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class RiskAction(Enum):
    """Types of risk management actions."""

    NONE = "NONE"
    SCALE_DOWN = "SCALE_DOWN"
    CLOSE_POSITION = "CLOSE_POSITION"
    LIQUIDATE_ALL = "LIQUIDATE_ALL"
    BLOCK_NEW_RISK = "BLOCK_NEW_RISK"


@dataclass
class ScenarioConfig:
    """Configuration for a single stress scenario."""

    name: str
    underlying_move: float  # e.g., -0.05 for -5%
    iv_change: float  # e.g., 5.0 for +5 vol points
    description: str = ""


@dataclass
class RiskConfig:
    """
    Configuration for risk management.
    """

    # Daily loss limits
    max_daily_loss: float = 0.02  # 2% NAV
    warning_daily_loss: float = 0.015  # 1.5% warning threshold

    # Greek limits
    vega_limit: float = 10000
    delta_tolerance: float = 100
    gamma_limit: float = 500

    # Position limits
    max_position_age_hours: int = 24
    max_position_pct: float = 0.10  # 10% of portfolio
    max_unrealized_loss_pct: float = 0.50  # 50% loss triggers close

    # Scenario risk
    max_scenario_loss_pct: float = 0.10  # 10% max loss in any scenario
    scenarios: List[ScenarioConfig] = field(
        default_factory=lambda: [
            ScenarioConfig("SPX_DOWN_5_VOL_UP", -0.05, 5.0, "SPX down 5%, IV up 5pts"),
            ScenarioConfig(
                "SPX_DOWN_10_VOL_UP", -0.10, 10.0, "SPX down 10%, IV up 10pts"
            ),
            ScenarioConfig("SPX_UP_5_VOL_DOWN", 0.05, -3.0, "SPX up 5%, IV down 3pts"),
            ScenarioConfig("VOL_SPIKE", 0.0, 8.0, "IV spike 8pts, flat spot"),
            ScenarioConfig("VOL_CRUSH", 0.0, -5.0, "IV crush 5pts, flat spot"),
        ]
    )

    # Regime-based adjustments
    regime_delta_multiplier: Dict[str, float] = field(
        default_factory=lambda: {
            "CALM": 1.2,
            "NORMAL": 1.0,
            "STRESSED": 0.7,
            "CRISIS": 0.3,
        }
    )
    regime_vega_multiplier: Dict[str, float] = field(
        default_factory=lambda: {
            "CALM": 1.2,
            "NORMAL": 1.0,
            "STRESSED": 0.5,
            "CRISIS": 0.2,
        }
    )

    # Enforcement
    hard_liquidate_on_breach: bool = True
    log_all_risk_events: bool = True


@dataclass
class ScenarioResult:
    """Result of a scenario stress test."""

    scenario: ScenarioConfig
    estimated_pnl: float
    pnl_pct_nav: float
    breaches_limit: bool
    delta_contribution: float
    gamma_contribution: float
    vega_contribution: float


@dataclass
class RiskEvent:
    """Record of a risk management event."""

    timestamp: datetime
    event_type: str
    rule_name: str
    details: Dict
    action_taken: RiskAction
    positions_affected: List[Symbol] = field(default_factory=list)


class ExposureLimitsRiskManagementModel(RiskManagementModel):
    """
    Institutional-grade risk management model with:

    1. Daily loss limits with hard enforcement
    2. Greek limits (delta, vega, gamma) with regime adjustment
    3. Scenario-based stress testing
    4. Position age and concentration limits
    5. Regime-aware risk scaling
    6. Comprehensive risk event logging

    All risk rules are hard - they generate actual liquidation orders.
    """

    def __init__(
        self,
        max_daily_loss: float = 0.02,
        vega_limit: float = 10000,
        delta_tolerance: float = 100,
        max_position_age_hours: int = 24,
        logger=None,
        regime_classifier=None,
        config: RiskConfig = None,
    ):
        """
        Initialize the risk management model.

        Args:
            max_daily_loss: Maximum daily loss as fraction of NAV
            vega_limit: Maximum portfolio vega exposure
            delta_tolerance: Maximum acceptable delta deviation
            max_position_age_hours: Maximum hours to hold position
            logger: StrategyLogger instance
            regime_classifier: VolatilityRegimeClassifier instance
            config: Full RiskConfig (overrides individual params)
        """
        if config:
            self.config = config
        else:
            self.config = RiskConfig(
                max_daily_loss=max_daily_loss,
                vega_limit=vega_limit,
                delta_tolerance=delta_tolerance,
                max_position_age_hours=max_position_age_hours,
            )

        self.logger = logger
        self.regime_classifier = regime_classifier

        # Daily P&L tracking
        self.daily_starting_value: Optional[float] = None
        self.last_reset_date: Optional[datetime] = None
        self.daily_pnl: float = 0
        self.daily_high_water: float = 0

        # Position tracking
        self.position_entry_times: Dict[Symbol, datetime] = {}

        # Risk event history
        self.risk_events: List[RiskEvent] = []

        # Greeks cache
        self.greeks_cache: Dict[Symbol, Dict] = {}
        self.greeks_cache_time: Optional[datetime] = None

        # Risk metrics
        self.risk_metrics = {
            "daily_pnl": 0,
            "daily_pnl_pct": 0,
            "max_drawdown": 0,
            "vega_breaches": 0,
            "delta_breaches": 0,
            "gamma_breaches": 0,
            "loss_limit_hits": 0,
            "scenario_breaches": 0,
            "positions_closed": 0,
            "risk_events_today": 0,
        }

        # Block new risk flag
        self.block_new_risk = False

    def ManageRisk(
        self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]
    ) -> List[PortfolioTarget]:
        """
        Main risk management entry point.

        Checks all risk rules and generates liquidation targets as needed.

        Args:
            algorithm: QuantConnect algorithm instance
            targets: Current portfolio targets from portfolio construction

        Returns:
            List of risk management targets (may include liquidations)
        """
        risk_targets = []

        # Update daily tracking
        self._update_daily_tracking(algorithm)

        # Clear stale Greeks cache
        if self.greeks_cache_time != algorithm.Time:
            self.greeks_cache.clear()
            self.greeks_cache_time = algorithm.Time

        # Get current regime
        regime = self._get_current_regime()

        # 1. Check daily loss limit (highest priority)
        if self._check_daily_loss_limit(algorithm):
            self._log_risk_event(
                algorithm,
                "DAILY_LOSS_BREACH",
                "DailyLossLimit",
                {
                    "daily_pnl": self.daily_pnl,
                    "daily_pnl_pct": self.risk_metrics["daily_pnl_pct"],
                    "limit": self.config.max_daily_loss,
                },
                RiskAction.LIQUIDATE_ALL,
            )

            if self.config.hard_liquidate_on_breach:
                risk_targets.extend(self._liquidate_all_positions(algorithm))
                return risk_targets

        # 2. Check scenario risk
        scenario_results = self._run_scenario_analysis(algorithm)
        scenario_targets = self._check_scenario_limits(algorithm, scenario_results)
        if scenario_targets:
            risk_targets.extend(scenario_targets)

        # 3. Check Greek limits (regime-adjusted)
        greek_targets = self._check_greek_limits(algorithm, regime)
        if greek_targets:
            risk_targets.extend(greek_targets)

        # 4. Check position age
        age_targets = self._check_position_age(algorithm)
        if age_targets:
            risk_targets.extend(age_targets)

        # 5. Check individual position risk
        position_targets = self._check_position_risk(algorithm)
        if position_targets:
            risk_targets.extend(position_targets)

        # 6. Filter out blocked targets if risk flag is set
        if self.block_new_risk:
            risk_targets = self._filter_new_risk(targets, risk_targets)

        return risk_targets

    def _update_daily_tracking(self, algorithm: QCAlgorithm) -> None:
        """Update daily P&L tracking with new day reset."""
        current_date = algorithm.Time.date()
        current_value = algorithm.Portfolio.TotalPortfolioValue

        # Reset on new day
        if self.last_reset_date != current_date:
            self.daily_starting_value = current_value
            self.daily_high_water = current_value
            self.last_reset_date = current_date
            self.risk_metrics["risk_events_today"] = 0
            self.block_new_risk = False

            if self.logger:
                self.logger.log(
                    f"Daily reset | Starting NAV: ${current_value:,.2f}", LogLevel.INFO
                )

        # Update P&L
        if self.daily_starting_value:
            self.daily_pnl = current_value - self.daily_starting_value
            self.risk_metrics["daily_pnl"] = self.daily_pnl
            self.risk_metrics["daily_pnl_pct"] = (
                self.daily_pnl / self.daily_starting_value
            )

            # Track high water for intraday drawdown
            if current_value > self.daily_high_water:
                self.daily_high_water = current_value

            intraday_drawdown = (
                self.daily_high_water - current_value
            ) / self.daily_high_water
            self.risk_metrics["max_drawdown"] = max(
                self.risk_metrics["max_drawdown"], intraday_drawdown
            )

    def _get_current_regime(self) -> Optional[str]:
        """Get current volatility regime."""
        if self.regime_classifier is None:
            return "NORMAL"

        snapshot = self.regime_classifier.get_current_regime()
        return snapshot.regime.value if snapshot else "NORMAL"

    def _check_daily_loss_limit(self, algorithm: QCAlgorithm) -> bool:
        """
        Check if daily loss limit has been breached.

        Returns:
            True if limit breached and liquidation needed
        """
        if self.daily_starting_value is None or self.daily_starting_value <= 0:
            return False

        current_value = algorithm.Portfolio.TotalPortfolioValue
        pnl_pct = (
            current_value - self.daily_starting_value
        ) / self.daily_starting_value

        # Check breach
        if pnl_pct < -self.config.max_daily_loss:
            self.risk_metrics["loss_limit_hits"] += 1
            self.block_new_risk = True

            if self.logger:
                self.logger.log(
                    f"DAILY LOSS LIMIT BREACHED | "
                    f"P&L: {pnl_pct:.2%} | Limit: {-self.config.max_daily_loss:.2%} | "
                    f"Liquidating all positions",
                    LogLevel.ERROR,
                )

            return True

        # Warning level
        if pnl_pct < -self.config.warning_daily_loss:
            if self.logger:
                self.logger.log(
                    f"Daily loss warning | P&L: {pnl_pct:.2%} | "
                    f"Warning: {-self.config.warning_daily_loss:.2%}",
                    LogLevel.WARNING,
                )

        return False

    def _run_scenario_analysis(self, algorithm: QCAlgorithm) -> List[ScenarioResult]:
        """
        Run all configured stress scenarios.

        Returns:
            List of ScenarioResult objects
        """
        results = []
        portfolio_value = algorithm.Portfolio.TotalPortfolioValue

        # Calculate current portfolio Greeks
        greeks = self._calculate_portfolio_greeks(algorithm)

        for scenario in self.config.scenarios:
            # Estimate P&L from scenario
            # Delta P&L: delta * underlying_move * underlying_price
            # This is simplified - assumes one underlying
            underlying_exposure = 0
            for holding in algorithm.Portfolio.Values:
                if holding.Symbol.SecurityType in [
                    SecurityType.Equity,
                    SecurityType.Index,
                ]:
                    underlying_exposure = holding.HoldingsValue
                    break

            # Approximate underlying price for delta contribution
            delta_pnl = greeks["delta"] * scenario.underlying_move * 100  # Simplified

            # Gamma P&L: 0.5 * gamma * (move)^2 * underlying_price^2
            gamma_pnl = 0.5 * greeks["gamma"] * (scenario.underlying_move**2) * 10000

            # Vega P&L: vega * iv_change (vega is per 1% move)
            vega_pnl = greeks["vega"] * scenario.iv_change

            # Total estimated P&L
            total_pnl = delta_pnl + gamma_pnl + vega_pnl
            pnl_pct = total_pnl / portfolio_value if portfolio_value > 0 else 0

            result = ScenarioResult(
                scenario=scenario,
                estimated_pnl=total_pnl,
                pnl_pct_nav=pnl_pct,
                breaches_limit=pnl_pct < -self.config.max_scenario_loss_pct,
                delta_contribution=delta_pnl,
                gamma_contribution=gamma_pnl,
                vega_contribution=vega_pnl,
            )
            results.append(result)

        return results

    def _check_scenario_limits(
        self, algorithm: QCAlgorithm, scenario_results: List[ScenarioResult]
    ) -> List[PortfolioTarget]:
        """
        Check scenario results against limits and generate actions.

        Args:
            algorithm: Algorithm instance
            scenario_results: List of scenario stress test results

        Returns:
            List of targets to reduce risk
        """
        targets = []

        for result in scenario_results:
            if result.breaches_limit:
                self.risk_metrics["scenario_breaches"] += 1

                self._log_risk_event(
                    algorithm,
                    "SCENARIO_BREACH",
                    f"Scenario_{result.scenario.name}",
                    {
                        "scenario": result.scenario.name,
                        "estimated_pnl": result.estimated_pnl,
                        "pnl_pct": result.pnl_pct_nav,
                        "delta_contrib": result.delta_contribution,
                        "vega_contrib": result.vega_contribution,
                        "gamma_contrib": result.gamma_contribution,
                        "limit": self.config.max_scenario_loss_pct,
                    },
                    RiskAction.SCALE_DOWN,
                )

                if self.logger:
                    self.logger.log(
                        f"SCENARIO BREACH: {result.scenario.name} | "
                        f"Est. P&L: ${result.estimated_pnl:,.0f} ({result.pnl_pct_nav:.2%}) | "
                        f"Limit: {-self.config.max_scenario_loss_pct:.2%}",
                        LogLevel.WARNING,
                    )

                # Scale down vega exposure (primary driver usually)
                if abs(result.vega_contribution) > abs(result.delta_contribution):
                    vega_targets = self._reduce_vega_exposure(algorithm, scale=0.7)
                    targets.extend(vega_targets)
                else:
                    # Reduce delta/gamma via position reduction
                    gamma_targets = self._reduce_gamma_exposure(algorithm, scale=0.7)
                    targets.extend(gamma_targets)

        return targets

    def _check_greek_limits(
        self, algorithm: QCAlgorithm, regime: str
    ) -> List[PortfolioTarget]:
        """
        Check Greek limits with regime adjustment.

        Args:
            algorithm: Algorithm instance
            regime: Current volatility regime

        Returns:
            List of targets to enforce limits
        """
        targets = []
        greeks = self._calculate_portfolio_greeks(algorithm)

        # Get regime-adjusted limits
        delta_mult = self.config.regime_delta_multiplier.get(regime, 1.0)
        vega_mult = self.config.regime_vega_multiplier.get(regime, 1.0)

        adjusted_delta_tol = self.config.delta_tolerance * delta_mult
        adjusted_vega_limit = self.config.vega_limit * vega_mult
        adjusted_gamma_limit = self.config.gamma_limit * delta_mult

        # Check vega
        if abs(greeks["vega"]) > adjusted_vega_limit:
            self.risk_metrics["vega_breaches"] += 1

            self._log_risk_event(
                algorithm,
                "VEGA_BREACH",
                "VegaLimit",
                {
                    "current_vega": greeks["vega"],
                    "limit": adjusted_vega_limit,
                    "base_limit": self.config.vega_limit,
                    "regime": regime,
                    "multiplier": vega_mult,
                },
                RiskAction.SCALE_DOWN,
            )

            if self.logger:
                self.logger.log(
                    f"VEGA LIMIT BREACH | Current: {greeks['vega']:.0f} | "
                    f"Limit: {adjusted_vega_limit:.0f} | Regime: {regime}",
                    LogLevel.WARNING,
                )

            vega_targets = self._reduce_vega_exposure(algorithm)
            targets.extend(vega_targets)

        # Check delta
        if abs(greeks["delta"]) > adjusted_delta_tol:
            self.risk_metrics["delta_breaches"] += 1

            self._log_risk_event(
                algorithm,
                "DELTA_BREACH",
                "DeltaTolerance",
                {
                    "current_delta": greeks["delta"],
                    "limit": adjusted_delta_tol,
                    "regime": regime,
                },
                RiskAction.SCALE_DOWN,
            )

            if self.logger:
                self.logger.log(
                    f"DELTA TOLERANCE BREACH | Current: {greeks['delta']:.0f} | "
                    f"Limit: {adjusted_delta_tol:.0f}",
                    LogLevel.WARNING,
                )

            delta_target = self._create_delta_hedge(algorithm, greeks["delta"])
            if delta_target:
                targets.append(delta_target)

        # Check gamma
        if abs(greeks["gamma"]) > adjusted_gamma_limit:
            self.risk_metrics["gamma_breaches"] += 1

            if self.logger:
                self.logger.log(
                    f"GAMMA LIMIT BREACH | Current: {greeks['gamma']:.1f} | "
                    f"Limit: {adjusted_gamma_limit:.1f}",
                    LogLevel.WARNING,
                )

            gamma_targets = self._reduce_gamma_exposure(algorithm)
            targets.extend(gamma_targets)

        return targets

    def _calculate_portfolio_greeks(self, algorithm: QCAlgorithm) -> Dict:
        """
        Calculate current portfolio Greeks using Lean Greeks where available.

        Returns:
            Dictionary with delta, vega, gamma, theta
        """
        greeks = {"delta": 0, "vega": 0, "gamma": 0, "theta": 0}

        for holding in algorithm.Portfolio.Values:
            if holding.Quantity == 0:
                continue

            symbol = holding.Symbol

            if symbol.SecurityType == SecurityType.Option:
                option_greeks = self._get_option_greeks(algorithm, symbol)
                if option_greeks:
                    multiplier = 100
                    qty = holding.Quantity

                    greeks["delta"] += option_greeks["delta"] * qty * multiplier
                    greeks["vega"] += option_greeks["vega"] * qty * multiplier
                    greeks["gamma"] += option_greeks["gamma"] * qty * multiplier
                    greeks["theta"] += option_greeks["theta"] * qty * multiplier
            else:
                # Equity/Index has delta = quantity
                greeks["delta"] += holding.Quantity

        return greeks

    def _get_option_greeks(
        self, algorithm: QCAlgorithm, option_symbol: Symbol
    ) -> Optional[Dict]:
        """
        Get option Greeks, preferring Lean's built-in values.

        Args:
            algorithm: Algorithm instance
            option_symbol: Option symbol

        Returns:
            Dictionary with Greeks or None
        """
        # Check cache
        if option_symbol in self.greeks_cache:
            return self.greeks_cache[option_symbol]

        security = algorithm.Securities.get(option_symbol)
        if not security:
            return None

        greeks = {}

        try:
            # Try Lean Greeks first
            if hasattr(security, "Greeks") and security.Greeks is not None:
                lean_greeks = security.Greeks

                if hasattr(lean_greeks, "Delta"):
                    greeks["delta"] = lean_greeks.Delta
                    greeks["vega"] = getattr(lean_greeks, "Vega", 0)
                    greeks["gamma"] = getattr(lean_greeks, "Gamma", 0)
                    greeks["theta"] = getattr(lean_greeks, "Theta", 0)

                    self.greeks_cache[option_symbol] = greeks
                    return greeks

            # Fallback calculation
            greeks = self._calculate_fallback_greeks(algorithm, option_symbol, security)

        except Exception:
            greeks = self._calculate_fallback_greeks(algorithm, option_symbol, security)

        if greeks:
            self.greeks_cache[option_symbol] = greeks

        return greeks

    def _calculate_fallback_greeks(
        self, algorithm: QCAlgorithm, option_symbol: Symbol, security
    ) -> Optional[Dict]:
        """Calculate approximate Greeks when Lean Greeks unavailable."""
        try:
            underlying = algorithm.Securities.get(option_symbol.Underlying)
            if not underlying or underlying.Price <= 0:
                return None

            underlying_price = underlying.Price
            strike = option_symbol.ID.StrikePrice
            tte = max(0.001, (option_symbol.ID.Date - algorithm.Time).days / 365.25)

            moneyness = underlying_price / strike
            iv = getattr(security, "ImpliedVolatility", 0.25) or 0.25

            # Approximate delta
            if option_symbol.ID.OptionRight == OptionRight.Call:
                delta = 0.5 + 0.5 * np.clip((moneyness - 1) * 5, -1, 1)
            else:
                delta = -0.5 + 0.5 * np.clip((moneyness - 1) * 5, -1, 1)

            # Vega
            atm_factor = np.exp(-8 * (moneyness - 1) ** 2)
            vega = underlying_price * np.sqrt(tte) * atm_factor * 0.01 * 0.4

            # Gamma
            gamma = (
                atm_factor / (underlying_price * iv * np.sqrt(tte)) * 0.01
                if tte > 0.01
                else 0
            )

            # Theta
            theta = -vega * iv / (2 * np.sqrt(tte)) if tte > 0.01 else 0

            return {"delta": delta, "vega": vega, "gamma": gamma, "theta": theta}

        except Exception:
            return None

    def _reduce_vega_exposure(
        self, algorithm: QCAlgorithm, scale: float = 0.5
    ) -> List[PortfolioTarget]:
        """
        Reduce vega exposure by scaling down positions.

        Args:
            algorithm: Algorithm instance
            scale: Scale factor for reduction (0.5 = reduce by 50%)

        Returns:
            List of position reduction targets
        """
        targets = []

        # Collect positions with vega contribution
        vega_positions = []
        for holding in algorithm.Portfolio.Values:
            if (
                holding.Symbol.SecurityType == SecurityType.Option
                and holding.Quantity != 0
            ):
                greeks = self._get_option_greeks(algorithm, holding.Symbol)
                if greeks:
                    vega_contrib = greeks["vega"] * holding.Quantity * 100
                    vega_positions.append(
                        (holding.Symbol, holding.Quantity, vega_contrib)
                    )

        # Sort by absolute vega
        vega_positions.sort(key=lambda x: abs(x[2]), reverse=True)

        # Reduce top contributors
        for symbol, quantity, vega_contrib in vega_positions[:5]:  # Top 5
            new_quantity = int(quantity * scale)

            if new_quantity != quantity:
                targets.append(PortfolioTarget(symbol, new_quantity))
                self.risk_metrics["positions_closed"] += 1

                if self.logger:
                    self.logger.log(
                        f"Reducing vega: {symbol} {quantity} -> {new_quantity}",
                        LogLevel.INFO,
                    )

        return targets

    def _reduce_gamma_exposure(
        self, algorithm: QCAlgorithm, scale: float = 0.5
    ) -> List[PortfolioTarget]:
        """
        Reduce gamma exposure by scaling down short-dated positions.

        Args:
            algorithm: Algorithm instance
            scale: Scale factor

        Returns:
            List of position reduction targets
        """
        targets = []

        # Collect positions with gamma contribution
        gamma_positions = []
        for holding in algorithm.Portfolio.Values:
            if (
                holding.Symbol.SecurityType == SecurityType.Option
                and holding.Quantity != 0
            ):
                greeks = self._get_option_greeks(algorithm, holding.Symbol)
                if greeks:
                    gamma_contrib = greeks["gamma"] * holding.Quantity * 100
                    gamma_positions.append(
                        (holding.Symbol, holding.Quantity, gamma_contrib)
                    )

        # Sort by absolute gamma
        gamma_positions.sort(key=lambda x: abs(x[2]), reverse=True)

        # Reduce top contributors
        for symbol, quantity, gamma_contrib in gamma_positions[:3]:
            new_quantity = int(quantity * scale)

            if new_quantity != quantity:
                targets.append(PortfolioTarget(symbol, new_quantity))

        return targets

    def _create_delta_hedge(
        self, algorithm: QCAlgorithm, current_delta: float
    ) -> Optional[PortfolioTarget]:
        """Create delta hedge using underlying."""
        # Find underlying
        underlyings = set()
        for holding in algorithm.Portfolio.Values:
            if (
                holding.Symbol.SecurityType == SecurityType.Option
                and holding.Quantity != 0
            ):
                underlyings.add(holding.Symbol.Underlying)

        if not underlyings:
            return None

        underlying = list(underlyings)[0]
        hedge_shares = -int(round(current_delta))

        if abs(hedge_shares) < 1:
            return None

        if self.logger:
            self.logger.log(
                f"Delta hedge: {underlying} shares={hedge_shares} (current delta={current_delta:.0f})",
                LogLevel.INFO,
            )

        return PortfolioTarget(underlying, hedge_shares)

    def _check_position_age(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """Check and close aged positions."""
        targets = []
        current_time = algorithm.Time

        for holding in algorithm.Portfolio.Values:
            if holding.Quantity == 0:
                continue

            symbol = holding.Symbol

            # Track entry time
            if symbol not in self.position_entry_times:
                self.position_entry_times[symbol] = current_time
                continue

            age_hours = (
                current_time - self.position_entry_times[symbol]
            ).total_seconds() / 3600

            if age_hours > self.config.max_position_age_hours:
                targets.append(PortfolioTarget(symbol, 0))
                self.risk_metrics["positions_closed"] += 1

                self._log_risk_event(
                    algorithm,
                    "POSITION_AGE",
                    "MaxPositionAge",
                    {
                        "symbol": str(symbol),
                        "age_hours": age_hours,
                        "limit": self.config.max_position_age_hours,
                    },
                    RiskAction.CLOSE_POSITION,
                    [symbol],
                )

                if self.logger:
                    self.logger.log(
                        f"Closing aged position: {symbol} | Age: {age_hours:.1f}h | "
                        f"Limit: {self.config.max_position_age_hours}h",
                        LogLevel.INFO,
                    )

        return targets

    def _check_position_risk(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """Check individual position concentration and loss limits."""
        targets = []
        portfolio_value = algorithm.Portfolio.TotalPortfolioValue

        for holding in algorithm.Portfolio.Values:
            if holding.Quantity == 0:
                continue

            symbol = holding.Symbol
            position_value = abs(holding.HoldingsValue)
            position_pct = (
                position_value / portfolio_value if portfolio_value > 0 else 0
            )

            # Concentration check
            if position_pct > self.config.max_position_pct:
                targets.append(PortfolioTarget(symbol, 0))
                self.risk_metrics["positions_closed"] += 1

                self._log_risk_event(
                    algorithm,
                    "CONCENTRATION",
                    "PositionConcentration",
                    {
                        "symbol": str(symbol),
                        "position_pct": position_pct,
                        "limit": self.config.max_position_pct,
                    },
                    RiskAction.CLOSE_POSITION,
                    [symbol],
                )

                if self.logger:
                    self.logger.log(
                        f"Closing concentrated position: {symbol} | "
                        f"Size: {position_pct:.1%} | Limit: {self.config.max_position_pct:.1%}",
                        LogLevel.WARNING,
                    )
                continue

            # Loss check
            unrealized_loss_pct = (
                -holding.UnrealizedProfitPercent
                if holding.UnrealizedProfitPercent < 0
                else 0
            )

            if unrealized_loss_pct > self.config.max_unrealized_loss_pct:
                targets.append(PortfolioTarget(symbol, 0))
                self.risk_metrics["positions_closed"] += 1

                self._log_risk_event(
                    algorithm,
                    "UNREALIZED_LOSS",
                    "MaxUnrealizedLoss",
                    {
                        "symbol": str(symbol),
                        "loss_pct": unrealized_loss_pct,
                        "limit": self.config.max_unrealized_loss_pct,
                    },
                    RiskAction.CLOSE_POSITION,
                    [symbol],
                )

                if self.logger:
                    self.logger.log(
                        f"Closing losing position: {symbol} | "
                        f"Loss: {unrealized_loss_pct:.1%} | Limit: {self.config.max_unrealized_loss_pct:.1%}",
                        LogLevel.WARNING,
                    )

        return targets

    def _liquidate_all_positions(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """Generate targets to liquidate all positions."""
        targets = []

        for holding in algorithm.Portfolio.Values:
            if holding.Quantity != 0:
                targets.append(PortfolioTarget(holding.Symbol, 0))
                self.risk_metrics["positions_closed"] += 1

        # Also call direct liquidation for immediate effect
        if self.config.hard_liquidate_on_breach:
            algorithm.Liquidate()

        return targets

    def _filter_new_risk(
        self,
        original_targets: List[PortfolioTarget],
        risk_targets: List[PortfolioTarget],
    ) -> List[PortfolioTarget]:
        """Filter out targets that would add new risk when blocked."""
        # Only allow risk-reducing targets
        reducing_targets = []

        for target in risk_targets:
            # Always allow closing positions
            if target.Quantity == 0:
                reducing_targets.append(target)

        return reducing_targets

    def _log_risk_event(
        self,
        algorithm: QCAlgorithm,
        event_type: str,
        rule_name: str,
        details: Dict,
        action: RiskAction,
        positions: List[Symbol] = None,
    ):
        """Log a risk event for audit trail."""
        event = RiskEvent(
            timestamp=algorithm.Time,
            event_type=event_type,
            rule_name=rule_name,
            details=details,
            action_taken=action,
            positions_affected=positions or [],
        )

        self.risk_events.append(event)
        self.risk_metrics["risk_events_today"] += 1

        # Keep last 1000 events
        if len(self.risk_events) > 1000:
            self.risk_events = self.risk_events[-1000:]

        if self.logger and self.config.log_all_risk_events:
            self.logger.log_risk_metrics(
                {
                    "event": event_type,
                    "rule": rule_name,
                    "action": action.value,
                    **details,
                }
            )

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes) -> None:
        """Handle security changes."""
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]
            if symbol in self.greeks_cache:
                del self.greeks_cache[symbol]

    def GetMetrics(self) -> Dict:
        """Get risk management metrics."""
        return self.risk_metrics.copy()

    def get_risk_events(self, last_n: int = None) -> List[RiskEvent]:
        """Get risk event history."""
        if last_n:
            return self.risk_events[-last_n:]
        return self.risk_events.copy()

    def get_scenario_summary(self, algorithm: QCAlgorithm) -> Dict:
        """Get summary of scenario analysis."""
        results = self._run_scenario_analysis(algorithm)

        return {
            "scenarios": [
                {
                    "name": r.scenario.name,
                    "description": r.scenario.description,
                    "estimated_pnl": r.estimated_pnl,
                    "pnl_pct": r.pnl_pct_nav,
                    "breaches": r.breaches_limit,
                }
                for r in results
            ],
            "worst_scenario": (
                min(results, key=lambda x: x.estimated_pnl).scenario.name
                if results
                else None
            ),
            "any_breach": any(r.breaches_limit for r in results),
        }
