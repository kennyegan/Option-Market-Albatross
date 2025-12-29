"""
Exposure Limits Risk Management Model
Enforces risk limits including daily loss, vega exposure, and delta neutrality.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from typing import List, Dict
import numpy as np
from datetime import datetime, timedelta


class ExposureLimitsRiskManagementModel(RiskManagementModel):
    """
    Risk management model that:
    1. Enforces maximum daily loss limits
    2. Monitors and caps vega exposure
    3. Maintains delta neutrality
    4. Implements time-based exits
    """
    
    def __init__(self,
                 max_daily_loss: float = 0.02,
                 vega_limit: float = 10000,
                 delta_tolerance: float = 100,
                 max_position_age_hours: int = 24,
                 logger = None):
        """
        Initialize the risk management model.
        
        Args:
            max_daily_loss: Maximum daily loss as fraction of NAV
            vega_limit: Maximum portfolio vega exposure
            delta_tolerance: Maximum acceptable delta deviation
            max_position_age_hours: Maximum hours to hold a position
            logger: Strategy logger instance
        """
        self.max_daily_loss = max_daily_loss
        self.vega_limit = vega_limit
        self.delta_tolerance = delta_tolerance
        self.max_position_age_hours = max_position_age_hours
        self.logger = logger
        
        # Track daily P&L
        self.daily_starting_value = None
        self.last_reset_date = None
        
        # Track position entry times
        self.position_entry_times = {}
        
        # Risk metrics
        self.risk_metrics = {
            'daily_pnl': 0,
            'max_drawdown': 0,
            'vega_breaches': 0,
            'delta_breaches': 0,
            'loss_limit_hits': 0
        }
    
    def ManageRisk(self, algorithm: QCAlgorithm, targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        """
        Manage portfolio risk and generate liquidation targets if needed.
        
        Args:
            algorithm: Algorithm instance
            targets: Current portfolio targets
            
        Returns:
            List of risk management targets
        """
        risk_targets = []
        
        # Update daily P&L tracking
        self._update_daily_pnl(algorithm)
        
        # Check daily loss limit
        if self._check_daily_loss_limit(algorithm):
            if self.logger:
                self.logger.log("Daily loss limit breached - liquidating all positions", LogLevel.ERROR)
            risk_targets.extend(self._liquidate_all_positions(algorithm))
            return risk_targets
        
        # Check Greek exposure limits
        greek_targets = self._check_greek_limits(algorithm)
        if greek_targets:
            risk_targets.extend(greek_targets)
        
        # Check position age
        age_targets = self._check_position_age(algorithm)
        if age_targets:
            risk_targets.extend(age_targets)
        
        # Check individual position risk
        position_targets = self._check_position_risk(algorithm)
        if position_targets:
            risk_targets.extend(position_targets)
        
        return risk_targets
    
    def _update_daily_pnl(self, algorithm: QCAlgorithm) -> None:
        """
        Update daily P&L tracking.
        
        Args:
            algorithm: Algorithm instance
        """
        current_date = algorithm.Time.date()
        
        # Reset daily tracking if new day
        if self.last_reset_date != current_date:
            self.daily_starting_value = algorithm.Portfolio.TotalPortfolioValue
            self.last_reset_date = current_date
            self.risk_metrics['daily_pnl'] = 0
            
            if self.logger:
                self.logger.log(f"Reset daily P&L tracking. Starting value: ${self.daily_starting_value:,.2f}", LogLevel.INFO)
    
    def _check_daily_loss_limit(self, algorithm: QCAlgorithm) -> bool:
        """
        Check if daily loss limit has been breached.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            True if limit breached
        """
        if self.daily_starting_value is None:
            return False
        
        current_value = algorithm.Portfolio.TotalPortfolioValue
        daily_pnl = (current_value - self.daily_starting_value) / self.daily_starting_value
        self.risk_metrics['daily_pnl'] = daily_pnl
        
        if daily_pnl < -self.max_daily_loss:
            self.risk_metrics['loss_limit_hits'] += 1
            
            if self.logger:
                self.logger.log(
                    f"Daily loss limit breached: {daily_pnl:.2%} < {-self.max_daily_loss:.2%}",
                    LogLevel.ERROR
                )
            
            return True
        
        return False
    
    def _check_greek_limits(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """
        Check Greek exposure limits and generate reduction targets if needed.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            List of targets to reduce exposure
        """
        targets = []
        
        # Calculate current Greeks
        portfolio_greeks = self._calculate_portfolio_greeks(algorithm)
        
        # Check vega limit
        if abs(portfolio_greeks['vega']) > self.vega_limit:
            self.risk_metrics['vega_breaches'] += 1
            
            if self.logger:
                self.logger.log(
                    f"Vega limit breached: {portfolio_greeks['vega']:.0f} > {self.vega_limit}",
                    LogLevel.WARNING
                )
            
            # Generate vega reduction targets
            vega_targets = self._reduce_vega_exposure(algorithm, portfolio_greeks['vega'])
            targets.extend(vega_targets)
        
        # Check delta neutrality
        if abs(portfolio_greeks['delta']) > self.delta_tolerance:
            self.risk_metrics['delta_breaches'] += 1
            
            if self.logger:
                self.logger.log(
                    f"Delta tolerance breached: {portfolio_greeks['delta']:.0f} > {self.delta_tolerance}",
                    LogLevel.WARNING
                )
            
            # Generate delta hedge
            delta_target = self._create_delta_hedge(algorithm, portfolio_greeks['delta'])
            if delta_target:
                targets.append(delta_target)
        
        return targets
    
    def _calculate_portfolio_greeks(self, algorithm: QCAlgorithm) -> Dict:
        """
        Calculate current portfolio Greeks.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            Dictionary of portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0,
            'vega': 0,
            'gamma': 0,
            'theta': 0
        }
        
        for holding in algorithm.Portfolio.Values:
            if holding.Quantity == 0:
                continue
            
            if holding.Symbol.SecurityType == SecurityType.Option:
                greeks = self._calculate_option_greeks(holding, algorithm)
                if greeks:
                    portfolio_greeks['delta'] += greeks['delta'] * holding.Quantity * 100
                    portfolio_greeks['vega'] += greeks['vega'] * holding.Quantity * 100
                    portfolio_greeks['gamma'] += greeks['gamma'] * holding.Quantity * 100
                    portfolio_greeks['theta'] += greeks['theta'] * holding.Quantity * 100
            else:
                # Underlying has delta of 1
                portfolio_greeks['delta'] += holding.Quantity
        
        return portfolio_greeks
    
    def _calculate_option_greeks(self, holding, algorithm: QCAlgorithm) -> Dict:
        """
        Calculate Greeks for an option position.
        
        Args:
            holding: Portfolio holding
            algorithm: Algorithm instance
            
        Returns:
            Dictionary of Greeks
        """
        try:
            option = holding.Symbol
            underlying_price = algorithm.Securities[option.Underlying].Price
            strike = option.ID.StrikePrice
            
            # Calculate moneyness
            moneyness = underlying_price / strike
            
            # Time to expiry
            tte = (option.ID.Date - algorithm.Time).days / 365.25
            
            # Simplified Greek calculations
            if option.ID.OptionRight == OptionRight.Call:
                delta = self._approximate_call_delta(moneyness, tte)
            else:
                delta = self._approximate_put_delta(moneyness, tte)
            
            vega = self._approximate_vega(moneyness, tte)
            gamma = self._approximate_gamma(moneyness, tte)
            theta = -vega * 0.01 / tte if tte > 0 else 0
            
            return {
                'delta': delta,
                'vega': vega,
                'gamma': gamma,
                'theta': theta
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error calculating Greeks: {e}", LogLevel.ERROR)
            return None
    
    def _approximate_call_delta(self, moneyness: float, tte: float) -> float:
        """Approximate call delta."""
        if moneyness > 1.1:
            return min(0.9, 0.5 + (moneyness - 1) * 2)
        elif moneyness < 0.9:
            return max(0.1, 0.5 - (1 - moneyness) * 2)
        else:
            return 0.5
    
    def _approximate_put_delta(self, moneyness: float, tte: float) -> float:
        """Approximate put delta."""
        return self._approximate_call_delta(moneyness, tte) - 1
    
    def _approximate_vega(self, moneyness: float, tte: float) -> float:
        """Approximate vega."""
        atm_factor = np.exp(-8 * (moneyness - 1) ** 2)
        time_factor = np.sqrt(tte) if tte > 0 else 0
        return 0.4 * atm_factor * time_factor
    
    def _approximate_gamma(self, moneyness: float, tte: float) -> float:
        """Approximate gamma."""
        atm_factor = np.exp(-8 * (moneyness - 1) ** 2)
        time_factor = 1 / np.sqrt(tte) if tte > 0.01 else 10
        return 0.05 * atm_factor * time_factor
    
    def _reduce_vega_exposure(self, algorithm: QCAlgorithm, current_vega: float) -> List[PortfolioTarget]:
        """
        Generate targets to reduce vega exposure.
        
        Args:
            algorithm: Algorithm instance
            current_vega: Current portfolio vega
            
        Returns:
            List of reduction targets
        """
        targets = []
        target_reduction = abs(current_vega) - self.vega_limit * 0.8  # Reduce to 80% of limit
        
        # Sort positions by vega contribution
        vega_positions = []
        
        for holding in algorithm.Portfolio.Values:
            if holding.Symbol.SecurityType == SecurityType.Option and holding.Quantity != 0:
                greeks = self._calculate_option_greeks(holding, algorithm)
                if greeks:
                    vega_contribution = greeks['vega'] * holding.Quantity * 100
                    vega_positions.append((holding.Symbol, holding.Quantity, vega_contribution))
        
        # Sort by absolute vega contribution
        vega_positions.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Reduce positions starting with highest vega
        reduced_vega = 0
        for symbol, quantity, vega_contrib in vega_positions:
            if reduced_vega >= target_reduction:
                break
            
            # Reduce position by 50%
            new_quantity = int(quantity * 0.5)
            reduction = quantity - new_quantity
            
            if reduction != 0:
                targets.append(PortfolioTarget(symbol, new_quantity))
                reduced_vega += abs(vega_contrib * 0.5)
                
                if self.logger:
                    self.logger.log(
                        f"Reducing vega exposure: {symbol} from {quantity} to {new_quantity}",
                        LogLevel.INFO
                    )
        
        return targets
    
    def _create_delta_hedge(self, algorithm: QCAlgorithm, current_delta: float) -> PortfolioTarget:
        """
        Create delta hedge target.
        
        Args:
            algorithm: Algorithm instance
            current_delta: Current portfolio delta
            
        Returns:
            Delta hedge target or None
        """
        # Find an underlying to hedge with
        underlyings = set()
        for holding in algorithm.Portfolio.Values:
            if holding.Symbol.SecurityType == SecurityType.Option and holding.Quantity != 0:
                underlyings.add(holding.Symbol.Underlying)
        
        if not underlyings:
            return None
        
        # Use first underlying (could be improved)
        underlying = list(underlyings)[0]
        hedge_shares = -int(round(current_delta))
        
        if abs(hedge_shares) < 1:
            return None
        
        if self.logger:
            self.logger.log(
                f"Creating delta hedge: {underlying} shares={hedge_shares}",
                LogLevel.INFO
            )
        
        return PortfolioTarget(underlying, hedge_shares)
    
    def _check_position_age(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """
        Check position age and close old positions.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            List of targets to close old positions
        """
        targets = []
        current_time = algorithm.Time
        
        for holding in algorithm.Portfolio.Values:
            if holding.Quantity == 0:
                continue
            
            # Check if we have entry time
            if holding.Symbol not in self.position_entry_times:
                self.position_entry_times[holding.Symbol] = current_time
                continue
            
            # Calculate position age
            position_age = current_time - self.position_entry_times[holding.Symbol]
            
            if position_age.total_seconds() > self.max_position_age_hours * 3600:
                targets.append(PortfolioTarget(holding.Symbol, 0))
                
                if self.logger:
                    self.logger.log(
                        f"Closing aged position: {holding.Symbol} "
                        f"Age={position_age.total_seconds()/3600:.1f} hours",
                        LogLevel.INFO
                    )
        
        return targets
    
    def _check_position_risk(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """
        Check individual position risk.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            List of targets to manage position risk
        """
        targets = []
        portfolio_value = algorithm.Portfolio.TotalPortfolioValue
        
        for holding in algorithm.Portfolio.Values:
            if holding.Quantity == 0:
                continue
            
            # Check position size relative to portfolio
            position_value = abs(holding.HoldingsValue)
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Close if position too large (risk concentration)
            if position_pct > 0.1:  # 10% of portfolio
                targets.append(PortfolioTarget(holding.Symbol, 0))
                
                if self.logger:
                    self.logger.log(
                        f"Closing concentrated position: {holding.Symbol} "
                        f"Size={position_pct:.1%} of portfolio",
                        LogLevel.WARNING
                    )
            
            # Check unrealized loss
            if holding.UnrealizedProfitPercent < -0.5:  # 50% loss
                targets.append(PortfolioTarget(holding.Symbol, 0))
                
                if self.logger:
                    self.logger.log(
                        f"Closing losing position: {holding.Symbol} "
                        f"Loss={holding.UnrealizedProfitPercent:.1%}",
                        LogLevel.WARNING
                    )
        
        return targets
    
    def _liquidate_all_positions(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """
        Generate targets to liquidate all positions.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            List of liquidation targets
        """
        targets = []
        
        for holding in algorithm.Portfolio.Values:
            if holding.Quantity != 0:
                targets.append(PortfolioTarget(holding.Symbol, 0))
        
        return targets
    
    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes) -> None:
        """
        Handle security changes.
        
        Args:
            algorithm: Algorithm instance
            changes: Security changes
        """
        # Clean up removed securities
        for security in changes.RemovedSecurities:
            if security.Symbol in self.position_entry_times:
                del self.position_entry_times[security.Symbol]
    
    def GetMetrics(self) -> Dict:
        """
        Get risk management metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return self.risk_metrics.copy()