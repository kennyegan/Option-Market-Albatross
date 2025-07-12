"""
Delta-Vega Neutral Portfolio Construction Model
Builds option portfolios with controlled greek exposures.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime, timedelta


class DeltaVegaNeutralPortfolioConstructionModel(PortfolioConstructionModel):
    """
    Portfolio construction model that:
    1. Maintains delta-neutral positions
    2. Caps vega exposure
    3. Sizes positions appropriately
    4. Creates hedged option baskets
    """
    
    def __init__(self,
                 max_position_size: float = 0.05,
                 vega_limit: float = 10000,
                 delta_tolerance: float = 100,
                 rebalance_threshold: float = 0.1,
                 logger = None):
        """
        Initialize the portfolio construction model.
        
        Args:
            max_position_size: Maximum position size as fraction of NAV
            vega_limit: Maximum portfolio vega exposure
            delta_tolerance: Acceptable delta deviation from neutral
            rebalance_threshold: Threshold for triggering rebalance (as % of limits)
            logger: Strategy logger instance
        """
        self.max_position_size = max_position_size
        self.vega_limit = vega_limit
        self.delta_tolerance = delta_tolerance
        self.rebalance_threshold = rebalance_threshold
        self.logger = logger
        
        # Track portfolio Greeks
        self.portfolio_greeks = {
            'delta': 0,
            'vega': 0,
            'gamma': 0,
            'theta': 0
        }
        
        # Track positions for efficient rebalancing
        self.active_positions = {}
        
    def CreateTargets(self, algorithm: QCAlgorithm, insights: List[Insight]) -> List[PortfolioTarget]:
        """
        Create portfolio targets from alpha insights.
        
        Args:
            algorithm: Algorithm instance
            insights: List of alpha insights
            
        Returns:
            List of portfolio targets
        """
        targets = []
        
        if not insights:
            # Check if rebalancing needed
            if self._needs_rebalancing(algorithm):
                targets = self._rebalance_portfolio(algorithm)
            return targets
        
        # Log insight processing
        if self.logger:
            self.logger.log(f"Processing {len(insights)} insights for portfolio construction", LogLevel.INFO)
        
        # Group insights by underlying
        insights_by_underlying = self._group_insights_by_underlying(insights)
        
        # Process each underlying's insights
        for underlying, underlying_insights in insights_by_underlying.items():
            basket_targets = self._create_option_basket(algorithm, underlying, underlying_insights)
            targets.extend(basket_targets)
        
        # Add hedging targets if needed
        hedge_targets = self._create_hedge_targets(algorithm, targets)
        targets.extend(hedge_targets)
        
        # Validate risk limits
        targets = self._validate_risk_limits(algorithm, targets)
        
        return targets
    
    def _group_insights_by_underlying(self, insights: List[Insight]) -> Dict:
        """
        Group insights by their underlying asset.
        
        Args:
            insights: List of insights
            
        Returns:
            Dictionary mapping underlying to its insights
        """
        grouped = {}
        
        for insight in insights:
            # Extract underlying from option symbol
            if insight.Symbol.SecurityType == SecurityType.Option:
                underlying = insight.Symbol.Underlying
            else:
                underlying = insight.Symbol
            
            if underlying not in grouped:
                grouped[underlying] = []
            grouped[underlying].append(insight)
        
        return grouped
    
    def _create_option_basket(self,
                             algorithm: QCAlgorithm,
                             underlying: Symbol,
                             insights: List[Insight]) -> List[PortfolioTarget]:
        """
        Create a delta-neutral option basket from insights.
        
        Args:
            algorithm: Algorithm instance
            underlying: Underlying symbol
            insights: Insights for this underlying
            
        Returns:
            List of portfolio targets
        """
        targets = []
        
        # Calculate portfolio value for position sizing
        portfolio_value = algorithm.Portfolio.TotalPortfolioValue
        max_position_value = portfolio_value * self.max_position_size
        
        # Sort insights by confidence
        sorted_insights = sorted(insights, key=lambda x: abs(x.Confidence), reverse=True)
        
        # Track basket Greeks
        basket_delta = 0
        basket_vega = 0
        
        for insight in sorted_insights:
            # Skip if we've reached vega limit
            if abs(basket_vega) >= self.vega_limit * 0.8:  # 80% of limit
                break
            
            # Get option contract details
            option_symbol = insight.Symbol
            security = algorithm.Securities.get(option_symbol)
            
            if not security or not security.HasData:
                continue
            
            # Calculate Greeks
            greeks = self._calculate_greeks(security, algorithm)
            if not greeks:
                continue
            
            # Determine position size
            position_size = self._calculate_position_size(
                insight,
                greeks,
                max_position_value,
                basket_vega,
                algorithm
            )
            
            if position_size == 0:
                continue
            
            # Create target
            target = PortfolioTarget(option_symbol, position_size)
            target.Insight = insight  # Attach insight for execution logic
            targets.append(target)
            
            # Update basket Greeks
            basket_delta += greeks['delta'] * position_size * 100  # Multiplier
            basket_vega += greeks['vega'] * position_size * 100
            
            if self.logger:
                self.logger.log(
                    f"Added to basket: {option_symbol} "
                    f"Size={position_size} "
                    f"Delta={greeks['delta']:.3f} "
                    f"Vega={greeks['vega']:.2f}",
                    LogLevel.DEBUG
                )
        
        # Create delta hedge if needed
        if abs(basket_delta) > self.delta_tolerance * 0.5:
            hedge_target = self._create_delta_hedge(algorithm, underlying, -basket_delta)
            if hedge_target:
                targets.append(hedge_target)
        
        return targets
    
    def _calculate_greeks(self, security, algorithm: QCAlgorithm) -> Dict:
        """
        Calculate option Greeks.
        
        Args:
            security: Option security
            algorithm: Algorithm instance
            
        Returns:
            Dictionary of Greeks or None
        """
        try:
            # In real implementation, we'd use Black-Scholes or get from data provider
            # For now, using approximations based on moneyness and time to expiry
            
            option = security.Symbol
            underlying_price = algorithm.Securities[option.Underlying].Price
            strike = option.ID.StrikePrice
            
            # Calculate moneyness
            moneyness = underlying_price / strike
            
            # Time to expiry in years
            tte = (option.ID.Date - algorithm.Time).days / 365.25
            
            # Approximate Greeks (simplified)
            if option.ID.OptionRight == OptionRight.Call:
                delta = self._approximate_call_delta(moneyness, tte)
                sign = 1
            else:
                delta = self._approximate_put_delta(moneyness, tte)
                sign = -1
            
            # Vega peaks at ATM
            vega = self._approximate_vega(moneyness, tte) * 100  # Per 1% vol move
            
            # Gamma also peaks at ATM
            gamma = self._approximate_gamma(moneyness, tte)
            
            # Theta (time decay)
            theta = -vega * 0.01 / tte if tte > 0 else 0  # Simplified
            
            return {
                'delta': delta,
                'vega': vega,
                'gamma': gamma,
                'theta': theta,
                'iv': security.ImpliedVolatility if hasattr(security, 'ImpliedVolatility') else 0.3
            }
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error calculating Greeks: {e}", LogLevel.ERROR)
            return None
    
    def _approximate_call_delta(self, moneyness: float, tte: float) -> float:
        """Approximate call delta based on moneyness and time."""
        # Simplified approximation
        if moneyness > 1.1:  # ITM
            return min(0.9, 0.5 + (moneyness - 1) * 2)
        elif moneyness < 0.9:  # OTM
            return max(0.1, 0.5 - (1 - moneyness) * 2)
        else:  # ATM
            return 0.5
    
    def _approximate_put_delta(self, moneyness: float, tte: float) -> float:
        """Approximate put delta based on moneyness and time."""
        # Put delta = Call delta - 1
        return self._approximate_call_delta(moneyness, tte) - 1
    
    def _approximate_vega(self, moneyness: float, tte: float) -> float:
        """Approximate vega based on moneyness and time."""
        # Vega peaks at ATM
        atm_factor = np.exp(-8 * (moneyness - 1) ** 2)
        time_factor = np.sqrt(tte) if tte > 0 else 0
        return 0.4 * atm_factor * time_factor
    
    def _approximate_gamma(self, moneyness: float, tte: float) -> float:
        """Approximate gamma based on moneyness and time."""
        # Gamma also peaks at ATM
        atm_factor = np.exp(-8 * (moneyness - 1) ** 2)
        time_factor = 1 / np.sqrt(tte) if tte > 0.01 else 10
        return 0.05 * atm_factor * time_factor
    
    def _calculate_position_size(self,
                                insight: Insight,
                                greeks: Dict,
                                max_position_value: float,
                                current_basket_vega: float,
                                algorithm: QCAlgorithm) -> int:
        """
        Calculate appropriate position size.
        
        Args:
            insight: Alpha insight
            greeks: Option Greeks
            max_position_value: Maximum position value
            current_basket_vega: Current basket vega exposure
            algorithm: Algorithm instance
            
        Returns:
            Number of contracts
        """
        # Get option price
        security = algorithm.Securities[insight.Symbol]
        option_price = security.Price
        
        if option_price <= 0:
            return 0
        
        # Base size on confidence and available risk budget
        confidence_factor = abs(insight.Confidence)
        
        # Calculate contract value
        contract_value = option_price * 100  # Multiplier
        
        # Maximum contracts based on position limit
        max_contracts_by_value = int(max_position_value * confidence_factor / contract_value)
        
        # Maximum contracts based on vega limit
        remaining_vega = self.vega_limit - abs(current_basket_vega)
        max_contracts_by_vega = int(remaining_vega / (abs(greeks['vega']) * 100)) if greeks['vega'] != 0 else max_contracts_by_value
        
        # Take minimum
        position_size = min(max_contracts_by_value, max_contracts_by_vega)
        
        # Apply direction based on insight
        if insight.Direction == InsightDirection.Down:
            position_size = -position_size
        
        return position_size
    
    def _create_delta_hedge(self,
                           algorithm: QCAlgorithm,
                           underlying: Symbol,
                           delta_to_hedge: float) -> PortfolioTarget:
        """
        Create delta hedge using underlying.
        
        Args:
            algorithm: Algorithm instance
            underlying: Underlying symbol
            delta_to_hedge: Delta amount to hedge
            
        Returns:
            Portfolio target for hedge or None
        """
        try:
            # Round to nearest share
            shares = int(round(delta_to_hedge))
            
            if abs(shares) < 1:
                return None
            
            if self.logger:
                self.logger.log(
                    f"Creating delta hedge: {underlying} "
                    f"Shares={shares} Delta={delta_to_hedge:.1f}",
                    LogLevel.INFO
                )
            
            return PortfolioTarget(underlying, shares)
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error creating delta hedge: {e}", LogLevel.ERROR)
            return None
    
    def _create_hedge_targets(self,
                             algorithm: QCAlgorithm,
                             targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        """
        Create additional hedging targets if needed.
        
        Args:
            algorithm: Algorithm instance
            targets: Current portfolio targets
            
        Returns:
            List of hedging targets
        """
        # This is handled in _create_option_basket for now
        return []
    
    def _validate_risk_limits(self,
                             algorithm: QCAlgorithm,
                             targets: List[PortfolioTarget]) -> List[PortfolioTarget]:
        """
        Validate targets against risk limits.
        
        Args:
            algorithm: Algorithm instance
            targets: Proposed portfolio targets
            
        Returns:
            Validated targets
        """
        # Calculate projected Greeks
        projected_delta = 0
        projected_vega = 0
        
        for target in targets:
            security = algorithm.Securities.get(target.Symbol)
            if not security:
                continue
            
            if target.Symbol.SecurityType == SecurityType.Option:
                greeks = self._calculate_greeks(security, algorithm)
                if greeks:
                    projected_delta += greeks['delta'] * target.Quantity * 100
                    projected_vega += greeks['vega'] * target.Quantity * 100
            else:
                # Underlying has delta of 1
                projected_delta += target.Quantity
        
        # Check limits
        if abs(projected_vega) > self.vega_limit:
            # Scale down all option targets
            scale_factor = self.vega_limit / abs(projected_vega) * 0.95
            
            if self.logger:
                self.logger.log(
                    f"Scaling down targets due to vega limit: "
                    f"Projected={projected_vega:.0f}, Limit={self.vega_limit}, "
                    f"Scale={scale_factor:.2f}",
                    LogLevel.WARNING
                )
            
            for target in targets:
                if target.Symbol.SecurityType == SecurityType.Option:
                    target.Quantity = int(target.Quantity * scale_factor)
        
        return targets
    
    def _needs_rebalancing(self, algorithm: QCAlgorithm) -> bool:
        """
        Check if portfolio needs rebalancing.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            True if rebalancing needed
        """
        # Calculate current portfolio Greeks
        current_greeks = self._calculate_portfolio_greeks(algorithm)
        
        # Check delta neutrality
        if abs(current_greeks['delta']) > self.delta_tolerance:
            return True
        
        # Check vega limit
        if abs(current_greeks['vega']) > self.vega_limit * (1 + self.rebalance_threshold):
            return True
        
        return False
    
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
                greeks = self._calculate_greeks(holding, algorithm)
                if greeks:
                    portfolio_greeks['delta'] += greeks['delta'] * holding.Quantity * 100
                    portfolio_greeks['vega'] += greeks['vega'] * holding.Quantity * 100
                    portfolio_greeks['gamma'] += greeks['gamma'] * holding.Quantity * 100
                    portfolio_greeks['theta'] += greeks['theta'] * holding.Quantity * 100
            else:
                # Underlying has delta of 1
                portfolio_greeks['delta'] += holding.Quantity
        
        return portfolio_greeks
    
    def _rebalance_portfolio(self, algorithm: QCAlgorithm) -> List[PortfolioTarget]:
        """
        Rebalance portfolio to maintain risk limits.
        
        Args:
            algorithm: Algorithm instance
            
        Returns:
            List of rebalancing targets
        """
        targets = []
        current_greeks = self._calculate_portfolio_greeks(algorithm)
        
        # Create delta hedge if needed
        if abs(current_greeks['delta']) > self.delta_tolerance:
            # Find appropriate underlying for hedge
            underlyings = set()
            for holding in algorithm.Portfolio.Values:
                if holding.Symbol.SecurityType == SecurityType.Option and holding.Quantity != 0:
                    underlyings.add(holding.Symbol.Underlying)
            
            if underlyings:
                # Use first underlying for hedge (could be improved)
                underlying = list(underlyings)[0]
                hedge_target = self._create_delta_hedge(algorithm, underlying, -current_greeks['delta'])
                if hedge_target:
                    targets.append(hedge_target)
        
        return targets