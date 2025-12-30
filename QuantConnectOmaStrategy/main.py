"""
QuantConnect OMA Strategy: IV/RV + Bid/Ask Arbitrage - Institutional Grade
Main algorithm entry point combining IV vs RV arbitrage with bid/ask spread capture.

Author: OMA Strategy Team
Version: 2.0 - Enhanced with regime awareness, scenario risk, vol targeting, TCA

This strategy implements:
1. IV/RV arbitrage with ensemble realized vol estimators
2. Regime-aware signal generation and risk management
3. Delta-vega neutral portfolio construction with factor buckets
4. Scenario-based stress testing
5. Smart execution with edge verification and TCA
"""

from AlgorithmImports import *
from alpha.iv_rv_spread_alpha import IVRVSpreadAlphaModel, AlphaConfig
from execution.smart_router import SmartExecutionModel, ExecutionConfig
from portfolio.delta_vega_neutral import DeltaVegaNeutralPortfolioConstructionModel, PortfolioConfig
from risk.exposure_limits import ExposureLimitsRiskManagementModel, RiskConfig, ScenarioConfig
from risk.vol_regime import VolatilityRegimeClassifier, RegimeConfig, VolatilityRegime
from data.realized_vol_calc import RealizedVolatilityCalculator
from utils.logger import StrategyLogger, LogLevel
import numpy as np
from datetime import timedelta


class OMAOptionsArbitrageAlgorithm(QCAlgorithm):
    """
    Main algorithm class for Options Market Arbitrage (OMA) strategy.
    
    This institutional-grade strategy combines:
    1. IV vs RV arbitrage signals with ensemble RV and uncertainty
    2. Bid/ask spread capture opportunities
    3. Regime-aware alpha generation (CALM/NORMAL/STRESSED/CRISIS)
    4. Delta-neutral portfolio construction with volatility targeting
    5. Factor bucket risk management (INDEX/TECH/SMALL_CAP/SINGLE_NAME)
    6. Scenario-based stress testing
    7. Smart execution with TCA tracking
    
    All components use Lean Greeks where available and fall back to
    approximations when necessary.
    """
    
    def Initialize(self):
        """Initialize the algorithm with strategy parameters and models."""
        # ================================================================
        # Basic Setup
        # ================================================================
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(1000000)
        self.SetTimeZone(TimeZones.NewYork)
        
        # ================================================================
        # Strategy Parameters (can be overridden via lean.json)
        # ================================================================
        # IV/RV Alpha parameters
        self.iv_rv_threshold = float(self.GetParameter("iv-rv-threshold") or 1.2)
        self.spread_threshold = float(self.GetParameter("spread-threshold") or 0.005)
        
        # Position sizing
        self.max_position_size = float(self.GetParameter("max-position-size") or 0.05)
        self.target_daily_vol = float(self.GetParameter("target-daily-vol") or 0.01)
        
        # Greek limits
        self.vega_limit = float(self.GetParameter("vega-limit") or 10000)
        self.delta_tolerance = float(self.GetParameter("delta-tolerance") or 100)
        self.gamma_limit = float(self.GetParameter("gamma-limit") or 500)
        
        # Risk parameters
        self.max_daily_loss = float(self.GetParameter("max-daily-loss") or 0.02)
        self.scenario_max_loss = float(self.GetParameter("scenario-max-loss") or 0.10)
        
        # Execution parameters
        self.min_edge_bps = float(self.GetParameter("min-edge-after-costs-bps") or 20)
        
        # Realized vol parameters
        self.rv_lookback = int(self.GetParameter("rv-lookback-default") or 10)
        self.ewma_decay = float(self.GetParameter("ewma-decay") or 0.94)
        
        # ================================================================
        # Initialize Logger (first, so other components can use it)
        # ================================================================
        log_level_str = self.GetParameter("log-level") or "INFO"
        log_level = getattr(LogLevel, log_level_str, LogLevel.INFO)
        
        self.logger = StrategyLogger(
            self,
            min_level=log_level,
            enable_performance_logging=True,
            risk_snapshot_interval_minutes=30
        )
        self.logger.log("="*60, LogLevel.INFO)
        self.logger.log("Initializing OMA Options Arbitrage Strategy v2.0", LogLevel.INFO)
        self.logger.log("="*60, LogLevel.INFO)
        
        # ================================================================
        # Initialize Data Components
        # ================================================================
        # Realized Volatility Calculator
        self.rv_calculator = RealizedVolatilityCalculator(
            self,
            lookback_days=self.rv_lookback,
            annualization_factor=252,
            use_log_returns=True,
            ewma_decay=self.ewma_decay,
            max_history_bars=120
        )
        self.logger.log(f"RV Calculator initialized: lookback={self.rv_lookback}d, ewma_decay={self.ewma_decay}", LogLevel.INFO)
        
        # ================================================================
        # Initialize Volatility Regime Classifier
        # ================================================================
        regime_config = RegimeConfig(
            vix_calm_upper=float(self.GetParameter("vol-regime-vix-calm") or 15),
            vix_normal_upper=float(self.GetParameter("vol-regime-vix-normal") or 20),
            vix_stressed_upper=float(self.GetParameter("vol-regime-vix-stressed") or 30),
            vix_spike_threshold=float(self.GetParameter("vol-regime-vix-spike") or 3),
            rv_calm_upper=float(self.GetParameter("vol-regime-rv-calm") or 0.12),
            rv_normal_upper=float(self.GetParameter("vol-regime-rv-normal") or 0.18),
            rv_stressed_upper=float(self.GetParameter("vol-regime-rv-stressed") or 0.30)
        )
        
        self.regime_classifier = VolatilityRegimeClassifier(
            self,
            rv_calculator=self.rv_calculator,
            config=regime_config
        )
        
        # Register regime change callback
        self.regime_classifier.on_regime_change(self._on_regime_change)
        self.logger.log("Volatility Regime Classifier initialized", LogLevel.INFO)
        
        # ================================================================
        # Initialize Universe
        # ================================================================
        self.InitializeUniverse()
        
        # ================================================================
        # Set Up Alpha Model
        # ================================================================
        alpha_config = AlphaConfig(
            iv_rv_threshold=self.iv_rv_threshold,
            iv_rv_max_signal=float(self.GetParameter("iv-rv-max-signal") or 2.0),
            spread_threshold=self.spread_threshold,
            moneyness_min=float(self.GetParameter("moneyness-min") or 0.90),
            moneyness_max=float(self.GetParameter("moneyness-max") or 1.10),
            min_dte=int(self.GetParameter("min-dte") or 1),
            max_dte=int(self.GetParameter("max-dte") or 45),
            optimal_dte_min=int(self.GetParameter("optimal-dte-min") or 7),
            optimal_dte_max=int(self.GetParameter("optimal-dte-max") or 21),
            max_rv_uncertainty_ratio=float(self.GetParameter("max-rv-uncertainty-ratio") or 0.2),
            disable_short_vol_in_crisis=True,
            regime_stress_multiplier=0.5
        )
        
        self.SetAlpha(IVRVSpreadAlphaModel(
            rv_calculator=self.rv_calculator,
            regime_classifier=self.regime_classifier,
            logger=self.logger,
            config=alpha_config
        ))
        self.logger.log(f"Alpha Model: IV/RV threshold={self.iv_rv_threshold}, moneyness=[{alpha_config.moneyness_min}-{alpha_config.moneyness_max}]", LogLevel.INFO)
        
        # ================================================================
        # Set Up Portfolio Construction Model
        # ================================================================
        portfolio_config = PortfolioConfig(
            max_position_size=self.max_position_size,
            vega_limit=self.vega_limit,
            delta_tolerance=self.delta_tolerance,
            gamma_limit=self.gamma_limit,
            target_daily_vol=self.target_daily_vol,
            vol_scaling_enabled=self.GetParameter("vol-scaling-enabled") != "false",
            min_vol_scale=float(self.GetParameter("min-vol-scale") or 0.2),
            max_vol_scale=float(self.GetParameter("max-vol-scale") or 2.0)
        )
        
        self.portfolio_model = DeltaVegaNeutralPortfolioConstructionModel(
            logger=self.logger,
            rv_calculator=self.rv_calculator,
            regime_classifier=self.regime_classifier,
            config=portfolio_config
        )
        self.SetPortfolioConstruction(self.portfolio_model)
        self.logger.log(f"Portfolio Model: vega_limit={self.vega_limit}, delta_tol={self.delta_tolerance}, target_vol={self.target_daily_vol:.1%}", LogLevel.INFO)
        
        # ================================================================
        # Set Up Execution Model
        # ================================================================
        execution_config = ExecutionConfig(
            simulated_latency_ms=int(self.GetParameter("simulated-latency-ms") or 50),
            limit_order_offset_bps=float(self.GetParameter("limit-order-offset-bps") or 10),
            min_absolute_tick_offset=float(self.GetParameter("min-absolute-tick-offset") or 0.05),
            min_edge_after_costs_bps=self.min_edge_bps,
            spread_cost_multiplier=float(self.GetParameter("spread-cost-multiplier") or 0.5),
            impact_buffer_bps=float(self.GetParameter("impact-buffer-bps") or 5),
            max_participation_pct=float(self.GetParameter("max-participation-pct") or 0.20),
            unfilled_timeout_seconds=int(self.GetParameter("unfilled-timeout-seconds") or 60),
            max_replace_attempts=int(self.GetParameter("max-replace-attempts") or 3)
        )
        
        self.execution_model = SmartExecutionModel(
            logger=self.logger,
            config=execution_config
        )
        self.SetExecution(self.execution_model)
        self.logger.log(f"Execution Model: min_edge={self.min_edge_bps}bps, participation={execution_config.max_participation_pct:.0%}", LogLevel.INFO)
        
        # ================================================================
        # Set Up Risk Management Model
        # ================================================================
        scenarios = [
            ScenarioConfig("SPX_DOWN_5_VOL_UP", -0.05, 5.0, "SPX down 5%, IV up 5pts"),
            ScenarioConfig("SPX_DOWN_10_VOL_UP", -0.10, 10.0, "SPX down 10%, IV up 10pts"),
            ScenarioConfig("SPX_UP_5_VOL_DOWN", 0.05, -3.0, "SPX up 5%, IV down 3pts"),
            ScenarioConfig("VOL_SPIKE", 0.0, 8.0, "IV spike 8pts, flat spot"),
            ScenarioConfig("VOL_CRUSH", 0.0, -5.0, "IV crush 5pts, flat spot")
        ]
        
        risk_config = RiskConfig(
            max_daily_loss=self.max_daily_loss,
            warning_daily_loss=self.max_daily_loss * 0.75,
            vega_limit=self.vega_limit,
            delta_tolerance=self.delta_tolerance,
            gamma_limit=self.gamma_limit,
            max_scenario_loss_pct=self.scenario_max_loss,
            scenarios=scenarios,
            max_position_age_hours=int(self.GetParameter("max-position-age-hours") or 24),
            max_position_pct=float(self.GetParameter("max-position-pct") or 0.10),
            max_unrealized_loss_pct=float(self.GetParameter("max-unrealized-loss-pct") or 0.50)
        )
        
        self.risk_model = ExposureLimitsRiskManagementModel(
            logger=self.logger,
            regime_classifier=self.regime_classifier,
            config=risk_config
        )
        self.SetRiskManagement(self.risk_model)
        self.logger.log(f"Risk Model: daily_loss={self.max_daily_loss:.1%}, scenario_loss={self.scenario_max_loss:.0%}", LogLevel.INFO)
        
        # ================================================================
        # Schedule Events
        # ================================================================
        # End of day position flatten
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(14, 55),
            self.CloseAllPositions
        )
        
        # Daily summary logging
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(15, 45),
            self.LogDailySummary
        )
        
        # Periodic risk snapshot
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(minutes=30)),
            self.LogRiskSnapshot
        )
        
        # ================================================================
        # Warm Up
        # ================================================================
        self.SetWarmUp(timedelta(days=25))
        
        # ================================================================
        # Track active chains
        # ================================================================
        self.active_chains = {}
        
        self.logger.log("="*60, LogLevel.INFO)
        self.logger.log("OMA Strategy Initialization Complete", LogLevel.INFO)
        self.logger.log("="*60, LogLevel.INFO)
    
    def InitializeUniverse(self):
        """Initialize the options universe with liquid contracts."""
        # Add SPX as primary underlying
        try:
            spx = self.AddIndex("SPX", Resolution.Minute)
            spx.SetDataNormalizationMode(DataNormalizationMode.Raw)
            self.underlying_symbol = spx.Symbol
            
            # Add option chain
            option = self.AddIndexOption(spx.Symbol, Resolution.Minute)
            option.SetFilter(self.OptionsFilter)
            
            self.logger.log(f"Added primary underlying: SPX", LogLevel.INFO)
        except Exception as e:
            self.logger.log(f"Failed to add SPX: {e}, falling back to SPY", LogLevel.WARNING)
            # Fallback to SPY
            spy = self.AddEquity("SPY", Resolution.Minute)
            self.underlying_symbol = spy.Symbol
            option = self.AddOption("SPY", Resolution.Minute)
            option.SetFilter(self.OptionsFilter)
        
        # Add VIX for regime classification
        try:
            vix = self.AddIndex("VIX", Resolution.Minute)
            self.vix_symbol = vix.Symbol
            self.regime_classifier.set_vix_symbol(self.vix_symbol)
            self.logger.log("Added VIX for regime classification", LogLevel.INFO)
        except Exception:
            self.vix_symbol = None
            self.logger.log("VIX not available - using IV proxy", LogLevel.WARNING)
        
        # Backup symbols for diversification
        self.backup_symbols = ["SPY", "QQQ", "IWM"]
        
    def OptionsFilter(self, universe):
        """
        Filter options universe for liquid contracts.
        
        Args:
            universe: OptionFilterUniverse object
            
        Returns:
            Filtered universe with liquid weekly options
        """
        return universe.IncludeWeeklys()\
                      .Strikes(-10, 10)\
                      .Expiration(timedelta(days=0), timedelta(days=45))\
                      .OnlyApplyFilterAtMarketOpen()
    
    def OnData(self, data):
        """
        Main data handler - processes option chains and updates calculations.
        
        Args:
            data: Slice object containing current market data
        """
        if self.IsWarmingUp:
            return
        
        # Update realized volatility calculations
        if self.underlying_symbol in data.Bars:
            self.rv_calculator.update(self.underlying_symbol, data.Bars[self.underlying_symbol])
        
        # Update volatility regime
        self._update_regime(data)
        
        # Process option chains
        for kvp in data.OptionChains:
            chain = kvp.Value
            self.ProcessOptionChain(chain)
    
    def _update_regime(self, data):
        """Update volatility regime classifier with latest data."""
        vix_value = None
        spx_price = None
        
        # Get VIX if available
        if self.vix_symbol and self.vix_symbol in data.Bars:
            vix_value = data.Bars[self.vix_symbol].Close
        
        # Get SPX price
        if self.underlying_symbol in data.Bars:
            spx_price = data.Bars[self.underlying_symbol].Close
        
        # Update regime
        self.regime_classifier.update(
            vix_value=vix_value,
            spx_price=spx_price,
            underlying_symbol=self.underlying_symbol
        )
    
    def _on_regime_change(self, old_regime: VolatilityRegime, new_regime: VolatilityRegime):
        """Handle regime change events."""
        self.logger.log_regime_change(old_regime.value, new_regime.value)
        
        # If entering crisis, log warning
        if new_regime == VolatilityRegime.CRISIS:
            self.logger.log(
                "CRISIS REGIME DETECTED - Short vol signals disabled, tightening risk limits",
                LogLevel.WARNING
            )
    
    def ProcessOptionChain(self, chain):
        """
        Process individual option chain for trading opportunities.
        
        Args:
            chain: OptionChain object
        """
        if len(chain) == 0:
            return
        
        # Filter for liquid contracts
        liquid_contracts = [
            contract for contract in chain
            if contract.OpenInterest > 1000 
            and contract.Volume > 1000
            and contract.BidSize > 0 
            and contract.AskSize > 0
            and contract.Bid > 0
            and contract.Ask > contract.Bid
        ]
        
        # Store active chains
        underlying = chain.Underlying.Symbol if chain.Underlying else chain.Symbol
        self.active_chains[underlying] = liquid_contracts
    
    def CloseAllPositions(self):
        """Close all positions at end of day (14:55 EST)."""
        self.logger.log("EOD FLATTEN: Closing all positions at 14:55 EST", LogLevel.INFO)
        
        # Log final Greeks before closing
        if self.portfolio_model.risk_snapshot:
            snapshot = self.portfolio_model.risk_snapshot
            self.logger.log(
                f"Pre-close Greeks: Delta={snapshot.total_delta:.0f}, "
                f"Vega={snapshot.total_vega:.0f}, Gamma={snapshot.total_gamma:.1f}",
                LogLevel.INFO
            )
        
        self.Liquidate()
    
    def LogDailySummary(self):
        """Log daily summary at end of day."""
        # Get TCA summary from execution model
        tca_summary = self.execution_model.get_tca_summary()
        
        # Get risk summary
        risk_summary = self.risk_model.GetMetrics()
        
        # Log daily summary
        self.logger.log_daily_summary(
            tca_summary=tca_summary,
            risk_summary=risk_summary
        )
    
    def LogRiskSnapshot(self):
        """Log periodic risk snapshot."""
        if self.portfolio_model.risk_snapshot is None:
            return
        
        snapshot = self.portfolio_model.risk_snapshot
        
        # Get scenario summary from risk model
        scenario_summary = self.risk_model.get_scenario_summary(self)
        
        self.logger.log_risk_snapshot({
            'delta': snapshot.total_delta,
            'vega': snapshot.total_vega,
            'gamma': snapshot.total_gamma,
            'theta': snapshot.total_theta,
            'delta_by_bucket': {k.value: v for k, v in snapshot.delta_by_bucket.items()},
            'vega_by_bucket': {k.value: v for k, v in snapshot.vega_by_bucket.items()},
            'delta_utilization': snapshot.delta_utilization,
            'vega_utilization': snapshot.vega_utilization,
            'gamma_utilization': snapshot.gamma_utilization,
            'vol_scale': snapshot.vol_scale_factor,
            'scenario_breaches': scenario_summary.get('any_breach', False),
            'worst_scenario': scenario_summary.get('worst_scenario', '')
        })
    
    def OnOrderEvent(self, orderEvent):
        """
        Handle order events for tracking and logging.
        
        Args:
            orderEvent: OrderEvent object
        """
        # Forward to execution model for TCA
        self.execution_model.OnOrderEvent(self, orderEvent)
        
        if orderEvent.Status == OrderStatus.Filled:
            self.logger.log(
                f"ORDER FILLED: {orderEvent.Symbol} "
                f"Qty: {orderEvent.FillQuantity} "
                f"Price: ${orderEvent.FillPrice:.2f}",
                LogLevel.INFO
            )
    
    def OnSecuritiesChanged(self, changes):
        """
        Handle universe changes.
        
        Args:
            changes: SecurityChanges object
        """
        for security in changes.AddedSecurities:
            if security.Type == SecurityType.Option:
                self.logger.log(f"Universe add: {security.Symbol}", LogLevel.DEBUG)
                
        for security in changes.RemovedSecurities:
            if security.Type == SecurityType.Option:
                self.logger.log(f"Universe remove: {security.Symbol}", LogLevel.DEBUG)
    
    def OnEndOfAlgorithm(self):
        """Called at end of algorithm - log final summary."""
        self.logger.log("="*60, LogLevel.INFO)
        self.logger.log("OMA Strategy Final Summary", LogLevel.INFO)
        self.logger.log("="*60, LogLevel.INFO)
        
        # Final TCA
        tca = self.execution_model.get_tca_summary()
        self.logger.log_tca_summary(tca)
        
        # Alpha diagnostics
        alpha_diag = self.logger.get_alpha_diagnostics_summary()
        self.logger.log(f"Alpha Diagnostics: {alpha_diag}", LogLevel.INFO)
        
        # Risk metrics
        risk_metrics = self.risk_model.GetMetrics()
        self.logger.log(f"Risk Metrics: {risk_metrics}", LogLevel.INFO)
        
        # Performance
        self.logger.log_performance("final")
        
        self.logger.log("="*60, LogLevel.INFO)
