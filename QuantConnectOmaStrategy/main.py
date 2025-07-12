"""
QuantConnect OMA Strategy: IV/RV + Bid/Ask Arbitrage
Main algorithm entry point that combines IV vs RV arbitrage with bid/ask spread capture.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from alpha.iv_rv_spread_alpha import IVRVSpreadAlphaModel
from execution.smart_router import SmartExecutionModel
from portfolio.delta_vega_neutral import DeltaVegaNeutralPortfolioConstructionModel
from risk.exposure_limits import ExposureLimitsRiskManagementModel
from data.realized_vol_calc import RealizedVolatilityCalculator
from utils.logger import StrategyLogger
import numpy as np
from datetime import timedelta


class OMAOptionsArbitrageAlgorithm(QCAlgorithm):
    """
    Main algorithm class for Options Market Arbitrage (OMA) strategy.
    
    This strategy combines:
    1. IV vs RV arbitrage signals
    2. Bid/ask spread capture opportunities
    3. Delta-neutral portfolio construction
    4. Vega exposure management
    """
    
    def Initialize(self):
        """Initialize the algorithm with strategy parameters and models."""
        # Basic setup
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(1000000)
        
        # Initialize logger
        self.logger = StrategyLogger(self)
        self.logger.log("Initializing OMA Options Arbitrage Strategy", LogLevel.INFO)
        
        # Strategy parameters
        self.iv_rv_threshold = 1.2  # IV must be 20% higher than RV
        self.spread_threshold = 0.005  # 0.5% bid/ask spread threshold
        self.max_position_size = 0.05  # 5% NAV per leg
        self.vega_limit = 10000  # Maximum vega exposure
        self.delta_tolerance = 100  # Delta neutrality tolerance
        self.max_daily_loss = 0.02  # 2% daily loss limit
        
        # Market hours and execution settings
        self.SetTimeZone(TimeZones.NewYork)
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(14, 55),
            self.CloseAllPositions
        )
        
        # Initialize realized vol calculator
        self.rv_calculator = RealizedVolatilityCalculator(self, lookback_days=10)
        
        # Add SPX options universe
        self.InitializeUniverse()
        
        # Set up models
        self.SetAlpha(IVRVSpreadAlphaModel(
            iv_rv_threshold=self.iv_rv_threshold,
            spread_threshold=self.spread_threshold,
            rv_calculator=self.rv_calculator,
            logger=self.logger
        ))
        
        self.SetPortfolioConstruction(DeltaVegaNeutralPortfolioConstructionModel(
            max_position_size=self.max_position_size,
            vega_limit=self.vega_limit,
            delta_tolerance=self.delta_tolerance,
            logger=self.logger
        ))
        
        self.SetExecution(SmartExecutionModel(
            latency_ms=50,
            logger=self.logger
        ))
        
        self.SetRiskManagement(ExposureLimitsRiskManagementModel(
            max_daily_loss=self.max_daily_loss,
            vega_limit=self.vega_limit,
            delta_tolerance=self.delta_tolerance,
            logger=self.logger
        ))
        
        # Warm up for realized volatility calculation
        self.SetWarmUp(timedelta(days=20))
        
    def InitializeUniverse(self):
        """Initialize the options universe with liquid contracts."""
        # Add SPX as primary underlying
        spx = self.AddIndex("SPX", Resolution.Minute)
        spx.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
        # Add option chain
        option = self.AddOption("SPX", Resolution.Minute)
        option.SetFilter(self.OptionsFilter)
        
        # Store reference to underlying
        self.underlying_symbol = spx.Symbol
        
        # If SPX not available, fall back to liquid equity options
        self.backup_symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
        self.active_chains = {}
        
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
        
        # Process option chains
        for kvp in data.OptionChains:
            chain = kvp.Value
            self.ProcessOptionChain(chain)
    
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
        ]
        
        # Store active chains for strategy models
        underlying = chain.Underlying.Symbol if chain.Underlying else chain.Symbol
        self.active_chains[underlying] = liquid_contracts
        
        self.logger.log(f"Processing {len(liquid_contracts)} liquid contracts for {underlying}", LogLevel.DEBUG)
    
    def CloseAllPositions(self):
        """Close all positions at end of day."""
        self.logger.log("Closing all positions at 2:55 PM EST", LogLevel.INFO)
        self.Liquidate()
    
    def OnOrderEvent(self, orderEvent):
        """
        Handle order events for tracking and logging.
        
        Args:
            orderEvent: OrderEvent object
        """
        if orderEvent.Status == OrderStatus.Filled:
            self.logger.log(
                f"Order filled: {orderEvent.Symbol} "
                f"Qty: {orderEvent.FillQuantity} "
                f"Price: {orderEvent.FillPrice}",
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
                self.logger.log(f"Added option: {security.Symbol}", LogLevel.DEBUG)
                
        for security in changes.RemovedSecurities:
            if security.Type == SecurityType.Option:
                self.logger.log(f"Removed option: {security.Symbol}", LogLevel.DEBUG)