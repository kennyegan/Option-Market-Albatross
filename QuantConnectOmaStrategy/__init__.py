# QuantConnect OMA Strategy Package
# Version 2.0 - Institutional Grade

"""
OMA Options Arbitrage Strategy

An institutional-grade options market-making / volatility arbitrage strategy
built on QuantConnect Lean framework.

Features:
- IV/RV arbitrage with ensemble realized vol estimators (Close-to-close, Parkinson, Garman-Klass, EWMA)
- Regime-aware signal generation (CALM/NORMAL/STRESSED/CRISIS)
- Delta-vega neutral portfolio construction with volatility targeting
- Factor bucket risk management (INDEX/TECH/SMALL_CAP/SINGLE_NAME)
- Scenario-based stress testing
- Smart execution with edge verification and TCA tracking
"""

__version__ = "2.0.0"
__author__ = "OMA Strategy Team"
