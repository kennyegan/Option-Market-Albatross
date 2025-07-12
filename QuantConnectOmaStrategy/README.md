# QuantConnect OMA Strategy: IV/RV + Bid/Ask Arbitrage

## 📋 Overview

This is a production-ready options arbitrage strategy for QuantConnect Lean that combines:
- **Implied vs Realized Volatility (IV/RV) arbitrage**: Captures premium when options are overpriced relative to historical volatility
- **Bid/Ask spread capture**: Places intelligent limit orders to profit from wide spreads
- **Delta-neutral portfolio construction**: Maintains market-neutral exposure
- **Vega-limited risk management**: Controls volatility risk exposure

## 🏗️ Architecture

```
QuantConnectOmaStrategy/
├── main.py                          # Algorithm entry point
├── alpha/
│   └── iv_rv_spread_alpha.py       # Signal generation for IV/RV and spread opportunities
├── execution/
│   └── smart_router.py             # Intelligent order routing with latency simulation
├── portfolio/
│   └── delta_vega_neutral.py       # Delta-neutral portfolio construction with vega limits
├── risk/
│   └── exposure_limits.py          # Risk management with daily loss limits and Greek controls
├── data/
│   └── realized_vol_calc.py        # Multi-method realized volatility calculation
├── utils/
│   └── logger.py                   # Centralized logging and performance tracking
├── requirements.txt                 # Python dependencies
├── lean.json                       # Lean CLI configuration
└── README.md                       # This file
```

## 🎯 Strategy Logic

### Alpha Generation
1. **IV/RV Arbitrage**
   - Calculates 10-day realized volatility using close-to-close, Parkinson, and Garman-Klass methods
   - Compares implied volatility from option prices to realized volatility
   - Generates short premium signals when IV > RV × 1.2 (20% premium)

2. **Bid/Ask Spread Capture**
   - Identifies options with spreads > 0.5% of mark price
   - Places limit orders inside the spread to capture edge

### Portfolio Construction
- Creates delta-neutral baskets by combining options with offsetting deltas
- Maintains vega exposure within ±10,000 limit
- Sizes positions up to 5% of NAV per leg
- Automatically hedges residual delta with underlying shares

### Execution
- Uses limit orders for spread capture opportunities
- Falls back to market orders for urgent risk management
- Simulates 50ms latency for realistic backtesting
- Adapts order size based on displayed liquidity

### Risk Management
- **Daily Loss Limit**: Auto-liquidates at -2% daily loss
- **Greek Limits**: 
  - Delta tolerance: ±100
  - Vega cap: ±10,000
- **Position Controls**:
  - Maximum position age: 24 hours
  - Individual position stop-loss: -50%
  - Concentration limit: 10% of portfolio
- **Time Exit**: Closes all positions at 2:55 PM EST

## 📊 Performance Metrics

The strategy tracks:
- Real-time P&L and drawdown
- Greek exposures (delta, vega, gamma, theta)
- Trade execution metrics (fill rates, latency)
- Signal generation statistics
- Error rates and types

## 🚀 Usage

### Local Development

1. Install QuantConnect Lean CLI:
```bash
pip install lean
```

2. Configure your Lean environment:
```bash
lean init
```

3. Run backtests:
```bash
cd QuantConnectOmaStrategy
lean backtest main.py
```

### QuantConnect Cloud

1. Upload all files to your QuantConnect project
2. Set `main.py` as the algorithm file
3. Configure parameters in the Algorithm Lab
4. Run backtest or deploy live

### Parameters

Key parameters (configurable in `lean.json` or Algorithm Lab):
- `iv_rv_threshold`: IV/RV ratio trigger (default: 1.2)
- `spread_threshold`: Minimum spread for capture (default: 0.5%)
- `max_position_size`: Maximum position as % of NAV (default: 5%)
- `vega_limit`: Maximum vega exposure (default: 10,000)
- `delta_tolerance`: Delta neutrality threshold (default: 100)
- `max_daily_loss`: Daily stop-loss (default: 2%)

## 📈 Backtesting

The strategy includes:
- SPX options as primary instrument
- Fallback to liquid equity options (SPY, QQQ, IWM, AAPL, MSFT)
- Weekly expirations preferred
- Liquidity filters: OI > 1000, Volume > 1000

### Sample Backtest Configuration
```python
self.SetStartDate(2023, 1, 1)
self.SetEndDate(2024, 1, 1)
self.SetCash(1000000)
```

## 🔧 Extending the Strategy

### Adding New Alpha Signals
1. Create new alpha model in `alpha/` directory
2. Inherit from `AlphaModel` base class
3. Implement `Update()` method returning `Insight` objects
4. Register in `main.py`

### Custom Risk Rules
1. Add logic to `risk/exposure_limits.py`
2. Implement in `ManageRisk()` method
3. Return `PortfolioTarget` objects for position adjustments

### Alternative Execution Algorithms
1. Create new execution model in `execution/`
2. Inherit from `ExecutionModel`
3. Implement order placement logic in `Execute()`

## ⚠️ Important Notes

1. **Greeks Approximation**: Current implementation uses simplified Greeks. For production, integrate with professional options pricing library
2. **Data Requirements**: Requires options data subscription with IV calculations
3. **Latency Simulation**: 50ms latency is simulated but may not reflect real conditions
4. **Transaction Costs**: Remember to account for options assignment/exercise fees

## 📝 Logging

The strategy uses a comprehensive logging system:
- Trade execution logs with entry/exit reasoning
- Performance checkpoints with drawdown tracking
- Greek exposure monitoring
- Error aggregation and reporting

Access logs through QuantConnect's logging interface or export via `logger.export_trade_log()`

## 🤝 Contributing

To contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## 📄 License

This strategy is provided as-is for educational and research purposes. Backtest thoroughly and understand the risks before live trading.

## 🚨 Risk Disclaimer

Options trading involves substantial risk of loss. This strategy:
- Can experience significant drawdowns during volatility regime changes
- Requires careful monitoring of Greek exposures
- May face liquidity challenges in stressed markets
- Should be paper traded extensively before live deployment

Always start with small position sizes and monitor carefully. Past performance does not guarantee future results.