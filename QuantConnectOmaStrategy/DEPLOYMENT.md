# QuantConnect OMA Strategy Deployment Guide

## 📋 Pre-Deployment Checklist

### 1. System Requirements
- [ ] Python 3.11 or higher
- [ ] QuantConnect Lean CLI installed
- [ ] Sufficient capital (minimum $100,000 recommended)
- [ ] Options data feed subscription
- [ ] Stable internet connection with low latency

### 2. Environment Setup
```bash
# Run the automated setup
python3 setup.py

# Or manual setup
pip install -r requirements.txt
lean init
```

### 3. Strategy Validation
- [ ] Backtest performance meets benchmarks (Sharpe > 1.5)
- [ ] Maximum drawdown within acceptable limits (<5%)
- [ ] Risk management systems tested and working
- [ ] All modules compile without errors

## 🧪 Testing Phases

### Phase 1: Historical Backtesting
```bash
# Test conservative parameters
python3 setup.py --create-sample
lean backtest --config lean_sample.json

# Test different market regimes
# Edit config/backtest_config.json for different scenarios
```

**Required Metrics:**
- Sharpe Ratio: > 1.5
- Maximum Drawdown: < 5%
- Win Rate: > 65%
- Calmar Ratio: > 3.0

### Phase 2: Paper Trading (30 days minimum)
```bash
# Use paper trading configuration
# Reduced position sizes and conservative parameters
```

**Monitor Daily:**
- P&L vs backtested expectations
- Greek exposures (delta, vega)
- Order fill rates and execution quality
- Risk limit adherence

### Phase 3: Live Trading (Small Size)
```bash
# Start with 10-20% of intended capital
# Conservative parameters initially
```

**Gradual Scale-Up:**
- Week 1-2: 10% target size
- Week 3-4: 25% target size
- Week 5-8: 50% target size
- Month 3+: Full size (if performance validates)

## ⚙️ Configuration for Different Environments

### Development Environment
```json
{
    "parameters": {
        "iv-rv-threshold": "1.3",
        "max-position-size": "0.01",
        "vega-limit": "1000",
        "max-daily-loss": "0.005"
    }
}
```

### Paper Trading Environment
```json
{
    "parameters": {
        "iv-rv-threshold": "1.2",
        "max-position-size": "0.02",
        "vega-limit": "3000",
        "max-daily-loss": "0.01"
    }
}
```

### Production Environment
```json
{
    "parameters": {
        "iv-rv-threshold": "1.2",
        "max-position-size": "0.05",
        "vega-limit": "10000",
        "max-daily-loss": "0.02"
    }
}
```

## 🚨 Risk Management Protocols

### Daily Risk Checks
1. **Pre-Market (8:30 AM EST)**
   - Review overnight positions
   - Check for corporate actions
   - Verify system connectivity
   - Confirm data feeds operational

2. **Market Open (9:30 AM EST)**
   - Monitor first 30 minutes for unusual activity
   - Verify delta neutrality
   - Check vega exposure limits

3. **Mid-Day (12:00 PM EST)**
   - Review P&L progress
   - Assess Greek exposures
   - Check for risk limit breaches

4. **Pre-Close (2:30 PM EST)**
   - Prepare for position exits
   - Review end-of-day procedures
   - Confirm risk limits intact

5. **Post-Market (4:30 PM EST)**
   - Analyze daily performance
   - Log any issues or anomalies
   - Prepare overnight risk report

### Emergency Procedures

#### Breach of Daily Loss Limit
1. Immediately liquidate all positions
2. Notify risk management team
3. Review logs for root cause
4. Suspend trading until issue resolved

#### Greek Exposure Breach
1. Hedge exposure immediately
2. Reduce position sizes
3. Review portfolio construction logic
4. Increase monitoring frequency

#### System Failure
1. Switch to manual monitoring
2. Use backup execution system
3. Reduce position sizes
4. Implement manual risk controls

## 📊 Monitoring and Alerting

### Real-Time Alerts
```python
# Configure alerts in your monitoring system
alerts = {
    "daily_pnl_threshold": -0.015,  # -1.5% daily loss
    "vega_exposure_warning": 8000,   # 80% of limit
    "delta_deviation": 75,           # 75% of tolerance
    "position_concentration": 0.08,  # 8% of portfolio
    "volume_drop": 0.5              # 50% volume decline
}
```

### Performance Dashboard
Track these metrics in real-time:
- Current P&L (daily, weekly, monthly)
- Greek exposures (delta, vega, gamma, theta)
- Position count and concentration
- Average trade duration
- Fill quality metrics
- Error rates by module

### Weekly Review Metrics
- Risk-adjusted returns (Sharpe, Sortino)
- Maximum drawdown periods
- Win/loss ratios by signal type
- Parameter sensitivity analysis
- Market regime performance

## 🔧 Maintenance Procedures

### Daily Maintenance
- Review error logs
- Update volatility calculations
- Verify data quality
- Check system performance

### Weekly Maintenance
- Analyze strategy performance
- Review parameter effectiveness
- Update universe of traded symbols
- Calibrate Greek calculations

### Monthly Maintenance
- Full system health check
- Strategy performance attribution
- Risk model validation
- Technology infrastructure review

## 🚀 Production Deployment Steps

### 1. Final Validation
```bash
# Run comprehensive tests
python3 setup.py --test-only
lean backtest

# Verify all systems
./scripts/health_check.sh  # Create this script
```

### 2. Live Environment Setup
```bash
# Create production configuration
cp lean.json lean_production.json
# Edit parameters for production

# Set up monitoring
# Configure alerting system
# Establish backup procedures
```

### 3. Go-Live Process
1. **T-1 Day**: Final system checks, team briefing
2. **T-Day 8:00 AM**: Pre-market startup procedures
3. **T-Day 9:30 AM**: Begin live trading with reduced size
4. **T-Day 4:00 PM**: End-of-day review and analysis

### 4. Post-Launch Monitoring
- **Days 1-7**: Intensive monitoring, daily reviews
- **Days 8-30**: Regular monitoring, weekly reviews
- **Days 31+**: Standard operational procedures

## 📋 Operational Procedures

### Start of Day
1. System health check
2. Data feed verification
3. Position reconciliation
4. Risk limit verification
5. Market conditions assessment

### During Trading Hours
1. Continuous monitoring of P&L and Greeks
2. Regular position and risk checks
3. Alert response procedures
4. Performance tracking

### End of Day
1. Position and P&L reconciliation
2. Risk exposure summary
3. Performance attribution
4. System log review
5. Next day preparation

## 🆘 Emergency Contacts

### Primary Contacts
- **Strategy Manager**: [Contact Information]
- **Risk Officer**: [Contact Information]  
- **Technology Lead**: [Contact Information]
- **Compliance Officer**: [Contact Information]

### Escalation Procedures
1. **Level 1**: Automated system response
2. **Level 2**: Strategy manager notification
3. **Level 3**: Risk officer escalation
4. **Level 4**: Senior management alert

## 📖 Documentation Requirements

### Trading Records
- All trades with entry/exit rationale
- Risk exposure logs
- System performance metrics
- Error and exception logs

### Compliance Records
- Daily risk reports
- Position limit compliance
- P&L attribution
- Model validation results

### Audit Trail
- Configuration changes
- Parameter modifications  
- System access logs
- Emergency procedures used

---

## ⚠️ Important Disclaimers

- This strategy involves substantial risk and may result in significant losses
- Thorough testing and validation are essential before live deployment
- Continuous monitoring and risk management are critical
- Past performance does not guarantee future results
- Ensure compliance with all applicable regulations

**Remember: Start small, scale gradually, and never risk more than you can afford to lose.**