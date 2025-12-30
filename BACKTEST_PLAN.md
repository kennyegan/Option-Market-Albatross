# Backtest Execution Plan

## 📋 Overview

This document outlines the **serious backtest plan** for OMA Strategy across different market regimes. Each window tests the strategy under specific conditions to validate robustness.

---

## 🎯 Backtest Windows

### 1. Calm Bull (Low Vol) - 2017-2019
**Period**: `2017-01-01` → `2019-12-31`  
**Expected Regime**: CALM/NORMAL  
**Market Conditions**: Low VIX, steady bull market, favorable for short-vol strategies

**Configuration**:
- IV/RV Threshold: 1.25 (higher - fewer signals in low vol)
- Max Position: 6% NAV
- Vega Limit: 12,000
- Target Daily Vol: 1.2%

**What to Look For**:
- High win rate (should be >70%)
- Consistent returns
- Low drawdowns
- Good Sharpe ratio (>2.0 expected)

---

### 2. Crash & COVID Spike - 2020
**Period**: `2020-01-01` → `2020-12-31`  
**Expected Regime**: CRISIS/STRESSED  
**Market Conditions**: March 2020 crash, VIX spike to 80+, extreme volatility

**Configuration**:
- IV/RV Threshold: 1.15 (lower - more selective)
- Max Position: 2% NAV (reduced risk)
- Vega Limit: 3,000 (tight)
- Target Daily Vol: 0.5% (very conservative)
- Short-vol disabled in CRISIS regime

**What to Look For**:
- Strategy should reduce/disable positions during March crash
- Daily loss limits should prevent major drawdowns
- Regime classifier should detect CRISIS
- Recovery after initial crash

---

### 3. High-Vol Bear / Regime Change - 2022
**Period**: `2022-01-01` → `2022-12-31`  
**Expected Regime**: STRESSED  
**Market Conditions**: Bear market, inflation, rate hikes, sustained elevated volatility

**Configuration**:
- IV/RV Threshold: 1.2 (standard)
- Max Position: 4% NAV (moderate)
- Vega Limit: 7,000
- Target Daily Vol: 0.8%
- Regime stress multiplier: 0.5

**What to Look For**:
- Regime adaptation (reduced size in stress)
- Scenario stress tests should trigger
- Managed drawdowns despite bear market
- Strategy should survive sustained stress

---

### 4. Mixed / Recent - 2023-2024
**Period**: `2023-01-01` → `2024-12-31` (or latest available)  
**Expected Regime**: MIXED  
**Market Conditions**: Recent period with mixed conditions

**Configuration**:
- IV/RV Threshold: 1.2 (standard)
- Max Position: 5% NAV
- Vega Limit: 10,000
- Target Daily Vol: 1.0%

**What to Look For**:
- Overall strategy performance
- Regime transitions handled well
- Consistent with other periods
- Good risk-adjusted returns

---

## ⚙️ Global Settings (All Backtests)

### Resolution
- **Underlying**: Minute
- **Options**: Minute
- **Do NOT use Tick** (too heavy, not needed)

### Universe
- **Primary**: SPX options
- **Backup**: SPY, QQQ, IWM, AAPL, MSFT (if SPX unavailable)
- **Filters**: OI > 1000, Volume > 1000, Bid > $0.05

### Warmup Period
- **60 days** (supports RV windows: 5d, 10d, 20d, 60d)
- No trades during warmup
- RV calculations building history

### Capital
- **$1,000,000** initial capital (all tests)

---

## 🚀 How to Run Each Backtest

### In QuantConnect Web Interface

#### Step 1: Select Window
Edit `main.py` to set dates:
```python
# For Calm Bull (2017-2019)
self.SetStartDate(2017, 1, 1)
self.SetEndDate(2019, 12, 31)

# For COVID (2020)
self.SetStartDate(2020, 1, 1)
self.SetEndDate(2020, 12, 31)

# For 2022 Bear
self.SetStartDate(2022, 1, 1)
self.SetEndDate(2022, 12, 31)

# For Recent (2023-2024)
self.SetStartDate(2023, 1, 1)
self.SetEndDate(2024, 12, 31)
```

#### Step 2: Update Parameters
Edit `lean.json` or use QuantConnect Parameters tab:
```json
{
    "parameters": {
        "iv-rv-threshold": "1.25",  // Adjust per window
        "max-position-size": "0.06", // Adjust per window
        "vega-limit": "12000",       // Adjust per window
        "warmup-days": "60"          // Always 60
    }
}
```

#### Step 3: Run Backtest
1. Click **"Backtest"** button
2. Wait for completion (10-30 minutes depending on period)
3. Save results with descriptive name: `OMA_CalmBull_2017-2019`

#### Step 4: Document Results
Record key metrics:
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Number of Trades
- Average Trade Duration

---

## 📊 Expected Results Summary

| Window | Expected Sharpe | Expected Max DD | Expected Win Rate | Key Test |
|--------|----------------|-----------------|-------------------|----------|
| **Calm Bull (2017-2019)** | > 2.0 | < 3% | > 70% | Favorable conditions |
| **COVID (2020)** | > 0.5 | < 8% | > 60% | Crisis survival |
| **2022 Bear** | > 1.0 | < 6% | > 65% | Sustained stress |
| **Recent (2023-2024)** | > 1.5 | < 5% | > 68% | Overall validation |

**Note**: These are targets based on theoretical analysis. Actual results will vary.

---

## 🔍 What to Analyze Per Window

### Performance Metrics
- [ ] Total Return (%)
- [ ] Annualized Return (%)
- [ ] Sharpe Ratio
- [ ] Sortino Ratio
- [ ] Maximum Drawdown (%)
- [ ] Win Rate (%)
- [ ] Profit Factor
- [ ] Average Trade Duration

### Risk Metrics
- [ ] Daily VaR (95%)
- [ ] Maximum Daily Loss
- [ ] Greek Exposure (delta, vega, gamma)
- [ ] Scenario Losses (all 5 scenarios)
- [ ] Regime Distribution (time in each regime)

### Execution Quality
- [ ] Fill Rate
- [ ] Average Slippage (bps)
- [ ] Spread Capture (bps)
- [ ] Order Type Distribution (limit vs market)
- [ ] Edge Blocked Count (trades rejected for insufficient edge)

### Strategy Behavior
- [ ] Signals Generated
- [ ] Trades Executed
- [ ] Positions Closed (EOD flatten working?)
- [ ] Risk Events Triggered
- [ ] Regime Changes Detected

---

## 📝 Running Order

**Recommended sequence** (easiest to hardest):

1. **Calm Bull (2017-2019)** - Start here, should perform well
2. **Recent (2023-2024)** - Validate current implementation
3. **2022 Bear** - Test sustained stress
4. **COVID (2020)** - Ultimate stress test

---

## 🎯 Success Criteria

### Per Window
- ✅ Strategy completes without errors
- ✅ Daily loss limits enforced (no breaches)
- ✅ EOD flatten working (positions closed at 2:55 PM)
- ✅ Regime classifier functioning
- ✅ Risk limits respected

### Overall
- ✅ Strategy survives all regimes
- ✅ Drawdowns controlled (< 10% even in worst case)
- ✅ Positive Sharpe in at least 3 of 4 windows
- ✅ No catastrophic failures

---

## 📈 After All Backtests

### Combine Results
Once each window is validated:
1. Run full-span backtest: `2017-01-01` → `2024-12-31`
2. Compare combined vs individual windows
3. Analyze regime transitions
4. Validate long-term performance

### Parameter Optimization
Based on results:
- Adjust IV/RV thresholds per regime
- Fine-tune position sizing
- Optimize risk limits
- Calibrate regime boundaries

---

## 🐛 Troubleshooting

### Issue: "No data for SPX options"
**Fix**: 
- Check data subscription in QuantConnect
- Use SPY fallback (already coded)
- Verify date range has data

### Issue: "Warmup period too short"
**Fix**: 
- Already set to 60 days
- If still issues, increase to 90 days

### Issue: "No signals generated"
**Fix**:
- Check IV/RV threshold (may be too high)
- Verify options data quality
- Check liquidity filters
- Review logs for signal generation

### Issue: "Excessive drawdowns"
**Fix**:
- Tighten daily loss limits
- Reduce position sizes
- Increase IV/RV threshold (more selective)
- Check scenario stress tests

---

## 📋 Checklist Before Each Backtest

- [ ] Dates updated in `main.py`
- [ ] Parameters updated in `lean.json` (or QuantConnect UI)
- [ ] Warmup set to 60 days
- [ ] Capital set to $1,000,000
- [ ] Resolution set to Minute
- [ ] Universe configured (SPX primary)
- [ ] All files uploaded to QuantConnect
- [ ] Ready to run!

---

## 🎉 Next Steps

1. **Run Calm Bull (2017-2019)** first - should be easiest
2. **Document results** in a spreadsheet or notes
3. **Compare across windows** to identify strengths/weaknesses
4. **Iterate** on parameters if needed
5. **Run full-span** once individual windows validated

---

**Good luck with your backtests!** 🚀

Each window will teach you something different about how the strategy behaves under various market conditions.

