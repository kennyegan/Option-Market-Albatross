# QuantConnect Backtesting Quick Start Guide

## ✅ Pre-Flight Check

Your strategy is **ready to run**! Here's what's configured:

- ✅ Start Date: 2023-01-01 (update per backtest window)
- ✅ End Date: 2024-12-31 (update per backtest window)
- ✅ Initial Capital: $1,000,000
- ✅ Warmup Period: 60 days (robust RV calculation)
- ✅ Universe: SPX options (with SPY fallback)
- ✅ Resolution: Minute (optimal balance)
- ✅ All modules initialized
- ✅ Parameters configured in `lean.json`

**📋 See [BACKTEST_PLAN.md](BACKTEST_PLAN.md) for the complete 4-window backtest plan.**

---

## 🚀 Step-by-Step: Running Your First Backtest

### Method 1: QuantConnect Cloud (Recommended - Easiest)

#### Step 1: Create Account
1. Go to [quantconnect.com](https://www.quantconnect.com)
2. Sign up (free tier available)
3. Verify email

#### Step 2: Create New Algorithm
1. Click **"Create Algorithm"** in dashboard
2. Select **"Blank Algorithm"** or **"Python"**
3. Name it: `OMA Options Arbitrage Strategy`

#### Step 3: Upload Your Code

**Option A: Direct Upload (Easiest)**
1. In QuantConnect editor, click **"Files"** tab
2. Click **"Upload"** button
3. Upload entire `QuantConnectOmaStrategy/` folder
   - Select all files: `main.py`, `alpha/`, `portfolio/`, `execution/`, `risk/`, `data/`, `utils/`, `config/`, `lean.json`
4. Wait for upload to complete

**Option B: Git Integration (Advanced)**
1. In QuantConnect, go to **Settings → Git**
2. Connect your GitHub repository
3. QuantConnect will sync automatically

#### Step 4: Set Main File
1. In QuantConnect editor, make sure `main.py` is set as the main file
2. The class `OMAOptionsArbitrageAlgorithm` should be detected automatically

#### Step 5: Configure Parameters (Optional)
1. Click **"Parameters"** tab in QuantConnect
2. Adjust any parameters from `lean.json` if needed
3. Or leave defaults (they're already in `lean.json`)

#### Step 6: Run Backtest
1. Click **"Backtest"** button (top right)
2. Wait for backtest to complete (may take 5-15 minutes)
3. Review results in the **"Results"** tab

#### Step 7: Review Results
- **Performance Metrics**: Sharpe, drawdown, returns
- **Charts**: Equity curve, drawdown, positions
- **Logs**: Check for any errors or warnings
- **Orders**: Review trade execution

---

### Method 2: Local Lean CLI (Advanced)

#### Step 1: Install Lean CLI
```bash
pip install lean
```

#### Step 2: Authenticate
```bash
lean login
# Follow prompts to authenticate with QuantConnect
```

#### Step 3: Initialize Project
```bash
cd QuantConnectOmaStrategy
lean init
```

#### Step 4: Download Data (Required for Local)
```bash
# Download SPX options data (this may take a while)
lean data download --dataset "QuantConnect/SPX Options" --start 2023-01-01 --end 2024-01-01
```

#### Step 5: Run Backtest
```bash
lean backtest
```

---

## ⚙️ Configuration Options

### Change Backtest Period

Edit `main.py`:
```python
self.SetStartDate(2023, 1, 1)  # Change start date
self.SetEndDate(2024, 1, 1)    # Change end date
```

### Adjust Strategy Parameters

Edit `lean.json` parameters section, or in QuantConnect web interface:
- `iv-rv-threshold`: 1.2 (IV must be 20% higher than RV)
- `max-position-size`: 0.05 (5% NAV per leg)
- `vega-limit`: 10000 (portfolio vega cap)
- `max-daily-loss`: 0.02 (2% daily loss limit)

### Use Different Backtest Scenario

Edit `config/backtest_config.json` to use:
- `conservative`: Tighter limits
- `aggressive`: Larger positions
- `volatility_crisis`: 2020 COVID period
- `low_vol_regime`: 2017 low volatility

---

## 🔍 What to Check After Backtest

### 1. Check Logs for Errors
- Look for any `ERROR` level messages
- Verify all modules loaded correctly
- Check for data issues

### 2. Review Performance Metrics
- **Sharpe Ratio**: Target > 1.5
- **Max Drawdown**: Should be < 5%
- **Win Rate**: Expect 65-75%
- **Total Return**: Check if reasonable

### 3. Verify Strategy Behavior
- **Positions**: Check that options are being traded
- **Greeks**: Verify delta is near-neutral
- **Regime**: Check if regime classifier is working
- **Risk Limits**: Verify daily loss limits are enforced

### 4. Check Execution Quality
- **Fill Rate**: Should be high (>80%)
- **Slippage**: Check TCA logs
- **Order Types**: Mix of limit and market orders

---

## 🐛 Common Issues & Fixes

### Issue: "No option chains found"
**Fix**: 
- Check data subscription (SPX options data may require subscription)
- Verify date range has data available
- Try SPY instead of SPX (already has fallback)

### Issue: "Module not found" errors
**Fix**:
- Ensure all files uploaded correctly
- Check `__init__.py` files exist in all directories
- Verify import paths are correct

### Issue: "Greeks calculation failed"
**Fix**:
- This is normal - fallback Greeks will be used
- Strategy has built-in fallback calculations
- Check logs to see if using Lean Greeks or fallback

### Issue: "No signals generated"
**Fix**:
- Check IV/RV threshold (may be too high)
- Verify options data is available
- Check liquidity filters (OI > 1000, Volume > 1000)
- Review logs for signal generation

### Issue: "Regime classifier not working"
**Fix**:
- VIX data may not be available in backtest
- Strategy will use IV proxy if VIX unavailable
- Check logs for regime status

---

## 📊 Expected First Run Behavior

### Warmup Period (First 20 days)
- Strategy is warming up for RV calculation
- No trades should occur
- Logs will show "Warming up"

### After Warmup
- RV calculations start
- Signals begin generating
- First trades should appear
- Delta hedging should activate

### Daily Schedule
- **Market Open**: Universe filtering, signal generation
- **Throughout Day**: Trading, risk checks, rebalancing
- **2:55 PM EST**: All positions closed (EOD flatten)
- **3:45 PM EST**: Daily summary logged

---

## 🎯 Next Steps After First Backtest

1. **Review Results**: Analyze performance vs expectations
2. **Check Logs**: Look for any warnings or errors
3. **Adjust Parameters**: Fine-tune based on results
4. **Test Different Periods**: Try 2020 (crisis) or 2017 (low vol)
5. **Paper Trade**: Run in paper trading mode before live

---

## 📞 Getting Help

- **QuantConnect Docs**: [docs.quantconnect.com](https://www.quantconnect.com/docs)
- **QuantConnect Forum**: [forum.quantconnect.com](https://forum.quantconnect.com)
- **GitHub Issues**: Open an issue in your repo

---

## ✅ Checklist Before Running

- [ ] QuantConnect account created
- [ ] All files uploaded to QuantConnect
- [ ] `main.py` set as main file
- [ ] Parameters reviewed (or using defaults)
- [ ] Data subscription active (if needed)
- [ ] Ready to run first backtest!

---

**You're all set! Click "Backtest" and watch your strategy run!** 🚀

