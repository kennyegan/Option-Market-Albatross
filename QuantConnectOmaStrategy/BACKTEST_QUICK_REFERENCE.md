# Backtest Quick Reference

Quick reference for running the 4 backtest windows.

## 🎯 The 4 Windows

### 1. Calm Bull (2017-2019)
```python
# In main.py
self.SetStartDate(2017, 1, 1)
self.SetEndDate(2019, 12, 31)
```

**Parameters** (in QuantConnect Parameters tab or `lean.json`):
```
iv-rv-threshold: 1.25
max-position-size: 0.06
vega-limit: 12000
target-daily-vol: 0.012
```

---

### 2. COVID Crash (2020)
```python
# In main.py
self.SetStartDate(2020, 1, 1)
self.SetEndDate(2020, 12, 31)
```

**Parameters**:
```
iv-rv-threshold: 1.15
max-position-size: 0.02
vega-limit: 3000
target-daily-vol: 0.005
max-daily-loss: 0.015
```

---

### 3. 2022 Bear Market
```python
# In main.py
self.SetStartDate(2022, 1, 1)
self.SetEndDate(2022, 12, 31)
```

**Parameters**:
```
iv-rv-threshold: 1.2
max-position-size: 0.04
vega-limit: 7000
target-daily-vol: 0.008
```

---

### 4. Recent (2023-2024)
```python
# In main.py
self.SetStartDate(2023, 1, 1)
self.SetEndDate(2024, 12, 31)
```

**Parameters** (default - already in `lean.json`):
```
iv-rv-threshold: 1.2
max-position-size: 0.05
vega-limit: 10000
target-daily-vol: 0.01
```

---

## ⚙️ Global Settings (All Windows)

**Always set these**:
- `warmup-days: 60`
- `resolution: Minute`
- `initial-capital: 1000000`

---

## 🚀 Quick Steps Per Backtest

1. **Update dates** in `main.py` (SetStartDate, SetEndDate)
2. **Update parameters** in QuantConnect Parameters tab
3. **Click Backtest**
4. **Save results** with descriptive name
5. **Document metrics** (Sharpe, DD, Win Rate, etc.)

---

## 📊 Expected Results (Targets)

| Window | Sharpe | Max DD | Win Rate |
|--------|--------|--------|----------|
| Calm Bull | > 2.0 | < 3% | > 70% |
| COVID | > 0.5 | < 8% | > 60% |
| 2022 Bear | > 1.0 | < 6% | > 65% |
| Recent | > 1.5 | < 5% | > 68% |

---

**Full details**: See [BACKTEST_PLAN.md](../../BACKTEST_PLAN.md)

