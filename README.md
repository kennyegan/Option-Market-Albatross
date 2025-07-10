# Option Market Albatross (OMA)

![Python](https://img.shields.io/badge/python-3.11-blue.svg)

Institutional-grade option market-making and bid/ask spread scalping framework.

## 📌 Version
Current release: **v0.1.0**

Track version history in the future via `CHANGELOG.md`.

## 🐍 Python Version
This project is developed using **Python 3.11**. Please ensure you're using Python 3.11+ to avoid compatibility issues.

## Architecture (Citadel-style)
- Modular pipeline: data → signal → execution → risk → log
- Uses Makefile + Conda + .env for structured deployment
- Configurable YAML-driven strategies

## Getting Started
```bash
conda create -n oma-bot python=3.11 -y
conda activate oma-bot
pip install -r requirements.txt
make run
```

## Roadmap
- Broker API integration (IBKR, Alpaca, Tradier)
- Backtest engine with realistic fill simulation
- Live P&L dashboard
- IV vs RV volatility models

MIT License — Fincept Quant Division
