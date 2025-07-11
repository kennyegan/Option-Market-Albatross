# Option Market Albatross (OMA)

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)

**Option Market Albatross (OMA)** is a production-grade, low-latency options arbitrage framework designed to simulate and deploy real-time strategies across U.S. equity derivatives markets. It implements **institutional infrastructure standards** modeled after internal systems used at **Citadel, Jane Street, and Optiver**.

OMA supports **backtesting, paper trading, and live deployment** — with modular, layered execution architecture combining the **flexibility of Python** and the **speed of C++**.

---

## 📌 Version

Current release: **v0.1.0**  
Track version history in `CHANGELOG.md`.

---

## 🧠 Project Purpose

> Build a realistic, institutional-quality options arbitrage framework for:
> - 🧪 Research & simulation with L2 fill models and latency logic  
> - 🧷 Passive bid/ask scalping and IV/RV arbitrage  
> - 📉 Vega- and delta-controlled risk exposure  
> - 🚀 Real-time execution with C++-level performance in the routing path

---

## 🐍 Python Version

Python 3.11+ is used for orchestration, configuration, monitoring, and high-level strategy logic.

---

## 🧱 C++ Integration

Performance-critical components — especially those involving:
- Order book queue simulation  
- Latency modeling  
- Execution fill simulation  
- Tick-level historical data replay  

are written in **C++17** and exposed to Python via:
- **Pybind11 bindings** (for cross-language modules)  
- Optional **shared memory IPC** between `oma-exec-core` and `oma-bot`

---

## 🏗️ Architecture (Citadel-style Layered Pipeline)


- Modular, fault-tolerant components
- Python handles strategy, config, orchestration
- C++ handles latency-sensitive execution paths
- Can be extended with FIX gateways, co-location engines, and data normalization daemons

---

## 🧰 Tech Stack

| Layer                  | Tool / Language / Service                                                       |
|------------------------|----------------------------------------------------------------------------------|
| **Strategy Logic**     | Python 3.11, YAML-based configuration                                            |
| **Execution Engine**   | C++17 core with `pybind11` Python interface                                      |
| **Backtesting Core**   | Custom tick/NBBO replay engine (C++), latency+slippage fill simulator            |
| **Broker API**         | `ib_insync` (IBKR), Tradier REST API, optional FIX route (future)               |
| **Market Data**        | Polygon.io, Tradier, IBKR L1+L2 feeds                                            |
| **Risk Engine**        | Delta/Vega tracking in Python, integrated kill switch logic                     |
| **Monitoring**         | Discord alerts, log files, real-time dashboard (planned: Streamlit/Grafana)     |
| **Deployment**         | Docker + systemd/pm2, optionally colocated servers (NY4 / Equinix)              |
| **C++ Build System**   | CMake + Conan for dependency management                                          |
| **Python Build**       | Conda + Makefile workflow                                                        |
| **IPC Layer**          | Pybind11 or shared memory for Python↔C++ calls                                  |
| **Data Storage**       | SQLite (local), Postgres (optional), flatfile logs                              |

---

## 📁 Folder Structure (Planned)

```bash
oma-bot/
├── strategy/ # Python strategy logic
│ ├── scanner.py # Finds wide-spread arbitrage candidates
│ ├── trader.py # Bid/ask management logic
│ └── config.yaml # Thresholds, Greeks filters, etc.
├── exec_core/ # C++17 order book simulation engine
│ ├── fill_sim.cpp # Queue modeling, partial fill engine
│ ├── latency_model.cpp # Realistic network delay simulator
│ └── CMakeLists.txt # C++ build config
├── bindings/ # pybind11 glue for Python/C++ bridge
│ └── fill_bindings.cpp
├── backtest/ # Orchestration of full backtests
│ └── backtest.py
├── risk/ # Delta/Vega monitors, position manager
│ └── risk_engine.py
├── infra/
│ ├── logger.py # Persistent trade & fill logs
│ ├── scheduler.py # Session scheduling & async control
│ └── discord_alerts.py # Live error + trade alerting
├── deployment/
│ ├── Dockerfile # Reproducible infra
│ └── makefile # One-line local setup
├── data/ # Live or replayed tick/NBBO datasets
├── .env # API keys (ignored by git)
├── requirements.txt # Python deps
├── README.md # You are here
└── CHANGELOG.md # Version history
```

---

## 🧪 Backtesting Features

- Replay **tick-level or NBBO** option data  
- Queue-priority fill simulation (C++)  
- Simulate **latency, slippage, cancel-replace delay**  
- Monte Carlo injection of jitter, network delay, and illiquidity  
- Full trade PnL + fill log recording for per-trade performance analytics  
- Parameter sweep testing (batch configs)

---

## 🚀 Deployment Modes

| Mode        | Description                                                            |
|-------------|------------------------------------------------------------------------|
| **Backtest** | Run offline with historical data and full latency-aware fill model     |
| **Paper**    | Execute trades via IBKR/Tradier in paper mode, with live quotes        |
| **Live**     | Deploy real capital with full kill switch, log, and position monitoring|

---

## ⚙️ Setup

```bash
# Create Python env
conda create -n oma-bot python=3.11 -y
conda activate oma-bot

# Build C++ core
cd exec_core && mkdir build && cd build
cmake ..
make -j4

# Install Python deps
pip install -r requirements.txt

# Run paper/live/backtest
make run
```
## 🏁 Final Notes
This is not a toy script bot.
OMA is designed to mirror institutional workflow for options arbitrage, with modular low-latency execution, realistic backtesting, and a hybrid Python/C++ codebase for ultimate flexibility and speed.


