"""
Strategy Logger Utility - Institutional Grade
Provides comprehensive logging with daily summaries, TCA, and structured output.

Author: OMA Strategy Team
Version: 2.0 - Enhanced with daily summaries, TCA integration, structured logging
"""

from AlgorithmImports import *
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from enum import Enum
import json
from collections import defaultdict


class LogLevel:
    """Log level constants matching QuantConnect's levels."""

    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class LogOutputMode(Enum):
    """Log output destination."""

    CONSOLE = "CONSOLE"
    FILE = "FILE"
    BOTH = "BOTH"


@dataclass
class DailySummary:
    """
    Daily summary of strategy performance and activity.
    """

    date: date

    # Portfolio metrics
    starting_nav: float = 0
    ending_nav: float = 0
    daily_pnl: float = 0
    daily_return_pct: float = 0
    cumulative_pnl: float = 0
    high_water_mark: float = 0
    drawdown_pct: float = 0

    # Greeks snapshot (end of day)
    eod_delta: float = 0
    eod_vega: float = 0
    eod_gamma: float = 0
    eod_theta: float = 0

    # Trading activity
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0

    # Risk events
    risk_events: int = 0
    greek_breaches: int = 0
    scenario_breaches: int = 0

    # Regime info
    regimes_seen: List[str] = field(default_factory=list)
    dominant_regime: str = "UNKNOWN"

    # TCA summary
    total_fills: int = 0
    avg_slippage_bps: float = 0
    avg_spread_capture_bps: float = 0
    net_execution_cost_bps: float = 0


@dataclass
class RiskSnapshot:
    """
    Point-in-time risk snapshot.
    """

    timestamp: datetime

    # Portfolio Greeks
    delta: float = 0
    vega: float = 0
    gamma: float = 0
    theta: float = 0

    # Per-bucket Greeks (optional)
    delta_by_bucket: Dict[str, float] = field(default_factory=dict)
    vega_by_bucket: Dict[str, float] = field(default_factory=dict)

    # Scenario losses
    scenario_losses: Dict[str, float] = field(default_factory=dict)
    worst_scenario: str = ""
    worst_scenario_loss: float = 0

    # Limits utilization
    delta_utilization: float = 0
    vega_utilization: float = 0
    gamma_utilization: float = 0


@dataclass
class AlphaDiagnostic:
    """
    Alpha model diagnostic record.
    """

    timestamp: datetime
    symbol: str
    signal_type: str
    signal_strength: float
    iv: float
    rv: float
    iv_rv_ratio: float
    moneyness: float
    dte: int
    regime: str
    spread_pct: float


class StrategyLogger:
    """
    Comprehensive strategy logger with:

    1. Daily summary generation
    2. Risk snapshot logging
    3. Alpha diagnostics tracking
    4. TCA integration
    5. Structured output (console/file/both)
    6. Event-based logging (not every tick)

    Key improvements over V1:
    - Daily summary with all metrics
    - Risk snapshots at configurable intervals
    - Alpha entry diagnostics
    - TCA summary integration
    - File logging with JSON format
    """

    def __init__(
        self,
        algorithm: QCAlgorithm,
        min_level: int = LogLevel.INFO,
        output_mode: LogOutputMode = LogOutputMode.CONSOLE,
        enable_performance_logging: bool = True,
        risk_snapshot_interval_minutes: int = 30,
        enable_file_logging: bool = False,
    ):
        """
        Initialize the strategy logger.

        Args:
            algorithm: QuantConnect algorithm instance
            min_level: Minimum log level to output
            output_mode: Where to send logs
            enable_performance_logging: Whether to log performance metrics
            risk_snapshot_interval_minutes: How often to snapshot risk
            enable_file_logging: Whether to write to file (in object store)
        """
        self.algorithm = algorithm
        self.min_level = min_level
        self.output_mode = output_mode
        self.enable_performance_logging = enable_performance_logging
        self.risk_snapshot_interval_minutes = risk_snapshot_interval_minutes
        self.enable_file_logging = enable_file_logging

        # Daily tracking
        self.current_date: Optional[date] = None
        self.daily_summaries: List[DailySummary] = []
        self.current_daily: Optional[DailySummary] = None
        self.starting_capital = 1000000  # Will be updated

        # Risk snapshots
        self.risk_snapshots: List[RiskSnapshot] = []
        self.last_risk_snapshot_time: Optional[datetime] = None

        # Alpha diagnostics
        self.alpha_diagnostics: List[AlphaDiagnostic] = []

        # Trade log
        self.trade_log: List[Dict] = []

        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_errors: List[Dict] = []

        # Performance checkpoints
        self.performance_checkpoints: Dict[str, Dict] = {}

        # File buffer for batch writing
        self.file_buffer: List[str] = []
        self.file_buffer_max = 100

        # Regime tracking for daily summary
        self.regimes_today: List[str] = []

        # Counters for daily summary
        self.daily_counters = {
            "signals": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "risk_events": 0,
            "greek_breaches": 0,
            "scenario_breaches": 0,
        }

        self.log("Strategy Logger initialized (v2.0)", LogLevel.INFO)

    def log(self, message: str, level: int = LogLevel.INFO, data: Dict = None) -> None:
        """
        Log a message with optional structured data.

        Args:
            message: Log message
            level: Log level
            data: Optional structured data to include
        """
        if level < self.min_level:
            return

        # Check for day rollover
        self._check_day_rollover()

        # Format timestamp
        timestamp = self.algorithm.Time.strftime("%Y-%m-%d %H:%M:%S")
        level_name = self._level_name(level)

        # Build log entry
        log_entry = f"[{timestamp}] [{level_name}] {message}"

        # Add structured data if provided
        if data:
            data_str = json.dumps(data, default=str, separators=(",", ":"))
            log_entry += f" | {data_str}"

        # Output to console
        if self.output_mode in [LogOutputMode.CONSOLE, LogOutputMode.BOTH]:
            if level == LogLevel.ERROR:
                self.algorithm.Error(log_entry)
            elif level == LogLevel.DEBUG:
                self.algorithm.Debug(log_entry)
            else:
                self.algorithm.Log(log_entry)

        # Output to file buffer
        if self.enable_file_logging and self.output_mode in [
            LogOutputMode.FILE,
            LogOutputMode.BOTH,
        ]:
            self._buffer_for_file(log_entry, data)

        # Track errors
        if level == LogLevel.ERROR:
            self._track_error(message)

    def log_trade(
        self,
        symbol: Symbol,
        action: str,
        quantity: int,
        price: float,
        reason: str = "",
        metadata: Dict = None,
    ) -> None:
        """
        Log a trade execution.

        Args:
            symbol: Traded symbol
            action: Trade action (BUY/SELL)
            quantity: Trade quantity
            price: Execution price
            reason: Reason for trade
            metadata: Additional trade metadata
        """
        trade_entry = {
            "time": self.algorithm.Time.isoformat(),
            "symbol": str(symbol),
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": (
                abs(quantity * price * 100)
                if symbol.SecurityType == SecurityType.Option
                else abs(quantity * price)
            ),
            "reason": reason,
            "portfolio_value": self.algorithm.Portfolio.TotalPortfolioValue,
        }

        if metadata:
            trade_entry.update(metadata)

        self.trade_log.append(trade_entry)

        # Update daily counter
        self.daily_counters["orders_filled"] += 1

        # Log
        self.log(
            f"TRADE: {action} {quantity} {symbol} @ ${price:.2f} - {reason}",
            LogLevel.INFO,
            trade_entry,
        )

    def log_signal(
        self, signal_type: str, symbol: Symbol, strength: float, metadata: Dict = None
    ) -> None:
        """
        Log alpha signal generation with diagnostics.

        Args:
            signal_type: Type of signal (IV_RV, SPREAD, etc.)
            symbol: Symbol generating signal
            strength: Signal strength
            metadata: Signal metadata (IV, RV, moneyness, DTE, regime, etc.)
        """
        self.daily_counters["signals"] += 1

        # Create diagnostic record
        if metadata:
            diag = AlphaDiagnostic(
                timestamp=self.algorithm.Time,
                symbol=str(symbol),
                signal_type=signal_type,
                signal_strength=strength,
                iv=metadata.get("iv", 0),
                rv=metadata.get("rv", 0),
                iv_rv_ratio=metadata.get("ratio", 0),
                moneyness=metadata.get("moneyness", 0),
                dte=metadata.get("dte", 0),
                regime=metadata.get("regime", "UNKNOWN"),
                spread_pct=metadata.get("spread_pct", 0),
            )
            self.alpha_diagnostics.append(diag)

            # Keep last 1000
            if len(self.alpha_diagnostics) > 1000:
                self.alpha_diagnostics = self.alpha_diagnostics[-1000:]

        # Log
        self.log(
            f"SIGNAL: {signal_type} {symbol} strength={strength:.2f}",
            LogLevel.DEBUG,
            metadata,
        )

    def log_risk_snapshot(self, snapshot_data: Dict) -> None:
        """
        Log a risk snapshot.

        Args:
            snapshot_data: Dictionary with risk metrics
        """
        # Check interval
        if self.last_risk_snapshot_time:
            elapsed = (
                self.algorithm.Time - self.last_risk_snapshot_time
            ).total_seconds() / 60
            if elapsed < self.risk_snapshot_interval_minutes:
                return

        self.last_risk_snapshot_time = self.algorithm.Time

        snapshot = RiskSnapshot(
            timestamp=self.algorithm.Time,
            delta=snapshot_data.get("delta", 0),
            vega=snapshot_data.get("vega", 0),
            gamma=snapshot_data.get("gamma", 0),
            theta=snapshot_data.get("theta", 0),
            delta_by_bucket=snapshot_data.get("delta_by_bucket", {}),
            vega_by_bucket=snapshot_data.get("vega_by_bucket", {}),
            scenario_losses=snapshot_data.get("scenario_losses", {}),
            worst_scenario=snapshot_data.get("worst_scenario", ""),
            worst_scenario_loss=snapshot_data.get("worst_scenario_loss", 0),
            delta_utilization=snapshot_data.get("delta_utilization", 0),
            vega_utilization=snapshot_data.get("vega_utilization", 0),
            gamma_utilization=snapshot_data.get("gamma_utilization", 0),
        )

        self.risk_snapshots.append(snapshot)

        # Keep last 500 snapshots
        if len(self.risk_snapshots) > 500:
            self.risk_snapshots = self.risk_snapshots[-500:]

        self.log(
            f"RISK SNAPSHOT | Delta: {snapshot.delta:.0f} | "
            f"Vega: {snapshot.vega:.0f} | Gamma: {snapshot.gamma:.1f}",
            LogLevel.INFO,
            snapshot_data,
        )

    def log_risk_metrics(self, risk_metrics: Dict) -> None:
        """
        Log risk management metrics.

        Args:
            risk_metrics: Dictionary of risk metrics
        """
        # Track risk events
        if risk_metrics.get("event"):
            self.daily_counters["risk_events"] += 1

            if (
                "VEGA" in str(risk_metrics.get("event", "")).upper()
                or "DELTA" in str(risk_metrics.get("event", "")).upper()
                or "GAMMA" in str(risk_metrics.get("event", "")).upper()
            ):
                self.daily_counters["greek_breaches"] += 1

            if "SCENARIO" in str(risk_metrics.get("event", "")).upper():
                self.daily_counters["scenario_breaches"] += 1

        self.log("RISK METRICS", LogLevel.INFO, risk_metrics)

    def log_greek_exposure(self, greeks: Dict) -> None:
        """
        Log portfolio Greek exposures.

        Args:
            greeks: Dictionary of Greek values
        """
        formatted = {
            "delta": f"{greeks.get('delta', 0):.1f}",
            "vega": f"{greeks.get('vega', 0):.0f}",
            "gamma": f"{greeks.get('gamma', 0):.3f}",
            "theta": f"{greeks.get('theta', 0):.0f}",
        }

        self.log(
            f"GREEKS | Delta: {formatted['delta']} | Vega: {formatted['vega']} | "
            f"Gamma: {formatted['gamma']} | Theta: {formatted['theta']}",
            LogLevel.INFO,
            greeks,
        )

    def log_regime_change(self, old_regime: str, new_regime: str) -> None:
        """
        Log volatility regime change.

        Args:
            old_regime: Previous regime
            new_regime: New regime
        """
        self.regimes_today.append(new_regime)

        self.log(
            f"REGIME CHANGE: {old_regime} -> {new_regime}",
            LogLevel.WARNING,
            {"old_regime": old_regime, "new_regime": new_regime},
        )

    def log_tca_summary(self, tca_summary: Dict) -> None:
        """
        Log TCA summary from execution model.

        Args:
            tca_summary: TCA summary dictionary
        """
        self.log(
            f"TCA SUMMARY | Fills: {tca_summary.get('total_fills', 0)} | "
            f"Avg Slippage: {tca_summary.get('avg_slippage_bps', 0):.1f}bps | "
            f"Spread Capture: {tca_summary.get('avg_spread_capture_bps', 0):.1f}bps | "
            f"Fill Rate: {tca_summary.get('fill_rate', 0):.1%}",
            LogLevel.INFO,
            tca_summary,
        )

    def log_performance(self, checkpoint: str = "default") -> None:
        """
        Log current performance metrics.

        Args:
            checkpoint: Named checkpoint for tracking
        """
        if not self.enable_performance_logging:
            return

        portfolio = self.algorithm.Portfolio
        current_value = portfolio.TotalPortfolioValue

        metrics = {
            "portfolio_value": current_value,
            "cash": portfolio.Cash,
            "holdings_value": portfolio.TotalHoldingsValue,
            "unrealized_profit": portfolio.TotalUnrealizedProfit,
            "total_fees": portfolio.TotalFees,
            "net_profit": portfolio.TotalProfit,
            "return_pct": (
                (current_value / self.starting_capital - 1)
                if self.starting_capital > 0
                else 0
            ),
        }

        # Track peak and drawdown
        if checkpoint not in self.performance_checkpoints:
            self.performance_checkpoints[checkpoint] = {
                "peak": current_value,
                "metrics": [],
            }

        cp = self.performance_checkpoints[checkpoint]
        if current_value > cp["peak"]:
            cp["peak"] = current_value

        drawdown = (cp["peak"] - current_value) / cp["peak"] if cp["peak"] > 0 else 0
        metrics["drawdown"] = drawdown

        cp["metrics"].append({"time": self.algorithm.Time, **metrics})

        self.log(
            f"PERFORMANCE | NAV: ${current_value:,.2f} | "
            f"Return: {metrics['return_pct']:.2%} | Drawdown: {drawdown:.2%}",
            LogLevel.INFO,
            metrics,
        )

    def log_daily_summary(
        self, tca_summary: Dict = None, risk_summary: Dict = None
    ) -> None:
        """
        Generate and log end-of-day summary.

        Args:
            tca_summary: TCA summary from execution model
            risk_summary: Risk summary from risk model
        """
        if self.current_daily is None:
            return

        portfolio = self.algorithm.Portfolio
        ending_nav = portfolio.TotalPortfolioValue

        # Update daily summary
        daily = self.current_daily
        daily.ending_nav = ending_nav
        daily.daily_pnl = ending_nav - daily.starting_nav
        daily.daily_return_pct = (
            daily.daily_pnl / daily.starting_nav if daily.starting_nav > 0 else 0
        )
        daily.cumulative_pnl = ending_nav - self.starting_capital

        # From counters
        daily.signals_generated = self.daily_counters["signals"]
        daily.orders_submitted = self.daily_counters["orders_submitted"]
        daily.orders_filled = self.daily_counters["orders_filled"]
        daily.orders_cancelled = self.daily_counters["orders_cancelled"]
        daily.risk_events = self.daily_counters["risk_events"]
        daily.greek_breaches = self.daily_counters["greek_breaches"]
        daily.scenario_breaches = self.daily_counters["scenario_breaches"]

        # Regime info
        daily.regimes_seen = list(set(self.regimes_today))
        if self.regimes_today:
            from collections import Counter

            regime_counts = Counter(self.regimes_today)
            daily.dominant_regime = regime_counts.most_common(1)[0][0]

        # TCA from execution model
        if tca_summary:
            daily.total_fills = tca_summary.get("total_fills", 0)
            daily.avg_slippage_bps = tca_summary.get("avg_slippage_bps", 0)
            daily.avg_spread_capture_bps = tca_summary.get("avg_spread_capture_bps", 0)
            daily.net_execution_cost_bps = (
                daily.avg_slippage_bps - daily.avg_spread_capture_bps
            )

        # Greeks from risk snapshot
        if self.risk_snapshots:
            last_snapshot = self.risk_snapshots[-1]
            daily.eod_delta = last_snapshot.delta
            daily.eod_vega = last_snapshot.vega
            daily.eod_gamma = last_snapshot.gamma
            daily.eod_theta = last_snapshot.theta

        # Store summary
        self.daily_summaries.append(daily)

        # Log summary
        self.log(
            f"\n{'='*60}\n"
            f"DAILY SUMMARY - {daily.date}\n"
            f"{'='*60}\n"
            f"Starting NAV: ${daily.starting_nav:,.2f}\n"
            f"Ending NAV:   ${daily.ending_nav:,.2f}\n"
            f"Daily P&L:    ${daily.daily_pnl:,.2f} ({daily.daily_return_pct:.2%})\n"
            f"Cumulative:   ${daily.cumulative_pnl:,.2f}\n"
            f"Drawdown:     {daily.drawdown_pct:.2%}\n"
            f"\n"
            f"Trading Activity:\n"
            f"  Signals:    {daily.signals_generated}\n"
            f"  Orders:     {daily.orders_filled} filled / {daily.orders_cancelled} cancelled\n"
            f"  Risk Events: {daily.risk_events}\n"
            f"\n"
            f"End-of-Day Greeks:\n"
            f"  Delta: {daily.eod_delta:.0f} | Vega: {daily.eod_vega:.0f} | "
            f"Gamma: {daily.eod_gamma:.1f}\n"
            f"\n"
            f"Regime: {daily.dominant_regime}\n"
            f"TCA: Slippage {daily.avg_slippage_bps:.1f}bps | "
            f"Spread Capture {daily.avg_spread_capture_bps:.1f}bps\n"
            f"{'='*60}",
            LogLevel.INFO,
        )

    def _check_day_rollover(self) -> None:
        """Check for day change and initialize daily tracking."""
        current = self.algorithm.Time.date()

        if self.current_date != current:
            # New day - finalize previous and start new
            if self.current_daily:
                # Don't log here - wait for explicit log_daily_summary call
                pass

            # Start new daily
            self.current_date = current
            self.current_daily = DailySummary(
                date=current, starting_nav=self.algorithm.Portfolio.TotalPortfolioValue
            )

            # Update starting capital on first day
            if self.starting_capital == 1000000 and self.current_daily.starting_nav > 0:
                self.starting_capital = self.current_daily.starting_nav

            # Reset counters
            self.daily_counters = {k: 0 for k in self.daily_counters}
            self.regimes_today = []

    def _track_error(self, message: str) -> None:
        """Track error for aggregation."""
        error_type = message.split(":")[0] if ":" in message else "General"
        self.error_counts[error_type] += 1

        self.last_errors.append({"time": self.algorithm.Time, "message": message})

        if len(self.last_errors) > 20:
            self.last_errors.pop(0)

    def _buffer_for_file(self, log_entry: str, data: Dict = None) -> None:
        """Buffer log entry for batch file writing."""
        file_entry = {
            "timestamp": self.algorithm.Time.isoformat(),
            "message": log_entry,
            "data": data,
        }

        self.file_buffer.append(json.dumps(file_entry))

        if len(self.file_buffer) >= self.file_buffer_max:
            self._flush_file_buffer()

    def _flush_file_buffer(self) -> None:
        """Flush file buffer to object store."""
        if not self.file_buffer:
            return

        try:
            # Write to object store (QuantConnect's cloud storage)
            date_str = self.algorithm.Time.strftime("%Y%m%d")
            key = f"logs/oma_strategy_{date_str}.jsonl"

            content = "\n".join(self.file_buffer)

            # Append to existing or create new
            if self.algorithm.ObjectStore.ContainsKey(key):
                existing = self.algorithm.ObjectStore.Read(key)
                content = existing + "\n" + content

            self.algorithm.ObjectStore.Save(key, content)
            self.file_buffer = []

        except Exception as e:
            self.algorithm.Debug(f"Failed to flush log buffer: {e}")

    def _level_name(self, level: int) -> str:
        """Get string name for log level."""
        level_names = {
            LogLevel.TRACE: "TRACE",
            LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARN",
            LogLevel.ERROR: "ERROR",
        }
        return level_names.get(level, "INFO")

    def get_summary(self) -> Dict:
        """
        Get summary of logging activity.

        Returns:
            Dictionary with logging summary
        """
        return {
            "error_counts": dict(self.error_counts),
            "last_errors": self.last_errors,
            "trade_count": len(self.trade_log),
            "daily_summaries_count": len(self.daily_summaries),
            "risk_snapshots_count": len(self.risk_snapshots),
            "alpha_diagnostics_count": len(self.alpha_diagnostics),
        }

    def get_alpha_diagnostics_summary(self) -> Dict:
        """
        Get summary of alpha diagnostics.

        Returns:
            Dictionary with alpha diagnostic statistics
        """
        if not self.alpha_diagnostics:
            return {}

        import numpy as np

        iv_rv_ratios = [
            d.iv_rv_ratio for d in self.alpha_diagnostics if d.iv_rv_ratio > 0
        ]
        moneyness = [d.moneyness for d in self.alpha_diagnostics]
        dte = [d.dte for d in self.alpha_diagnostics]

        # Regime distribution
        from collections import Counter

        regime_dist = dict(Counter([d.regime for d in self.alpha_diagnostics]))

        return {
            "total_signals": len(self.alpha_diagnostics),
            "iv_rv_ratio_mean": np.mean(iv_rv_ratios) if iv_rv_ratios else 0,
            "iv_rv_ratio_median": np.median(iv_rv_ratios) if iv_rv_ratios else 0,
            "iv_rv_ratio_std": np.std(iv_rv_ratios) if iv_rv_ratios else 0,
            "avg_moneyness": np.mean(moneyness) if moneyness else 0,
            "avg_dte": np.mean(dte) if dte else 0,
            "regime_distribution": regime_dist,
        }

    def export_trade_log(self) -> List[Dict]:
        """
        Export trade log for analysis.

        Returns:
            List of trade entries
        """
        return self.trade_log.copy()

    def export_daily_summaries(self) -> List[Dict]:
        """
        Export daily summaries.

        Returns:
            List of daily summary dictionaries
        """
        return [asdict(s) for s in self.daily_summaries]

    def set_min_level(self, level: int) -> None:
        """
        Update minimum log level.

        Args:
            level: New minimum log level
        """
        self.min_level = level
        self.log(f"Log level updated to {self._level_name(level)}", LogLevel.INFO)

    def start_timer(self, operation: str) -> None:
        """Start a performance timer."""
        if not hasattr(self.algorithm, "_timers"):
            self.algorithm._timers = {}
        self.algorithm._timers[operation] = self.algorithm.Time

    def end_timer(self, operation: str) -> float:
        """End a performance timer and return elapsed time."""
        if not hasattr(self.algorithm, "_timers"):
            return 0

        if operation not in self.algorithm._timers:
            return 0

        start_time = self.algorithm._timers[operation]
        elapsed = (self.algorithm.Time - start_time).total_seconds()

        self.log(f"Timer '{operation}': {elapsed:.3f}s", LogLevel.DEBUG)

        del self.algorithm._timers[operation]
        return elapsed
