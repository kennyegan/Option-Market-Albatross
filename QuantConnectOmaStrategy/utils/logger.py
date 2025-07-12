"""
Strategy Logger Utility
Provides consistent logging functionality across all strategy modules.

Author: OMA Strategy Team
Version: 1.0
"""

from AlgorithmImports import *
from typing import Dict, List
from datetime import datetime
import json


class LogLevel:
    """Log level constants matching QuantConnect's levels."""
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    

class StrategyLogger:
    """
    Centralized logger for the OMA strategy with features:
    1. Consistent formatting
    2. Log level filtering
    3. Performance metrics tracking
    4. Trade logging
    5. Error aggregation
    """
    
    def __init__(self, 
                 algorithm: QCAlgorithm,
                 min_level: int = LogLevel.INFO,
                 enable_performance_logging: bool = True):
        """
        Initialize the strategy logger.
        
        Args:
            algorithm: Algorithm instance
            min_level: Minimum log level to output
            enable_performance_logging: Whether to log performance metrics
        """
        self.algorithm = algorithm
        self.min_level = min_level
        self.enable_performance_logging = enable_performance_logging
        
        # Track errors for reporting
        self.error_counts = {}
        self.last_errors = []
        
        # Performance tracking
        self.performance_checkpoints = {}
        
        # Trade logging
        self.trade_log = []
        
        # Log initialization
        self.log("Strategy Logger initialized", LogLevel.INFO)
    
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
        
        # Format timestamp
        timestamp = self.algorithm.Time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Build log entry
        log_entry = f"[{timestamp}] [{self._level_name(level)}] {message}"
        
        # Add structured data if provided
        if data:
            log_entry += f" | Data: {json.dumps(data, default=str)}"
        
        # Use appropriate QC logging method
        if level == LogLevel.ERROR:
            self.algorithm.Error(log_entry)
            self._track_error(message)
        elif level == LogLevel.DEBUG:
            self.algorithm.Debug(log_entry)
        else:
            self.algorithm.Log(log_entry)
    
    def log_trade(self, 
                  symbol: Symbol,
                  action: str,
                  quantity: int,
                  price: float,
                  reason: str = "",
                  metadata: Dict = None) -> None:
        """
        Log a trade execution.
        
        Args:
            symbol: Traded symbol
            action: Trade action (BUY/SELL/OPEN/CLOSE)
            quantity: Trade quantity
            price: Execution price
            reason: Reason for trade
            metadata: Additional trade metadata
        """
        trade_entry = {
            'time': self.algorithm.Time,
            'symbol': str(symbol),
            'action': action,
            'quantity': quantity,
            'price': price,
            'reason': reason,
            'portfolio_value': self.algorithm.Portfolio.TotalPortfolioValue
        }
        
        if metadata:
            trade_entry.update(metadata)
        
        self.trade_log.append(trade_entry)
        
        # Log to main log
        self.log(
            f"TRADE: {action} {quantity} {symbol} @ ${price:.2f} - {reason}",
            LogLevel.INFO,
            trade_entry
        )
    
    def log_performance(self, checkpoint: str = "default") -> None:
        """
        Log current performance metrics.
        
        Args:
            checkpoint: Named checkpoint for tracking
        """
        if not self.enable_performance_logging:
            return
        
        metrics = {
            'portfolio_value': self.algorithm.Portfolio.TotalPortfolioValue,
            'cash': self.algorithm.Portfolio.Cash,
            'holdings_value': self.algorithm.Portfolio.TotalHoldingsValue,
            'unrealized_profit': self.algorithm.Portfolio.TotalUnrealizedProfit,
            'total_fees': self.algorithm.Portfolio.TotalFees,
            'net_profit': self.algorithm.Portfolio.TotalProfit,
            'return': self.algorithm.Portfolio.TotalPortfolioValue / 1000000 - 1  # Assuming $1M start
        }
        
        # Calculate drawdown
        if checkpoint not in self.performance_checkpoints:
            self.performance_checkpoints[checkpoint] = {
                'peak': metrics['portfolio_value'],
                'metrics': []
            }
        
        checkpoint_data = self.performance_checkpoints[checkpoint]
        if metrics['portfolio_value'] > checkpoint_data['peak']:
            checkpoint_data['peak'] = metrics['portfolio_value']
        
        drawdown = (checkpoint_data['peak'] - metrics['portfolio_value']) / checkpoint_data['peak']
        metrics['drawdown'] = drawdown
        
        # Store metrics
        checkpoint_data['metrics'].append({
            'time': self.algorithm.Time,
            **metrics
        })
        
        # Log summary
        self.log(
            f"Performance Update - Value: ${metrics['portfolio_value']:,.2f} "
            f"Return: {metrics['return']:.2%} Drawdown: {drawdown:.2%}",
            LogLevel.INFO,
            metrics
        )
    
    def log_risk_metrics(self, risk_metrics: Dict) -> None:
        """
        Log risk management metrics.
        
        Args:
            risk_metrics: Dictionary of risk metrics
        """
        self.log(
            "Risk Metrics Update",
            LogLevel.INFO,
            risk_metrics
        )
    
    def log_greek_exposure(self, greeks: Dict) -> None:
        """
        Log portfolio Greek exposures.
        
        Args:
            greeks: Dictionary of Greek values
        """
        formatted_greeks = {
            'delta': f"{greeks.get('delta', 0):.1f}",
            'vega': f"{greeks.get('vega', 0):.0f}",
            'gamma': f"{greeks.get('gamma', 0):.3f}",
            'theta': f"{greeks.get('theta', 0):.0f}"
        }
        
        self.log(
            f"Greek Exposure - Delta: {formatted_greeks['delta']} "
            f"Vega: {formatted_greeks['vega']} "
            f"Gamma: {formatted_greeks['gamma']} "
            f"Theta: {formatted_greeks['theta']}",
            LogLevel.INFO,
            greeks
        )
    
    def log_signal(self,
                   signal_type: str,
                   symbol: Symbol,
                   strength: float,
                   metadata: Dict = None) -> None:
        """
        Log alpha signal generation.
        
        Args:
            signal_type: Type of signal (IV_RV, SPREAD, etc.)
            symbol: Symbol generating signal
            strength: Signal strength
            metadata: Additional signal data
        """
        signal_data = {
            'type': signal_type,
            'symbol': str(symbol),
            'strength': strength,
            'time': self.algorithm.Time
        }
        
        if metadata:
            signal_data.update(metadata)
        
        self.log(
            f"Signal Generated - {signal_type} for {symbol} "
            f"Strength: {strength:.2f}",
            LogLevel.DEBUG,
            signal_data
        )
    
    def start_timer(self, operation: str) -> None:
        """
        Start a performance timer.
        
        Args:
            operation: Name of operation being timed
        """
        if hasattr(self.algorithm, '_timers'):
            self.algorithm._timers[operation] = self.algorithm.Time
    
    def end_timer(self, operation: str) -> float:
        """
        End a performance timer and log result.
        
        Args:
            operation: Name of operation being timed
            
        Returns:
            Elapsed time in seconds
        """
        if not hasattr(self.algorithm, '_timers'):
            return 0
        
        if operation not in self.algorithm._timers:
            return 0
        
        start_time = self.algorithm._timers[operation]
        elapsed = (self.algorithm.Time - start_time).total_seconds()
        
        self.log(
            f"Operation '{operation}' completed in {elapsed:.3f}s",
            LogLevel.DEBUG
        )
        
        del self.algorithm._timers[operation]
        return elapsed
    
    def _track_error(self, message: str) -> None:
        """Track error for aggregation."""
        # Count error types
        error_type = message.split(':')[0] if ':' in message else 'General'
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Keep last N errors
        self.last_errors.append({
            'time': self.algorithm.Time,
            'message': message
        })
        if len(self.last_errors) > 10:
            self.last_errors.pop(0)
    
    def _level_name(self, level: int) -> str:
        """Get string name for log level."""
        level_names = {
            LogLevel.TRACE: "TRACE",
            LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARN",
            LogLevel.ERROR: "ERROR"
        }
        return level_names.get(level, "INFO")
    
    def get_summary(self) -> Dict:
        """
        Get summary of logging activity.
        
        Returns:
            Dictionary with logging summary
        """
        summary = {
            'error_counts': self.error_counts,
            'last_errors': self.last_errors,
            'trade_count': len(self.trade_log),
            'checkpoints': list(self.performance_checkpoints.keys())
        }
        
        # Add latest performance metrics
        for checkpoint, data in self.performance_checkpoints.items():
            if data['metrics']:
                latest = data['metrics'][-1]
                summary[f'{checkpoint}_performance'] = {
                    'value': latest['portfolio_value'],
                    'return': latest['return'],
                    'drawdown': latest['drawdown']
                }
        
        return summary
    
    def export_trade_log(self) -> List[Dict]:
        """
        Export trade log for analysis.
        
        Returns:
            List of trade entries
        """
        return self.trade_log.copy()
    
    def set_min_level(self, level: int) -> None:
        """
        Update minimum log level.
        
        Args:
            level: New minimum log level
        """
        self.min_level = level
        self.log(f"Log level updated to {self._level_name(level)}", LogLevel.INFO)