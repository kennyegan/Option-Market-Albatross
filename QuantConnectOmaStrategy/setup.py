#!/usr/bin/env python3
"""
QuantConnect OMA Strategy Setup Script
Automates environment setup and dependency installation.

Author: OMA Strategy Team
Version: 1.0
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.11 or higher."""
    if sys.version_info < (3, 11):
        print("❌ Error: Python 3.11 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")


def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)


def check_lean_installation():
    """Check if QuantConnect Lean CLI is installed."""
    try:
        result = subprocess.run(["lean", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Lean CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Lean CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Lean CLI not installed")
        return False


def install_lean_cli():
    """Install QuantConnect Lean CLI."""
    print("🔧 Installing QuantConnect Lean CLI...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lean"])
        print("✅ Lean CLI installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Lean CLI: {e}")
        sys.exit(1)


def setup_lean_config():
    """Initialize Lean configuration if needed."""
    lean_config_path = Path.home() / ".lean" / "config.json"

    if not lean_config_path.exists():
        print("🔧 Setting up Lean configuration...")
        try:
            subprocess.check_call(["lean", "init"])
            print("✅ Lean configuration initialized")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not initialize Lean config: {e}")
            print("You may need to run 'lean init' manually")


def validate_strategy_files():
    """Validate that all required strategy files exist."""
    required_files = [
        "main.py",
        "lean.json",
        "requirements.txt",
        "alpha/iv_rv_spread_alpha.py",
        "execution/smart_router.py",
        "portfolio/delta_vega_neutral.py",
        "risk/exposure_limits.py",
        "data/realized_vol_calc.py",
        "utils/logger.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)

    print("✅ All strategy files present")


def test_strategy_compilation():
    """Test that the strategy compiles without errors."""
    print("🧪 Testing strategy compilation...")
    try:
        subprocess.check_call([sys.executable, "-m", "py_compile", "main.py"])
        print("✅ Main strategy file compiles successfully")

        # Test all Python files
        python_files = list(Path(".").rglob("*.py"))
        for py_file in python_files:
            if "__pycache__" not in str(py_file):
                subprocess.check_call(
                    [sys.executable, "-m", "py_compile", str(py_file)]
                )

        print("✅ All strategy modules compile successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation error: {e}")
        sys.exit(1)


def create_sample_backtest():
    """Create a sample backtest configuration for testing."""
    sample_config = {
        "algorithm-id": "oma-options-arbitrage-sample",
        "algorithm-language": "Python",
        "parameters": {
            "iv-rv-threshold": "1.2",
            "spread-threshold": "0.005",
            "max-position-size": "0.02",  # Smaller for testing
            "vega-limit": "5000",  # Reduced for testing
            "delta-tolerance": "50",
            "max-daily-loss": "0.01",  # Conservative for testing
        },
        "environments": {
            "backtesting": {
                "live-mode": False,
                "setup-handler": "QuantConnect.Lean.Engine.Setup.BacktestingSetupHandler",
                "result-handler": "QuantConnect.Lean.Engine.Results.BacktestingResultHandler",
                "data-feed-handler": "QuantConnect.Lean.Engine.DataFeeds.FileSystemDataFeed",
                "real-time-handler": "QuantConnect.Lean.Engine.RealTime.BacktestingRealTimeHandler",
                "history-provider": "QuantConnect.Lean.Engine.HistoricalData.SubscriptionDataReaderHistoryProvider",
                "transaction-handler": "QuantConnect.Lean.Engine.TransactionHandlers.BacktestingTransactionHandler",
            }
        },
        "data-folder": "../../../Data/",
        "debugging": False,
        "debugging-method": "LocalCmdline",
        "log-handler": "QuantConnect.Logging.FileLogHandler",
        "messaging-handler": "QuantConnect.Messaging.EventMessagingHandler",
    }

    with open("lean_sample.json", "w") as f:
        json.dump(sample_config, f, indent=4)

    print("✅ Sample backtest configuration created (lean_sample.json)")


def main():
    parser = argparse.ArgumentParser(description="Setup QuantConnect OMA Strategy")
    parser.add_argument(
        "--skip-lean", action="store_true", help="Skip Lean CLI installation"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only run tests, skip installation"
    )
    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample configuration"
    )

    args = parser.parse_args()

    print("🚀 QuantConnect OMA Strategy Setup")
    print("=" * 50)

    # Always check Python version and validate files
    check_python_version()
    validate_strategy_files()

    if not args.test_only:
        # Install dependencies
        install_dependencies()

        # Handle Lean CLI
        if not args.skip_lean:
            if not check_lean_installation():
                install_lean_cli()
            setup_lean_config()

    # Test compilation
    test_strategy_compilation()

    # Create sample config if requested
    if args.create_sample:
        create_sample_backtest()

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review and adjust parameters in lean.json")
    print("2. Run backtest: lean backtest")
    print("3. Check results and logs")
    print("4. Paper trade before going live")

    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main()
