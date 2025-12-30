#!/bin/bash
# OMA Strategy Setup Script
# Automates environment setup for Option Market Albatross

set -e  # Exit on error

echo "🚀 Option Market Albatross - Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✅ Conda found${NC}"
    USE_CONDA=true
else
    echo -e "${YELLOW}⚠️  Conda not found, will use venv instead${NC}"
    USE_CONDA=false
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.11+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Python version: $PYTHON_VERSION${NC}"
fi

# Create environment
if [ "$USE_CONDA" = true ]; then
    echo ""
    echo "📦 Creating Conda environment..."
    conda env create -f environment.yml
    echo -e "${GREEN}✅ Conda environment 'oma-bot' created${NC}"
    echo ""
    echo "To activate: conda activate oma-bot"
else
    echo ""
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    echo ""
    echo "To activate: source venv/bin/activate"
fi

# Install root dependencies
echo ""
echo "📦 Installing root dependencies..."
if [ "$USE_CONDA" = true ]; then
    conda activate oma-bot || source activate oma-bot
else
    source venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✅ Root dependencies installed${NC}"

# Install strategy dependencies
echo ""
echo "📦 Installing strategy dependencies..."
cd QuantConnectOmaStrategy
pip install -r requirements.txt
cd ..
echo -e "${GREEN}✅ Strategy dependencies installed${NC}"

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python3 << EOF
try:
    import numpy
    import pandas
    import scipy
    print("✅ Core dependencies: OK")
    
    import sys
    sys.path.insert(0, 'QuantConnectOmaStrategy')
    from alpha.iv_rv_spread_alpha import IVRVSpreadAlphaModel
    from portfolio.delta_vega_neutral import DeltaVegaNeutralPortfolioConstructionModel
    print("✅ Strategy modules: OK")
    print("\n🎉 Setup completed successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "📋 Next Steps:"
echo ""
echo "1. Activate environment:"
if [ "$USE_CONDA" = true ]; then
    echo "   conda activate oma-bot"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Upload to QuantConnect:"
echo "   - Sign up at https://www.quantconnect.com"
echo "   - Upload QuantConnectOmaStrategy/ folder"
echo "   - Run backtest in cloud"
echo ""
echo "3. Or use Lean CLI locally:"
echo "   pip install lean"
echo "   lean init"
echo "   lean backtest"
echo ""
echo "📖 See INSTALL.md for detailed instructions"
echo ""

