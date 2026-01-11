#!/bin/bash
# Quick Start Script for Polymarket Arbitrage Scanner
# Run: chmod +x setup.sh && ./setup.sh

echo "================================================"
echo "Polymarket Arbitrage Scanner - Setup"
echo "================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Test import
echo ""
echo "Testing imports..."
python3 -c "import requests; from rich.console import Console; print('✓ All imports successful')"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Quick Start Commands:"
echo ""
echo "  1. Scan for opportunities:"
echo "     python3 polymarket_arb.py"
echo ""
echo "  2. Continuous monitoring:"
echo "     python3 polymarket_arb.py --watch"
echo ""
echo "  3. Real-time price monitor:"
echo "     python3 price_monitor.py"
echo ""
echo "For trading, set these environment variables:"
echo "  export POLY_PRIVATE_KEY='your-key'"
echo "  export POLY_FUNDER_ADDRESS='your-address'"
echo ""
echo "Get your key from: https://reveal.magic.link/polymarket"
echo ""
