#!/bin/bash

# MARLIN v3 Training Monitor Launcher
echo "🔍 MARLIN v3 Training Monitor"
echo "============================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create .venv first."
    exit 1
fi

# Default monitoring duration
HOURS=${1:-8}

echo "⚙️  Starting monitoring for $HOURS hours"
echo "   Monitor updates every 30 seconds"
echo "   Press Ctrl+C to stop monitoring"
echo ""

# Activate virtual environment and run monitor
source .venv/bin/activate && python monitor_training.py $HOURS 