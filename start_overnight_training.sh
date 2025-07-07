#!/bin/bash

# MARLIN v3 Overnight CLI Training Launcher
# Usage: ./start_overnight_training.sh [max_runs] [max_hours]
# Example: ./start_overnight_training.sh 15 6

echo "🌙 MARLIN v3 Overnight CLI Training Launcher"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create .venv first."
    exit 1
fi

# Default parameters
MAX_RUNS=${1:-20}
MAX_HOURS=${2:-8}

echo "⚙️  Configuration:"
echo "   Max runs: $MAX_RUNS"
echo "   Max hours: $MAX_HOURS"
echo "   Training script: code/train_contextual_transformer_cli.py"
echo "   Results log: data/overnight_training_log.csv"
echo "   Best model dir: model/best_cli/"

echo ""
read -p "🚀 Start overnight training? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "✅ Starting overnight training..."
    
    # Activate virtual environment and run training
    source .venv/bin/activate && python code/overnight_cli_training.py $MAX_RUNS $MAX_HOURS
    
    echo ""
    echo "🏁 Overnight training complete!"
    echo "📊 Check results in data/overnight_training_log.csv"
    echo "🏆 Best model saved in model/best_cli/"
else
    echo "⏹️  Training cancelled."
fi 