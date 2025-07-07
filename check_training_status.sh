#!/bin/bash

# Quick training status checker
echo "🔍 MARLIN v3 Training Status Check"
echo "=================================="

# Check for active training processes
PIDS=$(pgrep -f train_contextual_transformer_cli.py)

if [ -z "$PIDS" ]; then
    echo "❌ No active training processes found"
    
    # Check if models exist
    if [ -f "model/klax_contextual_transformer_cli.pth" ]; then
        echo "✅ CLI model file exists"
        echo "📁 Model: model/klax_contextual_transformer_cli.pth"
        echo "📁 Scaler: model/contextual_cli_scaler.pkl"
        echo "📊 Size: $(ls -lh model/klax_contextual_transformer_cli.pth | awk '{print $5}')"
        echo "🕐 Modified: $(stat -f "%Sm" model/klax_contextual_transformer_cli.pth)"
    else
        echo "❌ No CLI model file found yet"
    fi
else
    echo "🟢 Active training processes:"
    for pid in $PIDS; do
        echo "   PID: $pid"
        # Show process info
        ps -p $pid -o pid,ppid,%cpu,%mem,etime,command
    done
    
    echo ""
    echo "📈 System Resources:"
    echo "   CPU Usage: $(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')"
    echo "   Memory: $(top -l 1 | grep "PhysMem" | awk '{print $2}')"
fi

echo ""
echo "📁 Model Directory:"
ls -la model/ | grep -E "(cli|contextual)"

echo ""
echo "💾 Database Status:"
if [ -f "data/klax_weather.db" ]; then
    echo "   Database size: $(ls -lh data/klax_weather.db | awk '{print $5}')"
    echo "   Last modified: $(stat -f "%Sm" data/klax_weather.db)"
else
    echo "   ❌ Database not found"
fi

echo ""
echo "📊 Recent logs (if any):"
if [ -f "data/overnight_training_log.csv" ]; then
    echo "   Log file exists: data/overnight_training_log.csv"
    tail -n 3 data/overnight_training_log.csv
else
    echo "   No overnight training log yet"
fi 