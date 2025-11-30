#!/bin/bash
set -e

echo "=========================================="
echo "   PHOENIX SYSTEM SELF-DIAGNOSTIC v5.0    "
echo "=========================================="

# 1. Set Environment
export PYTHONPATH=$PYTHONPATH:.

# 2. Run Unit & Safety Tests
echo "[1/2] Running Safety & Integrity Tests..."
python -m pytest tests/test_safety_mechanisms.py -v

# 3. Run System Integration (Dry Run)
echo "[2/2] Running Pipeline Integration Check (Dry Run)..."
python Phoenix_project/run_training.py --mode backtest --dry-run

echo "✅ SELF-CHECK COMPLETE. ALL SYSTEMS GREEN."
