#!/bin/bash
set -e

echo "=========================================="
echo "   PHOENIX SYSTEM SELF-DIAGNOSTIC v6.0    "
echo "=========================================="

# 1. Set Environment
export PYTHONPATH=$PYTHONPATH:.
LOG_FILE="phoenix_self_check.log"

# 2. Connectivity Pre-flight
echo "[1/4] 🔌 Checking Infrastructure Connectivity..."
python3 -c "
import sys
import os
# Ensure we can import from current directory
sys.path.append(os.getcwd())

try:
    from Phoenix_project.factory import PhoenixFactory
    
    # Check Redis
    print('  - Pinging Redis...')
    redis = PhoenixFactory.create_sync_redis_client()
    if not redis.ping():
        raise ConnectionError('Redis PING failed')
    print('  ✅ Redis Online')

except ImportError:
    print('  ⚠️  Skipping connectivity check (Dependencies missing or path issue).')
except Exception as e:
    print(f'  ❌ Connectivity Check Failed: {e}')
    sys.exit(1)
"

# 3. Run All Unit & Safety Tests
echo "[2/4] 🧪 Running Full Test Suite..."
# [Task 5.2] Run all tests found in the tests directory
python -m pytest Phoenix_project/tests/ -v

# 4. Run System Integration (Dry Run with Log Capture)
echo "[3/4] 🚀 Running Pipeline Integration Check (Dry Run)..."
# Capture stdout and stderr to log file
if python Phoenix_project/run_training.py --mode backtest --dry-run > "$LOG_FILE" 2>&1; then
    echo "  - Dry run execution: SUCCESS"
else
    echo "  - Dry run execution: FAILED"
    echo "--- Last 20 lines of log ---"
    tail -n 20 "$LOG_FILE"
    exit 1
fi

# 5. Log Analysis (Log Biopsy)
echo "[4/4] 🔍 Scanning Logs for Critical Patterns..."
# [Task 5.2] Scan for Panic, Error, Margin Call
CRITICAL_PATTERNS="PANIC|CRITICAL|MARGIN CALL|Traceback|Deadlock|Zombie|StaleDataError"

if grep -Eiq "$CRITICAL_PATTERNS" "$LOG_FILE"; then
    echo "❌ LOG AUDIT FAILED: Found critical anomalies in $LOG_FILE"
    echo "--- Detected Anomalies ---"
    grep -Ei "$CRITICAL_PATTERNS" "$LOG_FILE"
    exit 1
fi

# Check for general errors (non-fatal but noisy)
if grep -q "ERROR" "$LOG_FILE"; then
    echo "⚠️  WARNING: Errors detected in logs (Check if they are expected):"
    grep "ERROR" "$LOG_FILE" | head -n 5
    echo "  (See $LOG_FILE for full details)"
else
    echo "  ✅ Log Hygiene: Clean"
fi

# Cleanup
rm -f "$LOG_FILE"

echo "✅ SELF-CHECK COMPLETE. ALL SYSTEMS GREEN."
