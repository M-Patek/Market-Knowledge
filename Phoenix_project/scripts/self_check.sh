#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "[1] Install Dependencies"
pip install -r requirements.txt

echo "[2] API Smoke Test"
python Phoenix_project/api/gateway.py &
PID=$!
sleep 2 # Wait for server to start
curl -s "http://12_7.0.0.1:5000/analyze?ticker=NVDA" | head
curl -s "http://127.0.0.1:5000/metrics" | head
kill $PID || true # Kill server, || true in case it already died

echo "[3] CLI Smoke Test"
# This command will now correctly fail the script if it errors
python Phoenix_project/run.py --ticker NVDA

echo "[4] Lint Check"
# This command will now correctly fail the script if ruff finds issues
ruff check Phoenix_project

echo "DONE: All checks passed."
