#!/bin/bash
# Phoenix_project/scripts/self_check.sh
# Task 22: Automated integration testing.

set -e

echo "Running automated integration test..."
python scripts/run_cli.py --ticker "NVDA"

echo "Running unit tests..."
pytest -q
