#!/bin/bash
# Phoenix_project/scripts/self_check.sh
# Task 22: Automated integration testing.

set -e

echo "Running automated integration test..."
# 修复：[FIX-10] 'run_cli.py' 不接受 '--ticker' 参数。
# 在我们知道正确的参数之前，先不带参数运行它。
python scripts/run_cli.py

echo "Running unit tests..."
pytest -q
