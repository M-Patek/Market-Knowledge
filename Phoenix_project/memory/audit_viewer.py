# Phoenix_project/memory/audit_viewer.py
import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("PhoenixProject.AuditViewer")

# This must match the output file from `audit/logger.py` (Task 17)
_log_path = os.getenv("FULL_TRACE_LOG_PATH", "logs/full_run_trace.jsonl")

def _load_reports() -> List[Dict[str, Any]]:
    """Helper to load all reports from the JSONL log file."""
    if not os.path.exists(_log_path):
        logger.warning(f"Audit log file not found at: {_log_path}")
        return []
    
    reports = []
    try:
        with open(_log_path, 'r') as f:
            for line in f:
                reports.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading audit log: {e}")
    return reports

def list_reports_by_task_id(task_id: str) -> List[Dict[str, Any]]:
    """Must be able to list historical reports by task_id."""
    all_reports = _load_reports()
    filtered_reports = [r for r in all_reports if r.get("task_id") == task_id]
    logger.info(f"Found {len(filtered_reports)} reports for task_id: {task_id}")
    return filtered_reports
