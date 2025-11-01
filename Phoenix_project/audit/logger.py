# Phoenix_project/audit/logger.py
import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger("PhoenixProject.AuditLogger")

# TODO: Load this path from the central config (Task 20)
_log_path = os.getenv("FULL_TRACE_LOG_PATH", "logs/full_run_trace.jsonl")

def save_run(metadata: dict, paths: list[dict], fusion: dict) -> None:
    """Log file must contain timestamp, task_id, Agent outputs, and fusion result."""
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata,
        "task_id": metadata.get("task_id", "UNKNOWN"),
        "agent_outputs": paths,
        "fusion_result": fusion
    }
    
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)

    try:
        with open(_log_path, 'a') as f:
            logger.info(f"AuditLogger: Logging full trace for task {log_entry['task_id']}")
            f.write(json.dumps(log_entry) + '\n')
    except IOError as e:
        logger.error(f"AuditLogger: Error writing to full trace log: {e}")
