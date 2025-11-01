import json
import os
import hashlib

# Configure logger for this module (Layer 12)
from monitor.logging import get_logger
logger = get_logger("PhoenixProject.CoTDatabase")


def _generate_hash(data: any) -> str:
    """
    Creates a stable SHA-256 hash for any JSON-serializable data.
    """
    if data is None:
        return "none"
    data_string = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()


# TODO: Load this path from the central config (Task 20)
_log_path = os.getenv("AUDIT_LOG_PATH", "logs/cot_trace.jsonl")

def save_trace(trace_json: dict) -> None:
    """Must be able to save a jsonl file after every reasoning run."""
    
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)

    try:
        with open(_log_path, 'a') as f:
            logger.info(f"CoTDatabase: Logging trace for task {trace_json.get('task_id', 'UNKNOWN')}")
            f.write(json.dumps(trace_json) + '\n')
    except IOError as e:
        logger.error(f"CoTDatabase: Error writing to trace log: {e}")
