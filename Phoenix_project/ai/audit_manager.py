import json
import hashlib

# Configure logger for this module (Layer 12)
from observability import get_logger
logger = get_logger(__name__)


def _generate_hash(data: any) -> str:
    """
    Creates a stable SHA-256 hash for any JSON-serializable data.
    """
    if data is None:
        return "none"
    data_string = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data_string.encode('utf-8')).hexdigest()


class AuditManager:
    """
    Manages an append-only audit log for traceability.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path

    def log(self, task_id: str, task_name: str, input_hash: str, output_hash: str, details: dict = None):
        """
        Logs a structured record to the audit trail, including input/output hashes
        for version tracking (Layer 12).
        """
        record = {
            "task_id": task_id,
            "task_name": task_name,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "details": details or {}
        }

        try:
            with open(self.log_path, 'a') as f:
                logger.info(f"AuditManager: Logging record for {task_id} ({task_name})")
                f.write(json.dumps(record) + '\n')
        except IOError as e:
            logger.error(f"AuditManager: Error writing to audit log: {e}")
