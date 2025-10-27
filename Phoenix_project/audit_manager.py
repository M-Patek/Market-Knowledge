# audit_manager.py
import os
import json
import logging
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from typing import Dict, Any

class AuditManager:
    """
    Handles the logging of critical system decisions for audit, traceability,
    and offline analysis, including shadow model performance.
    """
    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.AuditManager")
        self.logger.info("AuditManager initialized.")

    def log_shadow_decision(self, champion_decision: Dict[str, Any], shadow_decision: Dict[str, Any]):
        """
        [Sub-Task 1.1.3] Logs the output of the champion and shadow models for parallel comparison.
        In a real system, this would write to a structured, queryable database (e.g., Elasticsearch).
        """
        log_record = {
            "comparison_type": "SHADOW_VS_CHAMPION",
            "champion_decision": champion_decision,
            "shadow_decision": shadow_decision
        }
        # For now, we log to a dedicated logger.
        self.logger.info(f"SHADOW_LOG: {json.dumps(log_record)}")

    def log_decision_audit_trail(self, decision_lineage: Dict[str, Any]):
        """
        [Sub-Task 3.2.1] Logs the full, complex JSON object representing the
        entire lineage of a single cognitive engine decision.
        """
        if not decision_lineage.get("decision_id"):
            self.logger.error("Failed to log audit trail: 'decision_id' is missing.")
            return

        self.logger.info(f"AUDIT_TRAIL: {json.dumps(decision_lineage)}")

    def archive_logs_to_s3(self, source_dir: str, bucket_name: str):
        """
        Archives all log files from a local directory to an S3 bucket and deletes them locally.
        """
        if not os.path.isdir(source_dir):
            self.logger.info(f"Audit log source directory '{source_dir}' not found. Skipping archiving.")
            return

        try:
            s3_client = boto3.client('s3')
            self.logger.info(f"Starting archival of audit logs from '{source_dir}' to S3 bucket '{bucket_name}'.")
        except (NoCredentialsError, PartialCredentialsError):
            self.logger.error("AWS credentials not found or incomplete. Skipping S3 archival.")
            return

        archived_count = 0
        log_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        if not log_files:
            self.logger.info("No audit logs found to archive.")
            return

        for filename in log_files:
            local_path = os.path.join(source_dir, filename)
            try:
                # Enforce server-side encryption for data at rest
                extra_args = {'ServerSideEncryption': 'AES256'}
                s3_client.upload_file(
                    local_path, bucket_name, filename, ExtraArgs=extra_args
                )
                self.logger.debug(f"Successfully uploaded {filename} to S3.")
                os.remove(local_path)
                archived_count += 1
            except ClientError as e:
                self.logger.error(f"Failed to upload {filename} to S3: {e}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while processing {filename}: {e}")

        self.logger.info(f"Completed S3 archival. Archived {archived_count} of {len(log_files)} log files.")
