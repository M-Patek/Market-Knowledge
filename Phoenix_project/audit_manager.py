# audit_manager.py
import os
import logging
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def archive_logs_to_s3(source_dir: str, bucket_name: str):
    """
    Archives all log files from a local directory to an S3 bucket and deletes them locally.

    Args:
        source_dir (str): The local directory containing the audit logs.
        bucket_name (str): The name of the target S3 bucket.
    """
    logger = logging.getLogger("PhoenixProject.AuditManager")

    if not os.path.isdir(source_dir):
        logger.info(f"Audit log source directory '{source_dir}' not found. Skipping archiving.")
        return

    try:
        s3_client = boto3.client('s3')
        logger.info(f"Starting archival of audit logs from '{source_dir}' to S3 bucket '{bucket_name}'.")
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("AWS credentials not found or incomplete. Skipping S3 archival.")
        return

    archived_count = 0
    log_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    if not log_files:
        logger.info("No audit logs found to archive.")
        return

    for filename in log_files:
        local_path = os.path.join(source_dir, filename)
        try:
            # Enforce server-side encryption for data at rest
            extra_args = {'ServerSideEncryption': 'AES256'}
            s3_client.upload_file(
                local_path, bucket_name, filename, ExtraArgs=extra_args
            )
            logger.debug(f"Successfully uploaded {filename} to S3.")
            os.remove(local_path)
            archived_count += 1
        except ClientError as e:
            logger.error(f"Failed to upload {filename} to S3: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {filename}: {e}")

    logger.info(f"Completed S3 archival. Archived {archived_count} of {len(log_files)} log files.")
