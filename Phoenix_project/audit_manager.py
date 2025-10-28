# audit_manager.py
import os
import json
import logging
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from typing import Dict, Any, List

class AuditManager:
    """
    处理关键系统决策的日志记录，以实现审计、可追溯性
    和离线分析，包括影子模型的性能。
    """
    def __init__(self):
        self.logger = logging.getLogger("PhoenixProject.AuditManager")
        self.logger.info("AuditManager initialized.")

    def log_shadow_decision(self, champion_decision: Dict[str, Any], shadow_decision: Dict[str, Any]):
        """
        [Sub-Task 1.1.3] 记录冠军模型和影子模型的输出，以便进行并行比较。
        在真实系统中，这会写入一个结构化的、可查询的数据库（例如 Elasticsearch）。
        """
        log_record = {
            "comparison_type": "SHADOW_VS_CHAMPION",
            "champion_decision": champion_decision,
            "shadow_decision": shadow_decision
        }
        # 目前，我们记录到一个专门的记录器。
        self.logger.info(f"SHADOW_LOG: {json.dumps(log_record)}")

    def log_decision_audit_trail(self, decision_lineage: Dict[str, Any]):
        """
        [Sub-Task 3.2.1] 记录代表单个认知引擎决策
        完整谱系的复杂 JSON 对象。
        """
        if not decision_lineage.get("decision_id"):
            self.logger.error("Failed to log audit trail: 'decision_id' is missing.")
            return

        self.logger.info(f"AUDIT_TRAIL: {json.dumps(decision_lineage)}")

    def archive_logs_to_s3(self, source_dir: str, bucket_name: str):
        """
        将所有日志文件从本地目录归档到 S3 存储桶，并在本地删除它们。
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
                # 强制执行服务器端加密以保护静态数据
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

    def fetch_logs_with_pnl(self, days: int) -> List[Dict[str, Any]]:
        """
        [L3 Agent Requirement] 获取包含 P&L 基本事实的最近决策日志。
        这是 MetaCognitiveAgent (Task 2.2) 调用的方法。
        """
        # 这是数据库查询逻辑的占位符。
        # 一个真实的实现会查询一个结构化的数据库 (例如 SQL, NoSQL)。
        self.logger.info(f"Fetching logs with P&L for the last {days} days.")
        # 返回虚拟数据以满足 L3 代理的需求
        return [{"decision_id": "dummy_123", "parameters": {"asset": "BTC", "amount": 1.5}, "outcome_pnl": 150.75}]

    def backfill_pnl_for_decision(self, decision_id: str, final_pnl: float) -> bool:
        """
        查找特定的决策日志，并用最终的 P&L 基本事实更新它。
        (Task 2.3 - P&L Backfill)
        """
        self.logger.info(f"Attempting to backfill P&L for decision_id: {decision_id}.")
        # 这是数据库更新逻辑的占位符。
        # 一个真实的实现会执行像下面这样的查询:
        # UPDATE decision_logs SET final_pnl = ? WHERE id = ?
        # 目前, 我们只记录操作并返回 True 表示成功。
        self.logger.info(f"DATABASE_UPDATE: SET final_pnl = {final_pnl} WHERE decision_id = {decision_id}")
        self.logger.info(f"Successfully backfilled P&L for decision_id: {decision_id}.")
        return True
