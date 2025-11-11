# Phoenix_project/storage/s3_client.py
# [主人喵的修复 11.11] 实现了 TBD (AWS 凭证和数据加载)

import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class S3Client:
    """
    [已实现]
    用于与 S3 (或兼容的) 对象存储（如 MinIO）交互的客户端。
    """

    def __init__(self, config: DictConfig):
        self.config = config.get("s3_client", {})
        self.bucket_name = self.config.get("bucket_name")
        
        # [实现] TBD: AWS 凭证处理
        # 最佳实践：不传递 access_key 或 secret_key。
        # Boto3 会自动从环境变量 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        # 或 EC2/ECS/EKS 上的 IAM 角色获取凭证。
        
        # 允许 MinIO/Localstack 的自定义端点
        endpoint_url = self.config.get("endpoint_url") 
        
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                region_name=self.config.get("region_name", "us-east-1")
            )
            # 验证连接 (可选，但有益)
            self.s3_client.list_buckets() 
            logger.info(f"S3Client initialized. Connected to endpoint: {endpoint_url or 'AWS S3 default'}")
            
        except NoCredentialsError:
            logger.error("S3Client failed: No AWS credentials found. Please configure environment variables or IAM roles.")
            raise
        except ClientError as e:
            logger.error(f"S3Client failed to connect: {e}", exc_info=True)
            raise

    def load_data(self, key: str) -> str | None:
        """
        [已实现] 从 S3 加载数据 (假设为 utf-8 文本/json)。
        """
        if not self.bucket_name:
            logger.error("Cannot load data: S3 bucket_name is not configured.")
            return None
            
        logger.debug(f"Loading data from S3: {self.bucket_name}/{key}")
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = response['Body'].read().decode('utf-8')
            return data
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"S3 load failed: Key '{key}' not found in bucket '{self.bucket_name}'.")
            else:
                logger.error(f"Failed to load data from S3 key '{key}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during S3 load: {e}", exc_info=True)
            return None

    def upload_data(self, key: str, data: str | bytes) -> bool:
        """
        [已实现] 将数据（文本或字节）上传到 S3。
        """
        if not self.bucket_name:
            logger.error("Cannot upload data: S3 bucket_name is not configured.")
            return False

        logger.debug(f"Uploading data to S3: {self.bucket_name}/{key}")
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=data)
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload data to S3 key '{key}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during S3 upload: {e}", exc_info=True)
            return False

    def upload_json(self, key: str, data: dict) -> bool:
        """
        [新] 辅助函数：将字典作为 JSON 上传。
        """
        try:
            json_data = json.dumps(data, indent=2)
            return self.upload_data(key, json_data)
        except TypeError as e:
            logger.error(f"Failed to serialize data to JSON for S3 upload: {e}")
            return False

    def load_json(self, key: str) -> dict | None:
        """
        [新] 辅助函数：从 S3 加载并解析 JSON。
        """
        data = self.load_data(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from S3 key '{key}': {e}")
                return None
        return None
