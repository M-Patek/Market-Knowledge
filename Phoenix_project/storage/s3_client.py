# Phoenix_project/storage/s3_client.py
import logging
import boto3
import pandas as pd
from botocore.exceptions import ClientError
import os

logger = logging.getLogger(__name__)

class S3Client:
    """
    (TBD) S3Client for reading/writing data from S3.
    """
    def __init__(self, config):
        self.config = config
        self.bucket_name = config.get("bucket_name", "phoenix-market-data")
        
        # (TBD: AWS 凭证应该通过 IAM 角色或环境变量来处理)
        # (我们不在这里硬编码)
        try:
            # [主人喵的修复 11.10] 移除 (mocked) 日志
            self.s3 = boto3.client(
                's3',
                # (TBD: 如果需要本地 (例如 MinIO) 测试，则配置 endpoint_url)
                # endpoint_url=config.get("endpoint_url", None),
                # aws_access_key_id=config.get("aws_access_key_id", None),
                # aws_secret_access_key=config.get("aws_secret_access_key", None)
            )
            # logger.info("(mocked) S3Client created...")
            logger.info(f"S3Client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Boto3 S3 client: {e}", exc_info=True)
            self.s3 = None


    def read_data(self, s3_key: str, local_path: str) -> pd.DataFrame | None:
        """
        Reads a file (e.g., Parquet, CSV) from S3 and loads it into a DataFrame.
        """
        if not self.s3:
            logger.error("S3 client not initialized. Cannot read data.")
            return None
            
        # (TBD: 我们应该下载到本地路径还是在内存中读取?)
        # (下载到本地路径更适合大文件)
        
        local_file = os.path.join(local_path, os.path.basename(s3_key))
        
        # [主人喵的修复 11.10] 取消模拟 S3 读取
        try:
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_file}...")
            self.s3.download_file(self.bucket_name, s3_key, local_file)
            
            # (TBD: 基于文件扩展名加载)
            if local_file.endswith(".parquet"):
                df = pd.read_parquet(local_file)
            elif local_file.endswith(".csv"):
                df = pd.read_csv(local_file)
            else:
                logger.warning(f"Unsupported file type for loading: {local_file}")
                # (TBD: 我们应该只下载还是也加载?)
                # (假设我们应该加载)
                return None
            
            logger.info(f"Successfully loaded data from {s3_key}")
            return df

        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Failed to read/load data {local_file}: {e}", exc_info=True)
            return None

        # (Mock implementation)
        # logger.warning(f"S3Client.read_data (mocked) for key: {s3_key}")
        # if s3_key.endswith(".parquet"):
        #     # (返回一个模拟的 DataFrame)
        #     return pd.DataFrame({
        #         "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        #         "price": [100.0, 101.0],
        #         "volume": [1000, 1200]
        #     })
        # return pd.DataFrame()


    def write_data(self, df: pd.DataFrame, s3_key: str, local_path: str) -> bool:
        """
        Writes a DataFrame to a local file (e.g., Parquet) and uploads it to S3.
        """
        if not self.s3:
            logger.error("S3 client not initialized. Cannot write data.")
            return False

        local_file = os.path.join(local_path, os.path.basename(s3_key))
        
        # [主人喵的修复 11.10] 取消模拟 S3 写入
        try:
            # 1. 写入本地文件
            logger.info(f"Writing DataFrame to local cache: {local_file}")
            if local_file.endswith(".parquet"):
                df.to_parquet(local_file, index=False)
            elif local_file.endswith(".csv"):
                df.to_csv(local_file, index=False)
            else:
                logger.error(f"Unsupported file type for writing: {local_file}")
                return False

            # 2. 上传到 S3
            logger.info(f"Uploading {local_file} to s3://{self.bucket_name}/{s3_key}...")
            self.s3.upload_file(local_file, self.bucket_name, s3_key)
            
            logger.info(f"Successfully wrote data to {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write/upload data to {s3_key}: {e}", exc_info=True)
            return False

        # (Mock implementation)
        # logger.warning(f"S3Client.write_data (mocked) for key: {s3_key}")
        # return True
