"""
Dataset Validation Script

This script is used to load a sample dataset (e.g., from a CSV or JSONL file)
and validate each entry against the Pydantic schemas.

This is crucial for ensuring data quality before ingesting a large
dataset into the vector or temporal databases.
"""
import sys
import os
import argparse
import logging
import json
import pandas as pd
from pydantic import ValidationError

# 修复：将项目根目录 (Phoenix_project) 添加到 sys.path
# 这允许脚本以 'python scripts/validate_dataset.py' 的方式运行
# 并正确解析 'from ai...' 或 'from core...'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 修复：现在使用绝对导入
from ai.data_adapter import DataAdapter
from core.schemas.data_schema import MarketEvent, TickerData

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [DatasetValidator] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_dataset_file(file_path: str, data_type: str):
    """
    Validates all entries in a given dataset file.

    Args:
        file_path (str): Path to the .jsonl or .csv file.
        data_type (str): The type of data to validate ('market_event' or 'ticker_data').
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Starting validation for file: {file_path}")
    logger.info(f"Expected data type: {data_type}")

    # 确定 Pydantic schema
    if data_type == 'market_event':
        schema_class = MarketEvent
    elif data_type == 'ticker_data':
        schema_class = TickerData
    else:
        logger.error(f"Unknown data type: {data_type}. Must be 'market_event' or 'ticker_data'.")
        return

    # DataAdapter 用于将原始字典转换为标准化的 schema
    # 注意：DataAdapter 内部可能有更复杂的逻辑
    # 为简单起见，我们这里直接使用 Pydantic 验证
    # adapter = DataAdapter() 

    valid_entries = 0
    invalid_entries = 0
    
    try:
        # 我们假设是 .jsonl (JSON Lines) 格式
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        raw_data = json.loads(line)
                        # 验证
                        schema_class(**raw_data)
                        valid_entries += 1
                    except ValidationError as e:
                        logger.warning(f"Validation FAILED for line {i+1}: {e.errors()}")
                        invalid_entries += 1
                    except json.JSONDecodeError:
                        logger.error(f"JSON decode FAILED for line {i+1}.")
                        invalid_entries += 1
        
        # 也可以添加对 CSV 的支持
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            # 必须处理 NaNs，它们在转为 dict 时会变成 float 'nan'
            df = df.where(pd.notnull(df), None)
            for i, row in enumerate(df.to_dict('records')):
                try:
                    # CSV 数据可能需要更多的数据类型转换
                    # 例如，时间戳和 JSON 字符串
                    if 'timestamp' in row and isinstance(row['timestamp'], str):
                        row['timestamp'] = pd.Timestamp(row['timestamp']).to_pydatetime()
                    
                    schema_class(**row)
                    valid_entries += 1
                except ValidationError as e:
                    logger.warning(f"Validation FAILED for CSV row {i+1}: {e.errors()}")
                    invalid_entries += 1
        else:
            logger.error(f"Unsupported file format: {file_path}. Please use .jsonl or .csv")
            return

    except Exception as e:
        logger.critical(f"A critical error occurred while processing the file: {e}", exc_info=True)
        return

    # --- 总结 ---
    logger.info("\n--- Dataset Validation Summary ---")
    logger.info(f"File processed: {file_path}")
    logger.info(f"Total entries valid: {valid_entries}")
    logger.info(f"Total entries invalid: {invalid_entries}")
    
    if invalid_entries == 0 and valid_entries > 0:
        logger.info("All entries are valid. Dataset is clean!")
    elif valid_entries == 0 and invalid_entries == 0:
        logger.warning("No data entries were found or processed.")
    else:
        logger.error(f"{invalid_entries} invalid entries found. Please review the warnings above.")

def main():
    parser = argparse.ArgumentParser(description="Phoenix Dataset Validator")
    parser.add_argument("file_path", 
                        type=str, 
                        help="Path to the dataset file (e.g., data/my_events.jsonl)")
    parser.add_argument("-t", "--type", 
                        type=str, 
                        required=True, 
                        choices=['market_event', 'ticker_data'],
                        help="The type of data in the file.")

    args = parser.parse_args()
    
    validate_dataset_file(args.file_path, args.type)

if __name__ == "__main__":
    main()
