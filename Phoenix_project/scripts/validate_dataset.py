import jsonschema
import json
import sys
from pathlib import Path

# 修正：添加项目根目录到 sys.path，以便导入根目录下的模块
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# --------------------------------------------------
# 原始脚本内容现在可以正常导入了
# --------------------------------------------------

from data_manager import DataManager
from monitor.logging import get_logger

logger = get_logger(__name__)

def validate_data_contract():
    """
    Ensures that the AI cache data strictly adheres to the schema defined in data_catalog.json.
    This is the Data Contract Enforcement gate for MLOps.
    """
    logger.info("Starting Data Contract Enforcement...")

    try:
        # 1. Load the DataManager to get access to the schema
        # 修正：使用基于 PROJECT_ROOT 的路径
        data_catalog_path = PROJECT_ROOT / "data_catalog.json"
        dm = DataManager(data_catalog_path=data_catalog_path)
        
        schema = dm.get_schema("fused_ai_analysis_schema")
        if not schema:
            logger.error("Failed to load 'fused_ai_analysis_schema' from data catalog.")
            return False
        
        logger.info("Successfully loaded 'fused_ai_analysis_schema' data contract.")

        # 2. Load the actual data cache
        # 修正：使用基于 PROJECT_ROOT 的路径
        # (假设 'asset_analysis_cache.json' 位于 'cache/' 目录)
        cache_path = PROJECT_ROOT / "cache" / "asset_analysis_cache.json"
        
        if not cache_path.exists():
            logger.warning(f"Cache file not found at {cache_path}. Skipping validation.")
            # 在 CI/CD 中，这可能是一个通过（pass）或失败（fail）的状态，取决于策略
            return True # 暂定为通过

        with open(cache_path, 'r') as f:
            data_cache = json.load(f)

        logger.info(f"Loaded data cache from {cache_path}.")

        # 3. Validate the data against the schema
        # 假设缓存是一个字典，其键是 Ticker，值是符合 schema 的分析对象
        if not isinstance(data_cache, dict):
            logger.error("Data cache is not a dictionary (key-value store). Validation failed.")
            return False

        all_valid = True
        for ticker, record in data_cache.items():
            try:
                jsonschema.validate(instance=record, schema=schema)
                logger.debug(f"Record for {ticker} PASSED validation.")
            except jsonschema.ValidationError as e:
                logger.error(f"Data Contract VIOLATION for ticker {ticker}: {e.message}")
                all_valid = False

        if all_valid:
            logger.info("Data Contract Enforcement PASSED. All records are valid.")
        else:
            logger.error("Data Contract Enforcement FAILED. Invalid records found.")
            
        return all_valid

    except FileNotFoundError:
        logger.error(f"Failed to load required file (data_catalog.json or cache).")
        return False
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from data catalog or cache.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    """
    This script is intended to be run as a CI/CD quality gate.
    If validation fails, it will exit with a non-zero status code,
    which should fail the CI pipeline.
    """
    if not validate_data_contract():
        logger.error("CI Quality Gate: FAILED")
        sys.exit(1)
    else:
        logger.info("CI Quality Gate: PASSED")
        sys.exit(0)
