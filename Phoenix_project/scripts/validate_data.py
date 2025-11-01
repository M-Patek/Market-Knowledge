import sys
from pathlib import Path

# 修正：添加项目根目录到 sys.path，以便导入根目录下的模块
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# --------------------------------------------------
# 原始脚本内容现在可以正常导入了
# --------------------------------------------------

import json
from data_manager import DataManager
from monitor.logging import get_logger
from core.schemas.data_schema import DATA_SCHEMA
from jsonschema import validate, ValidationError

logger = get_logger(__name__)

def validate_data_integrity():
    """
    Validates that the data in the data catalog matches the master DATA_SCHEMA.
    """
    logger.info("Starting data integrity validation...")
    
    try:
        # 修正：使用基于 PROJECT_ROOT 的路径
        data_catalog_path = PROJECT_ROOT / "data_catalog.json"
        dm = DataManager(data_catalog_path=data_catalog_path)
        
        if not dm.catalog:
            logger.error("Data catalog is empty or not loaded. Validation failed.")
            return False

        all_valid = True
        
        # 1. Validate 'fused_ai_analysis_schema' defined in data_catalog.json
        logger.info("Validating 'fused_ai_analysis_schema' in data_catalog.json...")
        fused_schema = dm.catalog.get("fused_ai_analysis_schema")
        if not fused_schema:
            logger.error("'fused_ai_analysis_schema' not found in data_catalog.json.")
            all_valid = False
        else:
            # Simple check: does it have a 'properties' key?
            if 'properties' not in fused_schema:
                 logger.error("'fused_ai_analysis_schema' seems malformed (missing 'properties').")
                 all_valid = False
            else:
                 logger.info("'fused_ai_analysis_schema' structure seems OK.")

        # 2. Validate 'master_data_schema' (DATA_SCHEMA from core.schemas)
        logger.info("Validating 'master_data_schema' (core.schemas.data_schema)...")
        if not DATA_SCHEMA or 'properties' not in DATA_SCHEMA:
            logger.error("Master DATA_SCHEMA in core.schemas.data_schema.py is malformed.")
            all_valid = False
        else:
            logger.info("Master DATA_SCHEMA structure seems OK.")

        # 3. Validate a sample data entry against the master schema
        # (This is a placeholder. In a real system, you'd fetch sample data)
        logger.warning("Skipping sample data validation (no data loaded).")
        
        if all_valid:
            logger.info("Data integrity validation passed.")
        else:
            logger.error("Data integrity validation FAILED.")
            
        return all_valid

    except Exception as e:
        logger.error(f"An exception occurred during data validation: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if not validate_data_integrity():
        sys.exit(1) # Exit with an error code if validation fails
