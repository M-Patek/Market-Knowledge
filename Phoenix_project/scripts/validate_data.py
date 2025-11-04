"""
Data Validation Script

This script provides utilities to validate the integrity and format
of data schemas (MarketEvent, TickerData, etc.) before ingestion
or in unit tests.
"""
import sys
import os
import json
import logging
from pydantic import ValidationError

# 修复：将项目根目录 (Phoenix_project) 添加到 sys.path
# 这允许脚本以 'python scripts/validate_data.py' 的方式运行
# 并正确解析 'from core...'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 修复：现在使用绝对导入
from Phoenix_project.core.schemas.data_schema import TickerData, MarketEvent, EconomicEvent

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Validator] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 示例数据 ---

EXAMPLE_TICKER_DATA = {
    "symbol": "AAPL",
    "timestamp": "2023-10-27T14:30:00Z",
    "open": 170.3,
    "high": 170.9,
    "low": 170.1,
    "close": 170.5,
    "volume": 1000000
}

EXAMPLE_MARKET_EVENT = {
    "id": "news_12345",
    "source": "bloomberg",
    "timestamp": "2023-10-27T14:32:00Z",
    "content": "Federal Reserve minutes show division on rate hikes.",
    "symbols": ["AAPL", "MSFT", "GOOG"],
    "metadata": {"sentiment": 0.2, "source_reliability": "high"}
}

EXAMPLE_ECONOMIC_EVENT = {
    "id": "econ_cpi_202310",
    "event_name": "US CPI (MoM)",
    "timestamp": "2023-10-27T12:30:00Z",
    "actual": 0.4,
    "forecast": 0.3,
    "previous": 0.6,
    "country": "USA",
    "impact": "high"
}

def validate_schema(data: dict, schema_class) -> bool:
    """
    Validates a dictionary against a given Pydantic schema class.

    Args:
        data (dict): The raw data dictionary.
        schema_class: The Pydantic BaseModel class (e.g., TickerData).

    Returns:
        bool: True if valid, False otherwise.
    """
    schema_name = schema_class.__name__
    logger.info(f"--- Validating schema: {schema_name} ---")
    try:
        # 尝试创建和验证 schema 实例
        schema_instance = schema_class(**data)
        
        logger.info(f"Validation SUCCESSFUL for {schema_name}.")
        logger.debug(f"Validated data: {schema_instance.model_dump_json(indent=2)}")
        return True
        
    except ValidationError as e:
        logger.error(f"Validation FAILED for {schema_name}!")
        logger.error("Details:")
        # 打印易于阅读的错误
        for error in e.errors():
            field = " -> ".join(map(str, error['loc']))
            msg = error['msg']
            logger.error(f"  Field: {field}")
            logger.error(f"  Error: {msg}")
        return False
    except Exception as e:
        logger.critical(f"An unexpected error occurred during validation: {e}", exc_info=True)
        return False

def main():
    """
    Runs validation on all example data structures.
    """
    logger.info("Starting data schema validation...")
    
    results = []
    
    # 1. 验证 TickerData
    results.append(validate_schema(EXAMPLE_TICKER_DATA, TickerData))
    
    # 2. 验证 MarketEvent
    results.append(validate_schema(EXAMPLE_MARKET_EVENT, MarketEvent))
    
    # 3. 验证 EconomicEvent
    results.append(validate_schema(EXAMPLE_ECONOMIC_EVENT, EconomicEvent))
    
    # 4. 验证一个无效的 TickerData (缺少字段)
    invalid_ticker = EXAMPLE_TICKER_DATA.copy()
    del invalid_ticker['close']
    logger.info("\nIntentionally testing invalid TickerData (missing 'close'):")
    results.append(not validate_schema(invalid_ticker, TickerData)) # 期望失败 (False)
    
    # 5. 验证一个无效的 MarketEvent (错误的数据类型)
    invalid_event = EXAMPLE_MARKET_EVENT.copy()
    invalid_event['symbols'] = "just a string" # 应该是 list
    logger.info("\nIntentionally testing invalid MarketEvent (wrong 'symbols' type):")
    results.append(not validate_schema(invalid_event, MarketEvent)) # 期望失败 (False)

    # --- 总结 ---
    logger.info("\n--- Validation Summary ---")
    if all(results):
        logger.info("All tests passed (including expected failures)!")
        logger.info("Data schemas are correctly defined and enforced.")
        sys.exit(0) # 成功退出
    else:
        logger.error("One or more validation tests failed.")
        sys.exit(1) # 失败退出

if __name__ == "__main__":
    main()
