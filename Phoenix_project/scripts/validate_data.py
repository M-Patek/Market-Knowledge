"""
Data Validation Script

This script provides utilities to validate the integrity and format
of data schemas (NewsData, MarketData, etc.) before ingestion
or in unit tests.
"""
import json
import logging
import sys # 修复：重新添加 sys 用于 sys.exit
from pydantic import ValidationError

# 修复：现在使用绝对导入
# 修复 (第 4 阶段): TickerData -> MarketData, EconomicEvent -> EconomicIndicator
from Phoenix_project.core.schemas.data_schema import MarketData, NewsData, EconomicIndicator

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [Validator] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 示例数据 ---

# 修复 (第 4 阶段): 重命名变量
EXAMPLE_MARKET_DATA = {
    "symbol": "AAPL",
    "timestamp": "2023-10-27T14:30:00Z",
    "open": 170.3,
    "high": 170.9,
    "low": 170.1,
    "close": 170.5,
    "volume": 1000000
}

# 修复 (第 3 阶段): 重命名变量
EXAMPLE_NEWS_DATA = {
    "id": "news_12345",
    "source": "bloomberg",
    "timestamp": "2023-10-27T14:32:00Z",
    "content": "Federal Reserve minutes show division on rate hikes.",
    "symbols": ["AAPL", "MSFT", "GOOG"],
    "metadata": {"sentiment": 0.2, "source_reliability": "high"}
}

# 修复 (第 4 阶段): 重命名变量
EXAMPLE_ECONOMIC_INDICATOR = {
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
        schema_class: The Pydantic BaseModel class (e.g., MarketData).

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
    
    # 1. 验证 MarketData
    # 修复 (第 4 阶段): 更新为 MarketData
    results.append(validate_schema(EXAMPLE_MARKET_DATA, MarketData))
    
    # 2. 验证 NewsData
    # 修复 (第 3 阶段): 更新为 NewsData
    results.append(validate_schema(EXAMPLE_NEWS_DATA, NewsData))
    
    # 3. 验证 EconomicIndicator
    # 修复 (第 4 阶段): 更新为 EconomicIndicator
    results.append(validate_schema(EXAMPLE_ECONOMIC_INDICATOR, EconomicIndicator))
    
    # 4. 验证一个无效的 MarketData (缺少字段)
    # 修复 (第 4 阶段): 重命名变量
    invalid_market_data = EXAMPLE_MARKET_DATA.copy()
    del invalid_market_data['close']
    logger.info("\nIntentionally testing invalid MarketData (missing 'close'):")
    # 修复 (第 4 阶段): 更新为 MarketData
    results.append(not validate_schema(invalid_market_data, MarketData)) # 期望失败 (False)
    
    # 5. 验证一个无效的 NewsData (错误的数据类型)
    # 修复 (第 3 阶段): 更新为 NewsData
    invalid_event = EXAMPLE_NEWS_DATA.copy()
    invalid_event['symbols'] = "just a string" # 应该是 list
    logger.info("\nIntentionally testing invalid NewsData (wrong 'symbols' type):")
    results.append(not validate_schema(invalid_event, NewsData)) # 期望失败 (False)

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
