# Phoenix_project/controller/error_handler.py
import logging
from typing import Any

logger = logging.getLogger("PhoenixProject.ErrorHandler")

# This should be loaded from config (Task 20, 'auto_retry')
MAX_RETRIES = 3 

def handle_failure(stage: str, error: Exception, attempt: int) -> bool:
    """
    Unified exception recovery.
    Must retry <= 3 times upon an API error.
    Returns True if a retry should be attempted, False otherwise.
    """
    logger.warning(f"Error encountered in stage '{stage}' (Attempt {attempt}): {error}")
    
    # TODO: 添加更复杂的逻辑来检查错误类型 (例如 503 vs 400)
    is_api_error = True # 模拟：目前假设所有错误都是 API 错误
    
    if is_api_error and attempt < MAX_RETRIES:
        logger.info(f"Decision: RETRYING stage '{stage}'")
        return True
    
    logger.error(f"Decision: FAILED stage '{stage}'. Max retries exceeded or error is non-retriable.")
    return False
