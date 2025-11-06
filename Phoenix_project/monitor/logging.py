"""
Observability and Logging Configuration (Layer 12)
"""

import logging
import sys
from typing import Optional, Any, Dict
from datetime import datetime

# 尝试导入 elasticsearch，如果失败则优雅降级
try:
    from elasticsearch import Elasticsearch, AsyncElasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    # 在日志中打印一次警告，以便开发人员知道
    # print("Warning: 'elasticsearch' library not found. Real ES logging is disabled.")

LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 全局日志级别
GLOBAL_LOG_LEVEL = logging.INFO

class ESLogger:
    """
    一个统一的日志记录器包装器。
    
    它封装了一个标准的 logging.Logger。
    根据要求，它充当一个占位符，以满足依赖此接口的模块
    （如 fusion, memory）的需求。
    
    它提供了标准日志记录器所没有的 log_info, log_warning 等方法，
    这可能是为了将来与特定的日志记录框架（如 structlog 或 ES）集成。
    """
    
    def __init__(self, name: str, level=logging.INFO):
        """
        初始化 ESLogger。
        
        Args:
            name (str): 日志记录器的名称。
            level (int): 日志级别。
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False # 防止日志向上传播到根记录器

        # 确保日志记录器已配置处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(LOGGING_FORMAT)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # --- 可选的真实 ES 客户端 (占位符) ---
        # self.es_client = None
        # self.es_index = "phoenix-logs"
        # config = {} # 在真实实现中，这里会传入 config
        # if ELASTICSEARCH_AVAILABLE and config.get("es_logging_enabled"):
        #     try:
        #         self.es_client = Elasticsearch(config.get("es_hosts"))
        #         self.es_index = config.get("es_index", "phoenix-logs")
        #         self.logger.info("Elasticsearch logging client initialized.")
        #     except Exception as e:
        #         self.logger.error(f"Failed to initialize Elasticsearch client: {e}")
        #         self.es_client = None

    def _log_to_es(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """(占位符) 将日志发送到 Elasticsearch。"""
        # if self.es_client:
        #     try:
        #         doc = {
        #             "@timestamp": datetime.utcnow().isoformat(),
        #             "level": level.upper(),
        #             "logger_name": self.logger.name,
        #             "message": message,
        #             **(extra or {})
        #         }
        #         self.es_client.index(index=self.es_index, document=doc)
        #     except Exception as e:
        #         # 不要在 ES 日志失败时再次调用 self.logger.error，
        #         # 因为这会触发另一次 _log_to_es 调用，导致无限循环。
        #         print(f"CRITICAL: Failed to send log to Elasticsearch: {e}")
        pass

    def log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        记录一条通用日志消息。
        
        Args:
            level (int): 日志级别 (e.g., logging.INFO)。
            message (str): 日志消息。
            extra (Optional[Dict[str, Any]]): 附加的结构化数据。
        """
        # 转发到标准日志记录器
        # 我们将 extra 传递给 'extra' 参数，以便 Formatter 可以使用它
        self.logger.log(level, message, extra=extra)
        
        # (可选) 转发到 ES
        # self._log_to_es(logging.getLevelName(level), message, extra)

    def log_debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 DEBUG 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.debug(message, extra=effective_extra)
        # self._log_to_es("DEBUG", message, effective_extra)

    def log_info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 INFO 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.info(message, extra=effective_extra)
        # self._log_to_es("INFO", message, effective_extra)

    def log_warning(self, message: str, exc_info=False, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 WARNING 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.warning(message, exc_info=exc_info, extra=effective_extra)
        # self._log_to_es("WARNING", message, effective_extra)

    def log_error(self, message: str, exc_info=True, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 ERROR 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.error(message, exc_info=exc_info, extra=effective_extra)
        # self._log_to_es("ERROR", message, effective_extra)

    def log_critical(self, message: str, exc_info=True, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 CRITICAL 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.critical(message, exc_info=exc_info, extra=effective_extra)
        # self._log_to_es("CRITICAL", message, effective_extra)

# --- 全局日志记录器实例缓存 ---
_loggers: Dict[str, ESLogger] = {}

def get_logger(name: str, level: Optional[int] = None) -> ESLogger:
    """
    获取一个 ESLogger 实例。
    
    这取代了旧的 get_logger，并返回我们新的包装器类。
    """
    global _loggers
    global GLOBAL_LOG_LEVEL

    if name not in _loggers:
        log_level = level if level is not None else GLOBAL_LOG_LEVEL
        _loggers[name] = ESLogger(name, level=log_level)
    
    # (可选) 如果传入了级别，则更新现有记录器的级别
    current_instance = _loggers[name]
    if level is not None and current_instance.logger.level != level:
         current_instance.logger.setLevel(level)

    return current_instance

def setup_logging(level=logging.INFO):
    """
    为应用程序设置全局日志级别。
    (被 run_training.py 和 worker.py 调用)
    """
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = level
    
    # (可选) 更新所有已创建的记录器的级别
    global _loggers
    for logger_instance in _loggers.values():
        logger_instance.logger.setLevel(level)
        
    # (可选) 配置根日志记录器（如果需要）
    root_logger = logging.getLogger()
    if not root_logger.handlers:
         handler = logging.StreamHandler(sys.stdout)
         handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
         root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # 使用标准 logging 模块记录一次，以确保根级别已设置
    logging.info(f"全局日志级别设置为: {logging.getLevelName(level)}")
