"""
Observability and Logging Configuration (Layer 12)
"""

import logging
import sys
import json
import contextvars
from typing import Optional, Any, Dict
from datetime import datetime

# 全局 Trace ID 上下文变量 (用于全链路追踪)
correlation_id_var = contextvars.ContextVar("correlation_id", default=None)

# 全局日志级别
GLOBAL_LOG_LEVEL = logging.INFO

class JSONFormatter(logging.Formatter):
    """
    结构化 JSON 日志格式化器 (符合 Cloud Native/Filebeat 标准)。
    自动包含 Trace ID 和 extra 字段。
    """
    def format(self, record):
        # 基础标准字段
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        
        # 自动提取 extra 字段 (过滤掉 LogRecord 的标准属性)
        # 这允许 logger.info("msg", extra={"user_id": 123}) 被正确序列化
        STANDARD_ATTRS = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
        }
        
        for key, value in record.__dict__.items():
            if key not in STANDARD_ATTRS and not key.startswith('_'):
                log_record[key] = value

        # 处理异常堆栈
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record, ensure_ascii=False)

class ESLogger:
    """
    一个统一的日志记录器包装器。
    它封装了一个标准的 logging.Logger。
    现在主要作为兼容层，实际输出由 JSONFormatter 接管。
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

        # 注意: 这里不再单独配置 Handler，而是依赖 setup_logging 配置的 Root Handler
        # 这样可以确保输出格式统一为 JSON

    def log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        记录一条通用日志消息。
        """
        self.logger.log(level, message, extra=extra)

    def log_debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 DEBUG 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.debug(message, extra=effective_extra)

    def log_info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 INFO 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.info(message, extra=effective_extra)

    def log_warning(self, message: str, exc_info=False, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 WARNING 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.warning(message, exc_info=exc_info, extra=effective_extra)

    def log_error(self, message: str, exc_info=True, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 ERROR 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.error(message, exc_info=exc_info, extra=effective_extra)

    def log_critical(self, message: str, exc_info=True, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """记录 CRITICAL 级别的日志。"""
        if extra is None: extra = {}
        effective_extra = {**extra, **kwargs}
        self.logger.critical(message, exc_info=exc_info, extra=effective_extra)

# --- 全局日志记录器实例缓存 ---
_loggers: Dict[str, ESLogger] = {}

def get_logger(name: str, level: Optional[int] = None) -> ESLogger:
    """
    获取一个 ESLogger 实例。
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

def setup_logging(level=logging.INFO, config=None):
    """
    初始化全局日志配置 (JSON 模式)。
    配置 Root Logger 以使用 JSONFormatter，使所有模块日志结构化。
    """
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = level
    
    root_logger = logging.getLogger()
    
    # 重置 Handlers，避免重复和格式冲突
    if root_logger.handlers:
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
            
    # 使用 JSONFormatter 的 StreamHandler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)
    
    root_logger.setLevel(level)
    
    # 更新缓存的 Logger
    global _loggers
    for logger_instance in _loggers.values():
        logger_instance.logger.setLevel(level)
        
    logging.info(f"Global logging configured (Level: {logging.getLevelName(level)}), JSON output enabled.")
