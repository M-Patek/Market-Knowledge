"""
Observability and Logging Configuration (Layer 12)
"""

import logging
import sys

LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOGGING_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
