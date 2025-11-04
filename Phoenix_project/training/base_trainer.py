# (原: ai/base_trainer.py)
# 导入路径 '..' 依然正确，因为 'training' 和 'ai' 都是 'Phoenix_project' 的子目录
from abc import ABC, abstractmethod
from typing import Dict, Any
# 保留：显式相对导入
from ..core.pipeline_state import PipelineState
# 保留：显式相对导入
from ..data_manager import DataManager
# 保留：显式相对导入
from ..monitor.logging import get_logger

logger = get_logger(__name__)
# ... existing code ...
