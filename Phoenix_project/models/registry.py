"""
[阶段 2] 已修改
模型注册表 (Model Registry)
负责原子性地管理生产模型的版本。
这是实现"自动回退"的关键：如果训练失败，"promote" 就不会发生，
系统会继续加载前一天稳定的模型。
"""

import json
import os
import torch # 假设模型是 torch 模型
from typing import Dict, Optional, Any

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.config.loader import load_config

logger = get_logger(__name__)

# 定义注册表文件的路径 (指向生产模型的 "指针" 文件)
CONFIG = load_config("system.yaml")
REGISTRY_BASE_DIR = CONFIG.get("models", {}).get("registry_path", "models/registry")
REGISTRY_FILE = os.path.join(REGISTRY_BASE_DIR, "production_models.json")
MODEL_ARTIFACTS_DIR = CONFIG.get("models", {}).get("artifacts_path", "models/artifacts")

# 确保目录存在
os.makedirs(REGISTRY_BASE_DIR, exist_ok=True)
os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)

class ModelRegistry:
    """
    管理生产模型的"生效"(promote)和加载。
    """

    def __init__(self):
        self.registry_path = REGISTRY_FILE
        self.artifacts_path = MODEL_ARTIFACTS_DIR
        self._ensure_registry_file()

    def _ensure_registry_file(self):
        """确保注册表 JSON 文件存在"""
        if not os.path.exists(self.registry_path):
            logger.warning(f"Production model registry file not found. Creating empty one at {self.registry_path}")
            try:
                with open(self.registry_path, 'w') as f:
                    json.dump({}, f)
            except IOError as e:
                logger.error(f"Failed to create registry file: {e}")

    def _read_registry(self) -> Dict[str, str]:
        """原子性地读取注册表文件"""
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                return data
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read production model registry! {e}")
            return {}

    def _write_registry(self, registry_data: Dict[str, str]) -> bool:
        """原子性地写入注册表文件"""
        try:
            # 写入临时文件
            temp_path = self.registry_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            # 原子性重命名
            os.rename(temp_path, self.registry_path)
            return True
        except IOError as e:
            logger.error(f"Failed to write to production model registry! {e}")
            return False

    def promote_model(self, model_name: str, candidate_artifact_path: str) -> bool:
        """
        [任务 2.1] 将一个"候选"模型提升为"生产"模型。
        这是一个原子性操作。

        Args:
            model_name (str): 模型的名称 (e.g., "drl", "gnn").
            candidate_artifact_path (str): 指向已验证的候选模型工件
                                           (e.g., "models/artifacts/drl_candidate_20251112.pkl")
        """
        if not os.path.exists(candidate_artifact_path):
            logger.error(f"Promotion failed: Candidate model file does not exist at {candidate_artifact_path}")
            return False

        logger.info(f"Promoting model '{model_name}' to production using artifact: {candidate_artifact_path}")
        
        # 读取、更新、写回
        registry_data = self._read_registry()
        registry_data[model_name] = candidate_artifact_path
        
        if self._write_registry(registry_data):
            logger.info(f"Successfully promoted '{model_name}'.")
            return True
        else:
            logger.error(f"Promotion failed for '{model_name}'.")
            return False

    def get_production_model_path(self, model_name: str) -> Optional[str]:
        """
        [任务 2.1] 获取"生产"模型的*路径*。
        
        如果训练失败 (promote_model 未被调用), 这将返回*前一天*的路径。
        如果模型从未被训练过, 它可能返回 None。
        """
        registry_data = self._read_registry()
        path = registry_data.get(model_name)
        
        if not path:
            logger.warning(f"No production model found in registry for '{model_name}'.")
            # [自动回退] 也许我们应该尝试加载一个 'default' 或 'baseline' 模型?
            # 暂时返回 None
            return None
        
        if not os.path.exists(path):
            logger.error(f"Registry points to missing model file for '{model_name}'! Path: {path}")
            return None
            
        return path

    def load_production_model(self, model_name: str) -> Optional[Any]:
        """
        [任务 2.2] 加载当前生效的"生产"模型。
        这是所有模型消费者 (如 Orchestrator) 应该调用的方法。
        """
        logger.info(f"Loading production model for '{model_name}'...")
        model_path = self.get_production_model_path(model_name)
        
        if not model_path:
            logger.error(f"Failed to load production model: No path found for '{model_name}'.")
            return None
            
        try:
            # 假设是 PyTorch 模型
            # 注意: DRL 模型可能需要不同的加载方式 (e.g., Stable Baselines)
            # GNN 也是
            # TODO: 抽象化加载逻辑 (e.g., joblib, torch.load, or SB3.load)
            
            # 临时假设 GNN 和 DRL 都可以用 torch.load
            model = torch.load(model_path)
            
            logger.info(f"Successfully loaded production model '{model_name}' from {model_path}")
            return model
            
        except Exception as e:
            logger.critical(f"CRITICAL FAILURE: Failed to load production model '{model_name}' from {model_path}. Error: {e}", exc_info=True)
            return None

# 创建一个单例 (Singleton) 供整个应用使用
registry = ModelRegistry()
