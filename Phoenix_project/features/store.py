import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from schemas.feature_schema import FeatureSchema

class FeatureStore:
    """
    (L1 Patched) 管理所有特征的定义、计算和检索。
    """
    def __init__(self, feature_list: List[FeatureSchema]):
        """
        初始化 FeatureStore。

        Args:
            feature_list: 一个 FeatureSchema 对象的列表。
        """
        self.logger = logging.getLogger("PhoenixProject.FeatureStore")
        self.features: Dict[str, FeatureSchema] = {f.name: f for f in feature_list}
        self.logger.info(f"FeatureStore initialized with {len(self.features)} features: {list(self.features.keys())}")

    def add_feature(self, feature: FeatureSchema):
        """动态添加一个新特征。"""
        if feature.name in self.features:
            self.logger.warning(f"Feature '{feature.name}' already exists. Overwriting.")
        self.features[feature.name] = feature

    def validate_feature_dependencies(self):
        """
        (L1) Validates the dependency graph of all registered features.
        Checks for undefined dependencies and circular dependencies.
        Raises:
            ValueError: If a dependency is not found or a circular dependency is detected.
        """
        self.logger.info("Validating feature dependency graph...")
        visiting = set()  # For detecting cycles (nodes currently in the recursion stack)
        visited = set()   # For tracking already validated nodes

        for feature_name in self.features:
            if feature_name not in visited:
                self._dfs_validate(feature_name, visiting, visited)

        self.logger.info("Feature dependency graph is valid.")

    def _dfs_validate(self, feature_name: str, visiting: set, visited: set):
        visiting.add(feature_name)

        for dep_name in self.features[feature_name].dependencies:
            if dep_name not in self.features:
                raise ValueError(f"Undefined dependency '{dep_name}' for feature '{feature_name}'.")
            if dep_name in visiting:
                raise ValueError(f"Circular dependency detected: '{feature_name}' -> '{dep_name}'")
            if dep_name not in visited:
                self._dfs_validate(dep_name, visiting, visited)

        visiting.remove(feature_name)
        visited.add(feature_name)

    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """
        递归地获取一个特征所需的所有依赖项。
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature '{feature_name}' not found in store.")
        
        deps_set = set()
        
        def _find_deps(f_name):
            if f_name in deps_set:
                return
            
            feat = self.features.get(f_name)
            if not feat:
                # 这是一个原始数据源，不是一个计算特征
                deps_set.add(f_name)
                return

            # 添加特征本身及其依赖项
            deps_set.add(f_name)
            for dep in feat.dependencies:
                _find_deps(dep)

        _find_deps(feature_name)
        return list(deps_set)

    def generate_features_for_ticker(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        为单个代码计算所有已注册的特征。
        """
        if data.empty:
            self.logger.warning(f"No data provided for {ticker}. Cannot generate features.")
            return {}

        # 这是一个占位符。一个真实的实现会：
        # 1. 构建一个依赖图（DAG）
        # 2. 按拓扑顺序执行特征计算
        # 3. 从 `data` DataFrame 中提取所需的列
        # 4. (L11) Resolve self.features[name].calc_fn (string) to a function via a Registry
        
        features = {}
        for name, feature in self.features.items():
            try:
                # 简化：假设每个特征都可以从原始 DataFrame 计算
                # This line is intentionally broken until L11 Registry is implemented
                # features[name] = feature.compute(data) 
                features[name] = np.random.rand() # Placeholder
            except Exception as e:
                self.logger.error(f"Failed to compute feature '{name}' for {ticker}: {e}")
                features[name] = None
        
        # 返回最近的特征值
        # 这是一个简化的假设；真实的实现会返回一个时间序列或特定日期的值
        final_features = {name: val[-1] if isinstance(val, (pd.Series, list, np.ndarray)) else val for name, val in features.items() if val is not None}
        
        self.logger.debug(f"Generated {len(final_features)} features for {ticker}.")
        return final_features

    def generate_features_from_alt_data(self, alt_data_df: pd.DataFrame):
        """
        [Task 2.1] 将原始替代数据转换为预测性特征
        并存储它们。
        """
        if alt_data_df.empty:
            self.logger.info("Received empty alternative data DataFrame. No features generated.")
            return
        
        self.logger.info(f"Received {len(alt_data_df)} rows of alternative data. (Placeholder for transformation logic).")
        # 占位符: 一个真实的实现会运行转换
        # (例如, self._transform_supply_chain_data(alt_data_df))
        # 并保存特征。
