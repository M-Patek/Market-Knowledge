import logging
import pandas as pd
from typing import Dict, Any, List
from datetime import date
from features.base import IFeature

class FeatureStore:
    """
    管理所有特征的定义、计算和检索。
    """
    def __init__(self, feature_list: List[IFeature]):
        """
        初始化 FeatureStore。

        Args:
            feature_list: 一个实现了 IFeature 接口的特征对象列表。
        """
        self.logger = logging.getLogger("PhoenixProject.FeatureStore")
        self.features = {f.name: f for f in feature_list}
        self.logger.info(f"FeatureStore initialized with {len(self.features)} features: {list(self.features.keys())}")

    def add_feature(self, feature: IFeature):
        """动态添加一个新特征。"""
        if feature.name in self.features:
            self.logger.warning(f"Feature '{feature.name}' already exists. Overwriting.")
        self.features[feature.name] = feature

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
        
        features = {}
        for name, feature in self.features.items():
            try:
                # 简化：假设每个特征都可以从原始 DataFrame 计算
                features[name] = feature.compute(data)
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
