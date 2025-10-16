import logging
import optuna
import numpy as np
from typing import Dict, Any

from .ai.walk_forward_trainer import WalkForwardTrainer

class HyperparameterOptimizer:
    """
    使用Optuna和向前滚动窗口验证来寻找最优的模型超参数。
    """
    def __init__(self, data, config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.HyperparameterOptimizer")
        self.data = data
        self.config = config
        self.optimizer_config = self.config.get('optimizer', {})

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna的目标函数。为给定的超参数试验运行一次完整的
        向前滚动窗口验证。
        """
        # 1. 动态定义超参数搜索空间
        trial_params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "num_heads": trial.suggest_int("num_heads", 2, 8, step=2),
            "ff_dim": trial.suggest_int("ff_dim", 32, 128, step=32),
        }
        
        self.logger.info(f"开始试验 {trial.number}，参数: {trial_params}")
        
        # 2. 运行完整的向前滚动窗口训练
        trainer = WalkForwardTrainer(self.data, trial_params, self.config)
        metrics = trainer.run() # 预期返回 {"sharpe_ratio": X, "avg_variance": Y}

        sharpe_ratio = metrics.get("sharpe_ratio")
        avg_variance = metrics.get("avg_variance")

        if sharpe_ratio is None or avg_variance is None or not np.isfinite(sharpe_ratio):
            self.logger.warning(f"试验 {trial.number} 得到无效指标。返回-10。")
            return -10.0 # 为失败的试验返回一个很差的分数

        # 复合目标函数：夏普比率 - λ * 平均后验方差
        variance_penalty_lambda = self.optimizer_config.get('variance_penalty_lambda', 2.0)
        objective_value = sharpe_ratio - (variance_penalty_lambda * avg_variance)
        
        self.logger.info(f"试验 {trial.number}: 夏普={sharpe_ratio:.4f}, 平均方差={avg_variance:.4f} -> 目标值={objective_value:.4f}")
        return objective_value

    def run(self):
        """
        启动超参数优化过程。
        """
        self.logger.info("--- 开始贝叶斯超参数优化 ---")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            self.objective,
            n_trials=self.optimizer_config.get('n_trials', 50),
            timeout=self.optimizer_config.get('timeout_seconds', 7200)
        )

        self.logger.info("优化完成。")
        self.logger.info(f"最佳试验: {study.best_trial.number}")
        self.logger.info(f"最佳目标值: {study.best_value}")
        self.logger.info(f"最佳参数: {study.best_params}")
        
        return study.best_params
