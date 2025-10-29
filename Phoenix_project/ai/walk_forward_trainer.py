import logging
import numpy as np
import pandas as pd
from .reasoning_ensemble import ReasoningEnsemble
from .base_trainer import BaseTrainer
from datetime import date, timedelta
import mlflow
from typing import Dict, Any, List

# --- DRL (Task 1.2) Imports ---
from execution.order_manager import OrderManager
from drl.trading_env import TradingEnv
from data.data_iterator import NumpyDataIterator as DataIterator
from stable_baselines3 import PPO

class WalkForwardTrainer(BaseTrainer):
    """
    实现了一个步进式训练和验证方法。
    
    这种方法模拟了一个真实的交易场景，其中模型在
    新数据可用时进行重新训练，并其性能在
    紧邻训练期后的样本外数据上进行评估。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 WalkForwardTrainer。

        Args:
            config (Dict[str, Any]): 配置字典，包含像
                                     'training_window_size', 'validation_window_size',
                                     'retraining_frequency', 和 'ensemble_config' 这样的参数。
        """
        super().__init__(config)
        self.logger = logging.getLogger("PhoenixProject.WalkForwardTrainer")
        self.training_window_size = config.get('training_window_size', 365)
        self.validation_window_size = config.get('validation_window_size', 90)
        self.retraining_frequency = config.get('retraining_frequency', 30) # 使用配置中的30天
        self.training_mode = config.get('training_mode', 'ensemble') # 'ensemble' or 'drl'
        # 存储上次成功再训练的日期
        self.last_retraining_date = config.get('last_retraining_date', date.min)
        
        # 初始化 ReasoningEnsemble，这是我们正在训练的模型
        self.ensemble_config = config.get('ensemble_config', {})
        self.drl_config = config.get('drl_config', {})
        
        if self.training_mode == 'ensemble':
            self.model = ReasoningEnsemble(self.ensemble_config)
        else:
            self.model = None # DRL agent will be instantiated per-fold
        
        self.logger.info(
            f"WalkForwardTrainer initialized: "
            f"Train window={self.training_window_size} days, "
            f"Validate window={self.validation_window_size} days, "
            f"Retrain every={self.retraining_frequency} days."
        )

    def check_retraining_status(self, current_date: date) -> Dict[str, Any]:
        """
        根据性能衰减或周期检查是否需要再训练。
        如果需要，生成一个带成本的 "再训练建议"。
        (Task 3.1 - Adaptive Retraining Approval Workflow)
        """
        self.logger.info(f"Checking retraining status for {current_date}...")
        
        # --- 性能衰减检查占位符 ---
        # performance_decay_trigger = self._check_performance_decay()
        
        # --- 检查周期 (使用配置中的 retraining_frequency) ---
        days_since_last_train = (current_date - self.last_retraining_date).days
        cycle_trigger = days_since_last_train >= self.retraining_frequency

        if cycle_trigger: # or performance_decay_trigger:
            self.logger.info(f"Retraining cycle trigger met ({days_since_last_train} days >= {self.retraining_frequency}).")
            
            # 定义 *下一个* 训练窗口以估算成本
            train_start_date = current_date
            train_end_date = current_date + timedelta(days=self.training_window_size)
            
            cost_data = self.estimate_api_cost(train_start_date, train_end_date)
            
            return {
                "recommendation_pending": True,
                "reason": f"Retraining cycle met ({self.retraining_frequency} days).",
                "cost_estimate": cost_data
            }

        return {"recommendation_pending": False}

    def train(self, data: np.ndarray) -> Any:
        """
        在给定的数据窗口上训练模型。
        
        在这个具体实现中，'训练' ReasoningEnsemble 可能涉及
        校准其组件或微调底层模型。
        """
        self.logger.info(f"Starting training on data with shape {data.shape}...")
        
        # 在真实场景中，这将是一个复杂的操作：
        # 1. 预处理数据 (例如, 特征工程, 缩放)
        # 2. 训练/微调集成中的每个模型 (L1 模型, BayesianFusionEngine 等)
        # 3. 校准概率模型。
        
        # 在这个例子中，我们将通过更新模型的 'internal_state'
        # 或 'version' 来模拟训练。
        simulated_training_artifacts = self.model.calibrate(data)
        
        self.logger.info("Training complete.")
        # 在成功训练后更新日期
        self.last_retraining_date = date.today() 
        return simulated_training_artifacts # 返回 "训练好" 的模型或其产物

    def evaluate(self, model_artifacts: Any, validation_data: np.ndarray) -> Dict[str, float]:
        """
        在样本外验证数据上评估训练好的模型。
        """
        self.logger.info(f"Starting evaluation on data with shape {validation_data.shape}...")
        
        simulated_returns = []
        
        # 这个循环模拟了逐日迭代验证窗口
        for i in range(len(validation_data)):
            # 获取当天的数据 (可能还有回看窗口)
            current_day_data = validation_data[max(0, i-30):i+1] # 示例: 使用30天回看
            
            # 模型做出决策 (例如, +1 为多头, -1 为空头, 0 为平仓)
            # 这模拟了 CognitiveEngine 的完整决策过程
            decision = self.model.make_decision(current_day_data) # 简化
            
            # 根据决策模拟当天的回报
            # 这是一个高度简化的 P&L 模拟
            actual_return = validation_data[i, -1] # 假设最后一列是目标回报
            daily_pnl = decision * actual_return
            simulated_returns.append(daily_pnl)

        # 从模拟回报中计算性能指标
        metrics = self._calculate_performance_metrics(simulated_returns)
        self.logger.info(f"Evaluation complete: {metrics}")
        return metrics

    def run_walk_forward_validation(self, full_dataset: np.ndarray) -> Dict[str, Any]:
        """
        在整个数据集上执行完整的步进式验证过程。
        """
        self.logger.info(f"Starting full walk-forward validation on dataset with {len(full_dataset)} days.")
        
        # Divert to DRL workflow if specified
        if self.training_mode == 'drl':
            return self.run_drl_walk_forward(full_dataset)

        # 计算步进式验证的步数
        total_data_days = len(full_dataset)
        days_per_step = self.retraining_frequency
        num_steps = (total_data_days - self.training_window_size) // days_per_step
        
        all_fold_metrics = []
        
        for i in range(num_steps):
            # 确定当前训练和验证窗口的索引
            train_start = i * days_per_step
            train_end = train_start + self.training_window_size
            val_start = train_end
            val_end = val_start + self.validation_window_size

            # 确保我们没有越界
            if val_end > total_data_days:
                break

            self.logger.info(f"--- Fold {i+1}/{num_steps} ---")
            self.logger.info(f"Training window: Days {train_start} to {train_end-1}")
            self.logger.info(f"Validation window: Days {val_start} to {val_end-1}")

            # 提取数据窗口
            train_data = full_dataset[train_start:train_end]
            validation_data = full_dataset[val_start:val_end]

            # 1. 训练模型
            trained_model_artifacts = self.train(train_data)
            
            # 2. 评估模型
            fold_metrics = self.evaluate(trained_model_artifacts, validation_data)
            all_fold_metrics.append(fold_metrics)
            
            # 将此折的指标记录到 MLflow
            mlflow.log_metrics(fold_metrics, step=i)

        # 聚合所有折的指标
        final_metrics = self._aggregate_metrics(all_fold_metrics)
        self.logger.info(f"--- Walk-forward validation complete ---")
        self.logger.info(f"Aggregated Metrics: {final_metrics}")
        
        # 将最终的聚合指标记录到 MLflow
        mlflow.log_metrics({f"agg_{k}": v for k, v in final_metrics.items()})
        
        return final_metrics

    def run_drl_walk_forward(self, full_dataset: np.ndarray) -> Dict[str, Any]:
        """
        (Task 1.2) 在整个数据集上为DRL代理执行完整的步进式验证过程。
        """
        self.logger.info(f"Starting DRL walk-forward validation on dataset with {len(full_dataset)} days.")

        total_data_days = len(full_dataset)
        days_per_step = self.retraining_frequency
        num_steps = (total_data_days - self.training_window_size) // days_per_step
        all_fold_metrics = []

        # --- DRL-specific setup from config ---
        env_params = self.drl_config.get('env_params', {})
        agent_params = self.drl_config.get('agent_params', {})
        column_map = self.drl_config.get('column_map', {}) # Defines how to map numpy columns to names
        train_timesteps = self.drl_config.get('train_timesteps', 10000)
        # Instantiate a single OrderManager based on config
        order_manager = OrderManager(**self.drl_config.get('order_manager_config', {}))

        for i in range(num_steps):
            train_start = i * days_per_step
            train_end = train_start + self.training_window_size
            val_start = train_end
            val_end = val_start + self.validation_window_size

            if val_end > total_data_days:
                break

            self.logger.info(f"--- DRL Fold {i+1}/{num_steps} ---")
            self.logger.info(f"Training window: Days {train_start} to {train_end-1}")
            self.logger.info(f"Validation window: Days {val_start} to {val_end-1}")

            train_data = full_dataset[train_start:train_end]
            validation_data = full_dataset[val_start:val_end]

            # 1. Setup Train Env & Agent
            train_iterator = DataIterator(train_data, column_map, ticker=env_params.get("trading_ticker"))
            train_env = TradingEnv(data_iterator=train_iterator, order_manager=order_manager, **env_params)
            agent = PPO("MlpPolicy", train_env, **agent_params)

            # 2. Train Agent
            self.logger.info(f"Training DRL agent for {train_timesteps} timesteps...")
            agent.learn(total_timesteps=train_timesteps)

            # 3. Setup Validation Env
            val_iterator = DataIterator(validation_data, column_map, ticker=env_params.get("trading_ticker"))
            val_env = TradingEnv(data_iterator=val_iterator, order_manager=order_manager, **env_params)

            # 4. Evaluate Agent
            fold_metrics = self._evaluate_drl_agent(agent, val_env)
            all_fold_metrics.append(fold_metrics)
            mlflow.log_metrics(fold_metrics, step=i)

        final_metrics = self._aggregate_metrics(all_fold_metrics)
        self.logger.info(f"--- DRL Walk-forward validation complete ---")
        self.logger.info(f"Aggregated Metrics: {final_metrics}")
        mlflow.log_metrics({f"agg_{k}": v for k, v in final_metrics.items()})
        return final_metrics

    def _evaluate_drl_agent(self, agent, env: TradingEnv) -> Dict[str, float]:
        """Helper to run a trained DRL agent on a validation environment."""
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
        
        # Extract metrics from the env's history
        metrics = self._calculate_performance_metrics(list(env.return_history))
        metrics["total_return"] = env.portfolio_value - env.initial_capital
        return metrics

    def _calculate_performance_metrics(self, returns: list) -> Dict[str, float]:
        """从回报列表中计算关键性能指标。"""
        if not returns or len(returns) == 0:
            return {"sharpe_ratio_dsr": 0, "max_drawdown": 0, "total_return": 0, "volatility": 0}
            
        returns_array = np.array(returns)
        total_return = np.sum(returns_array)
        
        # 夏普比率 (简化, 假设每日回报和252个交易日)
        mean_return = np.mean(returns_array)
        std_dev = np.std(returns_array)
        sharpe_ratio = (mean_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumsum(returns_array)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            "sharpe_ratio_dsr": sharpe_ratio, # DSR = 每日夏普比率
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "volatility": std_dev
        }

    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合所有折的指标 (例如, 通过平均)。"""
        if not all_metrics:
            return {}
            
        df = pd.DataFrame(all_metrics)
        return df.mean().to_dict()

    def _perform_bootstrap_test(self, validation_returns: list) -> Dict[str, float]:
        """
        在验证回报上执行平稳 bootstrap 测试，以创建
        夏普比率的置信区间。
        """
        # 这是一个复杂统计测试的占位符
        # 例如, 使用像 `arch` 这样的库或自定义实现
        self.logger.info("Performing stationary bootstrap test on returns...")
        
        # 模拟结果
        lower_bound = 0.85
        upper_bound = 1.25
        
        return {'sharpe_ci_lower': lower_bound, 'sharpe_ci_upper': upper_bound}

    def estimate_api_cost(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        [Sub-Task 1.2.3] 估算给定日期范围内的 AI 分析成本。
        (由 Task 3.4 的缓存逻辑支持)
        """
        self.logger.info(f"Estimating API cost for retraining from {start_date} to {end_date}...")
        
        # 1. 计算范围内的营业日数
        total_days = (end_date - start_date).days
        business_days = np.busday_count(start_date.isoformat(), end_date.isoformat())

        # 2. TODO: 查询 "AI 特征缓存" (Task 3.4) 找出这些营业日中有多少已经有数据。
        # (假设 data_manager.load_features_from_parquet 将被调用)
        #
        # 这是一个占位符逻辑。一个真实的实现会调用 data_manager。
        #
        simulated_cache_hit_rate = 0.90 # 假设由于 Task 3.4，缓存命中率很高
        cached_days = int(business_days * simulated_cache_hit_rate)
        missing_days = business_days - cached_days

        # 3. 将缺失天数乘以估算的每日成本。
        daily_cost = self.config.get('daily_average_analysis_cost', 0.50) # 默认 $0.50/天
        estimated_cost = missing_days * daily_cost

        self.logger.info(f"Cost estimation complete: {missing_days}/{business_days} business days require analysis. Estimated cost: ${estimated_cost:.2f}")
        
        return {"estimated_api_cost": estimated_cost, "business_days_total": business_days, "business_days_to_analyze": missing_days}
