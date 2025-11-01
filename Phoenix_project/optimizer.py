import logging
from typing import Dict, Any, Callable
from functools import partial

# 修正：导入 'optuna' 库，这个库之前缺失了
import optuna

from backtesting.engine import BacktestEngine
from data_manager import DataManager

logger = logging.getLogger(__name__)

class Optimizer:
    """
    使用 Optuna 框架的超参数优化器。
    
    它通过在回测引擎上运行多次试验来
    为策略（例如 RomanLegionStrategy）找到最佳参数。
    (Task 17 - 参数优化)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        backtest_engine: BacktestEngine,
        data_manager: DataManager
    ):
        """
        初始化优化器。
        
        Args:
            config (Dict[str, Any]): 'optimizer' 部分的配置。
            backtest_engine (BacktestEngine): 用于运行每次试验的回测引擎。
            data_manager (DataManager): 用于为回测获取数据。
        """
        self.config = config.get('optimizer', {})
        self.backtest_engine = backtest_engine
        self.data_manager = data_manager
        
        self.study_name = self.config.get('study_name', 'phoenix_optimization')
        self.n_trials = self.config.get('n_trials', 100)
        self.storage_url = self.config.get('storage_url', 'sqlite:///optuna_study.db')
        
        logger.info(f"Optimizer initialized: Study='{self.study_name}', Trials={self.n_trials}")

    def run_optimization(self) -> optuna.Study:
        """
        执行完整的优化过程。
        
        Returns:
            optuna.Study: 完成的 Optuna 研究对象。
        """
        logger.info(f"Starting optimization study '{self.study_name}' for {self.n_trials} trials...")
        
        try:
            # 1. 创建或加载一个 Optuna 研究
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                load_if_exists=True,
                direction='maximize' # 我们想要最大化夏普比率
            )

            # 2. 定义目标函数
            # 我们使用 partial 来 "冻结" objective 函数的 self, data 参数
            objective_func = partial(
                self._objective,
                strategy_class=self.backtest_engine.strategy_class # 假设引擎持有策略类
            )

            # 3. 运行优化
            study.optimize(
                objective_func,
                n_trials=self.n_trials,
                timeout=self.config.get('timeout_seconds', None)
            )

            # 4. 记录最佳结果
            logger.info("Optimization complete.")
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value (Sharpe Ratio): {study.best_value}")
            logger.info(f"Best params: {study.best_params}")
            
            return study

        except Exception as e:
            logger.error(f"Optimization study '{self.study_name}' failed: {e}", exc_info=True)
            raise

    def _objective(self, trial: optuna.Trial, strategy_class: Any) -> float:
        """
        Optuna 的目标函数。
        
        对于给定的试验，它：
        1. 建议一组超参数。
        2. 使用这些参数运行回测。
        3. 返回要最大化的指标 (例如夏普比率)。
        
        Args:
            trial: 当前的 Optuna 试验对象。
            strategy_class: 要实例化的策略类 (例如 RomanLegionStrategy)。
            
        Returns:
            float: 该试验的夏普比率。
        """
        
        # 1. 建议超参数
        # (这需要与策略的 __init__ 匹配)
        params_config = self.config.get('parameters', {})
        strategy_params = {}
        
        # 示例：为 SMA 周期采样
        if 'sma_short' in params_config:
            strategy_params['sma_short'] = trial.suggest_int(
                'sma_short', 
                params_config['sma_short']['low'], 
                params_config['sma_short']['high']
            )
        if 'sma_long' in params_config:
             strategy_params['sma_long'] = trial.suggest_int(
                'sma_long', 
                params_config['sma_long']['low'], 
                params_config['sma_long']['high']
            )
        # 示例：为 RSI 阈值采样
        if 'rsi_overbought' in params_config:
            strategy_params['rsi_overbought'] = trial.suggest_int(
                'rsi_overbought', 
                params_config['rsi_overbought']['low'], 
                params_config['rsi_overbought']['high']
            )
            
        logger.debug(f"Trial {trial.number}: Testing params {strategy_params}")

        try:
            # 2. 准备回测数据 (从 DataManager)
            # (这应该使用配置中的标准回测日期/资产)
            backtest_data = self.data_manager.get_data_for_backtest(
                assets=self.config.get('assets', ['SPY']),
                start_date=self.config.get('start_date', '2020-01-01'),
                end_date=self.config.get('end_date', '2023-01-01')
            )
            
            # 3. 合并配置
            # 创建一个新的配置 dict，注入建议的参数
            trial_config = self.backtest_engine.config.copy() # 使用引擎的基础配置
            trial_config['strategy']['parameters'] = strategy_params

            # 4. 运行回测
            results_df = self.backtest_engine.run(
                data=backtest_data,
                strategy_class=strategy_class, # 传递策略类
                config=trial_config # 传递特定于试验的配置
            )
            
            # 5. 提取要最大化的指标
            sharpe_ratio = results_df.get('Sharpe Ratio', 0.0)
            
            # 处理回测失败或夏普比率为 NaN 的情况
            if pd.isna(sharpe_ratio):
                sharpe_ratio = 0.0
                
            logger.debug(f"Trial {trial.number}: Sharpe Ratio = {sharpe_ratio}")
            
            return sharpe_ratio

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}", exc_info=True)
            # 告诉 Optuna 这次试验失败了
            raise optuna.exceptions.TrialPruned()
