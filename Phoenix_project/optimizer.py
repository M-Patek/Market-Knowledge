"""
Hyperparameter Optimizer

Uses Bayesian optimization (e.g., via scikit-optimize) to find the
best set of parameters for a given strategy or model.
"""
import logging
from typing import List, Dict, Any, Callable
# 修复：[FIX-8] 导入 skopt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# 修复：[FIX-8] 添加 pandas 导入
import pandas as pd

# 修复：[FIX-8] 导入路径从 '.backtesting.engine' 更改
from training.backtest_engine import BacktestingEngine
# 修复：[FIX-10] 导入 'load_config'
from config.loader import load_config

logger = logging.getLogger(__name__)

class Optimizer:
    """
    Wraps the optimization process, handling parameter spaces
    and objective function evaluation.
    """

    def __init__(self, 
                 param_space_config: List[Dict[str, Any]],
                 objective_function: Callable,
                 # 修复：[FIX-10] config 是一个 dict，而不是 loader
                 base_config: Dict[str, Any]):
        """
        Initializes the Optimizer.

        Args:
            param_space_config: A list of dictionaries defining the
                                search space, e.g.,
                                [{'type': 'Real', 'name': 'rsi_window', 'low': 5, 'high': 30}]
            objective_function: The function to minimize (e.g., -Sharpe Ratio).
                                It must accept (config, **params).
            base_config: The main config dict, which will be *copied*
                         and *mutated* with new params for each run.
        """
        self.param_space = self._build_param_space(param_space_config)
        self.param_names = [p.name for p in self.param_space]
        self.objective_function = objective_function
        # 修复：[FIX-10] 存储 base_config
        self.base_config = base_config
        
        logger.info(f"Optimizer initialized with parameter space: {self.param_names}")

    def _build_param_space(self, config: List[Dict[str, Any]]) -> List:
        """Converts the config dict into a list of skopt.space dimensions."""
        space = []
        for p in config:
            p_type = p.pop('type')
            p_name = p.pop('name')
            if p_type == 'Real':
                space.append(Real(name=p_name, **p))
            elif p_type == 'Integer':
                space.append(Integer(name=p_name, **p))
            elif p_type == 'Categorical':
                space.append(Categorical(name=p_name, **p))
            else:
                raise ValueError(f"Unsupported parameter type: {p_type}")
        return space

    @use_named_args(dimensions=None) # Will be set dynamically
    def _objective(self, **params) -> float:
        """
        Internal wrapper for the objective function.
        
        This function receives the parameters from `gp_minimize`,
        updates a *copy* of the config, and runs the objective function.
        """
        logger.debug(f"Testing parameters: {params}")
        
        try:
            # 1. Create a deep copy of the config to mutate
            # 修复：[FIX-10] 复制 base_config 字典
            import copy
            temp_config = copy.deepcopy(self.base_config)
            
            # 2. Update the config with the new parameters
            # Example: temp_config['strategy']['rsi_window'] = params['rsi_window']
            
            # For this example, let's assume we update a 'strategy' block
            strategy_config = temp_config.get('strategy', {})
            strategy_config.update(params)
            temp_config['strategy'] = strategy_config
            
            # 3. Run the user-provided objective function
            # 修复：[FIX-10] 传入 temp_config
            result = self.objective_function(config=temp_config, **params)
            
            # 4. Handle NaN/Inf results (common in backtesting)
            # 修复：[FIX-8] 使用 pd.isna 并检查无穷大
            if pd.isna(result) or not pd.Series(result).is_finite().all():
                logger.warning(f"Objective function returned invalid value (NaN/Inf) for params: {params}. Returning +inf.")
                # We are minimizing, so return a very bad score
                return float('inf')
                
            logger.info(f"Params: {params} -> Objective Score: {result:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during objective evaluation with params {params}: {e}", exc_info=True)
            # Return a very bad score on failure
            return float('inf')

    def run_optimization(self, n_calls: int = 50, n_initial_points: int = 10) -> Dict[str, Any]:
        """
        Executes the Bayesian optimization process.

        Args:
            n_calls: Total number of optimization runs.
            n_initial_points: Number of random points to sample before
                              building the surrogate model.
        Returns:
            A dictionary containing the best parameters and the best score.
        """
        logger.info(f"Starting optimization: n_calls={n_calls}, n_initial_points={n_initial_points}")
        
        # Dynamically set the dimensions for the @use_named_args decorator
        self._objective.__skopt_dimensions__ = self.param_space
        
        result = gp_minimize(
            func=self._objective,
            dimensions=self.param_space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func="EI", # Expected Improvement
            random_state=42,
            verbose=False
        )
        
        best_params = {name: val for name, val in zip(self.param_names, result.x)}
        best_score = result.fun
        
        logger.info(f"Optimization complete.")
        logger.info(f"Best Score: {best_score:.4f}")
        logger.info(f"Best Parameters: {best_params}")
        
        return {
            "best_score": best_score,
            "best_params": best_params,
            "optimization_result": result
        }

# --- Example Objective Function (to be defined by the user) ---

def example_sharpe_objective(config: Dict[str, Any], **params) -> float:
    """
    An example objective function that runs a backtest and returns
    the negative Sharpe ratio (since we want to minimize).
    
    Args:
        config: The *mutated* config dict with new params.
        **params: The parameters being tested.
        
    Returns:
        The score to be minimized (e.g., -Sharpe Ratio).
    """
    
    # 1. Initialize components using the *temp_config*
    # (This is a simplified example)
    
    # data_iterator = DataIterator(config.get('data'))
    # pipeline_state = PipelineState(config.get('portfolio'))
    # strategy_handler = MyStrategy(config=config, pipeline_state=pipeline_state)
    # ... (build other components)
    
    # engine = BacktestingEngine(
    #     data_iterator=data_iterator,
    #     strategy_handler=strategy_handler,
    #     ...
    # )
    
    # 2. Run the backtest
    # results = engine.run()
    
    # 3. Get the metric to optimize
    # sharpe_ratio = results.get('sharpe_ratio', 0.0)
    
    # --- Mocked Result ---
    # Simulate a result based on params for demonstration
    rsi_window = params.get('rsi_window', 14)
    # Simulate a simple curve where the best Sharpe is at window=20
    sharpe_ratio = 1.5 - ((rsi_window - 20) ** 2) / 200.0
    
    # We want to *minimize* the *negative* Sharpe ratio
    return -sharpe_ratio
