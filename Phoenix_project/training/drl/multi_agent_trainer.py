# training/drl/multi_agent_trainer.py
import ray
import pandas as pd
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium import spaces # [任务 2] 导入 gym spaces
import os # 导入 os
import json # 导入 json
from typing import Dict, Any # [任务 A.1] 导入类型提示
from datetime import datetime, timedelta # 导入 datetime

# [任务 2] 导入我们新的 (已修复的) gym.Env
from Phoenix_project.training.drl.trading_env import TradingEnv
from Phoenix_project.monitor.logging import get_logger
# [任务 A.1] 导入所需的数据和配置加载器
from Phoenix_project.data_manager import DataManager
from Phoenix_project.features.store import FeatureStore
from Phoenix_project.config.loader import ConfigLoader

logger = get_logger("DRLMultiAgentTrainer")

# --- [任务 1] 准备“营养液” (数据准备) ---
# [任务 A.1] 重构 load_nutrition_data 以使用真实数据源
def load_nutrition_data(
    data_manager: DataManager,
    feature_store: FeatureStore,
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    [任务 A.1 已重构]
    加载用于 DRL 训练的 L1（价格）和 L2（营养）特征。
    此函数现在查询 DataManager 获取价格，查询 FeatureStore 获取 L2 特征，
    然后将它们合并。
    
    不再生成模拟数据。
    """
    logger.info(f"正在为 {symbol} 从 {start_date} 到 {end_date} 加载 L1 价格和 L2 营养数据...")

    try:
        # 1. 从 DataManager 加载 L1 价格数据
        market_data_dict = data_manager.get_market_data([symbol], start_date, end_date)
        if not market_data_dict or symbol not in market_data_dict or market_data_dict[symbol].empty:
            logger.error(f"DataManager 未能返回 {symbol} 的市场数据。")
            raise ValueError(f"DataManager 未找到 {symbol} 的 L1 价格数据")
        
        # 仅选择 'close' 价格并重命名为 'price' 以匹配 env 期望
        price_df = market_data_dict[symbol][['close']].rename(columns={"close": "price"})
        price_df.index.name = "timestamp" # 确保索引有名称以便合并

        # 2. 从 FeatureStore 加载 L2 营养数据
        # (我们假设 L2 特征集被命名为 'l2_nutrition_features')
        l2_features_df = feature_store.get_features(
            feature_set_name='l2_nutrition_features',
            entity_ids=[symbol],
            start_time=start_date,
            end_time=end_date
        )
        
        required_cols = ["l2_sentiment", "l2_confidence"]
        if l2_features_df is None or l2_features_df.empty:
            logger.error("FeatureStore 为 'l2_nutrition_features' 返回了空数据。")
            raise ValueError("FeatureStore 未找到 L2 营养数据。(FeatureStore 可能是占位符)")
        
        if not all(col in l2_features_df.columns for col in required_cols):
            logger.error(f"FeatureStore 数据中缺少所需列 (需要: {required_cols})。")
            raise ValueError("FeatureStore 数据缺少 L2 列")
        
        # 确保 l2_features_df 以 timestamp 为索引
        if "timestamp" in l2_features_df.columns:
             l2_features_df = l2_features_df.set_index("timestamp")
        
        logger.info(f"成功加载 {len(price_df)} 行 L1 价格数据和 {len(l2_features_df)} 行 L2 特征数据。")

        # 3. 合并 L1 和 L2 数据
        merged_df = pd.merge(price_df, l2_features_df[required_cols], left_index=True, right_index=True, how='inner')
        
        if merged_df.empty:
            logger.error("L1 价格和 L2 特征的时间戳没有重叠。")
            raise ValueError("合并 L1 和 L2 数据后为空")
            
        logger.info(f"成功合并 {len(merged_df)} 行训练数据。")
        return merged_df

    except Exception as e:
        logger.error(f"加载 L1/L2 营养数据失败: {e}", exc_info=True)
        # 抛出异常，而不是返回模拟数据
        raise

# --- [任务 2] 重构“起搏器” (新训练器) ---

def get_agent_action_space_config(agent_id: str) -> dict:
    """
    为每个 L3 智能体定义其独特的动作空间。
    [任务 2] 这是必需的，因为 Alpha/Risk/Exec 有不同的输出。
    """
    if agent_id == "alpha_agent":
        # Alpha: 决定目标 *权重* (连续值, 0.0 到 1.0)
        # [任务 A.2] 匹配回测逻辑，允许做空 (-1.0 到 1.0)
        return {"type": "continuous", "low": -1.0, "high": 1.0, "shape": (1,)}
    
    elif agent_id == "risk_agent":
        # Risk: 决定风险 *标量* (连续值, 0.0 到 1.0)
        return {"type": "continuous", "low": 0.0, "high": 1.0, "shape": (1,)}
    
    elif agent_id == "execution_agent":
        # Exec: 决定执行 *风格* (离散值)
        # 示例: 0=市价单 (Market), 1=限价单 (Limit), 2=TWAP
        return {"type": "discrete", "n": 3}
    
    else:
        raise ValueError(f"未知的 agent_id: {agent_id}")

def run_training_session(
    data_df: pd.DataFrame,
    training_iterations: int = 100,
    save_dir: str = "models/drl_agents_v2_rllib" # [任务 4.3] 新的输出目录
):
    """
    [任务 2] 重构后的训练器。
    它不再使用 PettingZoo，而是循环训练三个独立的 (Gym) PPO 模型。
    """
    
    # (定义我们要训练的三个独立智能体)
    agent_ids = ["alpha_agent", "risk_agent", "execution_agent"]
    
    results = {}

    for agent_id in agent_ids:
        logger.info(f"--- [任务 2] 开始为 {agent_id} 配置训练 ---")
        
        # 1. 获取该智能体特定的动作空间
        action_config_raw = get_agent_action_space_config(agent_id)
        
        # [任务 A.1] 转换动作为 RLLib gym.spaces
        if action_config_raw["type"] == "discrete":
            action_space = spaces.Discrete(action_config_raw["n"])
        elif action_config_raw["type"] == "continuous":
            action_space = spaces.Box(
                low=action_config_raw.get("low", -1.0),
                high=action_config_raw.get("high", 1.0),
                shape=action_config_raw.get("shape", (1,)),
                dtype=np.float32
            )
        else:
             raise ValueError(f"不支持的动作空间类型: {action_config_raw['type']}")

        
        # 2. 定义环境配置 (传递 L2 数据和智能体特定配置)
        env_config = {
            "df": data_df,
            "agent_id": agent_id,
            "initial_balance": 100000.0,
            "action_space_config": action_config_raw # 传递原始配置字典
        }
        
        # 3. 配置 PPO (不再是 MultiAgent，只是标准的 PPO)
        config = (
            PPOConfig()
            .environment(
                env=TradingEnv, # [任务 2] 使用我们新的 gym.Env
                env_config=env_config,
                # [任务 A.1] 明确设置动作空间
                action_space=action_space
            )
            .framework("torch")
            .rollouts(num_rollout_workers=2)
            # (可以添加更多 RLLib 特定配置, 例如 .training(), .resources() ...)
        )

        # 4. 使用 Ray Tune 运行训练器
        logger.info(f"--- [任务 3] 正在为 {agent_id} 启动 Tuner ---")
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=ray.train.RunConfig(
                stop={"training_iteration": training_iterations},
                local_dir=f"{save_dir}/{agent_id}", # [任务 4.3] 保存到特定子目录
                name=f"{agent_id}_training",
            ),
        )
        
        result_grid = tuner.fit()
        
        # 5. 获取最佳检查点
        best_result = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
        best_checkpoint = best_result.checkpoint
        logger.info(f"--- [任务 3] {agent_id} 训练完成 ---")
        logger.info(f"最佳检查点保存在: {best_checkpoint.path}")
        
        results[agent_id] = best_checkpoint.path

    logger.info("--- [任务 3] 所有 L3 智能体训练已完成 ---")
    logger.info(f"Alpha Agent Checkpoint: {results.get('alpha_agent')}")
    logger.info(f"Risk Agent Checkpoint: {results.get('risk_agent')}")
    logger.info(f"Execution Agent Checkpoint: {results.get('execution_agent')}")
    logger.info(f"[任务 4.3] 主人喵！请将这些新路径更新到您的 system.yaml 中！")
    
    return results


def load_data_catalog(config_loader: ConfigLoader) -> Dict[str, Any]:
    """ [任务 A.1] 辅助函数：加载数据目录 """
    catalog_path = config_loader.load_config('system.yaml').get("data_catalog_path", "data_catalog.json")
    catalog_file_path = os.path.join(config_loader.config_path, catalog_path)
    if not os.path.exists(catalog_file_path):
        logger.error(f"Data catalog not found at {catalog_file_path}")
        return {}
    try:
        with open(catalog_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data catalog: {e}", exc_info=True)
        return {}

if __name__ == "__main__":
    logger.info("--- 启动 L3 DRL 训练 (已重构) ---")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    try:
        # [任务 A.1] 重构 __main__ 以实例化依赖项
        
        # 1. 设置配置和依赖项
        # 假设配置文件在 'config' 目录或由环境变量指定
        config_path = os.environ.get('PHOENIX_CONFIG_PATH', 'config')
        if not os.path.isdir(config_path):
             # 回退到项目根目录（假设脚本是从 Phoenix_project 运行的）
             config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
             if not os.path.isdir(config_path):
                 logger.warning(f"Config path {config_path} not found. Using default 'config'.")
                 config_path = 'config'

        config_loader = ConfigLoader(config_path)
        system_config = config_loader.load_config('system.yaml')
        data_catalog = load_data_catalog(config_loader)
        
        data_manager = DataManager(config_loader, data_catalog)
        
        # 确保 fs_base_path 是相对于项目根目录的
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        fs_base_path = system_config.get("data_store", {}).get("local_base_path", "data")
        if not os.path.isabs(fs_base_path):
            fs_base_path = os.path.join(project_root, fs_base_path)
            
        feature_store = FeatureStore(base_path=fs_base_path)

        # 2. 定义训练参数 (用于 __main__ 测试)
        symbol = "AAPL"
        start_date = datetime.now() - timedelta(days=730) # 2 年
        end_date = datetime.now()
        
        # 3. [任务 1] 加载“营养液” (使用新函数)
        training_data = load_nutrition_data(
            data_manager=data_manager,
            feature_store=feature_store,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        # 4. [任务 2 & 3] 运行新的训练流程
        # 确保 save_dir 也是相对于项目根目录的
        save_dir_path = os.path.join(project_root, "models/drl_agents_v2_rllib")
        run_training_session(
            data_df=training_data,
            training_iterations=10, # (设为 10 次迭代用于测试, 生产中应更高)
            save_dir=save_dir_path
        )

    except Exception as e:
        logger.critical(f"DRL 训练的主脚本失败: {e}", exc_info=True)
        logger.critical("请确保 FeatureStore (l2_nutrition_features) 中有数据，并且 DataManager 配置正确。")

    finally:
        ray.shutdown()
        logger.info("--- L3 DRL 训练已关闭 ---")
