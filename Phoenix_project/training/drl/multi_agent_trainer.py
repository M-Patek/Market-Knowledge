# training/drl/multi_agent_trainer.py
import ray
import pandas as pd
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium import spaces # [任务 2] 导入 gym spaces

# [任务 2] 导入我们新的 (已修复的) gym.Env
from Phoenix_project.training.drl.trading_env import TradingEnv
from Phoenix_project.monitor.logging import get_logger

logger = get_logger("DRLMultiAgentTrainer")

# --- [任务 1] 准备“营养液” (数据准备) ---
def load_nutrition_data(data_path: str = "data/historical_l2_features.csv") -> pd.DataFrame:
    """
    (模拟) 加载包含 L2 特征的训练数据。
    
    [任务 1] 主人喵！您必须用真实的加载逻辑替换这里。
    您需要运行一次历史 L2 认知引擎，将价格数据与 L2 结果
    (l2_sentiment, l2_confidence) 合并到这个 DataFrame 中。
    """
    logger.info(f"正在尝试从 {data_path} 加载训练数据...")
    try:
        # 尝试加载真实数据 (如果存在)
        df = pd.read_csv(data_path)
        logger.info(f"成功加载了 {len(df)} 行真实训练数据。")
        # [任务 1] 验证所需列
        required_cols = ["price", "l2_sentiment", "l2_confidence"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"数据中缺少所需列 (需要: {required_cols})。将退回使用模拟数据。")
            raise FileNotFoundError # (跳转到 except 块)
        return df
        
    except FileNotFoundError:
        logger.warning(f"未找到 {data_path}。正在创建模拟“营养液” (L2 数据)...")
        # [任务 1] 如果找不到文件，则创建模拟数据
        steps = 1000
        data = {
            "price": np.random.rand(steps) * 100 + 150,
            # L2 情感 (例如 -1.0 到 1.0)
            "l2_sentiment": np.random.randn(steps) * 0.5, 
             # L2 信心 (例如 0.0 到 1.0)
            "l2_confidence": np.random.rand(steps),
        }
        df = pd.DataFrame(data)
        df["l2_sentiment"] = np.clip(df["l2_sentiment"], -1.0, 1.0)
        return df

# --- [任务 2] 重构“起搏器” (新训练器) ---

def get_agent_action_space_config(agent_id: str) -> dict:
    """
    为每个 L3 智能体定义其独特的动作空间。
    [任务 2] 这是必需的，因为 Alpha/Risk/Exec 有不同的输出。
    """
    if agent_id == "alpha_agent":
        # Alpha: 决定目标 *权重* (连续值, 0.0 到 1.0)
        return {"type": "continuous", "shape": (1,)}
    
    elif agent_id == "risk_agent":
        # Risk: 决定风险 *标量* (连续值, 0.0 到 1.0)
        return {"type": "continuous", "shape": (1,)}
    
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
        action_config = get_agent_action_space_config(agent_id)
        
        # 2. 定义环境配置 (传递 L2 数据和智能体特定配置)
        env_config = {
            "df": data_df,
            "agent_id": agent_id,
            "initial_balance": 100000.0,
            "action_space_config": action_config
        }
        
        # 3. 配置 PPO (不再是 MultiAgent，只是标准的 PPO)
        config = (
            PPOConfig()
            .environment(
                env=TradingEnv, # [任务 2] 使用我们新的 gym.Env
                env_config=env_config
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


if __name__ == "__main__":
    logger.info("--- 启动 L3 DRL 训练 (已重构) ---")
    
    # (确保 Ray 已初始化)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # [任务 1] 加载“营养液”
    training_data = load_nutrition_data(
        data_path="data/historical_l2_features.csv" # (确保此文件存在)
    )

    # [任务 2 & 3] 运行新的训练流程
    run_training_session(
        data_df=training_data,
        training_iterations=10, # (设为 10 次迭代用于测试, 生产中应更高)
        save_dir="models/drl_agents_v2_rllib"
    )

    ray.shutdown()
    logger.info("--- L3 DRL 训练已关闭 ---")
