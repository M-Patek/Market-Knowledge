# training/drl/multi_agent_trainer.py
# [Beta 修复] 多资产面板加载与动态空间适配
# [Phase III Fix] Trainer-Environment Alignment

import ray
import pandas as pd
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium import spaces
import os
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

from Phoenix_project.training.drl.trading_env import TradingEnv
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.data_manager import DataManager
from Phoenix_project.features.store import FeatureStore
from Phoenix_project.config.loader import ConfigLoader

logger = get_logger("DRLMultiAgentTrainer")

# --- [Beta 修复] 加载多资产面板数据 ---
def load_nutrition_panel(
    data_manager: DataManager,
    feature_store: FeatureStore,
    symbols: List[str], # [Beta] Changed from single symbol to list
    start_date: datetime,
    end_date: datetime
) -> Dict[str, pd.DataFrame]: # Returns a dict of DataFrames or a MultiIndex DF
    """
    [Beta 修复] 加载多资产的 L1+L2 数据，并对齐时间戳。
    为 TradingEnv 提供完整的市场视图。
    """
    logger.info(f"Loading nutrition panel for {len(symbols)} assets: {symbols}")
    
    panel_data = {}
    
    # 1. 批量获取 L1 数据
    market_data_dict = data_manager.get_market_data(symbols, start_date, end_date) # Assuming get_market_data handles list
    
    for sym in symbols:
        try:
            # L1
            if sym not in market_data_dict or market_data_dict[sym] is None or market_data_dict[sym].empty:
                logger.warning(f"Missing L1 data for {sym}. Skipping.")
                continue
            
            df = market_data_dict[sym][['close', 'volume']].rename(columns={"close": "price"})
            
            # L2 (Optional / Mock if missing for now)
            # 真实场景应从 FeatureStore 批量获取
            l2_df = feature_store.get_features('l2_nutrition_features', [sym], start_date, end_date)
            
            if l2_df is not None and not l2_df.empty:
                if "timestamp" in l2_df.columns: l2_df = l2_df.set_index("timestamp")
                df = df.join(l2_df[["l2_sentiment", "l2_confidence"]], how='left').fillna(method='ffill').fillna(0)
            else:
                # [Fallback] If FeatureStore is empty, fill default L2 to allow training start
                df["l2_sentiment"] = 0.0
                df["l2_confidence"] = 0.5
            
            panel_data[sym] = df
            
        except Exception as e:
            logger.error(f"Error processing data for {sym}: {e}")

    # [Critical] Ensure timestamp alignment (Intersection)
    # 简单的对齐策略：取所有资产的时间戳交集
    if not panel_data:
        raise ValueError("No data loaded for any symbol.")
        
    common_index = panel_data[symbols[0]].index
    for sym in panel_data:
        common_index = common_index.intersection(panel_data[sym].index)
    
    if common_index.empty:
        raise ValueError("Time timestamps do not overlap across assets.")
        
    aligned_panel = {sym: df.loc[common_index] for sym, df in panel_data.items()}
    logger.info(f"Panel loaded. Common timestamps: {len(common_index)}")
    
    # 转换格式以适应 DataIterator (e.g., list of dicts per timestamp)
    # 这里为了简化，我们返回 aligned_panel 字典，TradingEnv 的 iterator 需要适配
    return aligned_panel

def get_agent_action_space_config(agent_id: str, num_assets: int) -> dict:
    """
    [Beta 修复] 动态适配资产数量
    """
    if agent_id == "alpha_agent":
        # Alpha: Output vector matching number of assets
        return {"type": "continuous", "low": -1.0, "high": 1.0, "shape": (num_assets,)}
    
    elif agent_id == "risk_agent":
        # Risk: Scalar global risk factor
        return {"type": "continuous", "low": 0.0, "high": 1.0, "shape": (1,)}
    
    elif agent_id == "execution_agent":
        # Exec: Discrete style per asset? Or global style? 
        # Assuming Global Style for simplicity, or (num_assets,) if per-asset.
        # Let's keep it global discrete for now.
        return {"type": "discrete", "n": 3}
    
    else:
        raise ValueError(f"Unknown agent_id: {agent_id}")

def run_training_session(
    panel_data: Dict[str, pd.DataFrame],
    asset_list: List[str],
    training_iterations: int = 100,
    save_dir: str = "models/drl_agents_v2_rllib"
):
    agent_ids = ["alpha_agent", "risk_agent", "execution_agent"]
    num_assets = len(asset_list)
    
    for agent_id in agent_ids:
        logger.info(f"--- Configuring training for {agent_id} ---")
        # [Beta 修复] Pass num_assets
        action_config_raw = get_agent_action_space_config(agent_id, num_assets)
        
        # Space Factory
        if action_config_raw["type"] == "discrete":
            action_space = spaces.Discrete(action_config_raw["n"])
        elif action_config_raw["type"] == "continuous":
            action_space = spaces.Box(
                low=action_config_raw["low"],
                high=action_config_raw["high"],
                shape=action_config_raw["shape"],
                dtype=np.float32
            )

        # Env Config
        env_config = {
            "panel_data": panel_data, # Pass full panel
            "asset_list": asset_list, # Explicit anchor
            "agent_id": agent_id,
            "initial_balance": 100000.0,
            "action_space_config": action_config_raw
        }
        
        config = (
            PPOConfig()
            .environment(
                env=TradingEnv,
                env_config=env_config,
                action_space=action_space
            )
            .framework("torch")
            .rollouts(num_rollout_workers=2)
        )
        
        logger.info(f"--- Starting Tuner for {agent_id} ---")
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=ray.train.RunConfig(
                stop={"training_iteration": training_iterations},
                local_dir=f"{save_dir}/{agent_id}",
                name=f"{agent_id}_training",
            ),
        )
        
        result_grid = tuner.fit()
        best_result = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
        logger.info(f"{agent_id} training complete. Checkpoint: {best_result.checkpoint.path}")

if __name__ == "__main__":
    logger.info("--- 启动 L3 DRL 训练 (已重构) ---")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    try:
        config_path = os.environ.get('PHOENIX_CONFIG_PATH', 'config')
        config_loader = ConfigLoader(config_path)
        
        # Mock FeatureStore/DataManager setup for standalone run
        # In real usage, these would be properly initialized
        # ...
        
        asset_list = ["AAPL", "MSFT", "GOOGL"]
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        # [Mock] For demonstration, we assume data loaded successfully
        # panel_data = load_nutrition_panel(..., asset_list, ...)
        
        # run_training_session(panel_data, asset_list)

    except Exception as e:
        logger.critical(f"DRL 训练的主脚本失败: {e}", exc_info=True)

    finally:
        ray.shutdown()
        logger.info("--- L3 DRL 训练已关闭 ---")
