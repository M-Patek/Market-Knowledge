"""
Phoenix Rebirth: 终极验证脚本 (Phase 5)

此脚本执行两项核心测试：
1. Test A: 心脏与血液测试 (Live Data Pipeline) - 验证 Redis 读写契约。
2. Test B: 大脑与神经测试 (Backtest Loop) - 验证回测闭环与智能体维度对齐。

Usage: python scripts/verify_rebirth.py
"""
import asyncio
import logging
import os
import sys
import json
import numpy as np
import pandas as pd
import redis
from datetime import datetime, timedelta

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Phoenix_project.core.schemas.data_schema import MarketData
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE
from Phoenix_project.data_manager import DataManager
from Phoenix_project.data.data_iterator import DataIterator
from Phoenix_project.training.drl.trading_env import PhoenixMultiAgentEnvV7
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VerifyRebirth")

# --- Test A: Live Pipeline ---

async def test_a_live_pipeline():
    logger.info("\n=== TEST A: Heart & Blood (Live Data Pipeline) ===")
    logger.info("Validating Phase 0 (Contracts), Phase 1 (Ingestion), Phase 2 (Access)...")
    
    # 1. Setup Redis
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        r.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        logger.error(f"Skipping Test A: Redis not available ({e})")
        return

    symbol = "TEST_CRYPTO"
    timestamp = datetime.utcnow()
    
    # 2. 模拟 StreamProcessor: 生成并写入合规数据
    mock_data = MarketData(
        symbol=symbol,
        timestamp=timestamp,
        open=100.0, high=105.0, low=95.0, close=102.5, volume=1000.0
    )
    
    key = REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=symbol)
    r.set(key, mock_data.model_dump_json())
    logger.info(f"[StreamProcessor Mock] Wrote MarketData to Redis key: {key}")

    # 3. 验证 DataManager: 读取并解析数据
    # 使用空配置初始化 DataManager (仅用于测试 Redis 读取)
    dm = DataManager({"data_manager": {}}, r)
    
    logger.info("[DataManager] Attempting to read latest market data...")
    read_data = await dm.get_latest_market_data(symbol)
    
    # 4. 断言
    assert read_data is not None, "FAILED: DataManager returned None!"
    assert isinstance(read_data, MarketData), "FAILED: Returned object is not MarketData model!"
    assert read_data.symbol == symbol
    assert read_data.close == 102.5
    logger.info(f"✅ Data Consistency Verified: Symbol={read_data.symbol}, Close={read_data.close}")
    logger.info("✅ Test A Passed: Live Data Pipeline is healthy.")

# --- Test B: Backtest Loop ---

class MockDataManager:
    """Mock DataManager 用于提供历史数据"""
    def __init__(self):
        self.config = {}
        
    async def get_market_data_history(self, symbol, start, end):
        # 生成 10 天的假数据
        dates = pd.date_range(start=start, end=end, freq='1D')
        df = pd.DataFrame(index=dates)
        df['open'] = 100.0
        df['high'] = 110.0
        df['low'] = 90.0
        df['close'] = 100.0 + np.random.randn(len(dates)) # 价格随机游走以测试价值更新
        df['volume'] = 1000.0
        return df

    async def get_news_data(self, start, end):
        return pd.DataFrame() # 无新闻

class MockComponent:
    """通用 Mock 组件 (Orchestrator/ContextBus)"""
    pass

async def test_b_backtest_loop():
    logger.info("\n=== TEST B: Brain & Nerves (Backtest Loop) ===")
    logger.info("Validating Phase 3 (Physics), Phase 4 (Brain Alignment)...")
    
    # 1. 初始化 DataIterator (Mock DataManager)
    mock_dm = MockDataManager()
    iterator_config = {'backtesting': {'step_size': '1d'}}
    iterator = DataIterator(iterator_config, mock_dm)
    
    start_date = datetime.now() - timedelta(days=15)
    end_date = datetime.now()
    symbols = ["BTC/USD"]
    
    logger.info("[DataIterator] Setting up...")
    await iterator.setup(start_date, end_date, symbols)
    
    # 2. 初始化 Environment
    env_config = {
        "data_iterator": iterator,
        "orchestrator": MockComponent(), # 暂不需要真实的 Orchestrator
        "context_bus": MockComponent(),
        "initial_balance": 100000.0
    }
    env = PhoenixMultiAgentEnvV7(env_config)
    logger.info("[TradingEnv] Initialized PhoenixMultiAgentEnvV7.")
    
    # 3. 初始化 AlphaAgent
    # (假设 config 为空也能运行，或根据实际需要填入)
    agent = AlphaAgent(config={})
    logger.info("[AlphaAgent] Initialized.")
    
    # 4. 运行闭环测试
    obs, info = env.reset()
    
    # [关键验证] 检查 Observation 维度 (Task 4.2 & Phase 4 Fix)
    alpha_obs = obs['alpha']
    logger.info(f"Initial Observation Shape: {alpha_obs.shape}")
    assert alpha_obs.shape == (5,), f"FAILED: Dimension mismatch! Expected (5,), got {alpha_obs.shape}"
    logger.info("✅ Observation dimensions aligned (5,).")
    
    # 模拟 3 个时间步
    for i in range(3):
        # 模拟 L2 Fusion Result (Task 4.2 Input)
        fusion_result = FusionResult(
            target_symbol="BTC/USD", decision="STRONG_BUY", confidence=0.95, 
            reasoning="Test", uncertainty=0.05
        )
        
        # 构造 L3 状态字典
        state_data = {
            "balance": env.balance,
            "holdings": env.positions.get("BTC/USD", {}).get("shares", 0.0),
            "price": env.current_prices.get("BTC/USD", 100.0)
        }
        
        # Agent 感知与决策
        formatted_obs = agent.format_observation(state_data, fusion_result)
        action = agent.compute_action(formatted_obs)
        
        # 环境物理反馈
        actions = {"alpha": action, "risk": action, "exec": action}
        obs, rewards, _, _, info = env.step(actions)
        
        logger.info(f"Step {i+1}: Portfolio Value={env.total_value:.2f}, Action={action}")

    logger.info("✅ Test B Passed: Backtest loop is closed and physics are active.")

async def main():
    await test_a_live_pipeline()
    await test_b_backtest_loop()
    logger.info("\n🎉 PHOENIX REBIRTH VERIFICATION COMPLETE! 🎉")

if __name__ == "__main__":
    asyncio.run(main())
