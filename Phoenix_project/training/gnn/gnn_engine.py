# [来自用户的最终版本]
# [阶段 3] 新文件
"""
GNN (图神经网络) 训练引擎。

负责夜间的 GNN 训练，包含 ET 感知的安全停止逻辑。
由 Celery worker (worker.py) 在 UTC 22:00 之后的链中触发。
"""

import datetime
import pytz # [阶段 1]
import time
import os
import asyncio # [GNN 写回] 导入 asyncio
import random # [GNN 写回] 用于模拟
import logging # (用于本地测试)
from typing import Any, Dict, List

# 导入项目模块 (存根)
# 假设这些模块存在于 Python 路径中
try:
    from Phoenix_project.monitor.logging import get_logger
    from Phoenix_project.monitor.metrics import METRICS
    from Phoenix_project.models.registry import registry, MODEL_ARTIFACTS_DIR
    # [GNN 写回] 导入 GraphDBClient 以便写回
    from Phoenix_project.ai.graph_db_client import GraphDBClient
except ImportError:
    # 回退 (Fallback) 逻辑，用于本地测试或模块尚未完全建立
    logger = logging.getLogger(__name__)
    logging.warning("Could not import Phoenix_project modules. Using mock objects.")
    
    # --- Mock Objects ---
    class MockMetrics:
        def increment_counter(self, *args, **kwargs):
            logger.info(f"METRICS.increment_counter({args}, {kwargs})")
    METRICS = MockMetrics()

    class MockRegistry:
        def promote_model(self, *args, **kwargs):
            logger.info(f"registry.promote_model({args}, {kwargs})")
        
        def get_production_model_path(self, *args, **kwargs):
            logger.info(f"registry.get_production_model_path({args}, {kwargs})")
            return None
    registry = MockRegistry()

    MODEL_ARTIFACTS_DIR = "models/artifacts"
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)

    class MockGraphDBClient:
        async def execute_write(self, *args, **kwargs):
            logger.info(f"GraphDBClient.execute_write({args}, {kwargs})")
            return True
        async def close(self):
            logger.info("GraphDBClient.close()")
    GraphDBClient = MockGraphDBClient # 覆盖
    # --- End Mock Objects ---

# [阶段 3] 关键时区和安全停止时间
MARKET_TZ = pytz.timezone("America/New_York")
# 09:00 ET (在 09:30 ET 开市前 30 分钟)
SAFE_STOP_TIME_ET = datetime.time(9, 0, 0) 

# --- 模拟 GNN 模型 ---
class MockGNNModel:
    """一个模拟的 GNN 模型，用于演示 save/promote 逻辑"""
    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata
        self.trained_at = datetime.datetime.now(pytz.UTC)

    def train_epoch(self):
        # 模拟训练一个 epoch
        logger.debug("GNN training epoch...")
        time.sleep(0.5) # 模拟工作

    def save(self, path: str):
        # 模拟保存
        logger.info(f"Saving mock GNN model to {path}")
        # 在真实场景中，这里会是 torch.save(self.model.state_dict(), path)
        with open(path, 'w') as f:
            f.write(f"Mock GNN Model. Trained at: {self.trained_at.isoformat()}")

    def predict_scores(self) -> List[Dict[str, Any]]:
        """
        [GNN 写回] 模拟 GNN 模型生成预测。
        """
        logger.info("GNN model generating predictions...")
        # 在真实场景中，这会加载 GDS 图并运行 GNN
        mock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        predictions = []
        for symbol in mock_symbols:
            predictions.append({
                "symbol": symbol,
                "score": round(random.uniform(0.1, 0.9), 4) # 模拟 risk_score
            })
        return predictions

# --- 辅助函数 ---

def _check_et_safety_window() -> bool:
    """
    [阶段 3] 检查是否已进入 ET 危险窗口 (市场即将开市)。
    
    Returns:
        bool: True 表示已进入危险窗口 (应停止), False 表示安全 (可继续)。
    """
    now_et = datetime.datetime.now(MARKET_TZ)
    
    # 检查是否为工作日 (周一=0, 周日=6)
    is_weekday = 0 <= now_et.weekday() <= 4
    
    # 危险窗口：在工作日，且时间在 09:00 ET 之后 (直到 ET 午夜)
    # (修正：应该是 >= SAFE_STOP_TIME_ET)
    is_danger_window = is_weekday and (now_et.time() >= SAFE_STOP_TIME_ET)
    
    return is_danger_window

async def _write_predictions_to_graph(predictions: List[Dict[str, Any]]):
    """
    [GNN 写回] 异步辅助函数，将 GNN 预测写回 Neo4j。
    """
    logger.info(f"Connecting to GraphDB to write {len(predictions)} GNN predictions...")
    graph_client = None
    try:
        graph_client = GraphDBClient()
        # (假设 GraphDBClient 在 __init__ 中处理连接)
        # (如果 GraphDBClient 是 Mock, verify_connectivity 可能不存在)
        # if hasattr(graph_client, 'verify_connectivity') and not await graph_client.verify_connectivity():
        #     logger.error("Failed to connect to Neo4j. GNN predictions will not be written back.")
        #     return

        # 批量写入查询
        # 这会找到 :Symbol 节点，如果 GNN 预测的属性还不存在，
        # 它会创建该属性 (e.g., gnn_predicted_risk_score)
        query = """
        UNWIND $predictions as pred
        MERGE (s:Symbol {id: pred.symbol})
        SET s.gnn_predicted_risk_score = pred.score,
            s.gnn_updated_at = timestamp()
        RETURN count(s) as updated_nodes
        """
        
        success = await graph_client.execute_write(query, params={"predictions": predictions})
        
        if success:
            logger.info("Successfully wrote GNN predictions back to Neo4j.")
            # [阶段 5] 写回成功！
            # GraphDBClient 的 schema 缓存会在下次 get_schema() 调用时失效并
            # 运行 'CALL apoc.meta.schema()'，
            # 此时 'gnn_predicted_risk_score' 将作为属性出现。
        else:
            logger.error("Neo4j execute_write (GNN predictions) failed.")

    except Exception as e:
        logger.error(f"Error during GNN prediction write-back: {e}", exc_info=True)
    finally:
        if graph_client and hasattr(graph_client, 'close'):
            await graph_client.close()

# --- 主训练函数 ---

def run_gnn_training_pipeline():
    """
    [阶段 3] GNN 夜间训练的主入口点。
    由 Celery (worker.py) 调用。
    """
    logger.info("[GNN Training] Nightly GNN training pipeline STARTED.")
    
    try:
        # 1. 导出图数据 (来自 Neo4j GDS)
        # (模拟)
        logger.info("[GNN Training] Exporting graph data from Neo4j GDS...")
        time.sleep(2) # 模拟 GDS 导出
        logger.info("[GNN Training] Graph data exported.")

        # 2. 训练 GNN 模型
        # (模拟)
        mock_model = MockGNNModel(metadata={"type": "GNN", "features": 128})
        num_epochs = 20 # 模拟 20 个 epochs
        training_success = False # 标志位
        
        for epoch in range(num_epochs):
            
            # [阶段 3] ET 感知安全逻辑
            # 必须在 *每个 epoch* 内部检查
            if _check_et_safety_window():
                logger.critical(
                    "[GNN Training] ET-Aware Safety Stop: "
                    f"已进入 {SAFE_STOP_TIME_ET.strftime('%H:%M')} ET 危险窗口 (市场即将开市)。"
                    "安全停止 GNN 训练。"
                )
                # 报警
                METRICS.increment_counter("nightly_pipeline_timeout_total", tags={"task": "gnn_training"})
                # 确保我们不会进入 "promote" 逻辑
                training_success = False 
                break # 退出训练循环

            # 模拟训练
            logger.info(f"[GNN Training] Running GNN epoch {epoch + 1}/{num_epochs}...")
            mock_model.train_epoch()
            # (模拟) ... 在这里评估验证集 ...
            
        else:
            # 'for...else' 块：仅在 'for' 循环 *正常完成* (未被 'break') 时执行
            logger.info("[GNN Training] GNN training completed successfully (not stopped).")
            training_success = True

        # 3. 暂存 (Save) 和 生效 (Promote)
        if training_success:
            logger.info("[GNN Training] Training successful. Proceeding to save and promote.")
            
            # 定义候选路径 (e.g., models/artifacts/gnn_candidate_20251112T0530.pth)
            timestamp_str = datetime.datetime.now(pytz.UTC).strftime("%Y%m%dT%H%M")
            candidate_path = os.path.join(MODEL_ARTIFACTS_DIR, f"gnn_candidate_{timestamp_str}.pth")

            # 1. 暂存 (Save)
            mock_model.save(candidate_path)
            
            # 2. 生效 (Promote) - [阶段 2]
            # 这是原子性更新。如果 GNN 训练失败或超时，
            # 此代码不会运行，Orchestrator 将自动加载昨天的模型。
            registry.promote_model("gnn", candidate_path)
            
            logger.info(f"[GNN Training] Successfully promoted new GNN model: {candidate_path}")
            
            # [GNN 写回] 
            # 3. 生成预测并写回 Neo4j
            # (在模型生效后执行)
            try:
                logger.info("[GNN Training] Writing GNN predictions back to Neo4j...")
                predictions = mock_model.predict_scores()
                if predictions:
                    # 我们在同步的 Celery 任务中，但 GDBClient 是异步的
                    # 所以我们使用 asyncio.run() 来执行这个一次性写回
                    asyncio.run(_write_predictions_to_graph(predictions))
                else:
                    logger.warning("[GNN Training] Mock model produced no predictions to write back.")
            except Exception as e:
                logger.error(f"[GNN Training] Failed during GNN write-back step: {e}", exc_info=True)
                # 我们不在这里 'raise'，因为训练和生效 (promote) 已经成功了
            
        else:
            logger.warning("[GNN Training] GNN training was stopped (timeout) or failed. Model will NOT be promoted.")
            # [自动回退]：我们什么都不做。注册表 (Registry) 仍指向旧模型。

        logger.info("[GNN Training] Nightly GNN training pipeline FINISHED.")

    except Exception as e:
        logger.error(f"[GNN Training] CRITICAL FAILURE in GNN pipeline: {e}", exc_info=True)
        # 向上抛出异常，以便 Celery 链 (chain) 知道它失败了
        raise

if __name__ == "__main__":
    # 用于本地测试
    logging.basicConfig(level=logging.INFO)
    logger.info("Running GNN Training Pipeline directly for testing...")
    # 添加一个基本的 MockGraphDBClient 如果它在 try/except 之外不可见
    if "GraphDBClient" not in globals():
        class MockGraphDBClient:
            async def execute_write(self, *args, **kwargs):
                logger.info(f"GraphDBClient.execute_write({args}, {kwargs})")
                return True
            async def close(self):
                logger.info("GraphDBClient.close()")
        GraphDBClient = MockGraphDBClient
        
    run_gnn_training_pipeline()
