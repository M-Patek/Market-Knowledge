import json
import logging
from typing import List, Any, Generator, AsyncGenerator
import time
import pandas as pd # <-- [FIX] 添加 pandas 导入

from Phoenix_project.agents.l1.base import L1Agent
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem

# 获取日志记录器
logger = logging.getLogger(__name__)

class TechnicalAnalystAgent(L1Agent):
    """
    L1 智能体：技术分析师
    分析价格图表、交易量和技术指标。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
    ):
        """
        初始化 TechnicalAnalystAgent。
        
        参数:
            agent_id (str): 智能体的唯一标识符。
            llm_client (Any): 用于与 LLM API 交互的客户端 (例如 EnsembleClient)。
            data_manager (Any): 用于检索数据的 DataManager。
        """
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            data_manager=data_manager
        )
        logger.info(f"[{self.agent_id}] TechnicalAnalystAgent initialized.")

    async def run(self, task: Task, context: List[Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        异步运行智能体以执行技术分析。
        
        参数:
            task (Task): 分配给智能体的任务。
            context (List[Any]): 智能体运行所需的附加上下文（例如，原始文档）。

        收益:
            AsyncGenerator[EvidenceItem, None]: 异步生成一个或多个 EvidenceItem。
        """
        logger.info(f"[{self.agent_id}] Running technical analysis for task: {task.task_id} on symbols: {task.symbols}")

        if not task.symbols:
            logger.warning(f"[{self.agent_id}] Task {task.task_id} has no symbols. Skipping.")
            return

        # 1. 准备 Prompt 的上下文
        symbol = task.symbols[0]
        
        # [FIX] 从 DataManager 获取 K 线数据，而不是依赖 context
        try:
            # (我们需要为技术分析定义一个合理的时间范围)
            # (暂时硬编码一个回溯期，例如 90 天。这应该来自配置。)
            end_date = pd.Timestamp.now(tz='UTC')
            start_date = end_date - pd.Timedelta(days=90)
            
            # get_market_data 返回 Dict[str, pd.DataFrame]
            market_data_dfs = self.data_manager.get_market_data(
                symbols=[symbol], 
                start_date=start_date, 
                end_date=end_date
            )
            
            klines_df = market_data_dfs.get(symbol)
            
            if klines_df is None or klines_df.empty:
                logger.warning(f"[{self.agent_id}] No kline data found for {symbol} from data_manager. Skipping.")
                return
            
            # (将 DataFrame 转换为 LLM 可读的格式，例如 JSON)
            # (只取几行数据，避免 prompt 过长)
            context_str = klines_df.tail(30).to_json(orient="records")
            
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to get kline data for {symbol}. Error: {e}", exc_info=True)
            return

        if not context_str:
             logger.warning(f"[{self.agent_id}] Kline data for {symbol} was empty after processing. Skipping.")
             return

        context_map = {
            "symbol": symbol,
            "context": context_str
        }
        
        # [关键] 设置此智能体对应的 Prompt 名称
        agent_prompt_name = "l1_technical_analyst" 

        try:
            # 2. 异步调用 LLM
            logger.debug(f"[{self.agent_id}] Calling LLM with prompt: {agent_prompt_name} for symbol: {symbol}")
            
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )
            
            if not response_str:
                logger.warning(f"[{self.agent_id}] LLM returned no response for task {task.task_id}.")
                return

            # 3. 解析和验证 JSON 输出
            logger.debug(f"[{self.agent_id}] Received LLM response (raw): {response_str[:200]}...")
            response_data = json.loads(response_str)
            
            # 4. 验证并生成 EvidenceItem
            evidence = EvidenceItem.model_validate(response_data)
            evidence.agent_id = self.agent_id
            
            logger.info(f"[{self.agent_id}] Successfully generated evidence for {symbol} with confidence {evidence.confidence}")
            yield evidence

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_id}] Failed to decode LLM JSON response for task {task.task_id}. Error: {e}")
            logger.debug(f"[{self.agent_id}] Faulty JSON string: {response_str}")
            return
        except Exception as e:
            logger.error(f"[{self.agent_id}] An error occurred during agent run for task {task.task_id}. Error: {e}")
            return
