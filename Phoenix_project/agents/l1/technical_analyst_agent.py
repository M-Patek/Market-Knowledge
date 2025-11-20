import json
import logging
from typing import List, Any, Generator, AsyncGenerator, Dict # [FIX] Add Dict
import time
import pandas as pd # <-- [FIX] 添加 pandas 导入
from Phoenix_project.agents.l1.base import L1Agent, logger
from Phoenix_project.core.schemas.data_schema import MarketData, EventData
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceSource
from Phoenix_project.core.pipeline_state import PipelineState

class TechnicalAnalystAgent(L1Agent):
    """
    L1 智能体，专注于技术分析。
    分析 K 线数据、交易量、波动性等，以识别趋势、支撑/阻力位和技术形态。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
        config: Dict[str, Any] = None # [FIX] Add config
    ):
        """
        初始化 TechnicalAnalystAgent。
        """
        super().__init__(
            agent_id=agent_id,
            role="Technical Analyst",
            prompt_template_name="l1_technical_analyst",
            llm_client=llm_client,
            data_manager=data_manager
        )
        self.config = config or {} # [FIX] Store config
        self.lookback_days = self.config.get('technical_lookback_days', 90) # [FIX] Get lookback from config
        logger.info(f"[{self.agent_id}] TechnicalAnalystAgent initialized with lookback: {self.lookback_days} days.") # [FIX] Log config

    async def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> AsyncGenerator[EvidenceItem, None]:
        """
        异步运行智能体以处理单个任务。
        [Refactored Phase 3.1] 对齐 BaseL1Agent 签名，使用 state.current_time 和 DataManager。
        """
        # 从 dependencies (任务载荷) 中提取上下文
        # 注意：Executor 传递的是 task['content']，我们需要确保 symbol 存在其中
        symbol = dependencies.get("symbol")
        task_id = dependencies.get("task_id", "unknown_task")
        event_id = dependencies.get("event_id", "unknown_event")
        description = dependencies.get("description", "Technical Analysis Task")
        
        logger.info(f"[{self.agent_id}] Running task: {task_id} for symbol: {symbol}")

        if not symbol:
            logger.warning(f"[{self.agent_id}] Task {task_id} missing 'symbol' in dependencies.")
            return

        # [FIX] 从 DataManager 获取 K 线数据，而不是依赖 context
        try:
            # [Time Machine] 强制使用仿真时间
            end_date = state.current_time
            # [FIX] Use configured lookback period instead of 90
            start_date = end_date - pd.Timedelta(days=self.lookback_days)
            
            # [Data Flow] 调用 DataManager 的标准历史接口 (Task 1.1/1.2)
            market_data_df = await self.data_manager.get_market_data_history(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            
            if market_data_df is None or market_data_df.empty:
                logger.warning(f"[{self.agent_id}] No market data found for {symbol} from {start_date} to {end_date}.")
                return

            # (这是一个简化的数据准备步骤)
            # (在实际应用中，我们会计算 TA 指标，如 MA, RSI, MACD)
            # (目前，我们只传递原始 K 线数据的摘要)
            
            data_summary = f"Technical data for {symbol} (last {self.lookback_days} days):\n"
            data_summary += f"Data points: {len(market_data_df)}\n"
            
            # 确保 DataFrame 有数据后再访问
            if not market_data_df.empty:
                # 假设时间索引或列名为 'timestamp'/'time'，DataManager 返回的 DF 应该已经处理过
                # 这里简单假设它是 DataFrame
                latest_close = market_data_df['close'].iloc[-1] if 'close' in market_data_df else 0.0
                high_max = market_data_df['high'].max() if 'high' in market_data_df else 0.0
                low_min = market_data_df['low'].min() if 'low' in market_data_df else 0.0
                vol_mean = market_data_df['volume'].mean() if 'volume' in market_data_df else 0.0

                data_summary += f"Latest Close: {latest_close:.2f}\n"
                data_summary += f"{self.lookback_days}d High: {high_max:.2f}\n"
                data_summary += f"{self.lookback_days}d Low: {low_min:.2f}\n"
                data_summary += f"{self.lookback_days}d Avg Volume: {vol_mean:.0f}\n"
            
            # (此处可以添加更多指标)
            # (例如：data_summary += f"RSI(14): {calculate_rsi(market_data_df, 14)}\n")

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error retrieving or processing market data for {symbol}: {e}", exc_info=True)
            return

        # 3. 生成提示
        prompt_data = {
            "symbol": symbol,
            "event_description": description,
            "market_data_summary": data_summary
        }
        
        try:
            prompt = self.render_prompt(prompt_data)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error rendering prompt: {e}", exc_info=True)
            return

        # 4. 调用 LLM
        try:
            llm_response = await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error calling LLM: {e}", exc_info=True)
            return

        # 5. 解析响应并生成 EvidenceItem
        # (假设 LLM 的响应是一个 JSON 字符串，包含 'analysis', 'sentiment', 'key_levels')
        try:
            # (这个解析逻辑非常简化)
            response_json = json.loads(llm_response)
            
            analysis_text = response_json.get("analysis", "No analysis provided.")
            sentiment = response_json.get("sentiment", "Neutral").capitalize()
            key_levels = response_json.get("key_levels", {}) # e.g., {"support": [100], "resistance": [120]}

            # (验证情感)
            if sentiment not in ["Bullish", "Bearish", "Neutral"]:
                sentiment = "Neutral"

            content = (
                f"**Technical Analysis ({sentiment}):** {analysis_text}\n"
                f"**Key Levels:** Support: {key_levels.get('support', 'N/A')}, Resistance: {key_levels.get('resistance', 'N/A')}"
            )
            
            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Technical Analysis Report",
                content=content,
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(), # [Time Machine] Use simulation time for evidence timestamp
                tags=["technical", "analysis", sentiment.lower()],
                raw_data=response_json
            )
            
            logger.info(f"[{self.agent_id}] Generated evidence for task {task_id}")
            yield evidence

        except json.JSONDecodeError:
            logger.warning(f"[{self.agent_id}] Failed to decode LLM JSON response: {llm_response}")
            # (如果 JSON 失败，我们将原始文本作为证据)
            evidence = EvidenceItem(
                agent_id=self.agent_id,
                task_id=task_id,
                event_id=event_id,
                symbol=symbol,
                headline="Technical Analysis (Raw)",
                content=f"Raw LLM Output (JSON parse failed): {llm_response}",
                data_source=EvidenceSource.AGENT_ANALYSIS,
                timestamp=state.current_time.timestamp(), # [Time Machine]
                tags=["technical", "raw", "error"],
                raw_data={"raw_text": llm_response}
            )
            yield evidence
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error parsing LLM response or creating evidence: {e}", exc_info=True)
