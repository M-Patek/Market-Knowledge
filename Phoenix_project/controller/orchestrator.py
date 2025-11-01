import asyncio
import logging
from typing import Dict, Any, Optional
from celery import Celery # 修正：导入 Celery

from data_manager import DataManager
from core.pipeline_state import PipelineState
from api.gemini_pool_manager import GeminiPoolManager
from strategy_handler import BaseStrategy
from execution.order_manager import OrderManager
from cognitive.engine import CognitiveEngine
from audit.logger import AuditLogger
from controller.error_handler import ErrorHandler

# 修正：不再导入 worker 任务，以避免循环依赖
# from worker import orchestrator_task 

class Orchestrator:
    """
    系统的中央协调器 (大脑)。
    
    它拥有所有核心组件的实例，并管理
    数据流、状态转换和决策循环。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: DataManager,
        pipeline_state: PipelineState,
        gemini_pool: GeminiPoolManager,
        strategy_handler: BaseStrategy,
        order_manager: OrderManager,
        celery_app: Celery # 修正：接收 Celery app 实例
    ):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.Orchestrator")
        self.data_manager = data_manager
        self.pipeline_state = pipeline_state
        self.gemini_pool = gemini_pool
        self.strategy_handler = strategy_handler
        self.order_manager = order_manager
        self.celery_app = celery_app # 修正：存储 Celery app
        
        # 初始化核心认知引擎
        self.cognitive_engine = CognitiveEngine(
            config=config,
            data_manager=data_manager,
            gemini_pool=gemini_pool,
            # 假设 AuditLogger 和 ErrorHandler 在这里创建
            audit_logger=AuditLogger(config.get('audit', {})),
            error_handler=ErrorHandler(config.get('error_handling', {}))
        )
        
        self.decision_loop_task: Optional[asyncio.Task] = None
        self._stop_loop = asyncio.Event()
        self.loop_frequency_seconds = config.get('orchestrator', {}).get('loop_frequency_seconds', 300) # 默认 5 分钟
        
        self.logger.info("Orchestrator initialized.")

    async def start_decision_loop(self):
        """
        启动主决策循环。
        """
        self.logger.info(f"Starting main decision loop (runs every {self.loop_frequency_seconds}s)")
        self._stop_loop.clear()
        
        while not self._stop_loop.is_set():
            try:
                await self.run_decision_cycle()
                
                # 等待下一次循环，或等待停止信号
                await asyncio.wait_for(
                    self._stop_loop.wait(), 
                    timeout=self.loop_frequency_seconds
                )
            except asyncio.TimeoutError:
                continue # 正常循环
            except Exception as e:
                self.logger.error(f"Error in decision loop: {e}", exc_info=True)
                # 发生严重错误时，冷却 60 秒
                await asyncio.sleep(60)

        logger.info("Decision loop stopped.")

    async def stop_decision_loop(self):
        """
        停止主决策循环。
        """
        self.logger.info("Stopping main decision loop...")
        self._stop_loop.set()
        if self.decision_loop_task:
            try:
                await self.decision_loop_task
            except asyncio.CancelledError:
                pass
        
    async def run_decision_cycle(self):
        """
        执行一个单独的、完整的决策周期。
        (例如，由 `start_decision_loop` 的计时器触发)
        """
        self.logger.info("--- Starting new decision cycle ---")
        
        # 1. 更新当前时间戳和状态
        current_time = pd.Timestamp.utcnow()
        self.pipeline_state.update_time(current_time)
        
        # 2. (可选) 检查系统健康状况
        # ...
        
        # 3. 运行策略的决策周期
        # 这将触发 CognitiveEngine.run_cycle()
        try:
            # 策略处理器 (RomanLegionStrategy) 包含认知引擎
            portfolio_decision = await self.strategy_handler.on_decision_cycle(
                current_time=current_time,
                state=self.pipeline_state
            )
            
            if portfolio_decision:
                # 4. 如果策略返回了决策，则将其传递给订单管理器
                self.logger.info(f"Decision cycle produced portfolio decision: {portfolio_decision.get('decision_id')}")
                
                # OrderManager 将计算差异并（模拟）执行订单
                await self.order_manager.process_portfolio_decision(
                    decision=portfolio_decision
                )
            else:
                self.logger.info("Decision cycle completed. No portfolio changes required.")

        except Exception as e:
            self.logger.error(f"Decision cycle failed: {e}", exc_info=True)
            # TODO: 将错误发送到 ErrorHandler
            
        self.logger.info("--- Decision cycle finished ---")

    async def process_realtime_event(self, event: Dict[str, Any]):
        """
        处理一个高优先级的实时事件 (例如，来自 EventDistributor 的新闻)。
        """
        self.logger.info(f"Processing real-time event: {event.get('event_id')}")
        
        try:
            # 1. (可选) 更新数据/状态
            # ...
            
            # 2. 将事件传递给策略以进行快速、战术性的响应
            tactical_signal = await self.strategy_handler.on_event(
                event=event,
                state=self.pipeline_state
            )
            
            if tactical_signal:
                # 3. 如果策略立即响应，则处理该战术信号
                self.logger.info(f"Event triggered tactical signal: {tactical_signal}")
                await self.order_manager.process_tactical_signal(tactical_signal)

        except Exception as e:
            self.logger.error(f"Real-time event processing failed: {e}", exc_info=True)

    def schedule_cognitive_workflow(self, task_description: str, context: Dict[str, Any]):
        """
        (由 Scheduler 或 API 调用)
        将一个完整的、重量级的认知工作流调度到 Celery worker 上。
        """
        self.logger.info(f"Scheduling cognitive workflow via Celery: {task_description}")
        
        try:
            # 修正：使用 .send_task() 按名称调用任务，
            # 而不是直接导入，以避免循环依赖。
            self.celery_app.send_task(
                'phoenix.run_cognitive_workflow',
                args=[task_description, context]
            )
            self.logger.info(f"Task '{task_description}' sent to Celery worker.")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule Celery task: {e}", exc_info=True)
            # TODO: ErrorHandler 回退逻辑

    async def run_cognitive_workflow(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        (由 Celery worker 或直接调用)
        实际执行重量级工作流。
        """
        self.logger.info(f"--- Running cognitive workflow: {task_description} ---")
        
        current_time = pd.Timestamp.utcnow()
        self.pipeline_state.update_time(current_time)
        
        try:
            portfolio_decision = await self.cognitive_engine.run_cycle(
                task_description=task_description,
                current_time=current_time,
                current_state=self.pipeline_state,
                context=context
            )
            
            if portfolio_decision:
                await self.order_manager.process_portfolio_decision(portfolio_decision)
                self.logger.info(f"Cognitive workflow complete. Decision: {portfolio_decision.get('decision_id')}")
                return portfolio_decision
            else:
                self.logger.info("Cognitive workflow complete. No decision made.")
                return {"status": "complete", "decision": None}

        except Exception as e:
            self.logger.error(f"Cognitive workflow failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def schedule_data_ingestion(self, sources: List[str]):
        """
        (由 Scheduler 调用)
        触发一个数据摄取任务。
        """
        self.logger.info(f"Scheduling data ingestion for sources: {sources}")
        # TODO: 这也可以是一个 Celery 任务
        try:
            # 暂时：同步调用 DataManager
            # (在未来，data_manager.ingest_data 应该是 async 的
            # 或者是一个 celery 任务)
            #
            # self.data_manager.ingest_data(sources)
            logger.warning("'schedule_data_ingestion' not yet implemented.")
            pass
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {e}", exc_info=True)
