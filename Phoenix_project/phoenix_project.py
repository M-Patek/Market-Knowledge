import os
import sys
import asyncio
from dotenv import load_dotenv

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 从 .env 文件加载环境变量
load_dotenv()

# 修正: 从新的 config.loader 模块导入 load_config 函数
from config.loader import load_config
from monitor.logging import get_logger
from controller.orchestrator import Orchestrator
from data_manager import DataManager
from core.pipeline_state import PipelineState
from strategy_handler import RomanLegionStrategy
from api.gemini_pool_manager import GeminiPoolManager
from events.stream_processor import StreamProcessor
from events.event_distributor import EventDistributor
from execution.order_manager import OrderManager

# --- 全局组件 ---
logger = get_logger('PhoenixMain')
config = None
orchestrator = None
gemini_pool = None
stream_processor = None
event_distributor = None

async def initialize_system():
    """
    初始化 Phoenix 系统的所有核心组件。
    """
    global config, orchestrator, gemini_pool, stream_processor, event_distributor, logger
    
    logger.info("--- PHOENIX PROJECT V2.0 INITIALIZATION START ---")
    
    try:
        # 1. 加载配置
        config_path = os.getenv('CONFIG_PATH', 'config/system.yaml')
        config = load_config(config_path) # 修正: 调用新加载器函数
        if config is None:
            logger.critical("加载 system.yaml 失败。正在退出。")
            sys.exit(1)
        logger.info(f"配置已从 {config_path} 加载")

        # 2. 初始化 Gemini API 池
        # 该池将在所有组件之间共享
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("未设置 GEMINI_API_KEY。LLM 功能将被禁用。")
        
        gemini_pool = GeminiPoolManager(
            api_key=gemini_api_key,
            pool_size=config.get('llm', {}).get('gemini_pool_size', 5)
        )
        logger.info(f"GeminiPoolManager 已初始化，大小为 {config.get('llm', {}).get('gemini_pool_size', 5)}")

        # 3. 初始化核心状态和数据管理
        pipeline_state = PipelineState()
        # 修正: DataManager 构造函数需要 cache_dir
        cache_dir = config.get('data_manager', {}).get('cache_dir', 'data_cache')
        data_manager = DataManager(config, pipeline_state, cache_dir=cache_dir)
        logger.info("PipelineState 和 DataManager 已初始化。")

        # 4. 初始化订单管理器 (执行层)
        order_manager = OrderManager(config.get('execution', {}))
        logger.info("OrderManager 已初始化。")

        # 5. 初始化策略处理器 (RomanLegion)
        logger.info("加载策略数据...")
        
        strategy = RomanLegionStrategy(
            config=config,
            data_manager=data_manager
        )
        logger.info("RomanLegionStrategy 已初始化。")

        # 6. 初始化协调器 (大脑)
        # 将所有共享组件传递给协调器
        orchestrator = Orchestrator(
            config=config,
            data_manager=data_manager,
            pipeline_state=pipeline_state,
            gemini_pool=gemini_pool,
            strategy_handler=strategy,
            order_manager=order_manager
        )
        logger.info("主协调器已初始化。")
        
        # 7. 初始化事件流和分发器
        stream_processor = StreamProcessor(config.get('event_stream', {}))
        event_distributor = EventDistributor(
            stream_processor=stream_processor,
            orchestrator=orchestrator,
            config=config.get('event_distributor', {})
        )
        logger.info("EventStreamProcessor 和 EventDistributor 已初始化。")

        logger.info("--- PHOENIX 系统初始化完成 ---")
        return True

    except Exception as e:
        logger.critical(f"系统初始化期间发生致命错误: {e}", exc_info=True)
        return False

async def run_system():
    """
    启动系统的主要异步循环。
    """
    global orchestrator, event_distributor, logger
    
    if not orchestrator or not event_distributor:
        logger.critical("系统未初始化。无法运行。")
        return

    logger.info("--- PHOENIX 系统正在启动主循环 ---")
    
    try:
        # 1. 启动协调器的主决策循环 (例如，每 5 分钟运行一次)
        orchestrator_task = asyncio.create_task(orchestrator.start_decision_loop())
        
        # 2. 启动事件分发器 (从流处理器消费)
        event_distributor_task = asyncio.create_task(event_distributor.start_consuming())
        
        logger.info("协调器和事件分发器循环正在运行。")
        
        # 等待任务完成 (或永远运行)
        await asyncio.gather(
            orchestrator_task,
            event_distributor_task
        )
        
    except asyncio.CancelledError:
        logger.info("主系统循环已取消。")
    except Exception as e:
        logger.error(f"主系统运行循环中发生错误: {e}", exc_info=True)
    finally:
        logger.info("--- PHOENIX 系统正在关闭 ---")
        await shutdown_system()

async def shutdown_system():
    """
    优雅地关闭所有系统组件。
    """
    global orchestrator, event_distributor, gemini_pool, logger
    
    logger.info("开始优雅关闭...")
    if event_distributor:
        await event_distributor.stop_consuming()
        logger.info("事件分发器已停止。")
        
    if orchestrator:
        await orchestrator.stop_decision_loop()
        logger.info("协调器循环已停止。")
        
    if gemini_pool:
        await gemini_pool.close()
        logger.info("Gemini API 池已关闭。")
    
    logger.info("--- PHOENIX 关闭完成 ---")

async def main():
    """
    应用程序的主入口点。
    """
    if await initialize_system():
        loop = asyncio.get_running_loop()
        try:
            # TODO: 添加信号处理器以实现优雅关闭
            await run_system()
        except KeyboardInterrupt:
            logger.info("收到 KeyboardInterrupt。正在关闭。")
            await shutdown_system()
    else:
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        logger.info(f"检测到 CLI 命令: {command}")
        # TODO: 实现 CLI 参数处理
        print(f"CLI 命令 '{command}' 尚未实现。启动主系统。")
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"应用程序运行失败: {e}", exc_info=True)
        sys.exit(1)

