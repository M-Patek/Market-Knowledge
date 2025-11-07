"""
[主人喵的修复 2] 新增文件

独立服务启动器：Kafka 消费者 (StreamProcessor)

此脚本被设计为作为一个独立的服务 (例如，一个专用的 Docker 容器) 运行。
它只负责一件事：运行 StreamProcessor 的永久循环，
从 Kafka 消费消息，并将它们推送到 EventDistributor (Redis 队列)。
"""
import os
import time
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.events.stream_processor import StreamProcessor
from Phoenix_project.events.event_distributor import EventDistributor
from Phoenix_project.monitor.logging import setup_logging, get_logger

# (配置重试逻辑)
RETRY_DELAY_SECONDS = 10 

def main():
    """
    主服务循环。
    """
    # 1. 初始化
    setup_logging()
    logger = get_logger("StreamProcessorService")
    logger.info("--- Starting StreamProcessor Service ---")
    
    stream_processor = None

    while True: # 保持服务永久运行
        try:
            # 2. 构建依赖项
            # (在循环内部构建，以便在 Redis 失败时可以重建)
            config_path = os.environ.get('PHOENIX_CONFIG_PATH', 'config')
            config_loader = ConfigLoader(config_path)
            
            event_distributor = EventDistributor()
            if not event_distributor.redis_client:
                raise ConnectionError("Failed to connect to EventDistributor (Redis). Retrying...")
                
            stream_processor = StreamProcessor(config_loader, event_distributor)
            
            # 3. 连接到 Kafka (这可能会失败并抛出异常)
            stream_processor.connect()
            
            # 4. 启动阻塞循环 (这可能会失败并抛出异常)
            logger.info("Kafka connected. Starting consumer loop...")
            stream_processor.start_consumer() # 阻塞
        
        except KeyboardInterrupt:
            logger.info("Service shutting down by user request.")
            break # 退出 while True 循环
            
        except Exception as e:
            logger.critical(f"StreamProcessor service failed: {e}. Retrying in {RETRY_DELAY_SECONDS}s...", exc_info=True)
            
        finally:
            # (清理旧的 consumer)
            if stream_processor and stream_processor.consumer:
                stream_processor.consumer.close()
                stream_processor = None
            
        time.sleep(RETRY_DELAY_SECONDS)

    logger.info("--- StreamProcessor Service Stopped ---")

if __name__ == "__main__":
    main()
