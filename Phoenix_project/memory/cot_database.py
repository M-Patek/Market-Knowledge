from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import aiofiles
import json
import os

# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.fusion_result import FusionResult
# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import ESLogger


class CoTDatabase:
    """
    一个简单的基于文件系统的数据库，用于存储思维链 (CoT)
    推理轨迹 (FusionResult 对象) 以供审计和分析。
    """

    def __init__(self, config: Dict[str, Any], logger: ESLogger):
        """
        初始化 CoTDatabase。

        Args:
            config: 包含配置的字典，
                    特别是 `db_path`。
            logger: ESLogger 实例，用于日志记录。
        """
        self.db_path = config.get("db_path", "./audit_traces")
        self.logger = logger
        self.lock = asyncio.Lock() # 用于 get_all_keys 的锁

        # 确保数据库目录存在
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.logger.log_info(
                f"CoTDatabase initialized. Storage path: {self.db_path}"
            )
        except OSError as e:
            self.logger.log_error(
                f"Failed to create CoTDatabase directory at {self.db_path}: {e}",
                exc_info=True,
            )
            raise

    def _get_filepath(self, event_id: str) -> str:
        """帮助程序：获取给定事件 ID 的文件路径。"""
        # 清理 event_id 以防止路径遍历问题
        filename = "".join(c for c in event_id if c.isalnum() or c in ("-", "_", "."))
        if not filename:
            filename = f"invalid_event_id_{hash(event_id)}"
        return os.path.join(self.db_path, f"{filename}.json")

    async def store_trace(
        self, event_id: str, trace_data: Dict[str, Any]
    ) -> bool:
        """
        存储给定事件的完整推理轨迹 (FusionResult)。

        Args:
            event_id: 事件的唯一标识符。
            trace_data: FusionResult 对象 (作为字典)。

        Returns:
            如果存储成功，则为 True，否则为 False。
        """
        if not event_id:
            self.logger.log_warning("store_trace called with empty event_id. Skipping.")
            return False

        filepath = self._get_filepath(event_id)
        self.logger.log_debug(f"Storing trace for event {event_id} to {filepath}")

        try:
            # 在异步环境中，文件写入应该使用锁，
            # 尽管 aiofiles 可能是线程安全的，但保险起见
            async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                # 我们序列化 Pydantic 模型 (作为 dict 传入)
                await f.write(json.dumps(trace_data, indent=2, default=str))
            self.logger.log_info(f"Successfully stored trace for event {event_id}.")
            return True
        except Exception as e:
            self.logger.log_error(
                f"Failed to store trace for event {event_id}: {e}", exc_info=True
            )
            return False

    async def retrieve_trace(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        检索特定事件 ID 的推理轨迹。

        Args:
            event_id: 事件的唯一标识符。

        Returns:
            如果找到，则为存储的轨迹字典，否则为 None。
        """
        filepath = self._get_filepath(event_id)
        self.logger.log_debug(f"Retrieving trace for event {event_id} from {filepath}")

        if not os.path.exists(filepath):
            self.logger.log_warning(f"No trace found for event {event_id} at {filepath}")
            return None

        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                data = await f.read()
            return json.loads(data)
        except Exception as e:
            self.logger.log_error(
                f"Failed to retrieve or parse trace for event {event_id}: {e}",
                exc_info=True,
            )
            return None

    async def search_traces(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        [已优化]
        在所有存储的轨迹中搜索关键字。
        警告：这是一个非常低效的 O(N) 文件系统扫描。
        不适用于生产环境，但实现了所请求的功能。
        """
        self.logger.log_debug(f"Searching traces for keywords: {keywords} (limit {limit})")
        if not keywords:
            return []
            
        all_keys = await self.get_all_keys()
        matches = []
        keywords_lower = [k.lower() for k in keywords]
        
        # (并行读取文件以提高速度)
        async def check_file(key):
            try:
                trace = await self.retrieve_trace(key)
                if not trace:
                    return None
                
                # 将整个 JSON 转换为小写字符串进行搜索
                content_to_search = json.dumps(trace).lower()
                
                # 如果任何关键字匹配，则返回轨迹
                if any(k in content_to_search for k in keywords_lower):
                    return trace
            except Exception as e:
                self.logger.log_warning(f"Failed to search trace {key}: {e}")
            return None

        tasks = [check_file(key) for key in all_keys]
        results = await asyncio.gather(*tasks)
        
        for trace in results:
            if trace:
                matches.append(trace)
                if len(matches) >= limit:
                    break
                    
        self.logger.log_info(f"Inefficient search_traces found {len(matches)} matches.")
        
        # (理想情况下，我们应该按相关性或日期排序，但目前只返回找到的前 N 个)
        return matches

    async def query_by_time(
        self, start_time: datetime, end_time: datetime, limit: int
    ) -> List[Dict[str, Any]]:
        """
        按时间低效地查询轨迹。
        警告：这是一个占位符。它扩展性很差。
        """
        self.logger.log_warning("query_by_time is highly inefficient on filesystem DB.")
        traces = []
        all_files = await self.get_all_keys()
        
        for event_id in all_files:
            if len(traces) >= limit:
                break
            trace = await self.retrieve_trace(event_id)
            if trace and "timestamp" in trace:
                try:
                    # 假设时间戳是标准 ISO 格式
                    ts_str = trace["timestamp"]
                    # 处理潜在的时区信息
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    
                    # 确保时间可比较 (例如，都是 aware 或都是 naive)
                    # 这是一个复杂的主题，这里简化了。
                    if (
                        ts.tzinfo is None
                        and start_time.tzinfo is not None
                        and end_time.tzinfo is not None
                    ):
                        # 临时：假设轨迹时间戳是 UTC (如果其他的是)
                        ts = ts.replace(tzinfo=datetime.timezone.utc)
                    
                    if start_time <= ts <= end_time:
                        traces.append(trace)
                except Exception as e:
                    self.logger.log_warning(
                        f"Could not parse timestamp for event {event_id}: {e}"
                    )
        return traces

    async def get_all_keys(self) -> List[str]:
        """
        低效地获取所有事件 ID (文件名)。
        警告：这是一个占位符。
        """
        try:
            async with self.lock:
                # (使用 os.listdir 是阻塞的，在异步代码中不好)
                # (使用 asyncio.to_thread 运行它)
                def list_dir_sync():
                    return [
                        f.replace(".json", "")
                        for f in os.listdir(self.db_path)
                        if f.endswith(".json")
                    ]
                
                files = await asyncio.to_thread(list_dir_sync)
            return files
        except Exception as e:
            self.logger.log_error(f"Failed to list all keys in CoTDatabase: {e}")
            return []
