"""
指标 (Metrics)
定义用于监控的指标接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

# --- 蓝图 1：导入 Prometheus 客户端 ---
from prometheus_client import Counter, Gauge, Histogram
# --- 结束：蓝图 1 ---

class IMetricsCollector(ABC):
    """
    指标收集器接口。
    """
    
    @abstractmethod
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """
        记录一个瞬时值 (e.g., portfolio_value)。
        """
        pass
        
    @abstractmethod
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """
        增加一个计数器 (e.g., orders_filled)。
        """
        pass
        
    @abstractmethod
    def timing(self, name: str, value: float, tags: Dict[str, str] = None):
        """
        记录一个持续时间 (e.g., cognitive_engine_latency)。
        """
        pass

# --- 蓝图 1：实现 PrometheusMetrics ---
class PrometheusMetrics(IMetricsCollector):
    """
    一个 IMetricsCollector 的实现，用于将指标暴露给 Prometheus。
    这现在是一个功能齐全的实现，而不是占位符。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.log_prefix = "PrometheusMetrics:"
        
        # 1. 定义蓝图中请求的特定指标
        self.api_request_latency = Histogram(
            'api_request_latency_seconds',
            'API 请求延迟 (Histogram)',
            ['method', 'path']
        )
        self.celery_task_status = Counter(
            'celery_task_status_total',
            'Celery 任务状态 (Counter)',
            ['task_name', 'status'] # status: 'success' or 'failure'
        )
        self.ai_analysis_errors = Counter(
            'ai_analysis_errors_total',
            'AI 分析错误 (Counter)',
            ['agent_id', 'error_type']
        )
        self.db_query_time = Gauge(
            'db_query_time_seconds',
            '数据库查询时间 (Gauge)',
            ['db_name', 'query_type']
        )
        
        # 2. 为 IMetricsCollector 接口的通用调用准备动态指标
        self.generic_gauges: Dict[str, Gauge] = {}
        self.generic_counters: Dict[str, Counter] = {}
        self.generic_histograms: Dict[str, Histogram] = {}
        
        print(f"INFO: {self.log_prefix} PrometheusMetrics (真实实现) 已初始化。")
        # 注意：start_http_server 不在这里调用。
        # 它由 worker.py 为 worker 指标调用，
        # 并由 interfaces/api_server.py 的中间件为 API 调用。

    def _get_or_create_metric(self, registry: Dict, metric_class, name: str, tags: Dict[str, str] = None):
        """辅助函数：如果指标尚不存在，则动态创建。"""
        label_keys = sorted(tags.keys()) if tags else []
        metric_key = f"{name}_{'_'.join(label_keys)}"
        
        if metric_key not in registry:
            registry[metric_key] = metric_class(name, f'通用指标 {name}', label_keys)
        return registry[metric_key]

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        tags = tags or {}
        # 映射到预定义指标
        if name == 'db_query_time' and 'db_name' in tags and 'query_type' in tags:
            self.db_query_time.labels(**tags).set(value)
        else:
            # 回退到通用
            metric = self._get_or_create_metric(self.generic_gauges, Gauge, name, tags)
            metric.labels(**tags).set(value)

    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        tags = tags or {}
        # 映射到预定义指标
        if name == 'celery_task_status' and 'task_name' in tags and 'status' in tags:
            self.celery_task_status.labels(**tags).inc(value)
        elif name == 'ai_analysis_errors' and 'agent_id' in tags and 'error_type' in tags:
            self.ai_analysis_errors.labels(**tags).inc(value)
        else:
            # 回退到通用
            metric = self._get_or_create_metric(self.generic_counters, Counter, name, tags)
            metric.labels(**tags).inc(value)
        
    def timing(self, name: str, value: float, tags: Dict[str, str] = None):
        tags = tags or {}
        # 映射到预定义指标
        if name == 'api_request_latency' and 'method' in tags and 'path' in tags:
            self.api_request_latency.labels(**tags).observe(value)
        else:
            # 回退到通用 (Histogram)
            metric = self._get_or_create_metric(self.generic_histograms, Histogram, name, tags)
            metric.labels(**tags).observe(value)
