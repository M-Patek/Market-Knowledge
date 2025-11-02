"""
指标 (Metrics)
定义用于监控的指标接口。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

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

# FIX (E6): 添加 PrometheusMetrics 占位符类
class PrometheusMetrics(IMetricsCollector):
    """
    (占位符)
    一个 IMetricsCollector 的实现，用于将指标暴露给 Prometheus。
    metrics_collector.py 试图导入这个类。
    """
    
    def __init__(self, config: Dict[str, Any]):
        print(f"INFO: (Stub) PrometheusMetrics Initialized.")
        # 在真实实现中，这里会启动一个 Prometheus 客户端
        # (e.g., from prometheus_client import start_http_server, Gauge)
        pass

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        print(f"METRIC (Gauge): {name}={value} (Tags: {tags})")
        pass
        
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        print(f"METRIC (Increment): {name}+{value} (Tags: {tags})")
        pass
        
    def timing(self, name: str, value: float, tags: Dict[str, str] = None):
        print(f"METRIC (Timing): {name}={value}ms (Tags: {tags})")
        pass
