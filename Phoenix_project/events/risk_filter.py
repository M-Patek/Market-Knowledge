"""
事件风险过滤器
在事件进入认知引擎之前，对其进行初步过滤。
(例如：过滤掉不相关的符号、低可信度来源的新闻)
"""
from typing import List, Dict, Any
from config.loader import ConfigLoader

# FIX (E1): 导入统一后的核心模式
from core.schemas.data_schema import MarketData, NewsData

class EventRiskFilter:
    """
    根据 config/event_filter_config.yaml 中的规则过滤传入的事件。
    """
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.get_event_filter_config()
        self.log_prefix = "EventRiskFilter:"
        
        self.min_news_confidence = self.config.get("min_news_confidence", 0.0)
        self.allowed_sources = set(self.config.get("allowed_sources", []))
        self.blocked_symbols = set(self.config.get("blocked_symbols", []))
        
        print(f"{self.log_prefix} Initialized.")
        if self.allowed_sources:
            print(f"{self.log_prefix} Allowed sources: {self.allowed_sources}")
        if self.blocked_symbols:
            print(f"{self.log_prefix} Blocked symbols: {self.blocked_symbols}")

    def filter_batch(self, data_batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        过滤整个数据批次。
        """
        filtered_batch = {
            "market_data": [],
            "news_data": []
            # economic_indicators 通常不过滤
        }
        
        # FIX (E1): 过滤 MarketData
        if "market_data" in data_batch:
            filtered_batch["market_data"] = [
                data for data in data_batch["market_data"]
                if self._filter_market_data(data)
            ]
            
        # FIX (E1): 过滤 NewsData
        if "news_data" in data_batch:
            filtered_batch["news_data"] = [
                data for data in data_batch["news_data"]
                if self._filter_news_data(data)
            ]
        
        # 保留其他数据类型
        for key, value in data_batch.items():
            if key not in filtered_batch:
                filtered_batch[key] = value
                
        return filtered_batch

    def _filter_market_data(self, data: MarketData) -> bool:
        """
        过滤市场数据的规则。
        """
        if data.symbol in self.blocked_symbols:
            print(f"{self.log_prefix} Blocking market data for {data.symbol}")
            return False
        
        # 示例：过滤掉异常的0成交量数据
        if data.volume <= 0:
            return False
            
        return True

    def _filter_news_data(self, data: NewsData) -> bool:
        """
        过滤新闻事件的规则。
        """
        # 1. 按来源过滤
        if self.allowed_sources and data.source not in self.allowed_sources:
            return False
            
        # 2. 按符号过滤
        if any(symbol in self.blocked_symbols for symbol in data.symbols):
            print(f"{self.log_prefix} Blocking news {data.id} due to blocked symbol")
            return False
            
        # 3. (假设) 按可信度过滤
        confidence = data.metadata.get("confidence_score", 1.0)
        if confidence < self.min_news_confidence:
            return False
            
        return True
