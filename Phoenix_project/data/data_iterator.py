"""
数据迭代器
负责按时间顺序模拟数据流，用于回测。
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator

# FIX (E1): 导入统一后的核心模式
from core.schemas.data_schema import MarketData, NewsData

class DataIterator:
    """
    一个生成器，用于模拟历史数据的时间流逝。
    它按时间戳顺序 'yield' 数据点或数据批次。
    """
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        market_data_dfs: Dict[str, pd.DataFrame],
        news_data_df: Optional[pd.DataFrame] = None,
        # economic_data_df: Optional[pd.DataFrame] = None, # 经济数据可以类似地添加
        step: timedelta = timedelta(days=1)
    ):
        """
        初始化迭代器。
        假设所有传入的 DataFrame 都已按时间戳索引。
        """
        self.current_time = start_date
        self.end_date = end_date
        self.step = step
        
        # 预处理市场数据
        self.market_data_iters = {
            symbol: df[ (df.index >= start_date) & (df.index <= end_date) ].to_dict('index')
            for symbol, df in market_data_dfs.items()
        }
        
        # 预处理新闻数据
        self.news_data = {}
        if news_data_df is not None:
            news_df = news_data_df[ (news_data_df.index >= start_date) & (news_data_df.index <= end_date) ]
            # 按日期分组，以便快速查找
            self.news_data = {
                date: group.to_dict('records')
                for date, group in news_df.groupby(news_df.index.date)
            }

        print(f"DataIterator initialized: {start_date} to {end_date} with step {step}")

    def __iter__(self) -> Generator[Dict[str, List[Any]], None, None]:
        """
        开始迭代。
        """
        while self.current_time <= self.end_date:
            
            current_date = self.current_time.date()
            batch_data = {
                "market_data": [],
                "news_data": []
            }
            
            # 1. 检索市场数据
            # 注意：真实的回测引擎会更精确地处理时间戳，这里为简化示例
            for symbol, data in self.market_data_iters.items():
                # 尝试找到与 current_time 完全匹配或最接近的数据点
                # 为简单起见，我们假设 market_data_iters 的键是 datetime 对象
                # 在实际应用中，pandas 的 asof() 或 reindex() 会更稳健
                
                # 这是一个简化的查找，假设索引是排序的
                found_data = data.get(self.current_time)
                if found_data:
                    # FIX (E1): 使用 MarketData 模式
                    batch_data["market_data"].append(
                        MarketData(
                            symbol=symbol,
                            timestamp=self.current_time,
                            open=found_data['open'],
                            high=found_data['high'],
                            low=found_data['low'],
                            close=found_data['close'],
                            volume=found_data['volume']
                        )
                    )

            # 2. 检索新闻数据 (当天发布的所有新闻)
            news_for_date = self.news_data.get(current_date, [])
            for news_item in news_for_date:
                # FIX (E1): 使用 NewsData 模式
                batch_data["news_data"].append(
                    NewsData(
                        id=news_item['id'],
                        source=news_item['source'],
                        timestamp=news_item['timestamp'], # 假设df中的timestamp是datetime对象
                        symbols=news_item.get('symbols', []),
                        content=news_item['content'],
                        headline=news_item.get('headline')
                    )
                )

            # 3. (可选) 检索经济数据...

            # 只有当这一天有数据时才 yield
            if batch_data["market_data"] or batch_data["news_data"]:
                yield batch_data
            
            # 前进到下一个时间步
            self.current_time += self.step
