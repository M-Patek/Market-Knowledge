"""
Phoenix Project 系统常量定义 (Phase 0 Task 0.2)
用于统一管理 Redis Keys、系统路径等核心常量。
"""

# --- Redis Key 模板 ---
# 实时市场数据 (Set/Hash) - 格式: phoenix:market:live:{symbol}
REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE = "phoenix:market:live:{symbol}"

# 新闻流 (List/Stream)
REDIS_KEY_NEWS_STREAM = "phoenix:news:stream"

# 事件分发队列 (List)
REDIS_KEY_EVENT_QUEUE = "phoenix_events"
