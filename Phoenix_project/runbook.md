Phoenix Project 运维手册 (Runbook)

本文档是用于诊断和解决 Phoenix Project 生产环境中常见问题的操作指南。

关键服务健康检查

Docker 容器 (docker ps):

确保所有核心服务都在运行 (phoenix_api, phoenix_worker, phoenix_stream_consumer, phoenix_data_producer)。

检查基础设施 (phoenix_redis, phoenix_postgres, phoenix_elastic, phoenix_neo4j, phoenix_kafka) 是否都在运行。

API 服务 (phoenix_api):

操作: curl http://localhost:8000/health

预期: {"status": "ok"}

Celery Worker (phoenix_worker):

操作: docker logs phoenix_worker

预期: 寻找 celery@...: Ready。寻找 Orchestrator Main Cycle START/END 日志，确认 run_main_cycle 正在被 controller/scheduler.py 调度。

Kafka & Zookeeper:

操作: (进入 Kafka 容器) kafka-topics --list --bootstrap-server localhost:29092

预期: 看到已创建的主题（例如 market_data_stream, news_events_stream）。

数据生产者 (phoenix_data_producer):

操作: docker logs phoenix_data_producer

预期: 看到连接到券商 WebSocket 并成功生产 Kafka 消息的日志。

流消费者 (phoenix_stream_consumer):

操作: docker logs phoenix_stream_consumer

预期: 看到从 Kafka 消费消息并将其推送到 Redis (EventDistributor) 的日志。

常见问题 (FAQ) 与故障排查

症状: API 返回 500 错误

检查 API 日志: docker logs phoenix_api

问题: ConnectionError (连接到 Redis, Postgres, Neo4j...)

解决: 检查 docker ps 确保目标数据库容器正在运行。检查 .env 文件中的 ..._URI 变量是否使用了正确的 Docker 服务名称（例如 phoenix_postgres 而不是 localhost）。

问题: GeminiPoolManager 相关的 ValueError (例如 GEMINI_API_KEYS 未设置)。

解决: 检查 .env 文件，确保 GEMINI_API_KEYS 已设置并包含至少一个有效的、用逗号分隔的 API 密钥。

症状: 系统没有做出任何决策 (Orchestrator 循环空转)

检查 Celery Worker 日志: docker logs phoenix_worker

寻找: Retrieved 0 new events from EventDistributor.

诊断路径:

Orchestrator 正在运行，但 EventDistributor (Redis 队列) 是空的。

检查上游 -> docker logs phoenix_stream_consumer

问题 (Consumer): KafkaError: No brokers available 或无法连接。

解决 (Consumer): 重启 phoenix_stream_consumer 和 phoenix_kafka。

问题 (Consumer): 消费者正在运行，但没有收到消息。

检查上游 -> docker logs phoenix_data_producer

问题 (Producer): 生产者未能连接到券商 (例如 Alpaca WebSocket 认证失败)。

解决 (Producer): 检查 .env 中的 ALPACA_API_KEY / ALPACA_API_SECRET。

问题 (Producer): 生产者未能连接到 Kafka。

解决 (Producer): 检查 .env 中的 KAFKA_BOOTSTRAP_SERVERS 是否设置为 kafka:29092。

[已更新] 症状: Critical Failure: Cognitive Engine Fails or Hangs

这是最严重的故障之一，意味着 orchestrator.py 在 run_main_cycle 期间失败。

检查 Celery Worker 日志: docker logs phoenix_worker

[新] 寻找特定错误:

日志: CognitiveEngine failed with a known error: ...

含义: 这是 CognitiveEngine (cognitive/engine.py) 内部的业务逻辑失败（例如，ReasoningEnsemble 失败或 FactChecker 崩溃）。

[新] 诊断: 这是由 orchestrator.py 中的 except CognitiveError 捕获的。错误消息将明确指出是哪个 AI 子系统（如 ReasoningEnsemble）失败了。请深入检查该子系统的日志。

寻找严重错误:

日志: Failed to run CognitiveEngine: ... 或 Orchestrator main cycle failed: ...

含义: 这是一个更深层次的系统崩溃。

[新] 诊断 (Sync/Async): 如果错误是关于 asyncio.run() 或 coroutine 相关的，这可能表明 cognitive_engine.py 中的某个深层异步调用失败了。

诊断 (配置): 检查 config/system.yaml。CognitiveEngine 依赖于 ai.reasoning_ensemble 和 evaluation.fact_checker 的配置。检查这些配置是否正确。

诊断 (API): 检查 GeminiPoolManager 日志。AI 引擎可能因为所有 API 密钥均已达到速率限制或失效而无法执行。

症状: 订单未被执行

检查 Celery Worker 日志: docker logs phoenix_worker

寻找: order_manager.process_target_portfolio 相关的日志。

诊断:

问题: DataManager 无法获取最新价格 (Could not retrieve latest market price for ...)。

解决: 检查 phoenix_data_producer 是否正在运行，并为该资产向 Redis (latest_prices) 推送数据。

问题: RiskManager 拒绝了交易 (RiskManager evaluation triggered adjustments...)。

解决: 这是正常行为。检查 risk_report 日志以了解原因。

问题: OrderManager 尝试下单但失败 (例如 alpaca_trade_api.rest.APIError)。

解决: 检查 Alpaca 账户状态、API 密钥权限和可用购买力。
