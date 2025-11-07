Phoenix Project - Operational Runbook
Version: 2.1 (已根据代码库 v2.0 更新)
Last Updated: 2025-11-08
Contact: Phoenix Project Team

系统概述 (System Overview)

Phoenix Project 是一个全面的量化交易研究平台。它旨在自动化整个回测流程，包括数据采集、特征计算、AI 增强分析、策略执行模拟和报告。

系统可以运行在两种主要模式下，如 phoenix_project.py 和 worker.py 所示：

回测模式 (Backtest Mode): 通过运行 python phoenix_project.py 触发。这将使用 config/system.yaml 中定义的参数（如 start_date, end_date）执行单次回测。

实时模式 (Live Mode): 通过 Celery worker (worker.py) 或主应用的 run_live() 方法启动。此模式依赖 controller/scheduler.py 中定义的调度任务来定期触发 Orchestrator 的 run_main_cycle。

先决条件与设置 (Prerequisites & Setup)

a. 克隆仓库

git clone <repository_url>
cd Phoenix_project


b. 环境设置

Python 版本: 推荐 Python 3.10 或更高版本 (如 Dockerfile 所示)。

安装依赖:

pip install -r requirements.txt


c. 配置环境变量

系统需要 API 密钥用于数据提供商和 AI 模型。

复制示例环境文件：

cp env.example .env


打开 .env 文件并填入您的实际 API 密钥 (例如 GEMINI_PRO_KEY, PINECONE_API_KEY, POSTGRES_PASSWORD 等)。变量名称必须与 config/system.yaml 和 docker-compose.yml 中引用的名称匹配。

核心操作 (Core Operations)

a. 配置运行

所有操作都由中央配置文件 config/system.yaml 控制。在运行之前，请检查此文件以配置：

system.environment: "development" 或 "production"

pipeline: (例如 fact_check_threshold)

ai: (例如 gemini-2.5-pro 模型)

temporal_db / tabular_db: 数据库连接（主机名应匹配 docker-compose.yml 服务名，如 elasticsearch, postgres_db）。

b. 运行回测 (Backtest Mode)

主入口点 phoenix_project.py 的 if __name__ == "__main__": 块被配置为运行回测。

python phoenix_project.py


预期输出:
控制台将显示来自 monitor/logging.py (loguru) 的结构化日志。您将看到系统初始化、LoopManager 运行回测，以及 Orchestrator 处理每个时间步的消息。

c. 运行实时模式 (Live Mode / Worker)

实时模式依赖于基础设施（Redis, Kafka, PG, ES, Neo4j）和 Celery worker。

启动基础设施 (推荐):

docker-compose up -d


启动 Celery Worker:
(确保您的 .env 文件已加载或环境变量已设置)

celery -A worker.celery_app worker --loglevel=info


启动 API 服务 (用于手动控制):
(在单独的终端中)

gunicorn -w 4 -b 0.0.0.0:8000 interfaces.api_server:app


注意: docker-compose.yml 也定义了 api 和 worker 服务，make run (或 docker-compose up -d --build) 是启动所有服务的首选方式。

d. 运行数据验证

在导入大型数据集之前，使用 scripts/validate_dataset.py 检查模式。

[重要] runbook (V2.0) 中的 market_event 类型已过时。请使用 news_data 或 market_data。

# 验证新闻/事件 JSONL 文件
python scripts/validate_dataset.py /path/to/your/news.jsonl --type news_data

# 验证市场数据 CSV 文件 (假设)
python scripts/validate_dataset.py /path/to/your/prices.csv --type market_data


e. 运行 CLI

使用 scripts/run_cli.py 手动与系统交互。

# 示例：手动注入一个新闻事件
python scripts/run_cli.py inject -s "Manual CLI" "突发新闻：美联储意外宣布降息。"

# 示例：手动触发一个计划任务 (如果 Orchestrator 已配置)
python scripts/run_cli.py trigger "daily_market_analysis"


验证与监控

a. 检查 API 端点

当 api 服务 (或 interfaces/api_server.py) 运行时 (默认在 http://localhost:8000)，您可以使用以下端点：

Health Check: GET /api/v1/health

检查 API 服务是否正在运行。

System Status: GET /api/v1/status

查询 ContextBus (或同等组件) 以获取系统当前状态的摘要。

Audit Logs: GET /api/v1/audit?limit=10

从 AuditViewer 检索最新的审计日志。

b. 审查 HTML 报告

(用于回测运行) 检查 output/renderer.py 和 training/engine.py (或 phoenix_project.py) 中配置的报告输出路径。默认情况下，这可能是 logs/backtest_report.html (取决于 BacktestingEngine 的实现)。

检查权益曲线图。

审查关键指标（夏普比率、最大回撤）。

c. 监控 (Prometheus)

[注意] runbook (V2.0) 中提到的 Prometheus 指标 (如 phoenix_cache_hits_total) 尚未完全实现。monitor/metrics.py 中的 PrometheusMetrics 类目前是一个占位符 (Stub)，它只将指标打印到控制台，而没有启动 HTTP /metrics 端点。

故障排除 (Troubleshooting)

1. 警报: Pydantic 验证错误

诊断: config/system.yaml 中的值无效或缺少必填字段，或者 scripts/validate_dataset.py 中的数据格式不正确。

响应:

仔细阅读控制台中的 Pydantic 错误消息。它会精确指出哪个字段不正确。

纠正 config/system.yaml 或数据文件，然后重新运行。

2. 警报: AI 分析失败 (例如 Gemini 错误)

诊断: GEMINI_API_KEY (在 .env 中定义) 可能无效，或者网络连接有问题，或者 ai/prompt_manager.py 无法加载提示 (例如 prompts/analyst.json)。

响应:

验证 .env 文件中的 GEMINI_API_KEY。

检查审计日志: RAG_ARCHITECTURE.md (V2.0) 提到的 ai_audit_logs/ 路径不正确。根据代码：

检查 memory/cot_database.py 中定义的 ./audit_traces 目录 (用于 CoT 数据库)。

检查 audit/logger.py 中定义的 logs/audit_trail.jsonl 文件 (用于 AuditLogger)。

3. 警报: 数据库连接失败 (PG, ES, Neo4j, Redis)

诊断: Docker 容器可能未运行，或者 .env / config/system.yaml 中的连接设置 (主机、密码) 不正确。

响应:

运行 docker-compose ps 确保所有服务 ( phoenix_redis, phoenix_postgres, phoenix_elastic, phoenix_neo4j, phoenix_kafka) 都在运行 (Running)。

检查 config/system.yaml 中的 temporal_db 和 tabular_db 的 host 是否与 docker-compose.yml 中的 container_name 匹配。

确保 .env 中的 POSTGRES_PASSWORD 与 docker-compose.yml 中的 POSTGRES_PASSWORD 匹配。
