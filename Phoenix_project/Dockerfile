# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FIX (E-DOCKER-2): 将暴露的端口从 5000 更改为 8000，以匹配 docker-compose.yml 中的 Gunicorn 命令
EXPOSE 8000

# (FIX) 替换 'COPY . .'
# 显式复制所有 Python 源代码目录和根 .py 文件
# 这可以防止对 README.md 或 Makefile 的更改破坏构建缓存

# --- [✅ 修复] 阶段 3 修复：保留这个主包复制 ---
# 假设 agents, ai, core 等都在 Phoenix_project 目录内部
COPY Phoenix_project /app/Phoenix_project

# --- [✅ 修复] 阶段 3 修复：删除以下所有重复/错误的 COPY 命令 ---
# (这些目录现在应该都在 /app/Phoenix_project 内部了)
# COPY agents /app/agents
# COPY ai /app/ai
# COPY audit /app/audit
# COPY cognitive /app/cognitive
# COPY controller /app/controller
# COPY core /app/core
# COPY data /app/data
# COPY evaluation /app/evaluation
# COPY events /app/events
# COPY execution /app/execution
# COPY features /app/features
# COPY fusion /app/fusion
# COPY interfaces /app/interfaces
# COPY memory /app/memory
# COPY models /app/models
# COPY monitor /app/monitor
# COPY output /app/output
# COPY prompts /app/prompts
# COPY reasoning /app/reasoning
# COPY sizing /app/sizing
# COPY storage /app/storage # [Fix III.3] 移除此行
# COPY templates /app/templates
# COPY training /app/training
# COPY utils /app/utils

# --- [✅ 修复] 阶段 3 修复：保留非 Python 包的根目录 ---
COPY config /app/config
COPY scripts /app/scripts

# 复制根目录中的所有 Python 文件和 .json/.yaml 配置文件
# 使用 shell 通配符来处理文件不存在的情况 (COPY 会失败)
COPY *.py .
COPY *.json .
COPY *.yaml .


# Default command (can be overridden in docker-compose)
# FIXED: 路径已修正，以匹配 /app 工作目录
CMD ["python", "phoenix_project.py"]
