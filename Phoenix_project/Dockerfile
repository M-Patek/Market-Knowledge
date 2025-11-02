# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: 移除了多余的 pip install 命令。
# 'prometheus-client', 'pytest', 'celery', 'pydantic' 已经包含在 requirements.txt 中。

# FIX (E-DOCKER-2): 将暴露的端口从 5000 更改为 8000，以匹配 docker-compose.yml 中的 Gunicorn 命令
EXPOSE 8000

# Copy the rest of the application code
# FIX (E-DOCKER-1): 修复了 'COPY' 指令的语法错误
COPY . .

# Default command (can be overridden in docker-compose)
# FIXED: 路径已修正，以匹配 /app 工作目录
CMD ["python", "phoenix_project.py"]
