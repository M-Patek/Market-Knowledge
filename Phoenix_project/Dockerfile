# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: 移除了多余的 pip install 命令。
# 'prometheus-client', 'pytest', 'celery', 'pydantic' 已经包含在 requirements.txt 中。

# Expose any necessary ports (e.g., for the API)
EXPOSE 5000

# Copy the rest of the application code
COPY . .

# Default command (can be overridden in docker-compose)
# FIXED: 路径已修正，以匹配 /app 工作目录
CMD ["python", "phoenix_project.py"]

