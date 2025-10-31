# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Layer 15 dependencies (Prometheus, pytest, celery, pydantic)
RUN pip install --no-cache-dir prometheus-client pytest celery pydantic

# Expose any necessary ports (e.g., for the API)
EXPOSE 5000

# Copy the rest of the application code
COPY . .

# Default command (can be overridden in docker-compose)
CMD ["python", "Phoenix_project/phoenix_project.py"]
