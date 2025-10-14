# Stage 1: Builder stage to install dependencies
FROM python:3.11 as builder

WORKDIR /usr/src/app

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final, slim image
FROM python:3.11-slim

WORKDIR /usr/src/app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY . .

# Command to run the application
CMD ["python", "phoenix_project.py"]
