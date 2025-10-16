# Stage 1: Builder stage to compile dependencies on an Alpine base
FROM python:3.11-alpine as builder

WORKDIR /usr/src/app

# [V2.0] Install build-base which contains compilers (gcc, etc.) needed for some Python packages
# that lack pre-compiled wheels for Alpine's musl libc.
RUN apk add --no-cache build-base

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final, deeply optimized production image
FROM python:3.11-alpine

WORKDIR /usr/src/app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY . .

# Command to run the application
CMD ["python", "phoenix_project.py"]

