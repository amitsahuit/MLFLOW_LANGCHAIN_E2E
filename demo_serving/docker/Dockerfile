# Multi-stage Dockerfile for MLflow Model Serving Application
# Stage 1: Base Python environment with dependencies
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development environment (optional)
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose ports
EXPOSE 8000 8001

# Development command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production environment
FROM base as production

# Copy source code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Copy configuration files
COPY .env.example .env

# Create necessary directories
RUN mkdir -p /tmp/mlflow_models /app/logs && \
    chown -R appuser:appuser /app /tmp/mlflow_models

# Switch to app user
USER appuser

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Stage 4: Model download stage (can be used separately)
FROM base as model-downloader

# Copy only necessary files for downloading
COPY app/core/config.py ./app/core/
COPY app/core/logging.py ./app/core/
COPY app/core/__init__.py ./app/core/
COPY app/services/s3_client.py ./app/services/
COPY app/services/__init__.py ./app/services/
COPY scripts/download_artifacts.py ./scripts/

# Copy configuration
COPY .env.example .env

# Switch to app user
USER appuser

# Command to download model artifacts
CMD ["python", "scripts/download_artifacts.py"]