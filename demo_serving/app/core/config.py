"""
Configuration management for the MLflow serving application.
"""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = {
        "protected_namespaces": (), 
        "extra": "allow",  # Allow extra fields from environment
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="http://127.0.0.1:5050", description="MLflow tracking server URI"
    )
    mlflow_model_uri: str = Field(
        default="models:/langchain-e2e-model/1", description="MLflow model URI"
    )
    mlflow_artifact_path: str = Field(
        default="s3://aiongenbucket/production_data/mlartifacts/1",
        description="S3 path to MLflow artifacts",
    )

    # S3 Configuration
    s3_endpoint_url: str = Field(
        default="http://localhost:9878", description="S3 endpoint URL"
    )
    s3_access_key: str = Field(default="hadoop", description="S3 access key")
    s3_secret_key: str = Field(default="hadoop", description="S3 secret key")
    s3_bucket_name: str = Field(default="aiongenbucket", description="S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="S3 region")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    log_level: str = Field(default="info", description="Logging level")
    reload: bool = Field(default=False, description="Enable auto-reload")

    # Model Configuration
    model_cache_dir: str = Field(
        default="/tmp/mlflow_models", description="Local directory for caching models"
    )
    max_model_cache_size: int = Field(
        default=1000000000, description="Maximum size of model cache in bytes"  # 1GB
    )
    model_timeout: int = Field(
        default=300, description="Model loading timeout in seconds"
    )

    # Health Check Configuration
    health_check_timeout: int = Field(
        default=30, description="Health check timeout in seconds"
    )
    readiness_check_timeout: int = Field(
        default=60, description="Readiness check timeout in seconds"
    )

    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8001, description="Prometheus metrics port")

    # Additional Configuration (from .env) - Commented out fields causing Pydantic validation errors
    # preload_model: bool = Field(default=False, description="Preload model on startup")
    # enable_cors: bool = Field(default=True, description="Enable CORS middleware")
    # debug: bool = Field(default=False, description="Enable debug mode")
    # request_timeout: int = Field(default=30, description="Request timeout in seconds")
    # max_batch_size: int = Field(default=10, description="Maximum batch size for predictions")
    # prediction_timeout: int = Field(default=10, description="Prediction timeout in seconds")
    # health_check_interval: int = Field(default=30, description="Health check interval in seconds")

    # Security
    allowed_hosts: List[str] = Field(
        default=["*"], description="Allowed hosts for CORS"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
