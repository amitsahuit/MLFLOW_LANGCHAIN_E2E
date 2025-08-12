"""
Pydantic models for API request and response validation.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    """Single prediction request model."""
    question: str = Field(
        ...,
        description="Question to ask the model",
        min_length=1,
        max_length=2000,
        example="What is machine learning?"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    questions: List[str] = Field(
        ...,
        description="List of questions to ask the model",
        min_items=1,
        max_items=100,
        example=["What is AI?", "How does ML work?"]
    )


class PredictionResponse(BaseModel):
    """Single prediction response model."""
    question: str = Field(..., description="Original question")
    response: str = Field(..., description="Model response")
    prediction_time: float = Field(..., description="Time taken for prediction in seconds")
    total_time: float = Field(..., description="Total request time in seconds")
    model_uri: str = Field(..., description="MLflow model URI used")
    timestamp: float = Field(..., description="Unix timestamp of prediction")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    results: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    total_time: float = Field(..., description="Total batch processing time")
    total_questions: int = Field(..., description="Number of questions processed")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    question: Optional[str] = Field(None, description="Question that caused the error")
    timestamp: float = Field(..., description="Unix timestamp of error")


class HealthResponse(BaseModel):
    """Health check response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Health status", example="healthy")
    timestamp: float = Field(..., description="Unix timestamp")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime: float = Field(..., description="Application uptime in seconds")


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    ready: bool = Field(..., description="Whether service is ready")
    timestamp: float = Field(..., description="Unix timestamp")
    model_status: str = Field(..., description="Model loading status")
    s3_connection: bool = Field(..., description="S3 connection status")
    cache_info: Dict[str, Any] = Field(..., description="Cache information")


class ModelInfoResponse(BaseModel):
    """Model information response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_uri: str = Field(..., description="MLflow model URI")
    model_metadata: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    cache_info: Dict[str, Any] = Field(..., description="Cache information")
    loaded_at: Optional[float] = Field(None, description="When model was loaded")
    prediction_stats: Dict[str, Any] = Field(..., description="Prediction statistics")


class MetricsResponse(BaseModel):
    """Metrics response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    predictions_total: int = Field(..., description="Total number of predictions")
    prediction_time_total: float = Field(..., description="Total prediction time")
    prediction_time_avg: float = Field(..., description="Average prediction time")
    model_loaded: bool = Field(..., description="Model loaded status")
    cache_size: int = Field(..., description="Cache size in bytes")
    uptime: float = Field(..., description="Application uptime")


class ReloadRequest(BaseModel):
    """Model reload request model."""
    force_refresh: bool = Field(
        default=False,
        description="Force refresh from S3"
    )


class ReloadResponse(BaseModel):
    """Model reload response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool = Field(..., description="Whether reload was successful")
    message: str = Field(..., description="Reload status message")
    timestamp: float = Field(..., description="Unix timestamp")
    model_metadata: Optional[Dict[str, Any]] = Field(None, description="Updated model metadata")