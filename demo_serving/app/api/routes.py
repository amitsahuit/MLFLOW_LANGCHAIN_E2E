"""
FastAPI routes for the MLflow serving application.
"""
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from .models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ReadinessResponse, ModelInfoResponse, MetricsResponse,
    ReloadRequest, ReloadResponse, ErrorResponse
)
from ..core import get_logger, settings
from ..core.model_loader import model_loader
from ..services.prediction import prediction_service
from ..services.s3_client import S3Client

router = APIRouter()
logger = get_logger("api_routes")

# Application startup time for uptime calculation
start_time = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - start_time


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make a single prediction."""
    try:
        logger.info(f"Received prediction request for question: {request.question[:50]}...")
        
        result = prediction_service.predict_single(request.question)
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result['error']}"
            )
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Make batch predictions."""
    try:
        logger.info(f"Received batch prediction request for {len(request.questions)} questions")
        
        start_time = time.time()
        results = prediction_service.predict_batch(request.questions)
        total_time = time.time() - start_time
        
        return BatchPredictionResponse(
            results=results,
            total_time=total_time,
            total_questions=len(request.questions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        model_loaded=model_loader.is_model_loaded(),
        uptime=get_uptime()
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness check endpoint."""
    try:
        # Check model status
        model_loaded = model_loader.is_model_loaded()
        
        # Test S3 connection
        s3_connection = True
        try:
            s3_client = S3Client()
            # Simple connection test - just list objects with minimal prefix
            s3_client.list_objects("")
        except Exception as e:
            logger.warning(f"S3 connection test failed: {e}")
            s3_connection = False
        
        # Get cache info
        cache_info = model_loader.get_cache_info()
        
        # Determine readiness
        ready = model_loaded or s3_connection  # Ready if model is loaded OR can access S3
        
        model_status = "loaded" if model_loaded else "not_loaded"
        
        return ReadinessResponse(
            ready=ready,
            timestamp=time.time(),
            model_status=model_status,
            s3_connection=s3_connection,
            cache_info=cache_info
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return ReadinessResponse(
            ready=False,
            timestamp=time.time(),
            model_status="error",
            s3_connection=False,
            cache_info={"error": str(e)}
        )


@router.get("/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Get model information."""
    try:
        model_metadata = model_loader.get_model_metadata()
        cache_info = model_loader.get_cache_info()
        prediction_stats = prediction_service.get_prediction_stats()
        
        loaded_at = None
        if model_metadata and "loaded_at" in model_metadata:
            loaded_at = model_metadata["loaded_at"]
        
        return ModelInfoResponse(
            model_uri=settings.mlflow_model_uri,
            model_metadata=model_metadata,
            cache_info=cache_info,
            loaded_at=loaded_at,
            prediction_stats=prediction_stats
        )
        
    except Exception as e:
        logger.error(f"Model info endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get application metrics."""
    try:
        prediction_stats = prediction_service.get_prediction_stats()
        cache_info = model_loader.get_cache_info()
        
        return MetricsResponse(
            predictions_total=prediction_stats.get("total_predictions", 0),
            prediction_time_total=prediction_stats.get("total_prediction_time", 0.0),
            prediction_time_avg=prediction_stats.get("average_prediction_time", 0.0),
            model_loaded=model_loader.is_model_loaded(),
            cache_size=cache_info.get("total_size", 0),
            uptime=get_uptime()
        )
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.post("/reload", response_model=ReloadResponse)
async def reload_model(
    request: ReloadRequest,
    background_tasks: BackgroundTasks
) -> ReloadResponse:
    """Reload the model."""
    try:
        logger.info(f"Model reload requested (force_refresh={request.force_refresh})")
        
        # Reload model
        model = model_loader.reload_model() if request.force_refresh else model_loader.load_model()
        
        # Get updated metadata
        model_metadata = model_loader.get_model_metadata()
        
        # Schedule cache cleanup in background
        if request.force_refresh:
            background_tasks.add_task(model_loader.cleanup_cache, keep_current=True)
        
        return ReloadResponse(
            success=True,
            message="Model reloaded successfully",
            timestamp=time.time(),
            model_metadata=model_metadata
        )
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        return ReloadResponse(
            success=False,
            message=f"Model reload failed: {str(e)}",
            timestamp=time.time(),
            model_metadata=None
        )


@router.delete("/cache")
async def clear_cache(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Clear model cache."""
    try:
        logger.info("Cache clear requested")
        
        # Get cache info before clearing
        cache_info_before = model_loader.get_cache_info()
        
        # Clear cache in background
        background_tasks.add_task(model_loader.cleanup_cache, keep_current=False)
        
        return {
            "success": True,
            "message": "Cache clearing initiated",
            "cache_info_before": cache_info_before,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/stats/reset")
async def reset_stats() -> Dict[str, Any]:
    """Reset prediction statistics."""
    try:
        logger.info("Stats reset requested")
        prediction_service.reset_stats()
        
        return {
            "success": True,
            "message": "Statistics reset successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Stats reset failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset stats: {str(e)}"
        )


# MLflow compatible endpoint for serving
@router.post("/invocations")
async def mlflow_invocations(data: Dict[str, Any]) -> Dict[str, Any]:
    """MLflow compatible prediction endpoint."""
    try:
        logger.info("Received MLflow invocations request")
        
        # Handle different input formats
        if "dataframe_split" in data:
            # MLflow DataFrame format
            df_data = data["dataframe_split"]
            if "data" in df_data and len(df_data["data"]) > 0:
                questions = [row[0] for row in df_data["data"]]
            else:
                raise ValueError("Invalid dataframe_split format")
        elif "instances" in data:
            # TensorFlow Serving format
            questions = [instance.get("question", "") for instance in data["instances"]]
        elif "inputs" in data:
            # Generic inputs format
            inputs = data["inputs"]
            if isinstance(inputs, list):
                questions = inputs
            elif isinstance(inputs, dict) and "question" in inputs:
                questions = [inputs["question"]]
            else:
                raise ValueError("Invalid inputs format")
        else:
            raise ValueError("Unsupported input format")
        
        # Make predictions
        if len(questions) == 1:
            result = prediction_service.predict_single(questions[0])
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return {"predictions": [result["response"]]}
        else:
            results = prediction_service.predict_batch(questions)
            responses = [r.get("response", f"Error: {r.get('error', 'Unknown error')}") 
                        for r in results]
            return {"predictions": responses}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MLflow invocations error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request format: {str(e)}"
        )