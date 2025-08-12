"""
FastAPI main application for MLflow model serving.
"""
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from .api import router
from .core import configure_logging, get_logger, settings
from .core.model_loader import model_loader
from .core.logging import RequestLoggingMiddleware

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')

# Application startup time
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = get_logger("app_lifespan")
    
    # Startup
    logger.info("🚀 Starting MLflow Model Serving Application")
    logger.info(f"📊 Configuration: {settings.mlflow_model_uri}")
    
    # Optionally pre-load model on startup
    try:
        if hasattr(settings, 'preload_model') and settings.preload_model:
            logger.info("⏳ Pre-loading model on startup...")
            model_loader.load_model()
            logger.info("✅ Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  Model pre-loading failed: {e}")
        logger.info("🔄 Model will be loaded on first request")
    
    logger.info("✅ Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application")
    
    # Cleanup model cache if needed
    try:
        model_loader.unload_model()
        logger.info("🧹 Model unloaded from memory")
    except Exception as e:
        logger.warning(f"⚠️  Model cleanup failed: {e}")
    
    logger.info("✅ Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="MLflow Model Serving API",
    description="FastAPI application for serving MLflow models with S3 integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure logging
configure_logging(settings.log_level)
logger = get_logger("main")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint compatibility - redirect to health
@app.get("/")
async def root():
    """Root endpoint - redirect to health check."""
    return {"message": "MLflow Model Serving API", "docs": "/docs", "health": "/api/v1/health"}


@app.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity testing."""
    return {"status": "pong", "timestamp": time.time()}


@app.get("/version")
async def version():
    """Get application version information."""
    return {
        "version": "1.0.0",
        "mlflow_model_uri": settings.mlflow_model_uri,
        "uptime": time.time() - startup_time,
        "timestamp": time.time()
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return generate_latest()


# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect Prometheus metrics for requests."""
    if not settings.metrics_enabled:
        return await call_next(request)
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        # Record error metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        REQUEST_DURATION.observe(time.time() - start_time)
        raise


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": time.time(),
            "path": request.url.path
        }
    )


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=settings.reload,
        access_log=True
    )