#!/usr/bin/env python3
"""
FastAPI Model Serving Server with Swagger Documentation
=====================================================
Custom FastAPI server that wraps MLflow models and provides:
- Interactive Swagger UI documentation
- Multiple API endpoints for different input formats
- Proper OpenAPI schema generation
- Health checks and monitoring endpoints
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import asyncio
import uvicorn
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, ConfigDict
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import mlflow
import mlflow.pyfunc
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model_instance = None
model_info = {}

# Pydantic models for API requests/responses
class QuestionRequest(BaseModel):
    """Single question request"""
    question: str = Field(..., description="The question to ask the AI model", example="What is machine learning?")
    max_new_tokens: Optional[int] = Field(100, description="Maximum number of new tokens to generate")
    temperature: Optional[float] = Field(0.8, description="Temperature for response generation (0.0-2.0)")

class BatchQuestionRequest(BaseModel):
    """Batch questions request"""
    questions: List[str] = Field(..., description="List of questions to ask the AI model", 
                                 example=["What is AI?", "How does machine learning work?"])
    max_new_tokens: Optional[int] = Field(100, description="Maximum number of new tokens to generate")
    temperature: Optional[float] = Field(0.8, description="Temperature for response generation (0.0-2.0)")

class QuestionResponse(BaseModel):
    """Single question response"""
    question: str = Field(..., description="The original question")
    response: str = Field(..., description="The AI model's response")
    response_time: float = Field(..., description="Response time in seconds")
    response_length: int = Field(..., description="Length of the response in characters")

class BatchQuestionResponse(BaseModel):
    """Batch questions response"""
    responses: List[QuestionResponse] = Field(..., description="List of question-response pairs")
    total_questions: int = Field(..., description="Total number of questions processed")
    total_time: float = Field(..., description="Total processing time in seconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status", example="healthy")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: str = Field(..., description="Name of the loaded model")
    model_version: Optional[str] = Field(None, description="Version of the loaded model")

class MLflowCompatibleRequest(BaseModel):
    """MLflow compatible request format"""
    model_config = ConfigDict(extra='allow')
    
    dataframe_split: Optional[Dict[str, Any]] = Field(None, description="MLflow dataframe split format")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Direct inputs dictionary")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FastAPI MLflow Model Server...")
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI MLflow Model Server...")

# Create FastAPI app
app = FastAPI(
    title="MLflow GenAI Model API",
    description="""
    Interactive API for MLflow deployed GenAI models with comprehensive documentation.
    
    ## Features
    - Single question answering
    - Batch question processing
    - MLflow compatible endpoints
    - Health monitoring
    - Interactive Swagger UI
    
    ## Model Information
    This API serves a HuggingFace transformer model deployed via MLflow for conversational AI tasks.
    """,
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc documentation
    openapi_url="/openapi.json",
    lifespan=lifespan
)

async def load_model():
    """Load the MLflow model"""
    global model_instance, model_info
    
    try:
        # Get model URI from environment variable
        model_uri = os.getenv("MLFLOW_MODEL_URI")
        if not model_uri:
            raise ValueError("MLFLOW_MODEL_URI environment variable not set")
        
        logger.info(f"Loading model from: {model_uri}")
        model_instance = mlflow.pyfunc.load_model(model_uri)
        
        # Extract model info
        model_info = {
            "name": os.getenv("MLFLOW_MODEL_NAME", "unknown"),
            "version": os.getenv("MLFLOW_MODEL_VERSION", "unknown"),
            "uri": model_uri
        }
        
        logger.info(f"Model loaded successfully: {model_info['name']} v{model_info['version']}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_info = {"error": str(e)}
        raise

def get_model():
    """Dependency to get model instance"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_instance else "unhealthy",
        model_loaded=model_instance is not None,
        model_name=model_info.get("name", "unknown"),
        model_version=model_info.get("version", "unknown")
    )

@app.post("/ask", response_model=QuestionResponse, tags=["Question Answering"])
async def ask_question(
    request: QuestionRequest,
    model = Depends(get_model)
):
    """
    Ask a single question to the AI model
    
    This endpoint accepts a single question and returns the AI model's response
    with timing and metadata information.
    """
    import time
    
    try:
        start_time = time.time()
        
        # Prepare input for MLflow model
        input_df = pd.DataFrame({"question": [request.question]})
        
        # Get prediction
        response = model.predict(input_df)
        
        # Handle response format
        if isinstance(response, list) and len(response) > 0:
            response_text = str(response[0])
        else:
            response_text = str(response)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return QuestionResponse(
            question=request.question,
            response=response_text,
            response_time=response_time,
            response_length=len(response_text)
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask/batch", response_model=BatchQuestionResponse, tags=["Question Answering"])
async def ask_batch_questions(
    request: BatchQuestionRequest,
    model = Depends(get_model)
):
    """
    Ask multiple questions to the AI model in batch
    
    This endpoint accepts a list of questions and returns all responses
    with timing and metadata information.
    """
    import time
    
    try:
        start_time = time.time()
        responses = []
        
        for question in request.questions:
            question_start = time.time()
            
            # Prepare input for MLflow model
            input_df = pd.DataFrame({"question": [question]})
            
            # Get prediction
            response = model.predict(input_df)
            
            # Handle response format
            if isinstance(response, list) and len(response) > 0:
                response_text = str(response[0])
            else:
                response_text = str(response)
            
            question_end = time.time()
            question_time = question_end - question_start
            
            responses.append(QuestionResponse(
                question=question,
                response=response_text,
                response_time=question_time,
                response_length=len(response_text)
            ))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return BatchQuestionResponse(
            responses=responses,
            total_questions=len(request.questions),
            total_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Error processing batch questions: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch questions: {str(e)}")

@app.post("/invocations", tags=["MLflow Compatible"])
async def mlflow_invocations(
    request: MLflowCompatibleRequest,
    model = Depends(get_model)
):
    """
    MLflow compatible invocations endpoint
    
    This endpoint maintains compatibility with MLflow's standard serving API.
    Accepts both dataframe_split and direct inputs formats.
    """
    import time
    
    try:
        start_time = time.time()
        
        # Handle different input formats
        if request.dataframe_split:
            # MLflow dataframe split format
            columns = request.dataframe_split.get("columns", [])
            data = request.dataframe_split.get("data", [])
            
            if "question" in columns:
                input_df = pd.DataFrame(data, columns=columns)
            else:
                raise ValueError("'question' column not found in dataframe_split")
                
        elif request.inputs:
            # Direct inputs format
            if isinstance(request.inputs, dict) and "question" in request.inputs:
                question = request.inputs["question"]
                if isinstance(question, list):
                    input_df = pd.DataFrame({"question": question})
                else:
                    input_df = pd.DataFrame({"question": [question]})
            else:
                raise ValueError("'question' field not found in inputs")
        else:
            raise ValueError("Either 'dataframe_split' or 'inputs' must be provided")
        
        # Get prediction
        response = model.predict(input_df)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Return in MLflow format
        return {
            "predictions": response if isinstance(response, list) else [response],
            "metadata": {
                "response_time": response_time,
                "model_name": model_info.get("name", "unknown"),
                "model_version": model_info.get("version", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Error in MLflow invocations: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/model/info", tags=["Model Information"])
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_info": model_info,
        "status": "loaded" if model_instance else "not_loaded",
        "endpoints": {
            "health": "/health",
            "single_question": "/ask",
            "batch_questions": "/ask/batch",
            "mlflow_compatible": "/invocations",
            "model_info": "/model/info",
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }

def main():
    """Main function to start the server"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    
    # Get configuration from environment
    host = os.getenv("MLFLOW_HOST", "0.0.0.0")
    port = int(os.getenv("MLFLOW_PORT", "5001"))
    
    print(f"""
🚀 Starting MLflow FastAPI Model Server
=======================================
Server: http://{host}:{port}
Swagger UI: http://{host}:{port}/docs
ReDoc: http://{host}:{port}/redoc
Health Check: http://{host}:{port}/health

Model URI: {os.getenv("MLFLOW_MODEL_URI", "Not set")}
    """)
    
    # Start server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()