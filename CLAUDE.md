# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLflow 3 End-to-End GenAI Lifecycle Management system demonstrating comprehensive MLflow integration with HuggingFace models, LangChain, and both local/S3 storage backends. The system implements a complete production-ready pipeline from experimentation to deployment and serving.

## Key Architecture

**Unified E2E Architecture:**
- **Main Orchestrator** (`e2e_mlflow.py`) - Central controller with multiple execution modes (experiment, deploy, serve, all)
- **Configuration Management** (`E2EConfig` dataclass) - Centralized config with S3/local storage flexibility
- **Model Wrapper** (`HuggingFaceModelWrapper`) - Custom PyFunc implementation for MLflow compatibility
- **Production Serving** (`fastapi_server.py`) - FastAPI server with Swagger UI and comprehensive APIs
- **Export Pipeline** (`demo_serving/`) - Standalone Docker deployment system for remote machines

**Core Integration Pattern:**
- HuggingFace GPT2/DialoGPT + LangChain + MLflow 3 native features
- Hybrid storage architecture: SQLite backend + S3/local artifact storage
- Complete observability through `@mlflow.trace` decorators and nested runs
- Multi-mode execution: development, testing, deployment, and serving

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start MLflow server (required for all operations)
mlflow server --host 127.0.0.1 --port 5050
```

### Core E2E Workflows

**Local Storage Mode:**
```bash
# Complete pipeline (experiment + deploy + test)
python e2e_mlflow.py --mode all --data-dir "mlflow_data"

# Individual stages
python e2e_mlflow.py --mode experiment   # Run experiment only
python e2e_mlflow.py --mode deploy      # Deploy model only
python e2e_mlflow.py --mode serve       # Get serving command

# FastAPI server with Swagger UI (recommended)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_MODEL_URI="models:/langchain-e2e-model/[VERSION]"
python fastapi_server.py  # Access UI: http://localhost:5001/docs
```

**S3 Storage Mode (with MinIO):**
```bash
# Start MinIO container
docker run -d -p 9878:9000 -e "MINIO_ROOT_USER=hadoop" -e "MINIO_ROOT_PASSWORD=hadoop" --name minio-server minio/minio server /data

# Configure S3 environment
export AWS_ACCESS_KEY_ID=hadoop
export AWS_SECRET_ACCESS_KEY=hadoop
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9878

# Start MLflow server with S3 artifacts
mlflow server \
  --host 127.0.0.1 \
  --port 5050 \
  --backend-store-uri "sqlite:///$(pwd)/local_mlflow_backend/mlflow.db" \
  --default-artifact-root "s3://aiongenbucket/production_data/mlartifacts"

# Run complete S3 workflow
python e2e_mlflow.py \
  --mode all \
  --data-dir "s3://aiongenbucket/production_data" \
  --s3-endpoint-url "http://localhost:9878"
```

### API Testing
```bash
# Interactive Swagger UI (recommended)
# Open: http://localhost:5001/docs

# FastAPI endpoints
curl -X POST http://localhost:5001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "max_new_tokens": 150}'

curl -X POST http://localhost:5001/ask/batch \
  -H "Content-Type: application/json" \
  -d '{"questions": ["What is AI?", "How does ML work?"]}'

# Health and model info
curl http://localhost:5001/health
curl http://localhost:5001/model/info
```

### Model Export for Remote Deployment
```bash
cd demo_serving

# Export model to standalone Docker package
python scripts/export_model_docker.py \
  --model-uri models:/langchain-e2e-model/1 \
  --s3-endpoint-url http://localhost:9878 \
  --s3-access-key hadoop \
  --s3-secret-key hadoop \
  --output-dir ./exported_model

# Deploy on remote machine (Docker only required)
cd exported_model
./build.sh && ./run.sh  # Access: http://localhost:8000/docs
```

## Configuration Management

**E2EConfig Dataclass** (`e2e_mlflow.py:95-120`) - Centralized configuration with:
- **Storage flexibility**: Local filesystem or S3/MinIO backends
- **Model settings**: Default GPT2, configurable HuggingFace models
- **MLflow integration**: Tracking URIs, experiment names, model registry
- **S3 configuration**: Endpoint URLs, credentials, bucket management

**Key Configuration Constants:**
- `MLFLOW_TRACKING_URI`: `http://127.0.0.1:5050`
- `BASE_MODEL_NAME`: `gpt2` (CPU-optimized)
- `REGISTERED_MODEL_NAME`: `langchain-e2e-model`
- FastAPI serving port: `5001`, export serving port: `8000`

## E2E Execution Modes

**Mode-Based Architecture** (`e2e_mlflow.py` main orchestrator):
1. **`all` mode**: Complete pipeline (experiment → deploy → test → serve)
2. **`experiment` mode**: Model loading, prompt registration, LangChain creation, evaluation
3. **`deploy` mode**: Model registration in MLflow Registry with PyFunc wrapper
4. **`test` mode**: Deployed model validation and testing
5. **`serve` mode**: Generate serving commands for production deployment
6. **`server-config` mode**: Display server configuration instructions

## Storage Architecture

**Hybrid Storage Design** (MLflow requirement):
- **Backend Store**: Always local SQLite (`local_mlflow_backend/mlflow.db`) - contains experiment metadata
- **Artifact Store**: Configurable local or S3 - contains model files and artifacts
- **Staging Directory**: `local_staging/` for temporary processing
- **Server Logs**: `local_mlflow_logs/` for debugging

**S3 Integration Pattern**:
```python
# S3 artifacts while maintaining local SQLite backend
--backend-store-uri "sqlite:///$(pwd)/local_mlflow_backend/mlflow.db"
--default-artifact-root "s3://bucket/path/mlartifacts"
```

## HuggingFaceModelWrapper Implementation

**Custom PyFunc Integration** (`e2e_mlflow.py:200+`):
- **`load_context()`**: Initialize HuggingFace model/tokenizer from MLflow artifacts
- **`predict()`**: Multi-format input handling (DataFrame, dict, string) with LangChain integration
- **MLflow tracing**: Automatic span tracking for all predictions
- **Error handling**: Graceful fallbacks and informative error messages

**Key Implementation Features:**
- CPU-optimized device mapping for accessibility
- Tokenizer padding with EOS token fallback
- LangChain pipeline creation for prompt templating
- Comprehensive input validation and format conversion

## FastAPI Production Server

**Enhanced API Server** (`fastapi_server.py`) with:
- **Interactive Swagger UI**: Auto-generated documentation at `/docs`
- **Multiple Endpoints**: `/ask`, `/ask/batch`, `/invocations`, `/health`, `/model/info`
- **Pydantic Models**: Type validation with `QuestionRequest`, `BatchQuestionRequest`
- **Production Features**: Async context managers, proper error handling, CORS support

**API Endpoints:**
- **`POST /ask`**: Single question with response metadata
- **`POST /ask/batch`**: Multiple questions with batch processing
- **`POST /invocations`**: MLflow-compatible endpoint for standard clients
- **`GET /health`**: Service health and model status
- **`GET /model/info`**: Comprehensive model metadata

## MLflow 3 Integration Patterns

**Advanced Tracing Architecture** (`@mlflow.trace` decorators):
- **Nested Runs**: Hierarchical tracking with parent/child relationships
- **Automatic Spans**: Function-level tracing for all model operations
- **Comprehensive Logging**: Parameters, metrics, artifacts, and tags
- **GenAI Scorers**: Correctness and RelevanceToQuery when available

**Prompt Management System**:
- **Native MLflow GenAI**: Prompt registry with versioning support
- **Artifact Fallback**: File-based storage when GenAI features unavailable
- **Multi-Template Support**: Various prompt styles with A/B testing capabilities

## Export and Remote Deployment

**Standalone Docker Export** (`demo_serving/` directory):
- **Model Packaging**: Complete artifact extraction from S3/local storage
- **Docker Containerization**: Fully self-contained deployment packages
- **Production FastAPI**: Independent server with monitoring and health checks
- **Zero Dependencies**: No MLflow server or S3 access required on target machine

**Export Workflow**:
1. **Artifact Download**: Extract model from MLflow Registry
2. **Package Creation**: Build Docker image with all dependencies
3. **Deployment Scripts**: Automated build, run, and test scripts
4. **Health Monitoring**: Built-in health checks and monitoring endpoints

## Key File References

**Primary Components:**
- `e2e_mlflow.py:95-120` - E2EConfig dataclass
- `e2e_mlflow.py:200+` - HuggingFaceModelWrapper implementation
- `e2e_mlflow.py:400+` - MLflow3E2ELifecycle orchestrator
- `fastapi_server.py:40-50` - Pydantic request models
- `demo_serving/scripts/export_model_docker.py` - Export functionality

**Architecture Documentation:**
- `docs/explaination_e2e.md` - Complete code flow diagram
- `demo_serving/EXPORT_GUIDE.md` - Remote deployment guide
- `README.md` - Comprehensive setup and usage instructions