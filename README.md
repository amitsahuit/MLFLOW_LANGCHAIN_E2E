# MLflow 3 End-to-End GenAI Lifecycle Management

## 🚀 S3 vs Local Storage Setup

### Prerequisites
- Docker (for MinIO/S3 emulation)
- Python 3.8+
- Virtual Environment

### Storage Architecture Overview
MLflow uses a **hybrid storage architecture** with specific requirements:

- **Backend Store (Metadata)**: MUST be local SQLite database or database server
  - Contains experiment metadata, run information, model registry data
  - Cannot use S3 URIs (MLflow limitation)
  - Always stored locally even in S3 mode
- **Artifact Store**: Configurable (Local filesystem or S3-compatible storage)
  - Contains model files, experiment artifacts, and large files
  - Can be stored in S3 when using S3 mode
- **Server Logs**: Always stored locally for debugging and monitoring
- **Temporary Files**: Model downloads and processing use system temporary directories

**Important**: When using S3 mode, some directories MUST remain local:
- `local_mlflow_backend/` - SQLite database (required by MLflow)
- `local_mlflow_logs/` - Server logs for debugging
- System temp directories for model processing

### 1. Local Storage Setup

#### Quick Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Start MLflow server with local storage
mlflow server \
  --host 127.0.0.1 \
  --port 5050 \
  --backend-store-uri "file://$PWD/local_mlflow_backend/mlflow.db" \
  --default-artifact-root "file://$PWD/mlflow_data/mlartifacts"

# Run E2E Lifecycle
.venv/bin/python e2e_mlflow.py --mode all --data-dir "mlflow_data"
```

### 2. S3 Storage Setup (MinIO/Ozone)

#### Prerequisites
- Docker
- AWS CLI compatible credentials

#### Step 1: Start MinIO Container
```bash
# Start MinIO S3-compatible storage
docker run -d \
  -p 9878:9000 \
  -p 9000:9000 \
  -e "MINIO_ROOT_USER=hadoop" \
  -e "MINIO_ROOT_PASSWORD=hadoop" \
  --name minio-server \
  minio/minio server /data
```

#### Step 2: Configure S3 Environment
```bash
# Set AWS environment variables
export AWS_ACCESS_KEY_ID=hadoop
export AWS_SECRET_ACCESS_KEY=hadoop
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9878

# Create bucket (if not exists)
aws --endpoint-url=http://localhost:9878 s3 mb s3://aiongenbucket
```

#### Step 3: Run MLflow with S3 Artifacts
```bash

# configure the server (IMPORTANT: Use full S3 path)
.venv/bin/python e2e_mlflow.py \
  --mode server-config \
  --data-dir "s3://aiongenbucket/production_data" \
  --s3-endpoint-url "http://localhost:9878" \
  --s3-bucket-name "aiongenbucket" \
  --s3-access-key "hadoop" \
  --s3-secret-key "hadoop"

# Start MLflow server with S3 artifact storage
mlflow server \
    --host 127.0.0.1 \
    --port 5050 \
    --backend-store-uri "sqlite:////Users/amitsahu/Desktop/aion_gen/mlflow/mlflow_POC/local_mlflow_backend/mlflow.db" \
    --default-artifact-root "s3://aiongenbucket/production_data/mlartifacts" \
    > "local_mlflow_logs/mlflow_server.log" 2>&1 &

# Run E2E Lifecycle with S3 storage
.venv/bin/python e2e_mlflow.py \
  --mode all \
  --model-name "gpt2" \
  --experiment-name "Langchain-E2E-Experiment" \
  --tracking-uri "http://127.0.0.1:5050" \
  --data-dir "s3://aiongenbucket/production_data" \
  --s3-endpoint-url "http://localhost:9878" \
  --s3-bucket-name "aiongenbucket"

```

### 3. Model Deployment and Serving

#### Step 1: Deploy Model to MLflow Registry
```bash
# Deploy model to registry
.venv/bin/python e2e_mlflow.py \
  --mode deploy \
  --data-dir "s3://aiongenbucket/production_data" \
  --s3-endpoint-url "http://localhost:9878"
```

#### Step 2: Get Serving Command
```bash
# Get the correct serving command with latest model version
.venv/bin/python e2e_mlflow.py --mode serve
```

#### Step 3: Serve the Model

**🚀 Option 1: FastAPI Server with Interactive Swagger UI (Recommended)**

```bash
# Set required environment variables
export AWS_ACCESS_KEY_ID=hadoop
export AWS_SECRET_ACCESS_KEY=hadoop
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9878
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_MODEL_URI="models:/langchain-e2e-model/[VERSION]"
export MLFLOW_MODEL_NAME=langchain-e2e-model
export MLFLOW_MODEL_VERSION=[VERSION]
export MLFLOW_HOST=0.0.0.0
export MLFLOW_PORT=5001

# Install FastAPI dependencies (if not already installed)
pip install fastapi uvicorn

# Start FastAPI server with Swagger UI
.venv/bin/python fastapi_server.py
```

**📊 FastAPI Server Features:**
- **Interactive Swagger UI**: `http://localhost:5001/docs`
- **ReDoc Documentation**: `http://localhost:5001/redoc`
- **Health Monitoring**: `http://localhost:5001/health`
- **Model Information**: `http://localhost:5001/model/info`

**API Endpoints Available:**
- `POST /ask` - Single question answering with response metadata
- `POST /ask/batch` - Batch question processing
- `POST /invocations` - MLflow compatible endpoint
- `GET /health` - Health check and model status
- `GET /model/info` - Comprehensive model information

**Option 2: Standard MLflow Serving (Basic)**
```bash
# Standard MLflow serving without Swagger UI
mlflow models serve -m models:/langchain-e2e-model/[VERSION] -p 5002 --host 0.0.0.0 --no-conda
```

#### Step 4: Test the API

**Using Interactive Swagger UI (Recommended):**
1. Open `http://localhost:5001/docs` in your browser
2. Click on any endpoint (e.g., `POST /ask`)
3. Click "Try it out"
4. Enter your question and click "Execute"
5. View the formatted response with metadata

**Using curl commands:**
```bash
# Test single question endpoint
curl -X POST http://localhost:5001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "max_new_tokens": 150, "temperature": 0.8}'

# Test batch questions endpoint
curl -X POST http://localhost:5001/ask/batch \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["What is AI?", "How does deep learning work?"],
    "max_new_tokens": 100,
    "temperature": 0.7
  }'

# Test MLflow compatible endpoint
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["question"], "data": [["What is AI?"]]}}'

# Health check
curl http://localhost:5001/health

# Model information
curl http://localhost:5001/model/info
```

**Response Format Examples:**

*Single Question Response:*
```json
{
  "question": "What is machine learning?",
  "response": "Machine learning is a subset of artificial intelligence...",
  "response_time": 2.34,
  "response_length": 247
}
```

*Batch Questions Response:*
```json
{
  "responses": [
    {
      "question": "What is AI?",
      "response": "Artificial Intelligence (AI) refers to...",
      "response_time": 1.89,
      "response_length": 198
    }
  ],
  "total_questions": 1,
  "total_time": 1.92
}
```

### 4. Complete Workflow Examples

#### Example 1: Local Storage Workflow
```bash
# 1. Start MLflow server
mlflow server --host 127.0.0.1 --port 5050

# 2. Run complete workflow
.venv/bin/python e2e_mlflow.py --mode all --data-dir "mlflow_data"

# 3. Get serving command
.venv/bin/python e2e_mlflow.py --mode serve

# 4A. Serve with FastAPI + Swagger UI (Recommended)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_MODEL_URI="models:/langchain-e2e-model/[VERSION]"
python fastapi_server.py
# Access Swagger UI: http://localhost:5001/docs

# 4B. Alternative: Standard MLflow serving
mlflow models serve -m models:/langchain-e2e-model/[VERSION] -p 5002 --host 0.0.0.0 --no-conda
```

#### Example 2: S3 Storage Workflow  
```bash
# 1. Start MinIO
docker run -d -p 9878:9000 -e "MINIO_ROOT_USER=hadoop" -e "MINIO_ROOT_PASSWORD=hadoop" --name minio-server minio/minio server /data

# 2. Configure environment
export AWS_ACCESS_KEY_ID=hadoop
export AWS_SECRET_ACCESS_KEY=hadoop  
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9878

# 3. Start MLflow server with S3 artifacts
mlflow server \
  --host 127.0.0.1 \
  --port 5050 \
  --backend-store-uri "sqlite:////Users/amitsahu/Desktop/aion_gen/mlflow/mlflow_POC/local_mlflow_backend/mlflow.db" \
  --default-artifact-root "s3://aiongenbucket/production_data/mlartifacts" \
  > "local_mlflow_logs/mlflow_server.log" 2>&1 &

# 4. Run deployment workflow
.venv/bin/python e2e_mlflow.py \
  --mode deploy \
  --data-dir "s3://aiongenbucket/production_data" \
  --s3-endpoint-url "http://localhost:9878"

# 5. Get serving command with correct version
.venv/bin/python e2e_mlflow.py --mode serve

# 6A. Serve with FastAPI + Swagger UI (Recommended)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_MODEL_URI="models:/langchain-e2e-model/[VERSION]"
export MLFLOW_MODEL_NAME=langchain-e2e-model
export MLFLOW_MODEL_VERSION=[VERSION]
python fastapi_server.py
# Access Swagger UI: http://localhost:5001/docs

# 6B. Alternative: Standard MLflow serving
mlflow models serve -m models:/langchain-e2e-model/[VERSION] -p 5002 --host 0.0.0.0 --no-conda
```

#### Example 3: Complete FastAPI Workflow with Testing
```bash
# 1. Setup and Deploy (following steps above)
# 2. Start FastAPI server
python fastapi_server.py

# 3. Test via Swagger UI (Interactive)
# Open: http://localhost:5001/docs
# Try the /ask endpoint with: {"question": "What is machine learning?"}

# 4. Test via CLI (Programmatic)
# Health check
curl http://localhost:5001/health

# Single question
curl -X POST http://localhost:5001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain neural networks", "max_new_tokens": 200}'

# Batch questions
curl -X POST http://localhost:5001/ask/batch \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is deep learning?", 
      "How do transformers work?",
      "Explain gradient descent"
    ],
    "max_new_tokens": 150
  }'
```

### 5. Troubleshooting

#### Model Deployment Issues

**"Registered Model with name=langchain-e2e-model not found"**
```bash
# Check if model exists in registry
.venv/bin/python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5050')
client = mlflow.MlflowClient()
models = client.search_registered_models()
for model in models:
    print(f'Model: {model.name}')
    versions = client.search_model_versions(f\"name='{model.name}'\")
    for version in versions:
        print(f'  Version {version.version}: {version.source}')
"

# If no models found, run deployment first:
.venv/bin/python e2e_mlflow.py --mode deploy --data-dir "s3://aiongenbucket/production_data" --s3-endpoint-url "http://localhost:9878"
```

**"Test status: Failed" in deployment**
- This usually indicates model loading issues
- Check MLflow server logs: `tail -f local_mlflow_logs/mlflow_server.log`
- Verify S3 connectivity and credentials

**Model serving fails with connection errors**
```bash
# Always get the latest serving command (don't use hardcoded version numbers):
.venv/bin/python e2e_mlflow.py --mode serve

# Make sure to set ALL required environment variables:
export AWS_ACCESS_KEY_ID=hadoop
export AWS_SECRET_ACCESS_KEY=hadoop
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9878
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050

# Then run the serve command from the output above
```

#### FastAPI Server Issues

**FastAPI server fails to start**
```bash
# Check if FastAPI dependencies are installed
pip install fastapi uvicorn

# Verify environment variables are set
echo $MLFLOW_MODEL_URI
echo $MLFLOW_TRACKING_URI

# Check if port 5001 is available
lsof -i :5001

# Start with verbose logging
python fastapi_server.py
```

**Empty Swagger UI or "Model not loaded" errors**
- Ensure MLflow tracking server is running: `curl http://127.0.0.1:5050/health`
- Verify model exists in registry: Check MLflow UI at `http://127.0.0.1:5050`
- Check environment variables are properly set
- Verify model URI format: `models:/model-name/version`

**Permission errors with S3 access**
- Verify AWS credentials: `aws --endpoint-url=http://localhost:9878 s3 ls`
- Check MinIO container is running: `docker ps | grep minio`
- Ensure all S3 environment variables are exported in current session

**Performance issues or slow responses**
- Check model size and complexity
- Monitor resource usage: `htop` or `top`
- Consider using smaller models for testing
- Use batch endpoints for multiple questions

#### S3 Connection Issues
- Verify MinIO/Ozone container is running: `docker ps | grep minio`
- Check AWS credentials are set correctly
- Confirm network accessibility: `curl http://localhost:9878`
- Use `--s3-endpoint-url` to specify correct endpoint

#### Artifact Storage Problems
- Ensure bucket exists: `aws --endpoint-url=http://localhost:9878 s3 ls s3://aiongenbucket`
- Check MLflow server configuration matches artifact store URI
- Verify S3 environment variables are set correctly

#### Local Directories Created in S3 Mode
**Problem**: `mlruns` and other directories appear in working directory despite S3 configuration

**Root Cause**: MLflow's hybrid architecture requires certain components to remain local:
- Backend store (SQLite database) cannot be stored in S3
- Temporary model files during processing
- Server logs and debugging information

**Expected Behavior in S3 Mode**:
```
Project Directory:
├── local_mlflow_backend/     # ✅ MUST remain local (SQLite database)
├── local_mlflow_logs/        # ✅ MUST remain local (server logs)
├── local_staging/            # ✅ Temporary files before S3 upload
└── .env                      # ✅ Configuration file

S3 Bucket (aiongenbucket/production_data/):
├── models/                   # ✅ Model artifacts in S3
├── prompts/                  # ✅ Prompt templates in S3
├── artifacts/                # ✅ Experiment results in S3
└── mlartifacts/             # ✅ MLflow artifacts in S3
```

**Fix for Duplicate Folders**: Ensure consistent S3 path usage:
- Use `--data-dir "s3://aiongenbucket/production_data"` (full path)
- NOT `--data-dir "s3://production_data"` (missing bucket name)

#### Why mlruns Folders Don't Appear in S3
**This is CORRECT behavior** - not a bug!

**MLflow's Architecture Requirement**:
- **Backend Store** (mlruns, experiments, model registry): MUST be local SQLite or database
- **Artifact Store** (model files, artifacts): CAN be S3

**Reason**: MLflow's backend store doesn't support S3 URIs due to performance and consistency requirements for metadata operations.

**What should be in S3**:
```
s3://aiongenbucket/production_data/
├── mlartifacts/          # ✅ Model artifacts and files
├── models/               # ✅ Downloaded model files  
├── prompts/              # ✅ Prompt templates
└── artifacts/            # ✅ Experiment results
```

**What stays local** (and should):
```
Local directories:
├── local_mlflow_backend/ # ✅ SQLite database (mlruns equivalent)
├── local_mlflow_logs/    # ✅ Server logs
└── /tmp/mlflow_temp*/    # ✅ Temporary processing files
```

### Security Considerations
- Use strong, unique credentials in production
- Enable TLS for S3 endpoints
- Implement IAM role-based access

### Performance Optimization
- Use region-local S3 endpoints
- Configure appropriate connection timeouts
- Monitor artifact download times

### 6. Export Model for Remote Deployment (NEW! 🚀)

For deploying your trained model on remote machines without S3 or MLflow server access:

#### Step 1: Export Model to Standalone Package
```bash
cd demo_serving

# Export your model from S3 to a standalone Docker package
python scripts/export_model_docker.py \
  --model-uri models:/langchain-e2e-model/1 \
  --s3-endpoint-url http://localhost:9878 \
  --s3-access-key hadoop \
  --s3-secret-key hadoop \
  --s3-bucket-name aiongenbucket \
  --output-dir ./exported_model
```

#### Step 2: Deploy on Remote Machine
```bash
# Transfer the exported_model/ directory to your remote machine
# Then on the remote machine (only Docker required):

cd exported_model
./build.sh      # Build Docker image
./run.sh        # Start container with Swagger UI  
./test.sh       # Test the API

# Access Swagger UI: http://localhost:8000/docs
```

**🎯 Key Benefits:**
- **Zero Dependencies**: No S3, MLflow server, or external services needed
- **Fully Standalone**: Model artifacts packaged inside container
- **Interactive Swagger UI**: Test and use your model through web interface  
- **Production Ready**: FastAPI with monitoring, health checks, and metrics
- **Deploy Anywhere**: Works on any machine with Docker

See [demo_serving/EXPORT_GUIDE.md](demo_serving/EXPORT_GUIDE.md) for complete documentation.

---

## License
Apache 2.0 - See LICENSE file for details