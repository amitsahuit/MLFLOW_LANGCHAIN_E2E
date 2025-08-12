# MLflow Model Export for Remote Deployment

This guide shows how to export your MLflow models from S3 storage and create standalone Docker containers that can be deployed on any remote machine with just Docker installed.

## 🎯 Use Case

You have trained and registered a model using MLflow with S3 storage, and now you want to:
1. **Export** the model artifacts from S3 
2. **Package** everything into a standalone Docker container
3. **Deploy** on a remote machine without requiring S3 access, MLflow server, or any external dependencies

## 🚀 Quick Start

### Step 1: Export Your Model

From your MLflow experiment machine (where you have S3 access):

```bash
cd demo_serving

# Export model from S3 to standalone Docker package
python scripts/export_model_docker.py \
  --model-uri models:/langchain-e2e-model/1 \
  --s3-endpoint-url http://localhost:9878 \
  --s3-access-key hadoop \
  --s3-secret-key hadoop \
  --s3-bucket-name aiongenbucket \
  --output-dir ./exported_model
```

**What this does:**
- Downloads model artifacts from S3
- Copies FastAPI application code
- Creates standalone Dockerfile and configuration 
- Generates deployment scripts
- Creates a completely self-contained package

### Step 2: Transfer to Remote Machine

Copy the exported package to your remote machine:

```bash
# Compress the export
tar -czf exported_model.tar.gz exported_model/

# Transfer to remote machine (replace with your method)
scp exported_model.tar.gz user@remote-machine:/path/to/deployment/
```

### Step 3: Deploy on Remote Machine

On the remote machine (only Docker required):

```bash
# Extract package
tar -xzf exported_model.tar.gz
cd exported_model

# Build Docker image
./build.sh

# Start container with Swagger UI
./run.sh

# Test the API
./test.sh
```

### Step 4: Access Swagger UI

Open your browser: **http://localhost:8000/docs**

🎉 You now have a fully functional MLflow model API with interactive Swagger documentation!

## 📋 Complete Example Workflow

### On Experiment Machine (with S3 access):

```bash
# Navigate to demo_serving directory
cd /path/to/mlflow/mlflow_POC/demo_serving

# Export your trained model
python scripts/export_model_docker.py \
  --model-uri models:/your-model-name/1 \
  --s3-endpoint-url http://localhost:9878 \
  --s3-access-key hadoop \
  --s3-secret-key hadoop \
  --s3-bucket-name aiongenbucket \
  --output-dir ./my_exported_model

# Check export results
ls -la my_exported_model/
# Should see: Dockerfile, app/, model/, scripts/, *.sh files, README.md

# Package for transfer
tar -czf my_model_package.tar.gz my_exported_model/
```

### On Remote/Production Machine:

```bash
# Extract and deploy
tar -xzf my_model_package.tar.gz
cd my_exported_model

# Build and run (single command)
./build.sh && ./run.sh

# Wait a moment, then test
sleep 10
./test.sh

# Access Swagger UI
echo "Open: http://localhost:8000/docs"
```

## 🛠️ Advanced Usage

### Using Makefile Commands

```bash
# Show export help
make export-model

# Show example command
make export-example

# Export and test locally (requires credentials)
make export-and-test
```

### Custom Export Options

```bash
# Export to different directory
python scripts/export_model_docker.py \
  --model-uri models:/my-model/2 \
  --s3-endpoint-url https://s3.amazonaws.com \
  --s3-access-key YOUR_AWS_KEY \
  --s3-secret-key YOUR_AWS_SECRET \
  --s3-bucket-name your-production-bucket \
  --output-dir ./production_export

# Export specific model version
python scripts/export_model_docker.py \
  --model-uri models:/chatbot-model/3 \
  [... other options ...]
```

### Different Deployment Scenarios

#### Scenario 1: Different Port
```bash
# On remote machine, use port 8080
docker run -d -p 8080:8000 --name mlflow-serving mlflow-model-serving:latest
# Access: http://localhost:8080/docs
```

#### Scenario 2: Cloud Deployment
```bash
# On cloud VM with public IP
docker run -d -p 80:8000 --name mlflow-serving mlflow-model-serving:latest
# Access: http://your-cloud-ip/docs
```

#### Scenario 3: Multiple Models
```bash
# Export different models to different directories
python scripts/export_model_docker.py --model-uri models:/model-a/1 --output-dir ./model_a_export
python scripts/export_model_docker.py --model-uri models:/model-b/1 --output-dir ./model_b_export

# Deploy on different ports
cd model_a_export && docker build -t model-a . && docker run -d -p 8001:8000 model-a
cd model_b_export && docker build -t model-b . && docker run -d -p 8002:8000 model-b
```

## 📊 API Endpoints Available

Once deployed, your remote container provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | **Interactive Swagger UI** |
| `/api/v1/predict` | POST | Single question prediction |
| `/api/v1/batch-predict` | POST | Batch predictions |
| `/api/v1/invocations` | POST | MLflow compatible endpoint |
| `/api/v1/health` | GET | Health check |
| `/api/v1/info` | GET | Model information |
| `/api/v1/metrics` | GET | Performance metrics |

### Example API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Batch predictions  
curl -X POST http://localhost:8000/api/v1/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"questions": ["What is AI?", "How does ML work?"]}'

# Health check
curl http://localhost:8000/api/v1/health
```

## 🔧 Troubleshooting

### Export Issues

**"Model not found in S3"**
```bash
# Check S3 connectivity
aws s3 ls s3://your-bucket --endpoint-url http://localhost:9878

# Verify model URI format
# Correct: models:/model-name/version
# Check MLflow UI for exact URI
```

**"Permission denied"**
```bash
# Check S3 credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
python scripts/export_model_docker.py [options...]
```

### Deployment Issues

**"Container won't start"**
```bash
docker logs mlflow-serving
# Check for missing files or configuration issues
```

**"Port already in use"**
```bash
# Use different port
docker run -d -p 8080:8000 --name mlflow-serving mlflow-model-serving:latest
```

**"Swagger UI empty"**
- Wait 30-60 seconds for model loading
- Check health: `curl http://localhost:8000/api/v1/health`
- Check logs: `docker logs mlflow-serving`

## 🎯 Benefits

✅ **No External Dependencies** - Runs anywhere with Docker  
✅ **Self-Contained** - Model artifacts included in container  
✅ **Production Ready** - FastAPI with comprehensive monitoring  
✅ **Interactive Testing** - Swagger UI for easy API exploration  
✅ **Multiple Formats** - Supports various input/output formats  
✅ **Lightweight** - Optimized for minimal resource usage  
✅ **Scalable** - Easy to replicate across multiple machines  

## 📁 Package Structure

After export, you get:

```
exported_model/
├── Dockerfile                 # Container build instructions
├── README.md                 # Deployment guide  
├── model/                    # Exported model artifacts
│   ├── MLmodel
│   ├── python_env.yaml
│   └── [model files...]
├── app/                      # FastAPI application
│   ├── main.py
│   ├── api/
│   └── [application code...]
├── scripts/                  # Utility scripts
├── requirements.txt          # Python dependencies
├── .env.standalone          # Standalone configuration
├── build.sh                 # Build Docker image
├── run.sh                   # Start container
├── test.sh                  # Test endpoints
└── model_metadata.json      # Export metadata
```

## 🌟 Production Considerations

- **Security**: Container runs as non-root user
- **Monitoring**: Built-in health checks and metrics
- **Logging**: Structured logging for debugging
- **Performance**: Optimized for CPU-based inference
- **Scaling**: Use Docker Compose or Kubernetes for multiple instances

## 📞 Support

For issues with:
- **Export Process**: Check S3 connectivity and credentials
- **Container Build**: Review Docker logs and requirements
- **API Usage**: Use Swagger UI at `/docs` for interactive testing
- **Performance**: Monitor `/api/v1/metrics` endpoint

---

**🚀 Ready to deploy your MLflow models anywhere with Docker!**