#!/usr/bin/env python3
"""
MLflow Model Export and Docker Container Builder

This script exports an MLflow model from S3/MinIO storage and creates a 
standalone Docker container that can be deployed on any remote machine
with just the model artifacts and serving infrastructure.
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

# Add parent directory to path to import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import Settings
from app.services.s3_client import S3Client


class ModelExporter:
    """Export MLflow models and create deployment-ready Docker containers."""
    
    def __init__(self, 
                 model_uri: str,
                 s3_endpoint_url: str,
                 s3_access_key: str,
                 s3_secret_key: str,
                 s3_bucket_name: str,
                 output_dir: str = "./exported_model"):
        
        self.model_uri = model_uri
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_bucket_name = s3_bucket_name
        self.output_dir = Path(output_dir)
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name='us-east-1'
        )
        
        print(f"🚀 ModelExporter initialized")
        print(f"📊 Model URI: {model_uri}")
        print(f"🗂️  S3 Endpoint: {s3_endpoint_url}")
        print(f"📦 S3 Bucket: {s3_bucket_name}")
    
    def export_model_artifacts(self) -> Dict[str, Any]:
        """Export model artifacts from S3 to local directory."""
        print(f"\n📁 Exporting model artifacts to: {self.output_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.output_dir / "model"
        model_dir.mkdir(exist_ok=True)
        
        # Parse model URI to get S3 path
        # Example: models:/langchain-e2e-model/1 -> find artifacts in MLflow registry
        if self.model_uri.startswith("models:/"):
            model_name = self.model_uri.split("/")[1]
            model_version = self.model_uri.split("/")[2]
            
            # Try to find artifacts by listing common MLflow paths
            possible_paths = [
                f"production_data/mlartifacts/1/{model_name}/artifacts/model/",
                f"production_data/artifacts/{model_name}/{model_version}/",
                f"mlartifacts/1/{model_name}/artifacts/model/",
                f"artifacts/{model_name}/{model_version}/",
                f"{model_name}/{model_version}/artifacts/model/",
                # Add more patterns as needed
            ]
            
            artifacts_path = None
            for path in possible_paths:
                try:
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.s3_bucket_name,
                        Prefix=path,
                        MaxKeys=1
                    )
                    if 'Contents' in response:
                        artifacts_path = path
                        print(f"✅ Found artifacts at: s3://{self.s3_bucket_name}/{path}")
                        break
                except ClientError:
                    continue
            
            if not artifacts_path:
                # Try to find by searching all mlartifacts
                try:
                    paginator = self.s3_client.get_paginator('list_objects_v2')
                    for page in paginator.paginate(Bucket=self.s3_bucket_name, 
                                                   Prefix="", 
                                                   Delimiter="/"):
                        for obj in page.get('Contents', []):
                            if model_name in obj['Key'] and 'mlartifacts' in obj['Key']:
                                # Extract the directory path
                                key_parts = obj['Key'].split('/')
                                if 'mlartifacts' in key_parts:
                                    artifacts_index = key_parts.index('mlartifacts')
                                    artifacts_path = '/'.join(key_parts[:artifacts_index + 3]) + '/artifacts/model/'
                                    print(f"✅ Found artifacts at: s3://{self.s3_bucket_name}/{artifacts_path}")
                                    break
                        if artifacts_path:
                            break
                except ClientError as e:
                    print(f"❌ Error searching for artifacts: {e}")
            
            if not artifacts_path:
                raise ValueError(f"Could not find model artifacts for {self.model_uri}")
        
        else:
            # Direct S3 path
            artifacts_path = self.model_uri.replace("s3://", "").replace(self.s3_bucket_name + "/", "")
        
        # Download all artifacts
        downloaded_files = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.s3_bucket_name, Prefix=artifacts_path):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    
                    # Skip directories
                    if key.endswith('/'):
                        continue
                    
                    # Calculate local path
                    relative_path = key.replace(artifacts_path, "")
                    local_path = model_dir / relative_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    print(f"⬇️  Downloading: {key}")
                    self.s3_client.download_file(self.s3_bucket_name, key, str(local_path))
                    downloaded_files.append(str(local_path))
            
            print(f"✅ Downloaded {len(downloaded_files)} files")
            
            # Create metadata file
            metadata = {
                "model_uri": self.model_uri,
                "s3_path": f"s3://{self.s3_bucket_name}/{artifacts_path}",
                "export_timestamp": __import__('time').time(),
                "downloaded_files": downloaded_files
            }
            
            metadata_file = self.output_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except ClientError as e:
            print(f"❌ Error downloading artifacts: {e}")
            raise
    
    def create_dockerfile(self) -> None:
        """Create a standalone Dockerfile for the exported model."""
        dockerfile_content = f'''# Standalone MLflow Model Serving Container
# Generated automatically by ModelExporter

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Copy exported model artifacts
COPY model/ ./model/

# Copy configuration
COPY .env.standalone .env

# Create necessary directories and set permissions
RUN mkdir -p /app/logs && \\
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
'''
        
        dockerfile_path = self.output_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"✅ Created Dockerfile: {dockerfile_path}")
    
    def create_standalone_config(self) -> None:
        """Create standalone configuration file."""
        config_content = f'''# Standalone MLflow Model Serving Configuration
# No external dependencies required

# Model Configuration - Uses local model files
MLFLOW_MODEL_URI=file:///app/model
MLFLOW_ARTIFACT_PATH=file:///app/model

# S3 Configuration - Disabled for standalone mode
S3_ENDPOINT_URL=
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_BUCKET_NAME=

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
WORKERS=1

# Cache Configuration
MODEL_CACHE_DIR=/app/model
PRELOAD_MODEL=true

# Metrics and Monitoring
METRICS_ENABLED=true

# Security
CORS_ORIGINS=["*"]

# Development (disabled in standalone)
RELOAD=false
'''
        
        config_path = self.output_dir / ".env.standalone"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created standalone config: {config_path}")
    
    def copy_application_code(self) -> None:
        """Copy necessary application code."""
        src_dir = Path(__file__).parent.parent
        
        # Directories to copy
        dirs_to_copy = ["app", "scripts"]
        
        for dir_name in dirs_to_copy:
            src_path = src_dir / dir_name
            dest_path = self.output_dir / dir_name
            
            if src_path.exists():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
                print(f"✅ Copied {dir_name}/ to {dest_path}")
        
        # Copy requirements.txt
        requirements_src = src_dir / "requirements.txt"
        requirements_dest = self.output_dir / "requirements.txt"
        if requirements_src.exists():
            shutil.copy2(requirements_src, requirements_dest)
            print(f"✅ Copied requirements.txt")
    
    def create_deployment_scripts(self) -> None:
        """Create deployment helper scripts."""
        
        # Build script
        build_script = '''#!/bin/bash
set -e

echo "🐳 Building MLflow Model Serving Docker Image..."

# Build the Docker image
docker build -t mlflow-model-serving:latest .

echo "✅ Docker image built successfully!"
echo "📊 Image info:"
docker images mlflow-model-serving:latest

echo ""
echo "🚀 To run the container:"
echo "docker run -d -p 8000:8000 --name mlflow-serving mlflow-model-serving:latest"
echo ""
echo "🌐 Then access Swagger UI at: http://localhost:8000/docs"
'''
        
        build_path = self.output_dir / "build.sh"
        with open(build_path, 'w') as f:
            f.write(build_script)
        build_path.chmod(0o755)
        
        # Run script
        run_script = '''#!/bin/bash
set -e

echo "🚀 Starting MLflow Model Serving Container..."

# Stop existing container if running
docker stop mlflow-serving 2>/dev/null || true
docker rm mlflow-serving 2>/dev/null || true

# Run the container
docker run -d \\
  --name mlflow-serving \\
  -p 8000:8000 \\
  mlflow-model-serving:latest

echo "✅ Container started successfully!"
echo ""
echo "📋 Container info:"
docker ps | grep mlflow-serving

echo ""
echo "🌐 Access points:"
echo "  • Swagger UI: http://localhost:8000/docs"
echo "  • Health Check: http://localhost:8000/api/v1/health"
echo "  • Model Info: http://localhost:8000/api/v1/info"
echo ""
echo "📊 To view logs:"
echo "docker logs -f mlflow-serving"
echo ""
echo "🛑 To stop:"
echo "docker stop mlflow-serving"
'''
        
        run_path = self.output_dir / "run.sh"
        with open(run_path, 'w') as f:
            f.write(run_script)
        run_path.chmod(0o755)
        
        # Test script
        test_script = '''#!/bin/bash
set -e

echo "🧪 Testing MLflow Model Serving API..."

BASE_URL="http://localhost:8000"

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
for i in {1..30}; do
  if curl -s "$BASE_URL/api/v1/health" > /dev/null; then
    echo "✅ Service is ready!"
    break
  fi
  echo "  Attempt $i/30..."
  sleep 2
done

echo ""
echo "🔍 Testing endpoints:"

# Health check
echo "1. Health Check:"
curl -s "$BASE_URL/api/v1/health" | jq '.'

echo ""
echo "2. Model Info:"
curl -s "$BASE_URL/api/v1/info" | jq '.'

echo ""
echo "3. Single Prediction:"
curl -s -X POST "$BASE_URL/api/v1/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"question": "What is machine learning?"}' | jq '.'

echo ""
echo "4. MLflow Compatible Endpoint:"
curl -s -X POST "$BASE_URL/api/v1/invocations" \\
  -H "Content-Type: application/json" \\
  -d '{"dataframe_split": {"columns": ["question"], "data": [["What is AI?"]]}}' | jq '.'

echo ""
echo "✅ All tests completed!"
echo "🌐 Open Swagger UI: http://localhost:8000/docs"
'''
        
        test_path = self.output_dir / "test.sh"
        with open(test_path, 'w') as f:
            f.write(test_script)
        test_path.chmod(0o755)
        
        print(f"✅ Created deployment scripts: build.sh, run.sh, test.sh")
    
    def create_readme(self) -> None:
        """Create README for the exported model."""
        readme_content = f'''# MLflow Model Serving - Standalone Deployment

This is a standalone Docker deployment package for the MLflow model: `{self.model_uri}`

## 🚀 Quick Start

### Prerequisites
- Docker installed and running
- Port 8000 available

### 1. Build the Docker Image
```bash
chmod +x build.sh
./build.sh
```

### 2. Run the Container
```bash
chmod +x run.sh
./run.sh
```

### 3. Test the API
```bash
chmod +x test.sh
./test.sh
```

### 4. Access Swagger UI
Open your browser and go to: **http://localhost:8000/docs**

## 📊 API Endpoints

- **Swagger UI**: `http://localhost:8000/docs`
- **Health Check**: `GET http://localhost:8000/api/v1/health`
- **Model Info**: `GET http://localhost:8000/api/v1/info`
- **Single Prediction**: `POST http://localhost:8000/api/v1/predict`
- **Batch Prediction**: `POST http://localhost:8000/api/v1/batch-predict`
- **MLflow Compatible**: `POST http://localhost:8000/api/v1/invocations`

## 🧪 Testing Examples

### Single Question via curl
```bash
curl -X POST http://localhost:8000/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "What is machine learning?"}}'
```

### Batch Questions
```bash
curl -X POST http://localhost:8000/api/v1/batch-predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "questions": [
      "What is AI?",
      "How does deep learning work?",
      "Explain neural networks"
    ]
  }}'
```

### MLflow Compatible Format
```bash
curl -X POST http://localhost:8000/api/v1/invocations \\
  -H "Content-Type: application/json" \\
  -d '{{
    "dataframe_split": {{
      "columns": ["question"],
      "data": [["What is artificial intelligence?"]]
    }}
  }}'
```

## 🔧 Manual Commands

```bash
# Build manually
docker build -t mlflow-model-serving:latest .

# Run manually
docker run -d -p 8000:8000 --name mlflow-serving mlflow-model-serving:latest

# View logs
docker logs -f mlflow-serving

# Stop container
docker stop mlflow-serving

# Remove container
docker rm mlflow-serving
```

## 📁 Package Contents

- `model/` - Exported MLflow model artifacts
- `app/` - FastAPI application code
- `scripts/` - Utility scripts
- `Dockerfile` - Container build configuration
- `.env.standalone` - Standalone configuration (no external dependencies)
- `requirements.txt` - Python dependencies
- `build.sh` - Build Docker image
- `run.sh` - Start container
- `test.sh` - Test API endpoints
- `model_metadata.json` - Export metadata

## 🌐 Remote Deployment

This package is completely self-contained and can be deployed on any machine with Docker:

1. Copy this entire directory to your target machine
2. Run `./build.sh` to build the image
3. Run `./run.sh` to start the service
4. Access the Swagger UI at `http://localhost:8000/docs`

No external dependencies (S3, MLflow server, etc.) are required!

## 📊 Model Information

- **Model URI**: `{self.model_uri}`
- **Export Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Container Port**: 8000
- **API Version**: v1

## 🆘 Troubleshooting

### Container won't start
```bash
docker logs mlflow-serving
```

### Port already in use
```bash
# Use different port
docker run -d -p 8080:8000 --name mlflow-serving mlflow-model-serving:latest
# Then access: http://localhost:8080/docs
```

### Health check fails
```bash
curl -v http://localhost:8000/api/v1/health
```
'''
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Created README: {readme_path}")
    
    def export_complete_package(self) -> Dict[str, Any]:
        """Export complete standalone deployment package."""
        print(f"\n🎯 Creating complete deployment package...")
        
        # 1. Export model artifacts
        metadata = self.export_model_artifacts()
        
        # 2. Copy application code
        self.copy_application_code()
        
        # 3. Create standalone configuration
        self.create_standalone_config()
        
        # 4. Create Dockerfile
        self.create_dockerfile()
        
        # 5. Create deployment scripts
        self.create_deployment_scripts()
        
        # 6. Create README
        self.create_readme()
        
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 Package location: {self.output_dir.absolute()}")
        print(f"📊 Model files: {len(metadata.get('downloaded_files', []))}")
        
        print(f"\n🚀 Next steps:")
        print(f"  1. cd {self.output_dir}")
        print(f"  2. ./build.sh")
        print(f"  3. ./run.sh")
        print(f"  4. Open http://localhost:8000/docs")
        
        return metadata


def main():
    """Main function to export model and create Docker package."""
    parser = argparse.ArgumentParser(
        description="Export MLflow model and create standalone Docker deployment package"
    )
    parser.add_argument(
        "--model-uri", 
        required=True,
        help="MLflow model URI (e.g., models:/my-model/1)"
    )
    parser.add_argument(
        "--s3-endpoint-url",
        required=True,
        help="S3 endpoint URL (e.g., http://localhost:9878)"
    )
    parser.add_argument(
        "--s3-access-key",
        required=True,
        help="S3 access key"
    )
    parser.add_argument(
        "--s3-secret-key", 
        required=True,
        help="S3 secret key"
    )
    parser.add_argument(
        "--s3-bucket-name",
        required=True,
        help="S3 bucket name"
    )
    parser.add_argument(
        "--output-dir",
        default="./exported_model",
        help="Output directory for the deployment package (default: ./exported_model)"
    )
    
    args = parser.parse_args()
    
    try:
        exporter = ModelExporter(
            model_uri=args.model_uri,
            s3_endpoint_url=args.s3_endpoint_url,
            s3_access_key=args.s3_access_key,
            s3_secret_key=args.s3_secret_key,
            s3_bucket_name=args.s3_bucket_name,
            output_dir=args.output_dir
        )
        
        exporter.export_complete_package()
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()