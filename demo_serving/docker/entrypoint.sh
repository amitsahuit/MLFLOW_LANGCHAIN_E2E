#!/bin/bash
set -e

# MLflow Model Serving Container Entrypoint Script

echo "🚀 Starting MLflow Model Serving Container"
echo "📊 Configuration:"
echo "  - Host: ${HOST:-0.0.0.0}"
echo "  - Port: ${PORT:-8000}"
echo "  - Log Level: ${LOG_LEVEL:-info}"
echo "  - Model URI: ${MLFLOW_MODEL_URI:-Not set}"
echo "  - S3 Endpoint: ${S3_ENDPOINT_URL:-Not set}"

# Function to wait for service availability
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local timeout=${4:-60}
    
    echo "⏳ Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name is available"
            return 0
        fi
        echo "  Attempt $i/$timeout - $service_name not ready"
        sleep 1
    done
    
    echo "❌ $service_name is not available after ${timeout}s"
    return 1
}

# Function to test S3 connectivity
test_s3_connection() {
    echo "🔍 Testing S3 connectivity..."
    
    python3 -c "
import boto3
import os
from botocore.exceptions import ClientError

try:
    client = boto3.client(
        's3',
        endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
        aws_access_key_id=os.environ.get('S3_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('S3_SECRET_KEY'),
        region_name=os.environ.get('S3_REGION', 'us-east-1'),
        verify=False
    )
    
    bucket = os.environ.get('S3_BUCKET_NAME')
    client.head_bucket(Bucket=bucket)
    print('✅ S3 connection successful')
except Exception as e:
    print(f'⚠️  S3 connection failed: {e}')
    print('🔄 Service will attempt to connect on first request')
"
}

# Function to create necessary directories
setup_directories() {
    echo "📁 Setting up directories..."
    
    mkdir -p "${MODEL_CACHE_DIR:-/tmp/mlflow_models}"
    mkdir -p /app/logs
    
    echo "✅ Directories created"
}

# Function to pre-download model (optional)
preload_model() {
    if [ "${PRELOAD_MODEL:-false}" = "true" ]; then
        echo "📥 Pre-loading model..."
        
        python3 -c "
try:
    from app.core.model_loader import model_loader
    model_loader.load_model()
    print('✅ Model pre-loaded successfully')
except Exception as e:
    print(f'⚠️  Model pre-loading failed: {e}')
    print('🔄 Model will be loaded on first request')
"
    fi
}

# Main startup sequence
main() {
    echo "🔧 Starting initialization..."
    
    # Setup directories
    setup_directories
    
    # Test S3 connection (non-blocking)
    test_s3_connection
    
    # Wait for external services if needed
    if [ -n "${WAIT_FOR_MLFLOW_SERVER}" ]; then
        MLFLOW_HOST=$(echo "$MLFLOW_TRACKING_URI" | sed 's|http://||' | sed 's|:[0-9]*||')
        MLFLOW_PORT=$(echo "$MLFLOW_TRACKING_URI" | sed 's|.*:||')
        wait_for_service "MLflow Server" "$MLFLOW_HOST" "$MLFLOW_PORT"
    fi
    
    if [ -n "${WAIT_FOR_S3}" ]; then
        S3_HOST=$(echo "$S3_ENDPOINT_URL" | sed 's|http://||' | sed 's|:[0-9]*||')
        S3_PORT=$(echo "$S3_ENDPOINT_URL" | sed 's|.*:||')
        wait_for_service "S3 Service" "$S3_HOST" "$S3_PORT"
    fi
    
    # Pre-load model if requested
    preload_model
    
    echo "✅ Initialization completed"
    echo "🚀 Starting FastAPI server..."
    
    # Start the application
    exec "$@"
}

# Handle different commands
case "$1" in
    "download-only")
        echo "📥 Download mode - downloading model artifacts only"
        python3 scripts/download_artifacts.py
        exit 0
        ;;
    "health-check")
        echo "🏥 Health check mode"
        python3 scripts/health_check.py
        exit $?
        ;;
    "shell")
        echo "🐚 Starting shell"
        exec /bin/bash
        ;;
    *)
        # Default: start the server
        main "$@"
        ;;
esac