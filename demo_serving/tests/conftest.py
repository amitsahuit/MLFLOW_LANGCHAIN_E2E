"""
Pytest configuration and fixtures for testing.
"""
import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ.update({
    "MLFLOW_TRACKING_URI": "http://test-mlflow:5000",
    "MLFLOW_MODEL_URI": "models:/test-model/1",
    "MLFLOW_ARTIFACT_PATH": "s3://test-bucket/test-artifacts/1",
    "S3_ENDPOINT_URL": "http://test-s3:9000",
    "S3_ACCESS_KEY": "test-key",
    "S3_SECRET_KEY": "test-secret",
    "S3_BUCKET_NAME": "test-bucket",
    "MODEL_CACHE_DIR": "/tmp/test_mlflow_models",
    "LOG_LEVEL": "debug",
    "METRICS_ENABLED": "false"
})

from app.main import app
from app.core import settings
from app.core.model_loader import model_loader
from app.services.prediction import prediction_service


@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_mlflow_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_s3_client():
    """Mock S3 client for testing."""
    with patch('app.services.s3_client.S3Client') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        
        # Mock successful operations
        mock_instance.list_objects.return_value = [
            {"Key": "model/MLmodel", "Size": 1000},
            {"Key": "model/model/model.pkl", "Size": 5000},
        ]
        mock_instance.download_file.return_value = True
        mock_instance.download_directory.return_value = True
        mock_instance.download_mlflow_artifacts.return_value = "/tmp/test_artifacts"
        mock_instance.verify_cache_integrity.return_value = True
        
        yield mock_instance


@pytest.fixture(scope="function")
def mock_mlflow_model():
    """Mock MLflow model for testing."""
    mock_model = Mock()
    mock_model.predict.return_value = ["This is a test response from the model."]
    mock_model.metadata = Mock()
    mock_model.metadata.signature = None
    return mock_model


@pytest.fixture(scope="function")
def mock_model_loader(mock_mlflow_model, temp_cache_dir):
    """Mock model loader with test model."""
    with patch('app.core.model_loader.model_loader') as mock_loader:
        mock_loader.get_model.return_value = mock_mlflow_model
        mock_loader.load_model.return_value = mock_mlflow_model
        mock_loader.is_model_loaded.return_value = True
        mock_loader.get_model_metadata.return_value = {
            "model_uri": "models:/test-model/1",
            "loaded_at": 1234567890.0,
            "cache_path": temp_cache_dir
        }
        mock_loader.get_cache_info.return_value = {
            "total_size": 1000000,
            "num_models": 1,
            "models": []
        }
        mock_loader.settings = settings
        
        yield mock_loader


@pytest.fixture(scope="function")
def mock_prediction_service():
    """Mock prediction service for testing."""
    with patch('app.services.prediction.prediction_service') as mock_service:
        mock_service.predict_single.return_value = {
            "question": "Test question",
            "response": "Test response",
            "prediction_time": 0.1,
            "total_time": 0.2,
            "model_uri": "models:/test-model/1",
            "timestamp": 1234567890.0
        }
        mock_service.predict_batch.return_value = [
            {
                "question": "Test question 1",
                "response": "Test response 1",
                "prediction_time": 0.1,
                "index": 0,
                "timestamp": 1234567890.0
            },
            {
                "question": "Test question 2", 
                "response": "Test response 2",
                "prediction_time": 0.1,
                "index": 1,
                "timestamp": 1234567890.0
            }
        ]
        mock_service.get_prediction_stats.return_value = {
            "total_predictions": 10,
            "total_prediction_time": 1.0,
            "average_prediction_time": 0.1,
            "model_loaded": True,
            "model_metadata": {"model_uri": "models:/test-model/1"}
        }
        
        yield mock_service


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks.",
        "What are the benefits of Python?",
        "How can I improve my coding skills?"
    ]


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {"question": "What is machine learning?"}


@pytest.fixture
def sample_batch_prediction_request():
    """Sample batch prediction request data."""
    return {
        "questions": [
            "What is AI?",
            "How does ML work?",
            "What are neural networks?"
        ]
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset any cached state in singletons
    if hasattr(model_loader, '_model'):
        model_loader._model = None
    if hasattr(model_loader, '_model_metadata'):
        model_loader._model_metadata = None
    
    if hasattr(prediction_service, '_prediction_count'):
        prediction_service._prediction_count = 0
    if hasattr(prediction_service, '_total_prediction_time'):
        prediction_service._total_prediction_time = 0.0
    
    yield


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "s3: mark test as requiring S3 connectivity"
    )