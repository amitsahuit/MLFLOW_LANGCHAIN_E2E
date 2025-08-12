"""
End-to-end integration tests for the MLflow serving application.
"""
import os
import pytest
import time
import requests
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock


@pytest.mark.integration
class TestE2EWorkflow:
    """End-to-end workflow tests."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_url = "http://localhost:8000"
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('app.services.s3_client.S3Client')
    @patch('app.core.model_loader.mlflow.pyfunc.load_model')
    def test_model_download_and_serving_workflow(self, mock_load_model, mock_s3_class):
        """Test complete workflow from model download to serving."""
        # Setup mocks
        mock_s3_instance = Mock()
        mock_s3_class.return_value = mock_s3_instance
        
        # Mock successful S3 operations
        artifacts_dir = os.path.join(self.temp_dir, "artifacts")
        model_dir = os.path.join(artifacts_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        Path(os.path.join(artifacts_dir, "MLmodel")).touch()
        Path(os.path.join(model_dir, "model.pkl")).touch()
        
        mock_s3_instance.download_mlflow_artifacts.return_value = artifacts_dir
        mock_s3_instance.list_objects.return_value = [
            {"Key": "model/MLmodel", "Size": 1000},
            {"Key": "model/model.pkl", "Size": 5000}
        ]
        mock_s3_instance.verify_cache_integrity.return_value = True
        
        # Mock MLflow model
        mock_model = Mock()
        mock_model.predict.return_value = ["This is a test response."]
        mock_model.metadata = Mock()
        mock_model.metadata.signature = None
        mock_load_model.return_value = mock_model
        
        # Import and create model loader with mocked dependencies
        from app.core.model_loader import ModelLoader
        model_loader = ModelLoader()
        model_loader.settings.model_cache_dir = self.temp_dir
        
        # Test model loading
        loaded_model = model_loader.load_model()
        assert loaded_model == mock_model
        assert model_loader.is_model_loaded()
        
        # Test prediction service
        from app.services.prediction import PredictionService
        prediction_service = PredictionService()
        
        # Mock the model loader in prediction service
        with patch('app.services.prediction.model_loader', model_loader):
            result = prediction_service.predict_single("What is AI?")
            
            assert result["question"] == "What is AI?"
            assert result["response"] == "This is a test response."
            assert "prediction_time" in result
            assert "total_time" in result
    
    def test_api_workflow_with_mocked_dependencies(self, test_client, mock_model_loader, mock_prediction_service):
        """Test complete API workflow with mocked dependencies."""
        # Test health check
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Test readiness check
        response = test_client.get("/api/v1/ready")
        assert response.status_code == 200
        
        # Test model info
        response = test_client.get("/api/v1/info")
        assert response.status_code == 200
        
        # Test single prediction
        response = test_client.post(
            "/api/v1/predict",
            json={"question": "What is machine learning?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "prediction_time" in data
        
        # Test batch prediction
        response = test_client.post(
            "/api/v1/batch-predict",
            json={"questions": ["Question 1", "Question 2"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        
        # Test metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        # Test model reload
        response = test_client.post(
            "/api/v1/reload",
            json={"force_refresh": False}
        )
        assert response.status_code == 200
    
    def test_error_scenarios(self, test_client):
        """Test error handling scenarios."""
        # Test invalid prediction request
        response = test_client.post(
            "/api/v1/predict",
            json={"invalid_field": "test"}
        )
        assert response.status_code == 422
        
        # Test empty question
        response = test_client.post(
            "/api/v1/predict",
            json={"question": ""}
        )
        assert response.status_code == 422
        
        # Test invalid batch request
        response = test_client.post(
            "/api/v1/batch-predict",
            json={"questions": []}
        )
        assert response.status_code == 422
        
        # Test nonexistent endpoint
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance integration tests."""
    
    def test_prediction_latency(self, test_client, mock_model_loader, mock_prediction_service):
        """Test prediction latency requirements."""
        start_time = time.time()
        
        response = test_client.post(
            "/api/v1/predict",
            json={"question": "What is artificial intelligence?"}
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        assert response.status_code == 200
        # Should respond within 1 second for mocked model
        assert latency < 1.0
        
        data = response.json()
        assert "prediction_time" in data
        assert "total_time" in data
    
    def test_concurrent_requests(self, test_client, mock_model_loader, mock_prediction_service):
        """Test handling concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return test_client.post(
                "/api/v1/predict",
                json={"question": "Test concurrent request"}
            )
        
        # Make 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
    
    def test_batch_processing_performance(self, test_client, mock_model_loader, mock_prediction_service):
        """Test batch processing performance."""
        large_batch = {
            "questions": [f"Question {i}" for i in range(50)]
        }
        
        start_time = time.time()
        response = test_client.post("/api/v1/batch-predict", json=large_batch)
        end_time = time.time()
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 50
        
        # Batch processing should be efficient
        total_time = end_time - start_time
        assert total_time < 5.0  # Should process 50 questions in under 5 seconds


@pytest.mark.integration
class TestHealthCheckIntegration:
    """Integration tests for health checking."""
    
    def test_health_check_script(self):
        """Test the health check script."""
        # This would normally test the actual script
        # For now, we'll test the health check logic
        from scripts.health_check import HealthChecker
        
        # Mock the requests to avoid actual HTTP calls
        with patch('scripts.health_check.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response
            
            checker = HealthChecker("http://localhost:8000")
            result = checker.check_health()
            
            assert result["success"] is True
            assert result["endpoint"] == "/api/v1/health"
    
    def test_comprehensive_health_check(self):
        """Test comprehensive health check."""
        from scripts.health_check import HealthChecker
        
        with patch('scripts.health_check.requests.get') as mock_get, \
             patch('scripts.health_check.requests.post') as mock_post:
            
            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response
            mock_post.return_value = mock_response
            
            checker = HealthChecker("http://localhost:8000")
            result = checker.run_comprehensive_check()
            
            assert result["overall_status"] in ["healthy", "degraded", "critical"]
            assert "checks" in result
            assert "health" in result["checks"]
            assert "readiness" in result["checks"]
            assert "prediction" in result["checks"]


@pytest.mark.integration
class TestCacheIntegration:
    """Integration tests for cache management."""
    
    def test_cache_lifecycle(self):
        """Test complete cache lifecycle."""
        temp_cache_dir = tempfile.mkdtemp()
        
        try:
            with patch('app.services.s3_client.S3Client') as mock_s3_class, \
                 patch('app.core.model_loader.mlflow.pyfunc.load_model') as mock_load_model:
                
                # Setup mocks
                mock_s3_instance = Mock()
                mock_s3_class.return_value = mock_s3_instance
                
                artifacts_dir = os.path.join(temp_cache_dir, "artifacts")
                model_dir = os.path.join(artifacts_dir, "model")
                os.makedirs(model_dir, exist_ok=True)
                Path(os.path.join(artifacts_dir, "MLmodel")).touch()
                
                mock_s3_instance.download_mlflow_artifacts.return_value = artifacts_dir
                mock_model = Mock()
                mock_load_model.return_value = mock_model
                
                # Test cache operations
                from app.core.model_loader import ModelLoader
                model_loader = ModelLoader()
                model_loader.settings.model_cache_dir = temp_cache_dir
                
                # Load model (creates cache)
                model_loader.load_model()
                
                # Check cache info
                cache_info = model_loader.get_cache_info()
                assert cache_info["num_models"] >= 0
                assert "total_size" in cache_info
                
                # Test cache cleanup
                model_loader.cleanup_cache(keep_current=False)
                
        finally:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    def test_environment_variable_override(self):
        """Test that environment variables override configuration."""
        with patch.dict(os.environ, {
            'MLFLOW_MODEL_URI': 'models:/test-env-model/1',
            'PORT': '9999',
            'LOG_LEVEL': 'debug'
        }):
            from app.core.config import get_settings
            settings = get_settings()
            
            assert settings.mlflow_model_uri == 'models:/test-env-model/1'
            assert settings.port == 9999
            assert settings.log_level == 'debug'
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        from app.core.config import Settings
        
        # Test valid configuration
        settings = Settings(
            mlflow_model_uri="models:/valid-model/1",
            s3_endpoint_url="http://localhost:9000",
            s3_bucket_name="test-bucket"
        )
        
        assert settings.mlflow_model_uri == "models:/valid-model/1"
        assert settings.s3_endpoint_url == "http://localhost:9000"
        assert settings.s3_bucket_name == "test-bucket"


@pytest.mark.integration
@pytest.mark.slow
class TestDockerIntegration:
    """Integration tests for Docker functionality."""
    
    def test_docker_build(self):
        """Test Docker image build (requires Docker)."""
        pytest.skip("Requires Docker environment")
    
    def test_docker_run(self):
        """Test Docker container run (requires Docker)."""
        pytest.skip("Requires Docker environment")