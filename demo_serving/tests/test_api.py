"""
Tests for FastAPI endpoints.
"""
import json
import pytest
from fastapi.testclient import TestClient


@pytest.mark.api
class TestHealthEndpoints:
    """Test health and readiness endpoints."""
    
    def test_health_endpoint(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_readiness_endpoint(self, test_client: TestClient, mock_model_loader, mock_s3_client):
        """Test readiness check endpoint."""
        response = test_client.get("/api/v1/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["ready"], bool)
        assert "timestamp" in data
        assert "model_status" in data
        assert "s3_connection" in data
        assert "cache_info" in data
    
    def test_info_endpoint(self, test_client: TestClient, mock_model_loader):
        """Test model info endpoint."""
        response = test_client.get("/api/v1/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_uri" in data
        assert "model_metadata" in data
        assert "cache_info" in data
        assert "prediction_stats" in data
    
    def test_metrics_endpoint(self, test_client: TestClient, mock_model_loader, mock_prediction_service):
        """Test metrics endpoint."""
        response = test_client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions_total" in data
        assert "prediction_time_total" in data
        assert "prediction_time_avg" in data
        assert "model_loaded" in data
        assert "cache_size" in data
        assert "uptime" in data


@pytest.mark.api
class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_single_prediction(self, test_client: TestClient, mock_model_loader, mock_prediction_service, sample_prediction_request):
        """Test single prediction endpoint."""
        response = test_client.post(
            "/api/v1/predict",
            json=sample_prediction_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["question"] == sample_prediction_request["question"]
        assert "response" in data
        assert "prediction_time" in data
        assert "total_time" in data
        assert "model_uri" in data
        assert "timestamp" in data
    
    def test_batch_prediction(self, test_client: TestClient, mock_model_loader, mock_prediction_service, sample_batch_prediction_request):
        """Test batch prediction endpoint."""
        response = test_client.post(
            "/api/v1/batch-predict",
            json=sample_batch_prediction_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "total_time" in data
        assert "total_questions" in data
        assert data["total_questions"] == len(sample_batch_prediction_request["questions"])
        assert len(data["results"]) == len(sample_batch_prediction_request["questions"])
    
    def test_mlflow_invocations_endpoint(self, test_client: TestClient, mock_model_loader, mock_prediction_service):
        """Test MLflow compatible invocations endpoint."""
        # Test DataFrame format
        request_data = {
            "dataframe_split": {
                "columns": ["question"],
                "data": [["What is machine learning?"]]
            }
        }
        
        response = test_client.post("/api/v1/invocations", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
    
    def test_prediction_validation_error(self, test_client: TestClient):
        """Test prediction endpoint with invalid request."""
        # Empty question
        response = test_client.post(
            "/api/v1/predict",
            json={"question": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_validation_error(self, test_client: TestClient):
        """Test batch prediction endpoint with invalid request."""
        # Empty questions list
        response = test_client.post(
            "/api/v1/batch-predict",
            json={"questions": []}
        )
        
        assert response.status_code == 422  # Validation error


@pytest.mark.api
class TestManagementEndpoints:
    """Test model management endpoints."""
    
    def test_reload_model(self, test_client: TestClient, mock_model_loader):
        """Test model reload endpoint."""
        response = test_client.post(
            "/api/v1/reload",
            json={"force_refresh": False}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "timestamp" in data
    
    def test_clear_cache(self, test_client: TestClient, mock_model_loader):
        """Test cache clear endpoint."""
        response = test_client.delete("/api/v1/cache")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "timestamp" in data
    
    def test_reset_stats(self, test_client: TestClient):
        """Test statistics reset endpoint."""
        response = test_client.post("/api/v1/stats/reset")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "message" in data
        assert "timestamp" in data


@pytest.mark.api
class TestUtilityEndpoints:
    """Test utility endpoints."""
    
    def test_root_endpoint(self, test_client: TestClient):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "docs" in data
        assert "health" in data
    
    def test_ping_endpoint(self, test_client: TestClient):
        """Test ping endpoint."""
        response = test_client.get("/ping")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "pong"
        assert "timestamp" in data
    
    def test_version_endpoint(self, test_client: TestClient):
        """Test version endpoint."""
        response = test_client.get("/version")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "version" in data
        assert "mlflow_model_uri" in data
        assert "uptime" in data
        assert "timestamp" in data


@pytest.mark.api
class TestErrorHandling:
    """Test error handling in API endpoints."""
    
    def test_404_not_found(self, test_client: TestClient):
        """Test 404 handling."""
        response = test_client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, test_client: TestClient):
        """Test method not allowed handling."""
        response = test_client.delete("/api/v1/health")  # DELETE not allowed on health
        
        assert response.status_code == 405
    
    def test_invalid_json(self, test_client: TestClient):
        """Test invalid JSON handling."""
        response = test_client.post(
            "/api/v1/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


@pytest.mark.api
class TestCORSHeaders:
    """Test CORS headers."""
    
    def test_cors_headers_present(self, test_client: TestClient):
        """Test that CORS headers are present."""
        response = test_client.options("/api/v1/health")
        
        # FastAPI automatically handles OPTIONS requests
        assert response.status_code in [200, 405]  # May vary based on CORS config
    
    def test_cors_preflight(self, test_client: TestClient):
        """Test CORS preflight request."""
        response = test_client.options(
            "/api/v1/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should not be blocked
        assert response.status_code in [200, 405]


@pytest.mark.api
@pytest.mark.slow
class TestPerformance:
    """Test API performance characteristics."""
    
    def test_concurrent_predictions(self, test_client: TestClient, mock_model_loader, mock_prediction_service):
        """Test concurrent prediction requests."""
        import concurrent.futures
        import time
        
        def make_prediction():
            return test_client.post(
                "/api/v1/predict",
                json={"question": "What is AI?"}
            )
        
        start_time = time.time()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        end_time = time.time()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
    
    def test_large_batch_prediction(self, test_client: TestClient, mock_model_loader, mock_prediction_service):
        """Test large batch prediction."""
        large_batch = {
            "questions": [f"Question {i}" for i in range(50)]
        }
        
        response = test_client.post("/api/v1/batch-predict", json=large_batch)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 50