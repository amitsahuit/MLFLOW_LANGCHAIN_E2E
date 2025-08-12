"""
Tests for model loading and prediction functionality.
"""
import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.core.model_loader import ModelLoader
from app.services.prediction import PredictionService


@pytest.mark.unit
class TestModelLoader:
    """Test ModelLoader functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_loader = ModelLoader()
        # Override cache directory for testing
        self.model_loader.settings.model_cache_dir = self.temp_dir
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_model_cache_path(self):
        """Test model cache path generation."""
        model_uri = "models:/test-model/1"
        cache_path = self.model_loader._get_model_cache_path(model_uri)
        
        assert isinstance(cache_path, str)
        assert "test-model" in cache_path
        assert cache_path.startswith(self.temp_dir)
    
    def test_is_cache_valid_missing_directory(self):
        """Test cache validation with missing directory."""
        non_existent_path = os.path.join(self.temp_dir, "nonexistent")
        assert not self.model_loader._is_cache_valid(non_existent_path)
    
    def test_is_cache_valid_missing_files(self):
        """Test cache validation with missing essential files."""
        cache_path = os.path.join(self.temp_dir, "test_cache")
        artifacts_dir = os.path.join(cache_path, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Missing MLmodel file
        assert not self.model_loader._is_cache_valid(cache_path)
    
    def test_is_cache_valid_complete_cache(self):
        """Test cache validation with complete cache."""
        cache_path = os.path.join(self.temp_dir, "test_cache")
        artifacts_dir = os.path.join(cache_path, "artifacts")
        model_dir = os.path.join(artifacts_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create essential files
        Path(os.path.join(artifacts_dir, "MLmodel")).touch()
        
        assert self.model_loader._is_cache_valid(cache_path)
    
    @patch('app.core.model_loader.mlflow.pyfunc.load_model')
    def test_load_model_from_cache(self, mock_load_model):
        """Test loading model from cache."""
        # Setup cache directory
        cache_path = os.path.join(self.temp_dir, "test_cache")
        artifacts_dir = os.path.join(cache_path, "artifacts")
        model_dir = os.path.join(artifacts_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        Path(os.path.join(artifacts_dir, "MLmodel")).touch()
        
        # Mock MLflow model
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        result = self.model_loader._load_model_from_cache(cache_path)
        
        assert result == mock_model
        mock_load_model.assert_called_once_with(artifacts_dir)
    
    def test_get_model_metadata(self):
        """Test model metadata extraction."""
        mock_model = Mock()
        mock_model.metadata = Mock()
        mock_model.metadata.signature = "test_signature"
        
        metadata = self.model_loader._get_model_metadata(mock_model)
        
        assert isinstance(metadata, dict)
        assert "model_uri" in metadata
        assert "loaded_at" in metadata
        assert "cache_path" in metadata
    
    def test_get_cache_info(self):
        """Test cache information retrieval."""
        # Create some test cache files
        test_model_dir = os.path.join(self.temp_dir, "test_model")
        os.makedirs(test_model_dir, exist_ok=True)
        test_file = os.path.join(test_model_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        cache_info = self.model_loader.get_cache_info()
        
        assert isinstance(cache_info, dict)
        assert "total_size" in cache_info
        assert "num_models" in cache_info
        assert "models" in cache_info
        assert "max_cache_size" in cache_info
    
    def test_model_lifecycle(self):
        """Test complete model lifecycle."""
        # Initially no model loaded
        assert not self.model_loader.is_model_loaded()
        assert self.model_loader.get_model() is None
        
        # Mock a model
        mock_model = Mock()
        self.model_loader._model = mock_model
        self.model_loader._model_metadata = {"test": "metadata"}
        
        # Now model should be loaded
        assert self.model_loader.is_model_loaded()
        assert self.model_loader.get_model() == mock_model
        
        # Unload model
        self.model_loader.unload_model()
        assert not self.model_loader.is_model_loaded()
        assert self.model_loader.get_model() is None


@pytest.mark.unit
class TestPredictionService:
    """Test PredictionService functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.prediction_service = PredictionService()
        self.prediction_service._prediction_count = 0
        self.prediction_service._total_prediction_time = 0.0
    
    @patch('app.services.prediction.model_loader')
    def test_predict_single_success(self, mock_model_loader):
        """Test successful single prediction."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = ["Test response"]
        mock_model_loader.get_model.return_value = mock_model
        mock_model_loader.settings.mlflow_model_uri = "models:/test-model/1"
        
        result = self.prediction_service.predict_single("Test question")
        
        assert result["question"] == "Test question"
        assert result["response"] == "Test response"
        assert "prediction_time" in result
        assert "total_time" in result
        assert "model_uri" in result
        assert "timestamp" in result
        assert "error" not in result
    
    @patch('app.services.prediction.model_loader')
    def test_predict_single_model_not_loaded(self, mock_model_loader):
        """Test single prediction when model not initially loaded."""
        # Mock model loading
        mock_model = Mock()
        mock_model.predict.return_value = ["Test response"]
        mock_model_loader.get_model.return_value = None  # Not loaded initially
        mock_model_loader.load_model.return_value = mock_model  # Loaded on demand
        mock_model_loader.settings.mlflow_model_uri = "models:/test-model/1"
        
        result = self.prediction_service.predict_single("Test question")
        
        assert result["response"] == "Test response"
        mock_model_loader.load_model.assert_called_once()
    
    @patch('app.services.prediction.model_loader')
    def test_predict_single_error(self, mock_model_loader):
        """Test single prediction with error."""
        # Mock model that raises an exception
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Test error")
        mock_model_loader.get_model.return_value = mock_model
        
        result = self.prediction_service.predict_single("Test question")
        
        assert result["question"] == "Test question"
        assert "error" in result
        assert result["error"] == "Test error"
        assert "timestamp" in result
    
    @patch('app.services.prediction.model_loader')
    def test_predict_batch_success(self, mock_model_loader):
        """Test successful batch prediction."""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = ["Response 1", "Response 2"]
        mock_model_loader.get_model.return_value = mock_model
        
        questions = ["Question 1", "Question 2"]
        results = self.prediction_service.predict_batch(questions)
        
        assert len(results) == 2
        assert results[0]["question"] == "Question 1"
        assert results[0]["response"] == "Response 1"
        assert results[1]["question"] == "Question 2"
        assert results[1]["response"] == "Response 2"
        
        for result in results:
            assert "prediction_time" in result
            assert "index" in result
            assert "timestamp" in result
    
    @patch('app.services.prediction.model_loader')
    def test_predict_batch_error(self, mock_model_loader):
        """Test batch prediction with error."""
        # Mock model that raises an exception
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Batch test error")
        mock_model_loader.get_model.return_value = mock_model
        
        questions = ["Question 1", "Question 2"]
        results = self.prediction_service.predict_batch(questions)
        
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result["question"] == questions[i]
            assert "error" in result
            assert result["error"] == "Batch test error"
            assert result["index"] == i
    
    def test_prediction_stats(self):
        """Test prediction statistics tracking."""
        # Initial stats
        stats = self.prediction_service.get_prediction_stats()
        assert stats["total_predictions"] == 0
        assert stats["total_prediction_time"] == 0.0
        assert stats["average_prediction_time"] == 0.0
        
        # Simulate some predictions
        self.prediction_service._prediction_count = 5
        self.prediction_service._total_prediction_time = 2.5
        
        stats = self.prediction_service.get_prediction_stats()
        assert stats["total_predictions"] == 5
        assert stats["total_prediction_time"] == 2.5
        assert stats["average_prediction_time"] == 0.5
        
        # Reset stats
        self.prediction_service.reset_stats()
        stats = self.prediction_service.get_prediction_stats()
        assert stats["total_predictions"] == 0
        assert stats["total_prediction_time"] == 0.0


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model loading and prediction."""
    
    @patch('app.services.s3_client.S3Client')
    @patch('app.core.model_loader.mlflow.pyfunc.load_model')
    def test_full_model_loading_flow(self, mock_load_model, mock_s3_class):
        """Test complete model loading flow."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Setup mocks
            mock_s3_instance = Mock()
            mock_s3_class.return_value = mock_s3_instance
            mock_s3_instance.download_mlflow_artifacts.return_value = os.path.join(temp_dir, "artifacts")
            
            # Create mock artifacts
            artifacts_dir = os.path.join(temp_dir, "artifacts")
            model_dir = os.path.join(artifacts_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            Path(os.path.join(artifacts_dir, "MLmodel")).touch()
            
            mock_model = Mock()
            mock_load_model.return_value = mock_model
            
            # Create model loader
            model_loader = ModelLoader()
            model_loader.settings.model_cache_dir = temp_dir
            
            # Load model
            loaded_model = model_loader.load_model()
            
            assert loaded_model == mock_model
            assert model_loader.is_model_loaded()
            
            # Test metadata
            metadata = model_loader.get_model_metadata()
            assert metadata is not None
            assert "model_uri" in metadata
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.s3
class TestS3Integration:
    """Tests requiring S3 connectivity (skipped in unit tests)."""
    
    def test_s3_download_integration(self):
        """Test actual S3 download (requires real S3 setup)."""
        pytest.skip("Requires actual S3 setup")
    
    def test_cache_verification_integration(self):
        """Test cache verification against real S3 (requires real S3 setup)."""
        pytest.skip("Requires actual S3 setup")