"""
MLflow model loader with HuggingFace integration and caching.
"""
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

from ..core import get_logger, settings
from ..services.s3_client import S3Client


class ModelLoader:
    """MLflow model loader with caching and HuggingFace integration."""
    
    def __init__(self):
        self.logger = get_logger("model_loader")
        self.settings = settings
        self.s3_client = S3Client()
        self._model = None
        self._model_metadata = None
        self._load_lock = threading.Lock()
        self._last_loaded = None
        
        # Ensure cache directory exists
        Path(self.settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_model_cache_path(self, model_uri: str) -> str:
        """Get local cache path for model."""
        # Create a safe directory name from model URI
        safe_name = model_uri.replace(':', '_').replace('/', '_').replace('models_', 'model_')
        return os.path.join(self.settings.model_cache_dir, safe_name)
    
    def _download_model_artifacts(self, force_refresh: bool = False) -> str:
        """Download model artifacts from S3 if needed."""
        cache_path = self._get_model_cache_path(self.settings.mlflow_model_uri)
        
        # Check if model is already cached and valid
        if not force_refresh and self._is_cache_valid(cache_path):
            self.logger.info(f"Using cached model from {cache_path}")
            return cache_path
        
        # Remove existing cache if force refresh
        if force_refresh and os.path.exists(cache_path):
            self.logger.info("Force refresh requested, removing cached model")
            shutil.rmtree(cache_path, ignore_errors=True)
        
        # Download from S3
        self.logger.info(f"Downloading model artifacts from {self.settings.mlflow_artifact_path}")
        
        try:
            artifact_dir = self.s3_client.download_mlflow_artifacts(
                artifact_path=self.settings.mlflow_artifact_path,
                local_cache_dir=cache_path
            )
            
            # Verify the downloaded artifacts contain required files
            self._verify_model_artifacts(artifact_dir)
            
            self.logger.info(f"Model artifacts downloaded to {artifact_dir}")
            return artifact_dir
            
        except Exception as e:
            self.logger.error(f"Failed to download model artifacts: {e}")
            # Clean up partial download
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path, ignore_errors=True)
            raise
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cached model is valid."""
        try:
            if not os.path.exists(cache_path):
                return False
            
            # Check if MLmodel file exists in subdirectories
            try:
                mlmodel_dir = self._find_mlmodel_path(cache_path)
                self.logger.debug(f"Found valid MLmodel at: {mlmodel_dir}")
            except ValueError:
                self.logger.warning("No valid MLmodel file found in cache")
                return False
            
            # Check cache age (optional - could implement TTL)
            cache_age = time.time() - os.path.getmtime(cache_path)
            max_cache_age = 24 * 3600  # 24 hours
            
            if cache_age > max_cache_age:
                self.logger.info(f"Cache is older than {max_cache_age}s, will refresh")
                return False
            
            self.logger.info("Model cache is valid")
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache validation failed: {e}")
            return False
    
    def _find_mlmodel_path(self, base_dir: str) -> str:
        """Find the MLmodel file in nested subdirectories."""
        base_path = Path(base_dir)
        
        # Look for MLmodel files in subdirectories
        mlmodel_files = list(base_path.rglob("MLmodel"))
        
        if not mlmodel_files:
            raise ValueError(f"No MLmodel file found in {base_dir} or its subdirectories")
        
        if len(mlmodel_files) == 1:
            return str(mlmodel_files[0].parent)
        
        # If multiple MLmodel files exist, use the most recently modified one
        latest_mlmodel = max(mlmodel_files, key=lambda p: p.stat().st_mtime)
        self.logger.info(f"Found {len(mlmodel_files)} MLmodel files, using most recent: {latest_mlmodel.parent}")
        
        return str(latest_mlmodel.parent)

    def _verify_model_artifacts(self, artifact_dir: str) -> None:
        """Verify that downloaded artifacts are complete."""
        artifact_path = Path(artifact_dir)
        
        # Find MLmodel file in subdirectories
        try:
            mlmodel_dir = self._find_mlmodel_path(str(artifact_path))
            self.logger.info(f"Found MLmodel at: {mlmodel_dir}")
        except ValueError as e:
            raise ValueError(f"MLmodel file not found in {artifact_dir}: {e}")
        
        self.logger.info("Model artifact verification passed")
    
    def _load_model_from_cache(self, cache_path: str) -> Any:
        """Load MLflow model from local cache."""
        try:
            # Find the MLmodel file in subdirectories
            mlmodel_dir = self._find_mlmodel_path(cache_path)
            
            self.logger.info(f"Loading MLflow model from {mlmodel_dir}")
            
            # Load the model using MLflow
            model = mlflow.pyfunc.load_model(mlmodel_dir)
            
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from cache: {e}")
            raise
    
    def _get_model_metadata(self, model) -> Dict[str, Any]:
        """Extract metadata from loaded model."""
        try:
            metadata = {
                'model_uri': self.settings.mlflow_model_uri,
                'loaded_at': time.time(),
                'cache_path': self._get_model_cache_path(self.settings.mlflow_model_uri)
            }
            
            # Try to get additional metadata from model
            if hasattr(model, 'metadata') and model.metadata:
                metadata.update({
                    'signature': str(model.metadata.signature) if model.metadata.signature else None,
                    'input_schema': str(model.metadata.get_input_schema()) if hasattr(model.metadata, 'get_input_schema') else None,
                    'output_schema': str(model.metadata.get_output_schema()) if hasattr(model.metadata, 'get_output_schema') else None,
                })
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract model metadata: {e}")
            return {
                'model_uri': self.settings.mlflow_model_uri,
                'loaded_at': time.time(),
                'error': str(e)
            }
    
    def load_model(self, force_refresh: bool = False) -> Any:
        """Load the MLflow model with caching."""
        with self._load_lock:
            # Return cached model if available and not forcing refresh
            if self._model is not None and not force_refresh:
                self.logger.info("Returning cached model")
                return self._model
            
            try:
                self.logger.info(f"Loading model: {self.settings.mlflow_model_uri}")
                
                # Download artifacts if needed
                cache_path = self._download_model_artifacts(force_refresh=force_refresh)
                
                # Load model from cache
                model = self._load_model_from_cache(cache_path)
                
                # Get model metadata
                metadata = self._get_model_metadata(model)
                
                # Cache the model
                self._model = model
                self._model_metadata = metadata
                self._last_loaded = time.time()
                
                self.logger.info("Model loaded and cached successfully")
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                # Don't cache failed loads
                self._model = None
                self._model_metadata = None
                raise
    
    def get_model(self) -> Optional[Any]:
        """Get the cached model if available."""
        return self._model
    
    def get_model_metadata(self) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        return self._model_metadata
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        with self._load_lock:
            self.logger.info("Unloading model from memory")
            self._model = None
            self._model_metadata = None
            self._last_loaded = None
    
    def reload_model(self) -> Any:
        """Reload the model (force refresh)."""
        self.logger.info("Reloading model")
        return self.load_model(force_refresh=True)
    
    def cleanup_cache(self, keep_current: bool = True) -> None:
        """Clean up old cached models."""
        try:
            cache_dir = Path(self.settings.model_cache_dir)
            if not cache_dir.exists():
                return
            
            current_cache_path = None
            if keep_current and self._model is not None:
                current_cache_path = self._get_model_cache_path(self.settings.mlflow_model_uri)
            
            total_size = 0
            removed_size = 0
            
            for item in cache_dir.iterdir():
                if item.is_dir():
                    item_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    total_size += item_size
                    
                    # Remove if not current model or if exceeding cache size
                    if (str(item) != current_cache_path or 
                        total_size > self.settings.max_model_cache_size):
                        
                        self.logger.info(f"Removing cached model: {item}")
                        shutil.rmtree(item, ignore_errors=True)
                        removed_size += item_size
            
            if removed_size > 0:
                self.logger.info(f"Cleaned up {removed_size} bytes from model cache")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache."""
        try:
            cache_dir = Path(self.settings.model_cache_dir)
            if not cache_dir.exists():
                return {'total_size': 0, 'num_models': 0, 'models': []}
            
            models = []
            total_size = 0
            
            for item in cache_dir.iterdir():
                if item.is_dir():
                    item_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    total_size += item_size
                    
                    models.append({
                        'path': str(item),
                        'size': item_size,
                        'created': item.stat().st_ctime,
                        'modified': item.stat().st_mtime,
                        'is_current': str(item) == self._get_model_cache_path(self.settings.mlflow_model_uri)
                    })
            
            return {
                'total_size': total_size,
                'num_models': len(models),
                'models': sorted(models, key=lambda x: x['modified'], reverse=True),
                'max_cache_size': self.settings.max_model_cache_size
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache info: {e}")
            return {'error': str(e)}


# Global model loader instance
model_loader = ModelLoader()