"""
Prediction service for handling model inference requests.
"""
import time
from typing import Any, Dict, List, Union
import pandas as pd

from ..core import get_logger
from ..core.model_loader import model_loader


class PredictionService:
    """Service for handling model predictions."""
    
    def __init__(self):
        self.logger = get_logger("prediction_service")
        self._prediction_count = 0
        self._total_prediction_time = 0.0
    
    def predict_single(self, question: str) -> Dict[str, Any]:
        """Make a single prediction."""
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            model = model_loader.get_model()
            if model is None:
                self.logger.info("Model not loaded, loading now...")
                model = model_loader.load_model()
            
            self.logger.info(f"Making prediction for question: {question[:100]}...")
            
            # Prepare input
            input_data = pd.DataFrame({"question": [question]})
            
            # Make prediction
            prediction_start = time.time()
            result = model.predict(input_data)
            prediction_time = time.time() - prediction_start
            
            # Handle result format
            if isinstance(result, list) and len(result) > 0:
                response = result[0]
            else:
                response = str(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Update metrics
            self._prediction_count += 1
            self._total_prediction_time += total_time
            
            self.logger.info(f"Prediction completed in {total_time:.3f}s")
            
            return {
                "question": question,
                "response": response,
                "prediction_time": prediction_time,
                "total_time": total_time,
                "model_uri": model_loader.settings.mlflow_model_uri,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                "question": question,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def predict_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            model = model_loader.get_model()
            if model is None:
                self.logger.info("Model not loaded, loading now...")
                model = model_loader.load_model()
            
            self.logger.info(f"Making batch prediction for {len(questions)} questions")
            
            # Prepare input
            input_data = pd.DataFrame({"question": questions})
            
            # Make prediction
            prediction_start = time.time()
            results = model.predict(input_data)
            prediction_time = time.time() - prediction_start
            
            # Handle results format
            if not isinstance(results, list):
                results = [results]
            
            # Ensure we have the same number of results as questions
            while len(results) < len(questions):
                results.append("No response generated")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Update metrics
            self._prediction_count += len(questions)
            self._total_prediction_time += total_time
            
            # Format response
            batch_results = []
            for i, (question, response) in enumerate(zip(questions, results)):
                batch_results.append({
                    "question": question,
                    "response": response,
                    "index": i,
                    "prediction_time": prediction_time / len(questions),  # Average per item
                    "timestamp": time.time()
                })
            
            self.logger.info(f"Batch prediction completed in {total_time:.3f}s")
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return [
                {
                    "question": question,
                    "error": str(e),
                    "index": i,
                    "timestamp": time.time()
                }
                for i, question in enumerate(questions)
            ]
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        return {
            "total_predictions": self._prediction_count,
            "total_prediction_time": self._total_prediction_time,
            "average_prediction_time": (
                self._total_prediction_time / self._prediction_count 
                if self._prediction_count > 0 else 0.0
            ),
            "model_loaded": model_loader.is_model_loaded(),
            "model_metadata": model_loader.get_model_metadata()
        }
    
    def reset_stats(self) -> None:
        """Reset prediction statistics."""
        self._prediction_count = 0
        self._total_prediction_time = 0.0
        self.logger.info("Prediction statistics reset")


# Global prediction service instance
prediction_service = PredictionService()