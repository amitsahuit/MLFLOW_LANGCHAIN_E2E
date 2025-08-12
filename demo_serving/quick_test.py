#!/usr/bin/env python3
"""
Quick test script to verify the fixes work without Docker Compose
"""

import sys
import os
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def test_config():
    """Test that the configuration loads without errors."""
    print("🔍 Testing configuration loading...")
    
    try:
        from app.core.config import settings
        print(f"✅ Configuration loaded successfully!")
        print(f"📊 Model URI: {settings.mlflow_model_uri}")
        print(f"🌐 Host: {settings.host}:{settings.port}")
        print(f"📝 Log Level: {settings.log_level}")
        print(f"🔧 Preload Model: {settings.preload_model}")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_models():
    """Test that Pydantic models work without namespace warnings."""
    print("\n🔍 Testing Pydantic models...")
    
    try:
        from app.api.models import HealthResponse, ModelInfoResponse, PredictionResponse
        
        # Test creating model instances
        health = HealthResponse(
            status="healthy",
            timestamp=1234567890.0,
            version="1.0.0",
            model_loaded=True,
            uptime=60.0
        )
        print(f"✅ HealthResponse created: {health.status}")
        
        # Test model with model_ prefix fields
        model_info = ModelInfoResponse(
            model_uri="models:/test/1",
            model_metadata={"test": "data"},
            cache_info={"size": 100},
            loaded_at=1234567890.0,
            prediction_stats={"count": 0}
        )
        print(f"✅ ModelInfoResponse created: {model_info.model_uri}")
        
        return True
    except Exception as e:
        print(f"❌ Pydantic models test failed: {e}")
        return False

def test_app_creation():
    """Test that the FastAPI app can be created."""
    print("\n🔍 Testing FastAPI app creation...")
    
    try:
        from app.main import create_app
        app = create_app()
        print(f"✅ FastAPI app created successfully!")
        print(f"📝 Title: {app.title}")
        print(f"🔗 Docs URL: {app.docs_url}")
        return True
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Quick Tests for Demo Serving Fixes")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config),
        ("Pydantic Models", test_models),
        ("FastAPI App Creation", test_app_creation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The fixes are working correctly.")
        print("\n✅ You can now run:")
        print("   docker-compose -f docker/docker-compose.yml --profile with-mlflow up -d")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)