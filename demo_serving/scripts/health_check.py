#!/usr/bin/env python3
"""
Health check script for MLflow model serving application.
"""
import os
import sys
import argparse
import time
import requests
from typing import Dict, Any

# Add app to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core import configure_logging, get_logger, settings


class HealthChecker:
    """Health checker for the MLflow serving application."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = get_logger("health_checker")
    
    def check_endpoint(self, endpoint: str, expected_status: int = 200) -> Dict[str, Any]:
        """Check a specific endpoint."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            self.logger.info(f"🔍 Checking {url}")
            
            start_time = time.time()
            response = requests.get(url, timeout=self.timeout)
            response_time = time.time() - start_time
            
            success = response.status_code == expected_status
            
            result = {
                "endpoint": endpoint,
                "url": url,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": success,
                "response_time": response_time,
                "timestamp": time.time()
            }
            
            if success:
                self.logger.info(f"✅ {endpoint} - OK ({response.status_code}) [{response_time:.3f}s]")
                try:
                    result["response_data"] = response.json()
                except:
                    result["response_data"] = response.text[:200]
            else:
                self.logger.warning(f"⚠️  {endpoint} - FAIL ({response.status_code}) [{response_time:.3f}s]")
                result["error"] = response.text[:200]
            
            return result
            
        except requests.exceptions.Timeout:
            self.logger.error(f"❌ {endpoint} - TIMEOUT (>{self.timeout}s)")
            return {
                "endpoint": endpoint,
                "url": url,
                "success": False,
                "error": "Timeout",
                "timestamp": time.time()
            }
        except requests.exceptions.ConnectionError:
            self.logger.error(f"❌ {endpoint} - CONNECTION ERROR")
            return {
                "endpoint": endpoint,
                "url": url,
                "success": False,
                "error": "Connection Error",
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"❌ {endpoint} - ERROR: {e}")
            return {
                "endpoint": endpoint,
                "url": url,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def check_health(self) -> Dict[str, Any]:
        """Check basic health endpoint."""
        return self.check_endpoint("/api/v1/health")
    
    def check_readiness(self) -> Dict[str, Any]:
        """Check readiness endpoint."""
        return self.check_endpoint("/api/v1/ready")
    
    def check_info(self) -> Dict[str, Any]:
        """Check model info endpoint."""
        return self.check_endpoint("/api/v1/info")
    
    def check_metrics(self) -> Dict[str, Any]:
        """Check metrics endpoint."""
        return self.check_endpoint("/api/v1/metrics")
    
    def check_prediction(self, question: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """Check prediction endpoint."""
        url = f"{self.base_url}/api/v1/predict"
        
        try:
            self.logger.info(f"🔍 Testing prediction with: {question[:50]}...")
            
            start_time = time.time()
            response = requests.post(
                url,
                json={"question": question},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response_time = time.time() - start_time
            
            success = response.status_code == 200
            
            result = {
                "endpoint": "/predict",
                "url": url,
                "status_code": response.status_code,
                "success": success,
                "response_time": response_time,
                "question": question,
                "timestamp": time.time()
            }
            
            if success:
                self.logger.info(f"✅ Prediction - OK ({response.status_code}) [{response_time:.3f}s]")
                try:
                    data = response.json()
                    result["response_data"] = data
                    result["response_length"] = len(data.get("response", ""))
                    self.logger.info(f"  Response: {data.get('response', 'N/A')[:100]}...")
                except:
                    result["response_data"] = response.text[:200]
            else:
                self.logger.warning(f"⚠️  Prediction - FAIL ({response.status_code}) [{response_time:.3f}s]")
                result["error"] = response.text[:200]
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Prediction test failed: {e}")
            return {
                "endpoint": "/predict",
                "url": url,
                "success": False,
                "error": str(e),
                "question": question,
                "timestamp": time.time()
            }
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        self.logger.info("🏥 Starting comprehensive health check")
        
        start_time = time.time()
        
        checks = {
            "health": self.check_health(),
            "readiness": self.check_readiness(),
            "info": self.check_info(),
            "metrics": self.check_metrics(),
            "prediction": self.check_prediction()
        }
        
        total_time = time.time() - start_time
        
        # Calculate overall status
        all_success = all(check.get("success", False) for check in checks.values())
        critical_success = checks["health"].get("success", False) and checks["prediction"].get("success", False)
        
        result = {
            "overall_status": "healthy" if all_success else ("critical" if not critical_success else "degraded"),
            "all_checks_passed": all_success,
            "critical_checks_passed": critical_success,
            "total_time": total_time,
            "timestamp": time.time(),
            "checks": checks
        }
        
        # Log summary
        status_emoji = "✅" if all_success else ("❌" if not critical_success else "⚠️")
        self.logger.info(f"{status_emoji} Health check completed: {result['overall_status'].upper()}")
        self.logger.info(f"📊 Summary: {sum(1 for c in checks.values() if c.get('success'))}/{len(checks)} checks passed in {total_time:.2f}s")
        
        return result


def main():
    """Main function for health check."""
    parser = argparse.ArgumentParser(
        description="Health check for MLflow model serving application"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the service"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--endpoint",
        choices=["health", "ready", "info", "metrics", "predict", "all"],
        default="all",
        help="Specific endpoint to check"
    )
    parser.add_argument(
        "--question",
        default="What is artificial intelligence?",
        help="Question for prediction test"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.log_level)
    logger = get_logger("health_check_main")
    
    # Create health checker
    checker = HealthChecker(args.url, args.timeout)
    
    try:
        if args.endpoint == "all":
            result = checker.run_comprehensive_check()
        elif args.endpoint == "health":
            result = checker.check_health()
        elif args.endpoint == "ready":
            result = checker.check_readiness()
        elif args.endpoint == "info":
            result = checker.check_info()
        elif args.endpoint == "metrics":
            result = checker.check_metrics()
        elif args.endpoint == "predict":
            result = checker.check_prediction(args.question)
        
        # Output results
        if args.json:
            import json
            print(json.dumps(result, indent=2, default=str))
        
        # Determine exit code
        if args.endpoint == "all":
            success = result.get("critical_checks_passed", False)
        else:
            success = result.get("success", False)
        
        return success
        
    except KeyboardInterrupt:
        logger.info("⚠️  Health check interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)