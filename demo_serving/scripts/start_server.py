#!/usr/bin/env python3
"""
Server startup script with enhanced configuration and monitoring.
"""
import os
import sys
import argparse
import signal
import time
import subprocess
from pathlib import Path

# Add app to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core import configure_logging, get_logger, settings


class ServerManager:
    """Manages the MLflow serving server lifecycle."""
    
    def __init__(self):
        self.logger = get_logger("server_manager")
        self.server_process = None
        self.shutdown_requested = False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"📨 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.stop_server()
    
    def validate_environment(self):
        """Validate environment configuration."""
        self.logger.info("🔍 Validating environment configuration...")
        
        required_vars = [
            "MLFLOW_MODEL_URI",
            "MLFLOW_ARTIFACT_PATH",
            "S3_ENDPOINT_URL",
            "S3_ACCESS_KEY",
            "S3_SECRET_KEY",
            "S3_BUCKET_NAME"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        # Validate paths
        model_cache_dir = settings.model_cache_dir
        Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Environment validation passed")
        return True
    
    def check_dependencies(self):
        """Check if all required dependencies are available."""
        self.logger.info("🔍 Checking dependencies...")
        
        try:
            import mlflow
            import transformers
            import torch
            import fastapi
            import uvicorn
            import boto3
            import langchain
            
            self.logger.info("✅ All dependencies available")
            return True
            
        except ImportError as e:
            self.logger.error(f"❌ Missing dependency: {e}")
            return False
    
    def start_server(self, 
                    host: str = None,
                    port: int = None,
                    workers: int = None,
                    log_level: str = None,
                    reload: bool = False,
                    preload_model: bool = False):
        """Start the FastAPI server."""
        
        # Use provided values or fall back to settings
        host = host or settings.host
        port = port or settings.port
        workers = workers or settings.workers
        log_level = log_level or settings.log_level
        
        self.logger.info("🚀 Starting MLflow Model Serving Server")
        self.logger.info(f"📊 Configuration:")
        self.logger.info(f"  - Host: {host}")
        self.logger.info(f"  - Port: {port}")
        self.logger.info(f"  - Workers: {workers}")
        self.logger.info(f"  - Log Level: {log_level}")
        self.logger.info(f"  - Reload: {reload}")
        self.logger.info(f"  - Preload Model: {preload_model}")
        
        # Set preload model environment variable
        if preload_model:
            os.environ["PRELOAD_MODEL"] = "true"
        
        # Build uvicorn command
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", host,
            "--port", str(port),
            "--log-level", log_level
        ]
        
        if reload:
            cmd.append("--reload")
        else:
            cmd.extend(["--workers", str(workers)])
        
        try:
            self.logger.info(f"🔧 Starting server with command: {' '.join(cmd)}")
            
            # Start the server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor server output
            self._monitor_server()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start server: {e}")
            return False
        
        return True
    
    def _monitor_server(self):
        """Monitor server process and handle output."""
        self.logger.info("📡 Monitoring server process...")
        
        try:
            for line in iter(self.server_process.stdout.readline, ''):
                if self.shutdown_requested:
                    break
                
                line = line.strip()
                if line:
                    # Forward server output to our logger
                    if "ERROR" in line or "CRITICAL" in line:
                        self.logger.error(f"SERVER: {line}")
                    elif "WARNING" in line:
                        self.logger.warning(f"SERVER: {line}")
                    else:
                        self.logger.info(f"SERVER: {line}")
            
            # Wait for process to complete
            if self.server_process:
                self.server_process.wait()
                
        except KeyboardInterrupt:
            self.logger.info("⚠️  Server monitoring interrupted")
        except Exception as e:
            self.logger.error(f"❌ Server monitoring error: {e}")
    
    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            self.logger.info("🛑 Stopping server process...")
            
            try:
                # Send SIGTERM for graceful shutdown
                self.server_process.terminate()
                
                # Wait for graceful shutdown (up to 30 seconds)
                try:
                    self.server_process.wait(timeout=30)
                    self.logger.info("✅ Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning("⏰ Graceful shutdown timeout, forcing kill...")
                    self.server_process.kill()
                    self.server_process.wait()
                    self.logger.info("✅ Server process killed")
                
            except Exception as e:
                self.logger.error(f"❌ Error stopping server: {e}")
            
            self.server_process = None
    
    def run(self, **kwargs):
        """Run the server with full lifecycle management."""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Validate environment
            if not self.validate_environment():
                return False
            
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Start server
            if not self.start_server(**kwargs):
                return False
            
            self.logger.info("✅ Server startup completed")
            return True
            
        except KeyboardInterrupt:
            self.logger.info("⚠️  Server startup interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"❌ Server startup failed: {e}")
            return False
        finally:
            # Cleanup
            self.stop_server()


def main():
    """Main function for server startup."""
    parser = argparse.ArgumentParser(
        description="Start MLflow model serving server"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Server host (default from config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default from config)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers (default from config)"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default=None,
        help="Log level (default from config)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)"
    )
    parser.add_argument(
        "--preload-model",
        action="store_true",
        help="Pre-load model on startup"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.log_level or settings.log_level)
    
    # Create server manager
    server_manager = ServerManager()
    
    # Start server
    success = server_manager.run(
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=args.reload,
        preload_model=args.preload_model
    )
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)