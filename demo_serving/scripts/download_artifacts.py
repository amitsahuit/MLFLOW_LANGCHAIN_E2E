#!/usr/bin/env python3
"""
Standalone script to download MLflow artifacts from S3.
"""
import os
import sys
import argparse
import time
from pathlib import Path

# Add app to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core import configure_logging, get_logger, settings
from app.services.s3_client import S3Client


def main():
    """Main function to download MLflow artifacts."""
    parser = argparse.ArgumentParser(
        description="Download MLflow artifacts from S3"
    )
    parser.add_argument(
        "--artifact-path",
        default=None,
        help="S3 artifact path (overrides config)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache directory (overrides config)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh from S3"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing cache"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.log_level)
    logger = get_logger("download_artifacts")
    
    logger.info("🚀 Starting MLflow artifact download")
    
    try:
        # Get configuration
        artifact_path = args.artifact_path or settings.mlflow_artifact_path
        cache_dir = args.cache_dir or settings.model_cache_dir
        
        logger.info(f"📊 Configuration:")
        logger.info(f"  - Artifact path: {artifact_path}")
        logger.info(f"  - Cache directory: {cache_dir}")
        logger.info(f"  - S3 endpoint: {settings.s3_endpoint_url}")
        logger.info(f"  - S3 bucket: {settings.s3_bucket_name}")
        
        # Create S3 client
        s3_client = S3Client()
        
        # Create cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        if args.verify_only:
            logger.info("🔍 Verifying existing cache...")
            
            # Extract S3 prefix from artifact path
            if artifact_path.startswith('s3://'):
                s3_path = artifact_path[5:]
                bucket_name, s3_prefix = s3_path.split('/', 1)
            else:
                raise ValueError(f"Invalid S3 path: {artifact_path}")
            
            # Check cache integrity
            cache_valid = s3_client.verify_cache_integrity(
                local_dir=cache_dir,
                s3_prefix=s3_prefix
            )
            
            if cache_valid:
                logger.info("✅ Cache verification passed")
                return True
            else:
                logger.warning("❌ Cache verification failed")
                return False
        
        # Download artifacts
        start_time = time.time()
        
        local_artifact_dir = s3_client.download_mlflow_artifacts(
            artifact_path=artifact_path,
            local_cache_dir=cache_dir
        )
        
        download_time = time.time() - start_time
        
        # Get download statistics
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(local_artifact_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        logger.info("✅ Download completed successfully")
        logger.info(f"📊 Statistics:")
        logger.info(f"  - Files downloaded: {file_count}")
        logger.info(f"  - Total size: {total_size / (1024*1024):.2f} MB")
        logger.info(f"  - Download time: {download_time:.2f} seconds")
        logger.info(f"  - Average speed: {(total_size / (1024*1024)) / download_time:.2f} MB/s")
        logger.info(f"  - Local path: {local_artifact_dir}")
        
        return True
        
    except KeyboardInterrupt:
        logger.info("⚠️  Download interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)