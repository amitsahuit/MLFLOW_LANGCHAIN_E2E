"""
S3 client service for downloading MLflow artifacts.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from ..core import get_logger, settings


class S3Client:
    """S3 client for downloading MLflow artifacts with retry logic."""
    
    def __init__(self):
        self.logger = get_logger("s3_client")
        self.settings = settings
        self._s3_client = None
        self._initialize_client(test_connection=False)  # Defer connection test
    
    def _initialize_client(self, test_connection: bool = True) -> None:
        """Initialize the S3 client with configuration."""
        try:
            config = Config(
                signature_version='s3v4',
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                },
                connect_timeout=10,
                read_timeout=30
            )
            
            self._s3_client = boto3.client(
                's3',
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key,
                aws_secret_access_key=self.settings.s3_secret_key,
                region_name=self.settings.s3_region,
                config=config,
                verify=False  # For local MinIO setups
            )
            
            # Test connection only if requested
            if test_connection:
                self._test_connection()
                self.logger.info("S3 client initialized and tested successfully")
            else:
                self.logger.info("S3 client initialized (connection test deferred)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            if test_connection:
                raise
            else:
                self.logger.warning("S3 client initialization failed, will retry on first use")
    
    def _test_connection(self, retries: int = 3) -> None:
        """Test S3 connection by checking bucket existence."""
        for attempt in range(retries):
            try:
                self._s3_client.head_bucket(Bucket=self.settings.s3_bucket_name)
                self.logger.info(f"Successfully connected to S3 bucket: {self.settings.s3_bucket_name}")
                return
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    # Try to create bucket if it doesn't exist
                    try:
                        self.logger.info(f"Bucket not found, attempting to create: {self.settings.s3_bucket_name}")
                        self._s3_client.create_bucket(Bucket=self.settings.s3_bucket_name)
                        self.logger.info(f"Successfully created S3 bucket: {self.settings.s3_bucket_name}")
                        return
                    except Exception as create_err:
                        self.logger.warning(f"Failed to create bucket: {create_err}")
                        if attempt < retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            raise
                else:
                    self.logger.error(f"S3 connection test failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        raise
            except NoCredentialsError:
                self.logger.error("S3 credentials not found or invalid")
                raise
            except Exception as e:
                self.logger.error(f"S3 connection test failed with unexpected error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise

    def _ensure_client_ready(self) -> None:
        """Ensure S3 client is initialized and ready."""
        if self._s3_client is None:
            self.logger.info("S3 client not initialized, initializing now...")
            self._initialize_client(test_connection=True)
        else:
            # Test connection if we haven't done so yet
            try:
                self._s3_client.head_bucket(Bucket=self.settings.s3_bucket_name)
            except Exception:
                self.logger.info("S3 connection test failed, reinitializing...")
                self._initialize_client(test_connection=True)
    
    def list_objects(self, prefix: str) -> List[Dict]:
        """List objects in S3 with given prefix."""
        self._ensure_client_ready()
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self.settings.s3_bucket_name,
                Prefix=prefix
            )
            
            objects = response.get('Contents', [])
            self.logger.info(f"Found {len(objects)} objects with prefix: {prefix}")
            return objects
            
        except ClientError as e:
            self.logger.error(f"Failed to list objects with prefix {prefix}: {e}")
            raise
    
    def download_file(self, s3_key: str, local_path: str, max_retries: int = 3) -> bool:
        """Download a single file from S3 with retry logic."""
        self._ensure_client_ready()
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading {s3_key} to {local_path} (attempt {attempt + 1})")
                
                self._s3_client.download_file(
                    self.settings.s3_bucket_name,
                    s3_key,
                    str(local_path)
                )
                
                # Verify file was downloaded
                if local_path.exists() and local_path.stat().st_size > 0:
                    self.logger.info(f"Successfully downloaded {s3_key}")
                    return True
                else:
                    self.logger.warning(f"Downloaded file {local_path} is empty or doesn't exist")
                    
            except ClientError as e:
                self.logger.error(f"Failed to download {s3_key} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error downloading {s3_key}: {e}")
                raise
        
        return False
    
    def download_directory(self, s3_prefix: str, local_dir: str, 
                          exclude_patterns: Optional[List[str]] = None) -> bool:
        """Download entire directory from S3."""
        exclude_patterns = exclude_patterns or []
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # List all objects with the prefix
            objects = self.list_objects(s3_prefix)
            
            if not objects:
                self.logger.warning(f"No objects found with prefix: {s3_prefix}")
                return False
            
            downloaded_count = 0
            total_size = 0
            
            for obj in objects:
                s3_key = obj['Key']
                
                # Skip directory markers (keys ending with '/')
                if s3_key.endswith('/'):
                    self.logger.debug(f"Skipping directory marker: {s3_key}")
                    continue
                
                # Skip if matches exclude patterns
                if any(pattern in s3_key for pattern in exclude_patterns):
                    self.logger.debug(f"Skipping {s3_key} (matches exclude pattern)")
                    continue
                
                # Calculate local path
                relative_path = s3_key.replace(s3_prefix, '').lstrip('/')
                if not relative_path:  # Skip if no relative path
                    continue
                    
                local_file_path = local_dir / relative_path
                
                # Download file
                if self.download_file(s3_key, str(local_file_path)):
                    downloaded_count += 1
                    total_size += obj['Size']
                    self.logger.debug(f"Downloaded {s3_key} -> {local_file_path}")
            
            self.logger.info(
                f"Successfully downloaded {downloaded_count} files "
                f"({total_size} bytes) from {s3_prefix} to {local_dir}"
            )
            return downloaded_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to download directory {s3_prefix}: {e}")
            # Clean up partial downloads
            if local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)
            raise
    
    def download_mlflow_artifacts(self, artifact_path: str, local_cache_dir: str) -> str:
        """Download MLflow artifacts from S3 to local cache."""
        # Parse S3 path
        if not artifact_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {artifact_path}")
        
        # Extract bucket and prefix from s3://bucket/prefix
        s3_path = artifact_path[5:]  # Remove 's3://'
        bucket_name, s3_prefix = s3_path.split('/', 1)
        
        # Ensure we're using the correct bucket
        if bucket_name != self.settings.s3_bucket_name:
            self.logger.warning(
                f"Artifact bucket ({bucket_name}) differs from configured bucket "
                f"({self.settings.s3_bucket_name}). Using configured bucket."
            )
        
        # Create local directory
        local_artifact_dir = Path(local_cache_dir) / "artifacts"
        local_artifact_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Downloading MLflow artifacts from {artifact_path}")
        
        # Download all artifacts
        success = self.download_directory(
            s3_prefix=s3_prefix,
            local_dir=str(local_artifact_dir),
            exclude_patterns=['.DS_Store', '__pycache__', '*.pyc']
        )
        
        if not success:
            raise RuntimeError(f"Failed to download artifacts from {artifact_path}")
        
        self.logger.info(f"MLflow artifacts downloaded to {local_artifact_dir}")
        return str(local_artifact_dir)
    
    def get_object_metadata(self, s3_key: str) -> Dict:
        """Get metadata for an S3 object."""
        try:
            response = self._s3_client.head_object(
                Bucket=self.settings.s3_bucket_name,
                Key=s3_key
            )
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'etag': response['ETag'],
                'content_type': response.get('ContentType', 'unknown')
            }
        except ClientError as e:
            self.logger.error(f"Failed to get metadata for {s3_key}: {e}")
            raise
    
    def verify_cache_integrity(self, local_dir: str, s3_prefix: str) -> bool:
        """Verify that local cache matches S3 contents."""
        try:
            local_dir = Path(local_dir)
            if not local_dir.exists():
                return False
            
            # Get S3 objects
            s3_objects = self.list_objects(s3_prefix)
            s3_files = {obj['Key'].replace(s3_prefix, '').lstrip('/'): obj 
                       for obj in s3_objects if not obj['Key'].endswith('/')}
            
            # Get local files
            local_files = {}
            for local_file in local_dir.rglob('*'):
                if local_file.is_file():
                    relative_path = str(local_file.relative_to(local_dir))
                    local_files[relative_path] = local_file
            
            # Compare file lists
            if set(s3_files.keys()) != set(local_files.keys()):
                self.logger.warning("File lists don't match between S3 and local cache")
                return False
            
            # Compare file sizes
            for file_path, s3_obj in s3_files.items():
                local_file = local_files[file_path]
                if local_file.stat().st_size != s3_obj['Size']:
                    self.logger.warning(f"Size mismatch for {file_path}")
                    return False
            
            self.logger.info("Cache integrity verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache integrity verification failed: {e}")
            return False