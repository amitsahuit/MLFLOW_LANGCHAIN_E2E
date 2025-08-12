"""
Tests for S3 client functionality.
"""
import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError

from app.services.s3_client import S3Client


@pytest.mark.unit
class TestS3Client:
    """Test S3Client functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('app.services.s3_client.boto3.client')
    def test_s3_client_initialization_success(self, mock_boto_client):
        """Test successful S3 client initialization."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        
        s3_client = S3Client()
        
        assert s3_client._s3_client == mock_client
        mock_client.head_bucket.assert_called_once()
    
    @patch('app.services.s3_client.boto3.client')
    def test_s3_client_initialization_connection_error(self, mock_boto_client):
        """Test S3 client initialization with connection error."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.side_effect = ClientError(
            error_response={'Error': {'Code': '404'}},
            operation_name='HeadBucket'
        )
        
        with pytest.raises(ClientError):
            S3Client()
    
    @patch('app.services.s3_client.boto3.client')
    def test_s3_client_no_credentials(self, mock_boto_client):
        """Test S3 client initialization with no credentials."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.side_effect = NoCredentialsError()
        
        with pytest.raises(NoCredentialsError):
            S3Client()
    
    @patch('app.services.s3_client.boto3.client')
    def test_list_objects_success(self, mock_boto_client):
        """Test successful object listing."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test/file1.txt', 'Size': 100},
                {'Key': 'test/file2.txt', 'Size': 200}
            ]
        }
        
        s3_client = S3Client()
        objects = s3_client.list_objects('test/')
        
        assert len(objects) == 2
        assert objects[0]['Key'] == 'test/file1.txt'
        assert objects[1]['Key'] == 'test/file2.txt'
    
    @patch('app.services.s3_client.boto3.client')
    def test_list_objects_empty(self, mock_boto_client):
        """Test object listing with no objects."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {}
        
        s3_client = S3Client()
        objects = s3_client.list_objects('empty/')
        
        assert len(objects) == 0
    
    @patch('app.services.s3_client.boto3.client')
    def test_download_file_success(self, mock_boto_client):
        """Test successful file download."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.download_file.return_value = None
        
        # Create a test file to simulate successful download
        test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        
        def mock_download_file(bucket, key, local_path):
            with open(local_path, 'w') as f:
                f.write("test content")
        
        mock_client.download_file.side_effect = mock_download_file
        
        s3_client = S3Client()
        result = s3_client.download_file('test/file.txt', test_file_path)
        
        assert result is True
        assert os.path.exists(test_file_path)
        with open(test_file_path, 'r') as f:
            assert f.read() == "test content"
    
    @patch('app.services.s3_client.boto3.client')
    def test_download_file_retry_logic(self, mock_boto_client):
        """Test file download retry logic."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        
        # First two attempts fail, third succeeds
        test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        
        call_count = 0
        def mock_download_file(bucket, key, local_path):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ClientError(
                    error_response={'Error': {'Code': 'ServiceUnavailable'}},
                    operation_name='GetObject'
                )
            with open(local_path, 'w') as f:
                f.write("test content")
        
        mock_client.download_file.side_effect = mock_download_file
        
        s3_client = S3Client()
        result = s3_client.download_file('test/file.txt', test_file_path, max_retries=3)
        
        assert result is True
        assert call_count == 3
        assert os.path.exists(test_file_path)
    
    @patch('app.services.s3_client.boto3.client')
    def test_download_file_max_retries_exceeded(self, mock_boto_client):
        """Test file download when max retries exceeded."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.download_file.side_effect = ClientError(
            error_response={'Error': {'Code': 'ServiceUnavailable'}},
            operation_name='GetObject'
        )
        
        s3_client = S3Client()
        test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        
        with pytest.raises(ClientError):
            s3_client.download_file('test/file.txt', test_file_path, max_retries=2)
    
    @patch('app.services.s3_client.boto3.client')
    def test_download_directory_success(self, mock_boto_client):
        """Test successful directory download."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test/dir/file1.txt', 'Size': 100},
                {'Key': 'test/dir/file2.txt', 'Size': 200},
                {'Key': 'test/dir/subdir/file3.txt', 'Size': 150}
            ]
        }
        
        downloaded_files = []
        def mock_download_file(bucket, key, local_path):
            downloaded_files.append((key, local_path))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                f.write(f"content of {key}")
        
        mock_client.download_file.side_effect = mock_download_file
        
        s3_client = S3Client()
        local_dir = os.path.join(self.temp_dir, "download_test")
        result = s3_client.download_directory('test/dir/', local_dir)
        
        assert result is True
        assert len(downloaded_files) == 3
        
        # Check that files were created in correct locations
        assert os.path.exists(os.path.join(local_dir, "file1.txt"))
        assert os.path.exists(os.path.join(local_dir, "file2.txt"))
        assert os.path.exists(os.path.join(local_dir, "subdir", "file3.txt"))
    
    @patch('app.services.s3_client.boto3.client')
    def test_download_directory_with_excludes(self, mock_boto_client):
        """Test directory download with exclude patterns."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test/dir/file1.txt', 'Size': 100},
                {'Key': 'test/dir/.DS_Store', 'Size': 50},
                {'Key': 'test/dir/__pycache__/cache.pyc', 'Size': 75}
            ]
        }
        
        downloaded_files = []
        def mock_download_file(bucket, key, local_path):
            downloaded_files.append(key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                f.write(f"content of {key}")
        
        mock_client.download_file.side_effect = mock_download_file
        
        s3_client = S3Client()
        local_dir = os.path.join(self.temp_dir, "download_test")
        result = s3_client.download_directory(
            'test/dir/', 
            local_dir,
            exclude_patterns=['.DS_Store', '__pycache__', '*.pyc']
        )
        
        assert result is True
        # Only file1.txt should be downloaded
        assert len(downloaded_files) == 1
        assert 'test/dir/file1.txt' in downloaded_files
    
    @patch('app.services.s3_client.boto3.client')
    def test_download_mlflow_artifacts(self, mock_boto_client):
        """Test MLflow artifacts download."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'mlartifacts/1/model/MLmodel', 'Size': 1000},
                {'Key': 'mlartifacts/1/model/model.pkl', 'Size': 5000}
            ]
        }
        
        def mock_download_file(bucket, key, local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                f.write(f"content of {key}")
        
        mock_client.download_file.side_effect = mock_download_file
        
        s3_client = S3Client()
        local_cache_dir = os.path.join(self.temp_dir, "cache")
        
        result_dir = s3_client.download_mlflow_artifacts(
            's3://test-bucket/mlartifacts/1',
            local_cache_dir
        )
        
        assert result_dir.endswith("artifacts")
        assert os.path.exists(result_dir)
    
    @patch('app.services.s3_client.boto3.client')
    def test_get_object_metadata(self, mock_boto_client):
        """Test getting object metadata."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.head_object.return_value = {
            'ContentLength': 1000,
            'LastModified': 'test_date',
            'ETag': 'test_etag',
            'ContentType': 'text/plain'
        }
        
        s3_client = S3Client()
        metadata = s3_client.get_object_metadata('test/file.txt')
        
        assert metadata['size'] == 1000
        assert metadata['last_modified'] == 'test_date'
        assert metadata['etag'] == 'test_etag'
        assert metadata['content_type'] == 'text/plain'
    
    @patch('app.services.s3_client.boto3.client')
    def test_verify_cache_integrity_success(self, mock_boto_client):
        """Test successful cache integrity verification."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test/file1.txt', 'Size': 100},
                {'Key': 'test/file2.txt', 'Size': 200}
            ]
        }
        
        # Create matching local files
        local_dir = os.path.join(self.temp_dir, "verify_test")
        os.makedirs(local_dir, exist_ok=True)
        
        with open(os.path.join(local_dir, "file1.txt"), 'w') as f:
            f.write("x" * 100)  # 100 bytes
        with open(os.path.join(local_dir, "file2.txt"), 'w') as f:
            f.write("x" * 200)  # 200 bytes
        
        s3_client = S3Client()
        result = s3_client.verify_cache_integrity(local_dir, 'test/')
        
        assert result is True
    
    @patch('app.services.s3_client.boto3.client')
    def test_verify_cache_integrity_size_mismatch(self, mock_boto_client):
        """Test cache integrity verification with size mismatch."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test/file1.txt', 'Size': 100}
            ]
        }
        
        # Create local file with different size
        local_dir = os.path.join(self.temp_dir, "verify_test")
        os.makedirs(local_dir, exist_ok=True)
        
        with open(os.path.join(local_dir, "file1.txt"), 'w') as f:
            f.write("x" * 50)  # Different size (50 vs 100)
        
        s3_client = S3Client()
        result = s3_client.verify_cache_integrity(local_dir, 'test/')
        
        assert result is False


@pytest.mark.s3
class TestS3ClientIntegration:
    """Integration tests requiring actual S3 connectivity."""
    
    def test_real_s3_connection(self):
        """Test connection to real S3/MinIO (requires setup)."""
        pytest.skip("Requires actual S3/MinIO setup")
    
    def test_real_s3_download(self):
        """Test downloading from real S3/MinIO (requires setup)."""
        pytest.skip("Requires actual S3/MinIO setup")