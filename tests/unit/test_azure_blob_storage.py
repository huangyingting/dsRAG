"""Unit tests for Azure Blob Storage FileSystem implementation."""

import os
import sys
import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Mock azure imports before importing AzureBlobStorage
sys.modules['azure.storage.blob'] = MagicMock()
sys.modules['azure.core.exceptions'] = MagicMock()

from dsrag.azure.blob_storage import AzureBlobStorage


class TestAzureBlobStorageInit(unittest.TestCase):
    """Test initialization of AzureBlobStorage."""
    
    @patch('dsrag.azure.blob_storage.BlobServiceClient')
    def test_init_with_connection_string(self, mock_blob_service):
        """Test initialization with connection string."""
        mock_client = MagicMock()
        mock_blob_service.from_connection_string.return_value = mock_client
        mock_container = MagicMock()
        mock_client.get_container_client.return_value = mock_container
        
        storage = AzureBlobStorage(
            base_path="/tmp/test",
            container_name="test-container",
            connection_string="test_connection_string"
        )
        
        self.assertEqual(storage.base_path, "/tmp/test")
        self.assertEqual(storage.container_name, "test-container")
        self.assertEqual(storage.connection_string, "test_connection_string")
        mock_blob_service.from_connection_string.assert_called_once_with("test_connection_string")
    
    @patch('dsrag.azure.blob_storage.BlobServiceClient')
    def test_init_with_account_credentials(self, mock_blob_service):
        """Test initialization with account name and key."""
        mock_client = MagicMock()
        mock_blob_service.return_value = mock_client
        mock_container = MagicMock()
        mock_client.get_container_client.return_value = mock_container
        
        storage = AzureBlobStorage(
            base_path="/tmp/test",
            container_name="test-container",
            account_name="testaccount",
            account_key="testkey"
        )
        
        self.assertEqual(storage.account_name, "testaccount")
        self.assertEqual(storage.account_key, "testkey")
        mock_blob_service.assert_called_once()
    
    @patch('dsrag.azure.blob_storage.BlobServiceClient')
    def test_init_without_credentials_raises_error(self, mock_blob_service):
        """Test that initialization without credentials raises ValueError."""
        with self.assertRaises(ValueError):
            AzureBlobStorage(
                base_path="/tmp/test",
                container_name="test-container"
            )


class TestAzureBlobStorageOperations(unittest.TestCase):
    """Test AzureBlobStorage operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('dsrag.azure.blob_storage.BlobServiceClient'):
            self.storage = AzureBlobStorage(
                base_path="/tmp/test",
                container_name="test-container",
                connection_string="test_connection_string"
            )
            self.storage.blob_service_client = MagicMock()
    
    def test_create_directory(self):
        """Test create_directory method."""
        # Mock delete_directory to avoid side effects
        with patch.object(self.storage, 'delete_directory'):
            self.storage.create_directory("kb1", "doc1")
            self.storage.delete_directory.assert_called_once_with("kb1", "doc1")
    
    def test_delete_directory(self):
        """Test delete_directory method."""
        mock_container = MagicMock()
        mock_blob1 = MagicMock()
        mock_blob1.name = "kb1/doc1/file1.jpg"
        mock_blob2 = MagicMock()
        mock_blob2.name = "kb1/doc1/file2.json"
        
        mock_container.list_blobs.return_value = [mock_blob1, mock_blob2]
        self.storage.blob_service_client.get_container_client.return_value = mock_container
        
        mock_blob_client1 = MagicMock()
        mock_blob_client2 = MagicMock()
        self.storage.blob_service_client.get_blob_client.side_effect = [
            mock_blob_client1, mock_blob_client2
        ]
        
        self.storage.delete_directory("kb1", "doc1")
        
        mock_container.list_blobs.assert_called_once_with(name_starts_with="kb1/doc1/")
        self.assertEqual(mock_blob_client1.delete_blob.call_count, 1)
        self.assertEqual(mock_blob_client2.delete_blob.call_count, 1)
    
    def test_delete_kb(self):
        """Test delete_kb method."""
        mock_container = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "kb1/file1.jpg"
        mock_container.list_blobs.return_value = [mock_blob]
        self.storage.blob_service_client.get_container_client.return_value = mock_container
        
        mock_blob_client = MagicMock()
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        self.storage.delete_kb("kb1")
        
        mock_container.list_blobs.assert_called_once_with(name_starts_with="kb1/")
        mock_blob_client.delete_blob.assert_called_once()
    
    def test_save_json(self):
        """Test save_json method."""
        mock_blob_client = MagicMock()
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        test_data = {"key": "value", "number": 123}
        self.storage.save_json("kb1", "doc1", "test.json", test_data)
        
        self.storage.blob_service_client.get_blob_client.assert_called_once_with(
            container="test-container",
            blob="kb1/doc1/test.json"
        )
        mock_blob_client.upload_blob.assert_called_once()
    
    def test_save_image(self):
        """Test save_image method."""
        mock_blob_client = MagicMock()
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        # Create a test image
        image = Image.new('RGB', (100, 100), color='red')
        
        self.storage.save_image("kb1", "doc1", "test.jpg", image)
        
        self.storage.blob_service_client.get_blob_client.assert_called_once_with(
            container="test-container",
            blob="kb1/doc1/test.jpg"
        )
        mock_blob_client.upload_blob.assert_called_once()
    
    @patch('builtins.open', create=True)
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_get_files(self, mock_makedirs, mock_exists, mock_open):
        """Test get_files method."""
        mock_exists.return_value = False
        mock_blob_client = MagicMock()
        mock_download = MagicMock()
        mock_download.readall.return_value = b"fake image data"
        mock_blob_client.download_blob.return_value = mock_download
        
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        result = self.storage.get_files("kb1", "doc1", 1, 2)
        
        # Should try to get 2 files (pages 1 and 2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all("page_" in path for path in result))
    
    def test_load_page_content(self):
        """Test load_page_content method."""
        mock_blob_client = MagicMock()
        mock_download = MagicMock()
        test_content = {"content": "Test page content"}
        mock_download.readall.return_value = json.dumps(test_content).encode('utf-8')
        mock_blob_client.download_blob.return_value = mock_download
        
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        result = self.storage.load_page_content("kb1", "doc1", 1)
        
        self.assertEqual(result, "Test page content")
    
    def test_load_data(self):
        """Test load_data method."""
        mock_blob_client = MagicMock()
        mock_download = MagicMock()
        test_data = {"key": "value", "items": [1, 2, 3]}
        mock_download.readall.return_value = json.dumps(test_data).encode('utf-8')
        mock_blob_client.download_blob.return_value = mock_download
        
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        result = self.storage.load_data("kb1", "doc1", "elements")
        
        self.assertEqual(result, test_data)
        self.storage.blob_service_client.get_blob_client.assert_called_once_with(
            container="test-container",
            blob="kb1/doc1/elements.json"
        )
    
    def test_to_dict(self):
        """Test to_dict serialization."""
        result = self.storage.to_dict()
        
        self.assertEqual(result['subclass_name'], 'AzureBlobStorage')
        self.assertEqual(result['base_path'], '/tmp/test')
        self.assertEqual(result['container_name'], 'test-container')
        self.assertIn('connection_string', result)


class TestAzureBlobStorageErrorHandling(unittest.TestCase):
    """Test error handling in AzureBlobStorage."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('dsrag.azure.blob_storage.BlobServiceClient'):
            self.storage = AzureBlobStorage(
                base_path="/tmp/test",
                container_name="test-container",
                connection_string="test_connection_string"
            )
            self.storage.blob_service_client = MagicMock()
    
    def test_log_error(self):
        """Test log_error method."""
        mock_blob_client = MagicMock()
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        error_data = {"error": "Test error", "code": 500}
        self.storage.log_error("kb1", "doc1", error_data)
        
        # Should create an error log blob
        mock_blob_client.upload_blob.assert_called_once()
    
    def test_save_page_content(self):
        """Test save_page_content method."""
        mock_blob_client = MagicMock()
        self.storage.blob_service_client.get_blob_client.return_value = mock_blob_client
        
        self.storage.save_page_content("kb1", "doc1", 1, "Test content")
        
        mock_blob_client.upload_blob.assert_called_once()
    
    def test_load_page_content_range(self):
        """Test load_page_content_range method."""
        with patch.object(self.storage, 'load_page_content') as mock_load:
            mock_load.side_effect = ["Content 1", "Content 2", "Content 3"]
            
            result = self.storage.load_page_content_range("kb1", "doc1", 1, 3)
            
            self.assertEqual(len(result), 3)
            self.assertEqual(result, ["Content 1", "Content 2", "Content 3"])
            self.assertEqual(mock_load.call_count, 3)


if __name__ == "__main__":
    unittest.main()
