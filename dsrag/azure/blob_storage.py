"""Azure Blob Storage implementation for dsRAG file system."""

import os
import io
import json
from typing import List, Optional
from datetime import datetime

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    raise ImportError(
        "Azure storage dependencies not found. Install with: pip install 'dsrag[azure-storage]'"
    )

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.file_parsing.file_system import FileSystem


class AzureBlobStorage(FileSystem):
    """
    Uses Azure Blob Storage to store and retrieve page image files and other data.
    """
    
    def __init__(
        self,
        base_path: str,
        container_name: str,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
    ):
        """
        Initialize Azure Blob Storage file system.
        
        Args:
            base_path: Local base path for temporary file downloads
            container_name: Name of the Azure Blob Storage container
            connection_string: Azure Storage connection string (preferred method)
            account_name: Azure Storage account name (alternative to connection_string)
            account_key: Azure Storage account key (alternative to connection_string)
        """
        super().__init__(base_path)
        self.container_name = container_name
        self.connection_string = connection_string
        self.account_name = account_name
        self.account_key = account_key
        
        # Initialize blob service client
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name and account_key:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=account_key
            )
        else:
            raise ValueError(
                "Either connection_string or both account_name and account_key must be provided"
            )
        
        # Ensure container exists
        self._ensure_container_exists()
    
    def _ensure_container_exists(self) -> None:
        """Create container if it doesn't exist."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
        except ResourceNotFoundError:
            self.blob_service_client.create_container(self.container_name)
    
    def _get_blob_client(self, blob_path: str):
        """Get a blob client for the specified path."""
        return self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_path
        )
    
    def create_directory(self, kb_id: str, doc_id: str) -> None:
        """
        This function is not needed for Azure Blob Storage as directories are virtual.
        We'll just ensure any existing files in this path are deleted.
        """
        prefix = f"{kb_id}/{doc_id}/"
        self.delete_directory(kb_id, doc_id)
    
    def delete_directory(self, kb_id: str, doc_id: str) -> None:
        """Delete all blobs in the specified directory."""
        prefix = f"{kb_id}/{doc_id}/"
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        try:
            blob_list = container_client.list_blobs(name_starts_with=prefix)
            for blob in blob_list:
                blob_client = self._get_blob_client(blob.name)
                blob_client.delete_blob()
        except Exception as e:
            print(f"Error deleting directory {prefix}: {e}")
    
    def delete_kb(self, kb_id: str) -> None:
        """Delete all blobs for a knowledge base."""
        prefix = f"{kb_id}/"
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        try:
            blob_list = container_client.list_blobs(name_starts_with=prefix)
            for blob in blob_list:
                blob_client = self._get_blob_client(blob.name)
                blob_client.delete_blob()
        except Exception as e:
            print(f"Error deleting knowledge base {kb_id}: {e}")
    
    def save_json(self, kb_id: str, doc_id: str, file_name: str, file: dict) -> None:
        """Save JSON data to Azure Blob Storage."""
        blob_path = f"{kb_id}/{doc_id}/{file_name}"
        json_data = json.dumps(file, indent=2)
        
        blob_client = self._get_blob_client(blob_path)
        try:
            content_settings = ContentSettings(content_type='application/json')
            blob_client.upload_blob(
                json_data,
                overwrite=True,
                content_settings=content_settings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upload JSON to Azure Blob Storage: {e}") from e
    
    def save_image(self, kb_id: str, doc_id: str, file_name: str, image: any) -> None:
        """Save image to Azure Blob Storage."""
        blob_path = f"{kb_id}/{doc_id}/{file_name}"
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        blob_client = self._get_blob_client(blob_path)
        try:
            content_settings = ContentSettings(content_type='image/jpeg')
            blob_client.upload_blob(
                buffer,
                overwrite=True,
                content_settings=content_settings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upload image to Azure Blob Storage: {e}") from e
    
    def get_files(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> List[str]:
        """
        Download files from Azure Blob Storage and return local paths.
        
        Args:
            kb_id: Knowledge base ID
            doc_id: Document ID
            page_start: Starting page number
            page_end: Ending page number (inclusive)
        
        Returns:
            List of local file paths
        """
        if page_start is None or page_end is None:
            return []
        
        file_paths = []
        output_folder = os.path.join(self.base_path, kb_id, doc_id)
        
        # Create local directory if needed
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except FileExistsError:
                pass
        
        # Try multiple extensions for backward compatibility
        for i in range(page_start, page_end + 1):
            found_file = False
            for ext in ['.jpg', '.jpeg', '.png']:
                blob_path = f"{kb_id}/{doc_id}/page_{i}{ext}"
                local_path = os.path.join(self.base_path, blob_path)
                
                blob_client = self._get_blob_client(blob_path)
                try:
                    # Download the blob
                    with open(local_path, 'wb') as f:
                        download_stream = blob_client.download_blob()
                        f.write(download_stream.readall())
                    file_paths.append(local_path)
                    found_file = True
                    break
                except ResourceNotFoundError:
                    continue
                except Exception as e:
                    print(f"Error downloading blob {blob_path}: {e}")
                    continue
            
            if not found_file:
                print(f"Warning: No image file found for page {i} in Azure Blob Storage")
        
        return file_paths
    
    def get_all_jpg_files(self, kb_id: str, doc_id: str) -> List[str]:
        """
        Get all image files from Azure Blob Storage and download them locally.
        
        Returns:
            Sorted list of local file paths
        """
        prefix = f"{kb_id}/{doc_id}/"
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        try:
            blob_list = container_client.list_blobs(name_starts_with=prefix)
            image_files = [
                blob.name for blob in blob_list
                if blob.name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if not image_files:
                return []
            
            # Create local directory
            output_folder = os.path.join(self.base_path, kb_id, doc_id)
            os.makedirs(output_folder, exist_ok=True)
            
            # Download each file
            local_file_paths = []
            for blob_name in image_files:
                local_path = os.path.join(self.base_path, blob_name)
                blob_client = self._get_blob_client(blob_name)
                
                try:
                    with open(local_path, 'wb') as f:
                        download_stream = blob_client.download_blob()
                        f.write(download_stream.readall())
                    local_file_paths.append(local_path)
                except Exception as e:
                    print(f"Error downloading blob {blob_name}: {e}")
                    continue
            
            # Sort by page number
            local_file_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            return local_file_paths
            
        except Exception as e:
            print(f"Error listing/downloading files from Azure Blob Storage: {e}")
            return []
    
    def log_error(self, kb_id: str, doc_id: str, error: dict) -> None:
        """Log error to Azure Blob Storage."""
        timestamp = datetime.now().isoformat()
        error_data = {
            'kb_id': kb_id,
            'doc_id': doc_id,
            'error': error,
            'timestamp': timestamp
        }
        
        # Save error log as JSON
        blob_path = f"{kb_id}/{doc_id}/errors/{timestamp}.json"
        blob_client = self._get_blob_client(blob_path)
        
        try:
            json_data = json.dumps(error_data, indent=2)
            content_settings = ContentSettings(content_type='application/json')
            blob_client.upload_blob(
                json_data,
                overwrite=True,
                content_settings=content_settings
            )
        except Exception as e:
            print(f"Failed to log error to Azure Blob Storage: {e}")
    
    def save_page_content(self, kb_id: str, doc_id: str, page_number: int, content: str) -> None:
        """Save page content to Azure Blob Storage."""
        blob_path = f"{kb_id}/{doc_id}/page_content_{page_number}.json"
        data = json.dumps({"content": content})
        
        blob_client = self._get_blob_client(blob_path)
        try:
            content_settings = ContentSettings(content_type='application/json')
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=content_settings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upload page content to Azure Blob Storage: {e}") from e
    
    def load_page_content(self, kb_id: str, doc_id: str, page_number: int) -> Optional[str]:
        """Load page content from Azure Blob Storage."""
        blob_path = f"{kb_id}/{doc_id}/page_content_{page_number}.json"
        blob_client = self._get_blob_client(blob_path)
        
        try:
            download_stream = blob_client.download_blob()
            data = json.loads(download_stream.readall().decode('utf-8'))
            return data["content"]
        except ResourceNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading page content from Azure Blob Storage: {e}")
            return None
    
    def load_page_content_range(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> list[str]:
        """Load page content for a range of pages."""
        page_contents = []
        for page_num in range(page_start, page_end + 1):
            content = self.load_page_content(kb_id, doc_id, page_num)
            if content is not None:
                page_contents.append(content)
        return page_contents
    
    def load_data(self, kb_id: str, doc_id: str, data_name: str) -> Optional[dict]:
        """Load JSON data from Azure Blob Storage."""
        blob_path = f"{kb_id}/{doc_id}/{data_name}.json"
        blob_client = self._get_blob_client(blob_path)
        
        try:
            download_stream = blob_client.download_blob()
            return json.loads(download_stream.readall().decode('utf-8'))
        except ResourceNotFoundError:
            print(f"Blob not found: {blob_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from blob {blob_path}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error loading data from Azure Blob Storage: {str(e)}")
            return None
    
    def to_dict(self):
        """Serialize configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "container_name": self.container_name,
            "connection_string": self.connection_string,
            "account_name": self.account_name,
            "account_key": self.account_key,
        })
        return base_dict
