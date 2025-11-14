"""Azure integrations for dsRAG.

This module provides Azure-specific implementations for dsRAG, including:
- Azure Blob Storage for file system storage
- Azure OpenAI for chat and embedding models
"""

from .blob_storage import AzureBlobStorage
from .azure_openai_chat import AzureOpenAIChatAPI
from .azure_openai_embedding import AzureOpenAIEmbedding

__all__ = [
    "AzureBlobStorage",
    "AzureOpenAIChatAPI",
    "AzureOpenAIEmbedding",
]
