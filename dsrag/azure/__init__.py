"""Azure integrations for dsRAG.

This module provides Azure-specific implementations for dsRAG, including:
- Azure Blob Storage for file system storage
- Azure OpenAI for chat and embedding models
- Azure OpenAI VLM for vision-language models
"""

from .blob_storage import AzureBlobStorage
from .azure_openai_chat import AzureOpenAIChatAPI
from .azure_openai_embedding import AzureOpenAIEmbedding
from .azure_openai_vlm import AzureOpenAIVLM

__all__ = [
    "AzureBlobStorage",
    "AzureOpenAIChatAPI",
    "AzureOpenAIEmbedding",
    "AzureOpenAIVLM",
]
