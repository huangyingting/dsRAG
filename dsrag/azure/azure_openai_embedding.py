"""Azure OpenAI Embedding implementation for dsRAG."""

import os
from typing import Optional, List

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError(
        "OpenAI package not found. Install with: pip install 'dsrag[openai]'"
    )

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dsrag.embedding import Embedding
from dsrag.database.vector.types import Vector


class AzureOpenAIEmbedding(Embedding):
    """
    Azure OpenAI Embedding implementation.
    
    Uses Azure OpenAI Service for text embeddings. Requires Azure OpenAI endpoint,
    API key, and deployment name to be configured.
    """
    
    def __init__(
        self,
        deployment_name: str,
        dimension: int = 1536,
        api_version: str = "2024-02-15-preview",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Azure OpenAI Embedding.
        
        Args:
            deployment_name: Name of the Azure OpenAI embedding deployment
            dimension: Embedding dimension (depends on the model used)
            api_version: Azure OpenAI API version
            azure_endpoint: Azure OpenAI endpoint URL (falls back to AZURE_OPENAI_ENDPOINT env var)
            api_key: Azure OpenAI API key (falls back to AZURE_OPENAI_API_KEY env var)
        """
        super().__init__(dimension)
        self.deployment_name = deployment_name
        self.api_version = api_version
        
        # Get credentials from parameters or environment
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint must be provided via azure_endpoint parameter "
                "or AZURE_OPENAI_ENDPOINT environment variable"
            )
        
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key must be provided via api_key parameter "
                "or AZURE_OPENAI_API_KEY environment variable"
            )
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )
    
    def get_embeddings(self, text: List[str], input_type: Optional[str] = None) -> List[Vector]:
        """
        Generate embeddings for text using Azure OpenAI.
        
        Args:
            text: Text or list of texts to embed
            input_type: Optional input type hint (not used by Azure OpenAI)
        
        Returns:
            Embedding vector(s)
        """
        # Ensure text is a list
        texts = [text] if isinstance(text, str) else text
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment_name,  # In Azure, this is the deployment name
        )
        
        embeddings = [embedding_item.embedding for embedding_item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings
    
    def to_dict(self):
        """Serialize configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'deployment_name': self.deployment_name,
            'api_version': self.api_version,
            'azure_endpoint': self.azure_endpoint,
            'api_key': self.api_key,
        })
        return base_dict
