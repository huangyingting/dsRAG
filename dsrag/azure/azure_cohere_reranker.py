"""Azure Cohere Reranker implementation for dsRAG.

This module provides support for Cohere reranking models deployed on Azure.
"""

import os
from typing import Optional, List
from scipy.stats import beta

try:
    import cohere
except ImportError:
    raise ImportError(
        "Cohere package not found. Install with: pip install 'dsrag[cohere]'"
    )

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dsrag.reranker import Reranker


class AzureCohereReranker(Reranker):
    """
    Azure Cohere Reranker implementation.
    
    Uses Cohere reranking models deployed on Azure. Requires Azure endpoint
    and API key to be configured.
    """
    
    def __init__(
        self,
        model: str = "Cohere-rerank-v3.5",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Azure Cohere Reranker.
        
        Args:
            model: Cohere reranking model name (e.g., "Cohere-rerank-v3.5")
            azure_endpoint: Azure Cohere endpoint URL (falls back to AZURE_COHERE_ENDPOINT env var)
            api_key: Azure Cohere API key (falls back to AZURE_COHERE_API_KEY or CO_API_KEY env var)
        """
        self.model = model
        
        # Get credentials from parameters or environment
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_COHERE_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        
        if not self.azure_endpoint:
            raise ValueError(
                "Azure Cohere endpoint must be provided via azure_endpoint parameter "
                "or AZURE_COHERE_ENDPOINT environment variable"
            )
        
        if not self.api_key:
            raise ValueError(
                "Azure Cohere API key must be provided via api_key parameter "
                "or AZURE_COHERE_API_KEY (or CO_API_KEY) environment variable"
            )
        
        # Initialize Cohere client with Azure endpoint
        self.client = cohere.Client(
            api_key=self.api_key,
            base_url=self.azure_endpoint,
        )
    
    def transform(self, x):
        """
        Transformation function to map the absolute relevance value to a value 
        that is more uniformly distributed between 0 and 1.
        
        This is critical for RSE to work properly, because it utilizes the 
        absolute relevance values to calculate the similarity scores.
        """
        a, b = 0.4, 0.4  # These can be adjusted to change the distribution shape
        return beta.cdf(x, a, b)
    
    def rerank_search_results(self, query: str, search_results: list) -> list:
        """
        Use Azure Cohere Rerank API to rerank the search results.
        
        Args:
            query: The search query string
            search_results: List of search result dictionaries with metadata
            
        Returns:
            List of reranked search results with updated similarity scores
        """
        documents = []
        for result in search_results:
            documents.append(f"{result['metadata']['chunk_header']}\n\n{result['metadata']['chunk_text']}")
        
        reranked_results = self.client.rerank(
            model=self.model, 
            query=query, 
            documents=documents
        )
        results = reranked_results.results
        reranked_indices = [result.index for result in results]
        reranked_similarity_scores = [result.relevance_score for result in results]
        reranked_search_results = [search_results[i] for i in reranked_indices]
        
        for i, result in enumerate(reranked_search_results):
            result['similarity'] = self.transform(reranked_similarity_scores[i])
        
        return reranked_search_results
    
    def to_dict(self):
        """Serialize configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model,
            'azure_endpoint': self.azure_endpoint,
            'api_key': self.api_key,
        })
        return base_dict
