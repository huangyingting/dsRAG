"""Example of using Azure Cohere Reranker with dsRAG.

This example demonstrates how to use Cohere reranking models deployed on Azure
with the dsRAG knowledge base.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from dsrag.knowledge_base import KnowledgeBase
from dsrag.azure import (
    AzureBlobStorage,
    AzureOpenAIChatAPI,
    AzureOpenAIEmbedding,
    AzureCohereReranker,
)

# Example 1: Using Azure Cohere Reranker with explicit configuration
def example_with_explicit_config():
    """Create a KB with Azure Cohere reranker using explicit configuration."""
    
    # Initialize Azure Cohere Reranker
    reranker = AzureCohereReranker(
        model="Cohere-rerank-v3.5",
        azure_endpoint="https://your-cohere-endpoint.azure.com/",
        api_key="your-cohere-api-key",
    )
    
    # Initialize other Azure components
    embedding = AzureOpenAIEmbedding(
        deployment_name=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        dimension=1536,
    )
    
    chat = AzureOpenAIChatAPI(
        deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    )
    
    storage = AzureBlobStorage(
        base_path="~/my_kb",
        container_name=os.environ["AZURE_STORAGE_CONTAINER_NAME"],
    )
    
    # Create Knowledge Base with Azure Cohere reranker
    kb = KnowledgeBase(
        kb_id="my_azure_cohere_kb",
        embedding_model=embedding,
        reranker=reranker,  # Use Azure Cohere reranker
        auto_context_model=chat,
        file_system=storage,
    )
    
    return kb


# Example 2: Using environment variables
def example_with_env_vars():
    """Create a KB with Azure Cohere reranker using environment variables."""
    
    # Set environment variables in .env file:
    # AZURE_COHERE_ENDPOINT=https://your-cohere-endpoint.azure.com/
    # AZURE_COHERE_API_KEY=your_cohere_api_key
    
    # Initialize Azure Cohere Reranker (will use env vars)
    reranker = AzureCohereReranker(
        model="Cohere-rerank-v3.5",  # or any other Cohere model
    )
    
    # Initialize other Azure components
    embedding = AzureOpenAIEmbedding(
        deployment_name=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        dimension=1536,
    )
    
    chat = AzureOpenAIChatAPI(
        deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    )
    
    storage = AzureBlobStorage(
        base_path="~/my_kb",
        container_name=os.environ["AZURE_STORAGE_CONTAINER_NAME"],
    )
    
    # Create Knowledge Base
    kb = KnowledgeBase(
        kb_id="my_azure_cohere_kb",
        embedding_model=embedding,
        reranker=reranker,
        auto_context_model=chat,
        file_system=storage,
    )
    
    return kb


# Example 3: Using Azure Cohere with an existing KB
def example_with_existing_kb():
    """Load an existing KB and use Azure Cohere reranker for queries."""
    
    # Initialize Azure Cohere Reranker
    reranker = AzureCohereReranker(
        model="Cohere-rerank-v3.5",
        # azure_endpoint and api_key will be loaded from env vars
    )
    
    # Load existing KB with Azure Cohere reranker
    kb = KnowledgeBase(
        kb_id="existing_kb",
        reranker=reranker,  # Override the stored reranker
    )
    
    # Query the KB (will use Azure Cohere for reranking)
    results = kb.query(
        search_queries=["What is the main topic?"],
        rse_params="balanced",
    )
    
    return results


if __name__ == "__main__":
    print("Azure Cohere Reranker Examples")
    print("=" * 50)
    print("\nTo use Azure Cohere reranker, you need:")
    print("1. Cohere model deployed on Azure")
    print("2. Azure Cohere endpoint URL")
    print("3. Azure Cohere API key")
    print("\nSet these in your .env file:")
    print("  AZURE_COHERE_ENDPOINT=https://your-endpoint.azure.com/")
    print("  AZURE_COHERE_API_KEY=your_api_key")
    print("\nThen use AzureCohereReranker in your KnowledgeBase!")
