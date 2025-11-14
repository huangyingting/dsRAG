"""
Example: Using dsRAG with Azure services

This example demonstrates how to use dsRAG with:
- Azure Blob Storage for file storage
- Azure OpenAI for chat completions
- Azure OpenAI for embeddings

Before running this example, set the following environment variables:
- AZURE_STORAGE_CONNECTION_STRING (or AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY)
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dsRAG components
from dsrag.knowledge_base import KnowledgeBase
from dsrag.azure import AzureBlobStorage, AzureOpenAIChatAPI, AzureOpenAIEmbedding
# AzureOpenAIVLM is imported conditionally in the example


def main():
    """Main example function."""
    
    print("=" * 80)
    print("dsRAG Azure Integration Example")
    print("=" * 80)
    
    # Configuration
    kb_id = "azure_example_kb"
    container_name = "dsrag-example"
    chat_deployment = "gpt-4"  # Change to your deployment name
    embedding_deployment = "text-embedding-ada-002"  # Change to your deployment name
    
    # Step 1: Initialize Azure Blob Storage
    print("\n1. Initializing Azure Blob Storage...")
    azure_storage = AzureBlobStorage(
        base_path=os.path.expanduser("~/dsrag_azure_example"),
        container_name=container_name,
        connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
    )
    print(f"   ✓ Azure Blob Storage initialized with container: {container_name}")
    
    # Step 2: Initialize Azure OpenAI Chat
    print("\n2. Initializing Azure OpenAI Chat...")
    azure_chat = AzureOpenAIChatAPI(
        deployment_name=chat_deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        temperature=0.2,
        max_tokens=1000,
    )
    print(f"   ✓ Azure OpenAI Chat initialized with deployment: {chat_deployment}")
    
    # Step 3: Initialize Azure OpenAI Embedding
    print("\n3. Initializing Azure OpenAI Embedding...")
    azure_embedding = AzureOpenAIEmbedding(
        deployment_name=embedding_deployment,
        dimension=1536,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    print(f"   ✓ Azure OpenAI Embedding initialized with deployment: {embedding_deployment}")
    
    # Step 3b: Initialize Azure OpenAI VLM (optional, for vision-based parsing)
    vlm_deployment = os.environ.get("AZURE_OPENAI_VLM_DEPLOYMENT", "gpt-4o")
    azure_vlm = None
    try:
        from dsrag.azure import AzureOpenAIVLM
        print(f"\n3b. Initializing Azure OpenAI VLM (optional)...")
        azure_vlm = AzureOpenAIVLM(
            deployment_name=vlm_deployment,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        )
        print(f"   ✓ Azure OpenAI VLM initialized with deployment: {vlm_deployment}")
    except Exception as e:
        print(f"   ⚠ Azure OpenAI VLM initialization skipped: {e}")
    
    # Step 4: Create Knowledge Base
    print(f"\n4. Creating Knowledge Base '{kb_id}'...")
    kb = KnowledgeBase(
        kb_id=kb_id,
        embedding_model=azure_embedding,
        auto_context_model=azure_chat,
        file_system=azure_storage,
        vlm_client=azure_vlm,  # Optional VLM client for vision-based parsing
        exists_ok=True,  # Allow loading existing KB
    )
    print(f"   ✓ Knowledge Base created successfully")
    
    # Step 5: Add a sample document
    print("\n5. Adding sample document...")
    sample_text = """
    Microsoft Azure: A Comprehensive Cloud Platform
    
    Microsoft Azure is a comprehensive cloud computing platform that provides a wide array
    of services to help businesses build, deploy, and manage applications. Azure offers
    infrastructure as a service (IaaS), platform as a service (PaaS), and software as a
    service (SaaS) solutions.
    
    Key Azure Services:
    
    1. Compute Services: Azure provides virtual machines, container services (AKS), and
       serverless computing (Azure Functions) to run applications.
    
    2. Storage Services: Azure Blob Storage offers scalable object storage for unstructured
       data. It's ideal for storing documents, images, and backups.
    
    3. AI and Machine Learning: Azure OpenAI Service provides access to powerful language
       models like GPT-4 for natural language processing, content generation, and more.
    
    4. Database Services: Azure offers managed database services including Azure SQL Database,
       Cosmos DB for NoSQL, and Azure Database for PostgreSQL.
    
    5. Networking: Azure Virtual Network enables secure communication between Azure resources
       and on-premises infrastructure.
    
    Azure OpenAI Service:
    
    Azure OpenAI Service combines OpenAI's cutting-edge models with Azure's enterprise-grade
    capabilities. It provides REST API access to GPT-4, GPT-3.5-Turbo, and embedding models.
    These models can be used for:
    - Content generation and summarization
    - Semantic search and retrieval
    - Code generation and analysis
    - Language translation
    - Question answering systems
    
    Azure Blob Storage:
    
    Azure Blob Storage is optimized for storing massive amounts of unstructured data. It offers:
    - Three storage tiers: Hot, Cool, and Archive
    - High availability and durability
    - Built-in security with encryption at rest
    - Global distribution with geo-replication
    - Integration with Azure CDN for fast content delivery
    
    Benefits of Using Azure:
    
    1. Scalability: Easily scale resources up or down based on demand
    2. Global Reach: Data centers in multiple regions worldwide
    3. Security: Enterprise-grade security and compliance certifications
    4. Cost Efficiency: Pay-as-you-go pricing with no upfront costs
    5. Integration: Seamless integration with other Microsoft services
    """
    
    kb.add_document(
        doc_id="azure_overview",
        text=sample_text,
        document_title="Microsoft Azure Overview",
    )
    print("   ✓ Document 'azure_overview' added successfully")
    
    # Step 6: Query the Knowledge Base
    print("\n6. Querying the Knowledge Base...")
    queries = [
        "What is Azure Blob Storage?",
        "What AI services does Azure provide?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")
        results = kb.query(
            search_queries=[query],
            rse_params="balanced",
        )
        
        if results:
            print(f"   Found {len(results)} relevant segments:")
            for j, result in enumerate(results, 1):
                print(f"\n   Segment {j}:")
                print(f"   - Document: {result['doc_id']}")
                print(f"   - Score: {result['score']:.4f}")
                print(f"   - Content preview: {result['content'][:200]}...")
        else:
            print("   No results found")
    
    # Step 7: Test Azure OpenAI Chat directly
    print("\n7. Testing Azure OpenAI Chat directly...")
    messages = [
        {"role": "system", "content": "You are a helpful Azure expert."},
        {"role": "user", "content": "In one sentence, what makes Azure Blob Storage useful?"}
    ]
    response = azure_chat.make_llm_call(messages)
    print(f"   Response: {response}")
    
    # Step 8: Test Azure OpenAI Embedding directly
    print("\n8. Testing Azure OpenAI Embedding directly...")
    test_texts = ["Azure is a cloud platform", "Machine learning in the cloud"]
    embeddings = azure_embedding.get_embeddings(test_texts)
    print(f"   ✓ Generated {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    
    # Step 9: Demonstrate persistence
    print("\n9. Testing persistence (save and load)...")
    print("   Saving knowledge base configuration...")
    # KB is automatically saved
    
    print("   Loading knowledge base from saved configuration...")
    kb_loaded = KnowledgeBase(kb_id=kb_id)
    print(f"   ✓ Knowledge base loaded successfully")
    print(f"   - File system type: {type(kb_loaded.file_system).__name__}")
    print(f"   - Embedding model type: {type(kb_loaded.embedding_model).__name__}")
    print(f"   - Chat model type: {type(kb_loaded.auto_context_model).__name__}")
    
    # Step 10: Summary
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nWhat we demonstrated:")
    print("  ✓ Azure Blob Storage for document and metadata storage")
    print("  ✓ Azure OpenAI for chat completions (AutoContext)")
    print("  ✓ Azure OpenAI for text embeddings")
    print("  ✓ Azure OpenAI VLM for vision-based document parsing (optional)")
    print("  ✓ Creating and querying a knowledge base")
    print("  ✓ Persistence and loading of configurations")
    print("\nCleanup:")
    print("  To delete the knowledge base, run:")
    print(f"    kb = KnowledgeBase(kb_id='{kb_id}')")
    print("    kb.delete()")
    print("=" * 80)


def cleanup_example():
    """Clean up the example knowledge base."""
    kb_id = "azure_example_kb"
    print(f"Cleaning up knowledge base '{kb_id}'...")
    
    try:
        kb = KnowledgeBase(kb_id=kb_id)
        kb.delete()
        print(f"✓ Knowledge base '{kb_id}' deleted successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="dsRAG Azure Integration Example")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up the example knowledge base"
    )
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_example()
    else:
        try:
            main()
        except Exception as e:
            print(f"\nError: {e}")
            print("\nPlease ensure you have:")
            print("  1. Set the required environment variables")
            print("  2. Valid Azure credentials")
            print("  3. Deployed models in Azure OpenAI")
            sys.exit(1)
