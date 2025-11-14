# Azure Integration for dsRAG

This module provides Azure-specific implementations for dsRAG, enabling seamless integration with Azure cloud services.

## Components

### 1. Azure Blob Storage (`AzureBlobStorage`)
A `FileSystem` implementation that uses Azure Blob Storage for storing documents, images, and metadata.

**Features:**
- Store and retrieve page images
- Save and load JSON metadata
- Support for error logging
- Compatible with all dsRAG knowledge base operations

### 2. Azure OpenAI Chat (`AzureOpenAIChatAPI`)
An `LLM` implementation that uses Azure OpenAI Service for chat completions.

**Features:**
- Compatible with GPT-4, GPT-3.5-Turbo, and other Azure OpenAI models
- Configurable temperature and token limits
- Seamless integration with dsRAG's AutoContext feature

### 3. Azure OpenAI Embedding (`AzureOpenAIEmbedding`)
An `Embedding` implementation that uses Azure OpenAI Service for text embeddings.

**Features:**
- Support for text-embedding-ada-002 and newer models
- Configurable embedding dimensions
- Batch embedding support

### 4. Azure OpenAI VLM (`AzureOpenAIVLM`)
A `VLM` (Vision Language Model) implementation that uses Azure OpenAI Service for image analysis.

**Features:**
- Support for GPT-4 Vision models (gpt-4-vision-preview, gpt-4o, gpt-4-turbo)
- Extract structured data from images
- Document parsing with visual understanding
- Configurable response schemas

## Installation

Install dsRAG with Azure support:

```bash
# Install with Azure storage support only
pip install "dsrag[azure-storage]"

# Install with Azure OpenAI support only (includes VLM support)
pip install "dsrag[azure-openai]"

# Install with all Azure components
pip install "dsrag[azure]"
```

## Configuration

### Environment Variables

Set the following environment variables for Azure services:

```bash
# Azure Blob Storage
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
# OR
export AZURE_STORAGE_ACCOUNT_NAME="your_account_name"
export AZURE_STORAGE_ACCOUNT_KEY="your_account_key"

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your_api_key"
```

### Container Name
You'll need to specify a container name for Azure Blob Storage. The container will be created automatically if it doesn't exist.

## Usage Examples

### Basic Usage with All Azure Components

```python
from dsrag.knowledge_base import KnowledgeBase
from dsrag.azure import (
    AzureBlobStorage, 
    AzureOpenAIChatAPI, 
    AzureOpenAIEmbedding,
    AzureOpenAIVLM
)

# Initialize Azure Blob Storage
azure_storage = AzureBlobStorage(
    base_path="~/dsrag_azure",
    container_name="dsrag-documents",
    connection_string="your_connection_string",  # or use account_name/account_key
)

# Initialize Azure OpenAI Chat
azure_chat = AzureOpenAIChatAPI(
    deployment_name="gpt-4",  # Your Azure OpenAI deployment name
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key="your_api_key",
    temperature=0.2,
    max_tokens=1000,
)

# Initialize Azure OpenAI Embedding
azure_embedding = AzureOpenAIEmbedding(
    deployment_name="text-embedding-ada-002",  # Your embedding deployment name
    dimension=1536,
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key="your_api_key",
)

# Initialize Azure OpenAI VLM (for document parsing with vision)
azure_vlm = AzureOpenAIVLM(
    deployment_name="gpt-4o",  # Your vision model deployment name
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key="your_api_key",
)

# Create a knowledge base with Azure components
kb = KnowledgeBase(
    kb_id="my_azure_kb",
    embedding_model=azure_embedding,
    auto_context_model=azure_chat,
    file_system=azure_storage,
    vlm_client=azure_vlm,  # Optional: for VLM-based document parsing
)

# Add documents with VLM parsing
kb.add_document(
    doc_id="my_document",
    file_path="path/to/document.pdf",
    document_title="My Document",
    file_parsing_config={
        "use_vlm": True,  # Enable VLM parsing
    }
)

# Add documents with text
kb.add_document(
    doc_id="my_text_doc",
    text="Your document text here...",
    document_title="My Text Document",
)

# Query the knowledge base
results = kb.query(
    search_queries=["What is this document about?"],
    rse_params="balanced",
)

for result in results:
    print(f"Document: {result['doc_id']}")
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']}")
```

### Using Environment Variables

```python
import os
from dsrag.knowledge_base import KnowledgeBase
from dsrag.azure import AzureBlobStorage, AzureOpenAIChatAPI, AzureOpenAIEmbedding

# Azure components will automatically use environment variables
azure_storage = AzureBlobStorage(
    base_path="~/dsrag_azure",
    container_name="dsrag-documents",
    # connection_string will be read from AZURE_STORAGE_CONNECTION_STRING
)

azure_chat = AzureOpenAIChatAPI(
    deployment_name="gpt-4",
    # azure_endpoint and api_key will be read from environment variables
)

azure_embedding = AzureOpenAIEmbedding(
    deployment_name="text-embedding-ada-002",
    dimension=1536,
    # azure_endpoint and api_key will be read from environment variables
)

kb = KnowledgeBase(
    kb_id="my_azure_kb",
    embedding_model=azure_embedding,
    auto_context_model=azure_chat,
    file_system=azure_storage,
)
```

### Using Only Azure Blob Storage

```python
from dsrag.knowledge_base import KnowledgeBase
from dsrag.azure import AzureBlobStorage
from dsrag.llm import OpenAIChatAPI
from dsrag.embedding import OpenAIEmbedding

# Use Azure for storage, but regular OpenAI for models
azure_storage = AzureBlobStorage(
    base_path="~/dsrag_azure",
    container_name="dsrag-documents",
    connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
)

kb = KnowledgeBase(
    kb_id="hybrid_kb",
    embedding_model=OpenAIEmbedding(),  # Regular OpenAI
    auto_context_model=OpenAIChatAPI(),  # Regular OpenAI
    file_system=azure_storage,  # Azure Blob Storage
)
```

### Persistence and Loading

```python
from dsrag.knowledge_base import KnowledgeBase

# Create and configure KB with Azure components (first time)
kb = KnowledgeBase(
    kb_id="persistent_kb",
    embedding_model=azure_embedding,
    auto_context_model=azure_chat,
    file_system=azure_storage,
)

# Later, load the KB - Azure components will be restored automatically
kb_loaded = KnowledgeBase(kb_id="persistent_kb")

# All Azure components are automatically restored from saved configuration
print(type(kb_loaded.file_system))  # <class 'AzureBlobStorage'>
print(type(kb_loaded.embedding_model))  # <class 'AzureOpenAIEmbedding'>
```

## Testing

### Unit Tests

Run unit tests for Azure Blob Storage:

```bash
python -m pytest tests/unit/test_azure_blob_storage.py -v
```

### Integration Tests

Integration tests require valid Azure credentials. Set up your environment variables first:

```bash
# Set Azure credentials
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
export AZURE_STORAGE_CONTAINER_NAME="test-container"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your_api_key"
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"

# Run integration tests
python -m pytest tests/integration/test_azure_integration.py -v
```

Or run tests directly:

```bash
python tests/integration/test_azure_integration.py
```

## Architecture

### Storage Architecture

The `AzureBlobStorage` class stores files in the following structure:

```
container/
├── kb_id_1/
│   ├── doc_id_1/
│   │   ├── page_1.jpg
│   │   ├── page_2.jpg
│   │   ├── page_content_1.json
│   │   ├── elements.json
│   │   └── errors/
│   │       └── timestamp.json
│   └── doc_id_2/
│       └── ...
└── kb_id_2/
    └── ...
```

### Serialization

All Azure components support serialization via `to_dict()` and `from_dict()` methods, enabling:
- Persistent storage of KB configurations
- Easy reconstruction of Azure components
- Version control of KB setups

## Best Practices

1. **Use environment variables** for credentials to avoid hardcoding sensitive information
2. **Choose appropriate container names** that reflect your application structure
3. **Monitor Azure costs** - storage and API calls have associated costs
4. **Use appropriate deployment names** that match your Azure OpenAI deployments
5. **Set proper embedding dimensions** based on your chosen embedding model
6. **Consider regional deployment** for lower latency
7. **Implement retry logic** for production applications

## Troubleshooting

### Common Issues

**Issue: "Azure storage dependencies not found"**
- Solution: Install with `pip install "dsrag[azure-storage]"`

**Issue: "OpenAI package not found"**
- Solution: Install with `pip install "dsrag[azure-openai]"` or `pip install "dsrag[openai]"`

**Issue: "Container not found"**
- Solution: The container is created automatically, but ensure your connection string has proper permissions

**Issue: "Deployment not found"**
- Solution: Verify your deployment name matches exactly with your Azure OpenAI deployment

**Issue: "Authentication failed"**
- Solution: Check that your API key and endpoint are correct and the key hasn't expired

## Limitations

- Azure Blob Storage requires internet connectivity
- API rate limits apply based on your Azure OpenAI tier
- Large files may have upload/download latency
- Costs scale with usage (storage, API calls, egress)

## Contributing

Contributions to improve Azure integration are welcome! Please ensure:
- All tests pass
- New features include tests
- Documentation is updated
- Code follows the existing patterns

## License

This Azure integration is part of dsRAG and follows the same MIT license.
