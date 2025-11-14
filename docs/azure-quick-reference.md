# Azure Integration Quick Reference

## Installation

```bash
pip install "dsrag[azure]"
```

## Environment Variables

```bash
# Storage
export AZURE_STORAGE_CONNECTION_STRING="..."
export AZURE_STORAGE_CONTAINER_NAME="dsrag-docs"

# OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
```

## Basic Usage

```python
from dsrag.knowledge_base import KnowledgeBase
from dsrag.azure import AzureBlobStorage, AzureOpenAIChatAPI, AzureOpenAIEmbedding

# Initialize components
storage = AzureBlobStorage(
    base_path="~/dsrag",
    container_name="dsrag-docs",
)

chat = AzureOpenAIChatAPI(
    deployment_name="gpt-4",
)

embedding = AzureOpenAIEmbedding(
    deployment_name="text-embedding-ada-002",
    dimension=1536,
)

# Create KB
kb = KnowledgeBase(
    kb_id="my_kb",
    embedding_model=embedding,
    auto_context_model=chat,
    file_system=storage,
)

# Add document
kb.add_document(
    doc_id="doc1",
    text="Your text here...",
)

# Query
results = kb.query(
    search_queries=["Your question?"],
)
```

## Class Reference

### AzureBlobStorage

**Constructor Parameters:**
- `base_path`: Local temp directory
- `container_name`: Azure container name (required)
- `connection_string`: Azure connection string
- OR `account_name` + `account_key`

**Key Methods:**
- `save_json(kb_id, doc_id, filename, data)`
- `load_data(kb_id, doc_id, data_name)`
- `save_image(kb_id, doc_id, filename, image)`
- `get_files(kb_id, doc_id, page_start, page_end)`

### AzureOpenAIChatAPI

**Constructor Parameters:**
- `deployment_name`: Azure deployment name (required)
- `azure_endpoint`: Azure endpoint URL
- `api_key`: Azure API key
- `api_version`: API version (default: "2024-02-15-preview")
- `temperature`: 0-2 (default: 0.2)
- `max_tokens`: Max response tokens (default: 1000)

**Key Methods:**
- `make_llm_call(chat_messages)`: Returns response string

### AzureOpenAIEmbedding

**Constructor Parameters:**
- `deployment_name`: Azure deployment name (required)
- `dimension`: Embedding dimension (required)
- `azure_endpoint`: Azure endpoint URL
- `api_key`: Azure API key
- `api_version`: API version (default: "2024-02-15-preview")

**Key Methods:**
- `get_embeddings(texts, input_type=None)`: Returns list of vectors

## Testing

```bash
# Unit tests (no Azure required)
python3 tests/unit/test_azure_blob_storage.py

# Integration tests (Azure required)
python3 tests/integration/test_azure_integration.py

# Example
python3 examples/azure_example.py
python3 examples/azure_example.py --cleanup
```

## Common Deployment Names

| Model Type | Common Deployment Names |
|-----------|------------------------|
| Chat | `gpt-4`, `gpt-35-turbo`, `gpt-4-turbo` |
| Embedding | `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large` |

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Azure storage dependencies not found" | `pip install "dsrag[azure-storage]"` |
| "Deployment not found" | Check deployment name in Azure OpenAI Studio |
| "Authentication failed" | Verify endpoint URL and API key |
| "Container not found" | Check connection string permissions |

## File Structure in Azure

```
container-name/
├── kb_id/
│   ├── doc_id/
│   │   ├── page_1.jpg
│   │   ├── page_content_1.json
│   │   └── elements.json
```

## Cost Estimates

- Storage: ~$0.01/GB/month
- Transactions: ~$0.05/10K ops
- GPT-4: ~$0.03-0.06/1K tokens
- Embeddings: ~$0.0001/1K tokens

Full test suite: **< $0.50**
