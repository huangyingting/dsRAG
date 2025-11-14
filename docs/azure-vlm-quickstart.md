# Azure OpenAI VLM Quick Start Guide

## What is Azure OpenAI VLM?

The `AzureOpenAIVLM` class enables you to use Azure OpenAI's vision-capable models (like GPT-4o) to analyze images and extract structured information from documents. This is particularly useful for:

- Parsing complex PDFs with diagrams, charts, and tables
- Extracting structured data from scanned documents
- Understanding visual content in knowledge base documents
- Document layout analysis and content extraction

## Installation

```bash
pip install "dsrag[azure-openai]"
```

## Basic Setup

### 1. Set Environment Variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_VLM_DEPLOYMENT="gpt-4o"  # Your vision model deployment
```

### 2. Deploy a Vision Model in Azure

In Azure OpenAI Studio:
1. Go to "Deployments"
2. Create a new deployment
3. Select a vision-capable model (gpt-4o, gpt-4-turbo, gpt-4-vision-preview)
4. Note the deployment name

## Usage Examples

### Example 1: Simple Image Analysis

```python
from dsrag.azure import AzureOpenAIVLM

# Initialize VLM
vlm = AzureOpenAIVLM(deployment_name="gpt-4o")

# Analyze an image
response = vlm.make_llm_call(
    image_path="path/to/image.jpg",
    system_message="Describe what you see in this image.",
    max_tokens=500,
    temperature=0.3,
)

print(response)
```

### Example 2: Structured Data Extraction

```python
from dsrag.azure import AzureOpenAIVLM

vlm = AzureOpenAIVLM(deployment_name="gpt-4o")

# Define a schema for structured output
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "main_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "has_diagrams": {"type": "boolean"}
    }
}

response = vlm.make_llm_call(
    image_path="path/to/document_page.png",
    system_message="Extract the title, main points, and indicate if there are diagrams.",
    response_schema=schema,
    max_tokens=1000,
    temperature=0.2,
)

print(response)  # JSON string matching the schema
```

### Example 3: Knowledge Base with VLM

```python
from dsrag.knowledge_base import KnowledgeBase
from dsrag.azure import (
    AzureBlobStorage,
    AzureOpenAIChatAPI,
    AzureOpenAIEmbedding,
    AzureOpenAIVLM
)

# Initialize all Azure components
storage = AzureBlobStorage(
    base_path="~/dsrag",
    container_name="my-docs",
)

chat = AzureOpenAIChatAPI(deployment_name="gpt-4")
embedding = AzureOpenAIEmbedding(deployment_name="text-embedding-ada-002", dimension=1536)
vlm = AzureOpenAIVLM(deployment_name="gpt-4o")

# Create knowledge base with VLM support
kb = KnowledgeBase(
    kb_id="visual_kb",
    embedding_model=embedding,
    auto_context_model=chat,
    file_system=storage,
    vlm_client=vlm,  # Enable VLM parsing
)

# Add a document with VLM parsing
kb.add_document(
    doc_id="technical_spec",
    file_path="path/to/technical_spec.pdf",
    document_title="Technical Specifications",
    file_parsing_config={
        "use_vlm": True,  # Enable VLM-based parsing
        "vlm_config": {
            "save_path": "/tmp/vlm_processing",
        }
    }
)

# Query as usual
results = kb.query(
    search_queries=["What are the key specifications?"],
)
```

### Example 4: Manual Credentials (No Environment Variables)

```python
from dsrag.azure import AzureOpenAIVLM

vlm = AzureOpenAIVLM(
    deployment_name="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key="your-api-key",
    api_version="2024-02-15-preview",
)

response = vlm.make_llm_call(
    image_path="diagram.png",
    system_message="Explain this diagram.",
)
```

## Configuration Options

### AzureOpenAIVLM Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `deployment_name` | str | Yes | - | Azure deployment name (must be vision model) |
| `azure_endpoint` | str | No | env var | Azure OpenAI endpoint URL |
| `api_key` | str | No | env var | Azure OpenAI API key |
| `api_version` | str | No | "2024-02-15-preview" | Azure API version |

### make_llm_call Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_path` | str | Yes | - | Path to image file |
| `system_message` | str | Yes | - | Prompt/instructions |
| `response_schema` | dict | No | None | JSON schema for structured output |
| `max_tokens` | int | No | 4000 | Maximum response tokens |
| `temperature` | float | No | 0.5 | Sampling temperature (0-2) |

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## Best Practices

### 1. Choose the Right Model

- **gpt-4o**: Best balance of speed and quality (recommended)
- **gpt-4-turbo**: Good for complex visual understanding
- **gpt-4-vision-preview**: Earlier version, may have limitations

### 2. Optimize Prompts

```python
# Good prompt - specific and clear
system_message = """
Extract the following information from this invoice:
- Invoice number
- Date
- Total amount
- Line items with quantities and prices

Return as JSON.
"""

# Less effective prompt - too vague
system_message = "Tell me about this document."
```

### 3. Use Structured Outputs

For consistent data extraction, always provide a `response_schema`:

```python
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "total": {"type": "number"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "price": {"type": "number"}
                }
            }
        }
    }
}
```

### 4. Handle Errors Gracefully

```python
try:
    response = vlm.make_llm_call(
        image_path="document.png",
        system_message="Extract data",
    )
except Exception as e:
    print(f"VLM call failed: {e}")
    # Fallback to non-VLM parsing
```

### 5. Monitor Costs

Vision models are more expensive than text-only models:
- Check your Azure OpenAI pricing
- Use appropriate `max_tokens` limits
- Consider caching results for repeated analyses

## Troubleshooting

### Error: "Azure OpenAI endpoint must be provided"

**Solution**: Set the environment variable or pass credentials:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### Error: "Deployment not found"

**Solution**: Verify your deployment name matches exactly in Azure OpenAI Studio.

### Error: "The model does not support vision"

**Solution**: Ensure you're using a vision-capable model (gpt-4o, gpt-4-turbo, gpt-4-vision-preview).

### Poor Quality Results

**Solutions**:
1. Improve image quality (higher resolution, better contrast)
2. Make prompts more specific
3. Use response schemas for structured output
4. Adjust temperature (lower = more deterministic)
5. Increase max_tokens if responses are truncated

## Integration with dsRAG

When used with dsRAG's knowledge base:

1. **VLM is optional**: KB works without VLM for text-only documents
2. **Per-document control**: Enable VLM parsing per document via `file_parsing_config`
3. **Automatic serialization**: VLM configuration is saved with the KB
4. **Fallback support**: Can configure fallback VLM for resilience

```python
# VLM is saved with KB configuration
kb = KnowledgeBase(kb_id="my_kb", vlm_client=vlm)

# Later, load KB and VLM is restored
kb_loaded = KnowledgeBase(kb_id="my_kb")
# kb_loaded.vlm_client is automatically restored
```

## Performance Tips

1. **Batch Processing**: Process multiple images sequentially
2. **Async Operations**: Consider async wrappers for parallel processing
3. **Image Preprocessing**: Optimize image size before sending
4. **Result Caching**: Cache VLM results to avoid redundant calls
5. **Rate Limiting**: Respect Azure OpenAI rate limits

## Examples in the Repo

See `examples/azure_example.py` for a complete working example with VLM support.

## Additional Resources

- [Azure OpenAI Vision Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision)
- [dsRAG Documentation](https://github.com/D-Star-AI/dsRAG)
- [VLM Best Practices](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest#best-practices)
