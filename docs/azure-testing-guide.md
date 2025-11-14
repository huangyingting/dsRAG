# Azure Integration Testing Guide

This guide provides instructions for testing the Azure integration components in dsRAG.

## Prerequisites

1. **Azure Account**: You need an active Azure subscription
2. **Azure OpenAI Service**: Access to Azure OpenAI Service with deployed models
3. **Azure Storage Account**: An Azure Storage account for Blob Storage

## Setup

### 1. Install Dependencies

Install dsRAG with Azure support:

```bash
# Install all Azure components
pip install "dsrag[azure]"

# Or install components separately
pip install "dsrag[azure-storage]"  # For Azure Blob Storage only
pip install "dsrag[azure-openai]"   # For Azure OpenAI only
```

### 2. Configure Environment Variables

Create a `.env` file in the project root or set the following environment variables:

```bash
# Azure Blob Storage (choose connection string OR account credentials)
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
# OR
export AZURE_STORAGE_ACCOUNT_NAME="your_storage_account_name"
export AZURE_STORAGE_ACCOUNT_KEY="your_storage_account_key"

# Azure Storage Container
export AZURE_STORAGE_CONTAINER_NAME="dsrag-test"

# Azure OpenAI Service
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
export AZURE_OPENAI_API_KEY="your_api_key"

# Azure OpenAI Deployments
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
```

### 3. Create Azure Resources

#### Azure Storage Account

1. Go to Azure Portal → Storage Accounts
2. Click "Create"
3. Fill in details and create
4. Navigate to "Access keys" to get connection string

#### Azure OpenAI Service

1. Go to Azure Portal → Azure OpenAI
2. Click "Create"
3. After creation, go to "Keys and Endpoint"
4. Copy the endpoint and API key

#### Deploy Models

1. In Azure OpenAI Studio, go to "Deployments"
2. Deploy required models:
   - A chat model (e.g., `gpt-4`, `gpt-35-turbo`)
   - An embedding model (e.g., `text-embedding-ada-002`)
3. Note the deployment names (they may differ from model names)

## Running Tests

### Unit Tests

Unit tests use mocking and don't require actual Azure resources:

```bash
python3 tests/unit/test_azure_blob_storage.py
```

Expected output:
```
test_azure_blob_storage_operations ... ok
test_azure_chat_serialization ... ok
test_azure_embedding_serialization ... ok
test_delete_directory ... ok
test_delete_kb ... ok
test_init_with_account_credentials ... ok
test_init_with_connection_string ... ok
test_log_error ... ok
test_save_image ... ok
test_save_json ... ok
test_to_dict ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.123s

OK
```

### Integration Tests

Integration tests require actual Azure resources and valid credentials:

```bash
# Ensure environment variables are set
source .env  # or load your environment variables

# Run integration tests
python3 tests/integration/test_azure_integration.py
```

Expected output:
```
test_001_create_kb_with_azure_components ... ok
test_002_add_document_to_azure_kb ... ok
test_003_query_azure_kb ... ok
test_004_azure_chat_basic_call ... ok
test_005_azure_embedding_basic_call ... ok
test_006_save_and_load_with_azure ... ok
test_007_azure_blob_storage_operations ... ok

----------------------------------------------------------------------
Ran 7 tests in 45.678s

OK
```

### Running the Example

Run the complete Azure example:

```bash
# Run the example
python3 examples/azure_example.py

# Clean up after running
python3 examples/azure_example.py --cleanup
```

## Test Coverage

### Unit Tests (`test_azure_blob_storage.py`)

Tests the following without requiring Azure resources:

1. **Initialization**
   - Connection string initialization
   - Account credentials initialization
   - Error handling for missing credentials

2. **CRUD Operations**
   - Create directory
   - Delete directory
   - Delete knowledge base
   - Save JSON files
   - Save images
   - Load data

3. **Serialization**
   - `to_dict()` method for all Azure components
   - Configuration persistence

### Integration Tests (`test_azure_integration.py`)

Tests real Azure integration:

1. **Knowledge Base Operations**
   - Create KB with Azure components
   - Add documents
   - Query documents
   - Save and load configurations

2. **Azure OpenAI Chat**
   - Basic chat completions
   - Message formatting
   - Response handling

3. **Azure OpenAI Embeddings**
   - Single and batch embeddings
   - Dimension verification
   - Vector operations

4. **Azure Blob Storage**
   - JSON save/load
   - Image save/load
   - Page content operations
   - Directory cleanup

## Troubleshooting

### Common Test Failures

#### "Azure storage dependencies not found"

**Problem**: Azure SDK not installed

**Solution**:
```bash
pip install "dsrag[azure-storage]"
```

#### "Missing required environment variables"

**Problem**: Environment variables not set

**Solution**: Ensure all required environment variables are set:
```bash
# Check if variables are set
echo $AZURE_STORAGE_CONNECTION_STRING
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_KEY
```

#### "Deployment not found"

**Problem**: Deployment name doesn't match Azure deployment

**Solution**: 
- Check your Azure OpenAI Studio for exact deployment names
- Update environment variables to match:
```bash
export AZURE_OPENAI_CHAT_DEPLOYMENT="your-exact-deployment-name"
```

#### "Authentication failed"

**Problem**: Invalid credentials or expired API key

**Solution**:
- Verify API key in Azure Portal
- Check endpoint URL format: `https://<resource-name>.openai.azure.com`
- Ensure no trailing slashes in endpoint URL

#### "Container not found"

**Problem**: Container doesn't exist and client can't create it

**Solution**:
- Ensure your storage account credentials have proper permissions
- Manually create the container in Azure Portal
- Verify connection string includes `AccountKey`

### Network Issues

If you're behind a proxy or firewall:

```bash
# Set proxy if needed
export HTTPS_PROXY="http://proxy.company.com:8080"
export HTTP_PROXY="http://proxy.company.com:8080"
```

### Rate Limiting

Azure OpenAI has rate limits. If tests fail due to rate limiting:

1. Wait a few minutes and retry
2. Check your quota in Azure Portal
3. Consider upgrading your Azure OpenAI tier
4. Add delays between test runs

## Cost Considerations

Running these tests will incur small Azure costs:

- **Storage**: ~$0.01 per GB per month
- **Blob operations**: ~$0.05 per 10,000 transactions
- **OpenAI API calls**: Varies by model
  - GPT-4: ~$0.03-0.06 per 1K tokens
  - Embeddings: ~$0.0001 per 1K tokens

Estimated cost for full test suite: **< $0.50**

To minimize costs:
- Run `--cleanup` after integration tests
- Delete test containers after testing
- Use smaller/cheaper models for testing

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Azure Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install ".[azure]"
    
    - name: Run unit tests
      run: |
        python3 tests/unit/test_azure_blob_storage.py
    
    - name: Run integration tests
      env:
        AZURE_STORAGE_CONNECTION_STRING: ${{ secrets.AZURE_STORAGE_CONNECTION_STRING }}
        AZURE_STORAGE_CONTAINER_NAME: ${{ secrets.AZURE_STORAGE_CONTAINER_NAME }}
        AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
        AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
        AZURE_OPENAI_CHAT_DEPLOYMENT: ${{ secrets.AZURE_OPENAI_CHAT_DEPLOYMENT }}
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT: ${{ secrets.AZURE_OPENAI_EMBEDDING_DEPLOYMENT }}
      run: |
        python3 tests/integration/test_azure_integration.py
```

## Best Practices

1. **Environment Isolation**: Use separate Azure resources for testing
2. **Cleanup**: Always clean up test resources to avoid costs
3. **Secrets Management**: Never commit credentials to version control
4. **Test Data**: Use small documents for faster tests
5. **Error Handling**: Check logs for detailed error messages

## Additional Resources

- [Azure Blob Storage Documentation](https://docs.microsoft.com/en-us/azure/storage/blobs/)
- [Azure OpenAI Service Documentation](https://docs.microsoft.com/en-us/azure/ai-services/openai/)
- [dsRAG Documentation](https://github.com/D-Star-AI/dsRAG)

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Azure service status
3. Check dsRAG GitHub issues
4. Open a new issue with:
   - Error messages
   - Test output
   - Environment details (OS, Python version, etc.)
