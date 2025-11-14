"""Integration tests for Azure components with dsRAG.

These tests verify that Azure Blob Storage, Azure OpenAI Chat, and Azure OpenAI Embeddings
work correctly with the KnowledgeBase class.

To run these tests, you need to set the following environment variables:
- AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY)
- AZURE_STORAGE_CONTAINER_NAME
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_CHAT_DEPLOYMENT
- AZURE_OPENAI_EMBEDDING_DEPLOYMENT
"""

import os
import sys
import unittest
import dotenv

dotenv.load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Check if Azure dependencies are available
try:
    from dsrag.azure.blob_storage import AzureBlobStorage
    from dsrag.azure.azure_openai_chat import AzureOpenAIChatAPI
    from dsrag.azure.azure_openai_embedding import AzureOpenAIEmbedding
    AZURE_AVAILABLE = True
except ImportError as e:
    AZURE_AVAILABLE = False
    AZURE_IMPORT_ERROR = str(e)

from dsrag.knowledge_base import KnowledgeBase


@unittest.skipUnless(AZURE_AVAILABLE, f"Azure dependencies not available: {AZURE_IMPORT_ERROR if not AZURE_AVAILABLE else ''}")
class TestAzureIntegration(unittest.TestCase):
    """Integration tests for Azure components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Check for required environment variables
        cls.required_env_vars = [
            "AZURE_STORAGE_CONTAINER_NAME",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        ]
        
        cls.missing_vars = [var for var in cls.required_env_vars if not os.environ.get(var)]
        
        if cls.missing_vars:
            raise unittest.SkipTest(
                f"Missing required environment variables: {', '.join(cls.missing_vars)}"
            )
        
        # Set up Azure components
        cls.base_path = os.path.expanduser("~/dsrag_test_azure")
        cls.container_name = os.environ["AZURE_STORAGE_CONTAINER_NAME"]
        
        # Initialize Azure Blob Storage
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if connection_string:
            cls.azure_storage = AzureBlobStorage(
                base_path=cls.base_path,
                container_name=cls.container_name,
                connection_string=connection_string,
            )
        else:
            account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
            account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            if not account_name or not account_key:
                raise unittest.SkipTest(
                    "Either AZURE_STORAGE_CONNECTION_STRING or both "
                    "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be set"
                )
            cls.azure_storage = AzureBlobStorage(
                base_path=cls.base_path,
                container_name=cls.container_name,
                account_name=account_name,
                account_key=account_key,
            )
        
        # Initialize Azure OpenAI components
        cls.azure_chat = AzureOpenAIChatAPI(
            deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            temperature=0.2,
            max_tokens=1000,
        )
        
        cls.azure_embedding = AzureOpenAIEmbedding(
            deployment_name=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            dimension=1536,  # Standard for text-embedding-ada-002
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
        
        cls.kb_id = "test_azure_kb"
    
    def test_001_create_kb_with_azure_components(self):
        """Test creating a knowledge base with Azure components."""
        kb = KnowledgeBase(
            kb_id=self.kb_id,
            embedding_model=self.azure_embedding,
            auto_context_model=self.azure_chat,
            file_system=self.azure_storage,
            exists_ok=False,
        )
        
        self.assertIsInstance(kb.embedding_model, AzureOpenAIEmbedding)
        self.assertIsInstance(kb.auto_context_model, AzureOpenAIChatAPI)
        self.assertIsInstance(kb.file_system, AzureBlobStorage)
    
    def test_002_add_document_to_azure_kb(self):
        """Test adding a document to a knowledge base with Azure components."""
        kb = KnowledgeBase(kb_id=self.kb_id)
        
        # Add a simple test document
        test_text = """
        Azure is a cloud computing platform by Microsoft. It provides a wide range of services
        including computing, analytics, storage, and networking. Users can pick and choose from
        these services to develop and scale new applications, or run existing applications in
        the public cloud.
        
        Azure OpenAI Service provides REST API access to OpenAI's powerful language models including
        the GPT-4, GPT-3.5-Turbo, and Embeddings model series. These models can be easily adapted
        to your specific task including but not limited to content generation, summarization,
        semantic search, and natural language to code translation.
        """
        
        kb.add_document(
            doc_id="azure_intro",
            text=test_text,
            document_title="Introduction to Azure",
        )
        
        # Verify document was added
        doc_ids = kb.chunk_db.get_all_doc_ids()
        self.assertIn("azure_intro", doc_ids)
    
    def test_003_query_azure_kb(self):
        """Test querying a knowledge base with Azure components."""
        kb = KnowledgeBase(kb_id=self.kb_id)
        
        # Query the knowledge base
        results = kb.query(
            search_queries=["What is Azure?"],
            rse_params="balanced",
        )
        
        # Should get at least one result
        self.assertGreater(len(results), 0)
        
        # Results should have required fields
        for result in results:
            self.assertIn("doc_id", result)
            self.assertIn("content", result)
            self.assertIn("score", result)
    
    def test_004_azure_chat_basic_call(self):
        """Test basic chat call with Azure OpenAI."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = self.azure_chat.make_llm_call(messages)
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # The response should mention "4"
        self.assertIn("4", response)
    
    def test_005_azure_embedding_basic_call(self):
        """Test basic embedding call with Azure OpenAI."""
        texts = ["This is a test sentence.", "Another test sentence."]
        
        embeddings = self.azure_embedding.get_embeddings(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), self.azure_embedding.dimension)
        self.assertEqual(len(embeddings[1]), self.azure_embedding.dimension)
    
    def test_006_save_and_load_with_azure(self):
        """Test saving and loading KB configuration with Azure components."""
        kb = KnowledgeBase(kb_id=self.kb_id)
        
        # Save is automatic, now try to load
        kb2 = KnowledgeBase(kb_id=self.kb_id)
        
        # Verify components were restored correctly
        self.assertIsInstance(kb2.embedding_model, AzureOpenAIEmbedding)
        self.assertIsInstance(kb2.auto_context_model, AzureOpenAIChatAPI)
        self.assertIsInstance(kb2.file_system, AzureBlobStorage)
        
        # Verify configurations match
        self.assertEqual(
            kb2.embedding_model.deployment_name,
            self.azure_embedding.deployment_name
        )
        self.assertEqual(
            kb2.auto_context_model.deployment_name,
            self.azure_chat.deployment_name
        )
        self.assertEqual(
            kb2.file_system.container_name,
            self.azure_storage.container_name
        )
    
    def test_007_azure_blob_storage_operations(self):
        """Test Azure Blob Storage file operations."""
        import json
        from PIL import Image
        
        test_kb_id = "test_storage_ops"
        test_doc_id = "test_doc"
        
        # Test JSON save/load
        test_data = {"key": "value", "items": [1, 2, 3]}
        self.azure_storage.save_json(test_kb_id, test_doc_id, "test.json", test_data)
        loaded_data = self.azure_storage.load_data(test_kb_id, test_doc_id, "test")
        self.assertEqual(loaded_data, test_data)
        
        # Test image save (create a simple test image)
        test_image = Image.new('RGB', (100, 100), color='blue')
        self.azure_storage.save_image(test_kb_id, test_doc_id, "test_image.jpg", test_image)
        
        # Test page content save/load
        test_content = "This is test page content."
        self.azure_storage.save_page_content(test_kb_id, test_doc_id, 1, test_content)
        loaded_content = self.azure_storage.load_page_content(test_kb_id, test_doc_id, 1)
        self.assertEqual(loaded_content, test_content)
        
        # Clean up
        self.azure_storage.delete_directory(test_kb_id, test_doc_id)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        try:
            # Delete the test knowledge base
            kb = KnowledgeBase(kb_id=cls.kb_id)
            kb.delete()
        except Exception as e:
            print(f"Error cleaning up test KB: {e}")
        
        try:
            # Clean up local temporary files
            import shutil
            if os.path.exists(cls.base_path):
                shutil.rmtree(cls.base_path)
        except Exception as e:
            print(f"Error cleaning up local files: {e}")


@unittest.skipUnless(AZURE_AVAILABLE, "Azure dependencies not available")
class TestAzureSerializationDeserialization(unittest.TestCase):
    """Test serialization and deserialization of Azure components."""
    
    def test_azure_blob_storage_serialization(self):
        """Test AzureBlobStorage to_dict and from_dict."""
        with unittest.mock.patch('dsrag.azure.blob_storage.BlobServiceClient'):
            storage = AzureBlobStorage(
                base_path="/tmp/test",
                container_name="test-container",
                connection_string="test_conn_str",
            )
            
            serialized = storage.to_dict()
            
            self.assertEqual(serialized['subclass_name'], 'AzureBlobStorage')
            self.assertEqual(serialized['base_path'], '/tmp/test')
            self.assertEqual(serialized['container_name'], 'test-container')
    
    def test_azure_chat_serialization(self):
        """Test AzureOpenAIChatAPI to_dict."""
        with unittest.mock.patch('dsrag.azure.azure_openai_chat.AzureOpenAI'):
            chat = AzureOpenAIChatAPI(
                deployment_name="gpt-4",
                azure_endpoint="https://test.openai.azure.com",
                api_key="test_key",
            )
            
            serialized = chat.to_dict()
            
            self.assertEqual(serialized['subclass_name'], 'AzureOpenAIChatAPI')
            self.assertEqual(serialized['deployment_name'], 'gpt-4')
            self.assertEqual(serialized['azure_endpoint'], 'https://test.openai.azure.com')
    
    def test_azure_embedding_serialization(self):
        """Test AzureOpenAIEmbedding to_dict."""
        with unittest.mock.patch('dsrag.azure.azure_openai_embedding.AzureOpenAI'):
            embedding = AzureOpenAIEmbedding(
                deployment_name="text-embedding-ada-002",
                dimension=1536,
                azure_endpoint="https://test.openai.azure.com",
                api_key="test_key",
            )
            
            serialized = embedding.to_dict()
            
            self.assertEqual(serialized['subclass_name'], 'AzureOpenAIEmbedding')
            self.assertEqual(serialized['deployment_name'], 'text-embedding-ada-002')
            self.assertEqual(serialized['dimension'], 1536)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
