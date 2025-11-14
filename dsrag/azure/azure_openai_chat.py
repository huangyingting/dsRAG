"""Azure OpenAI Chat API implementation for dsRAG."""

import os
from typing import Optional

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError(
        "OpenAI package not found. Install with: pip install 'dsrag[openai]'"
    )

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dsrag.llm import LLM


class AzureOpenAIChatAPI(LLM):
    """
    Azure OpenAI Chat API implementation.
    
    Uses Azure OpenAI Service for chat completions. Requires Azure OpenAI endpoint,
    API key, and deployment name to be configured.
    """
    
    def __init__(
        self,
        deployment_name: str,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.2,
        max_tokens: int = 1000,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Azure OpenAI Chat API.
        
        Args:
            deployment_name: Name of the Azure OpenAI deployment
            api_version: Azure OpenAI API version
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            azure_endpoint: Azure OpenAI endpoint URL (falls back to AZURE_OPENAI_ENDPOINT env var)
            api_key: Azure OpenAI API key (falls back to AZURE_OPENAI_API_KEY env var)
        """
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
    
    def make_llm_call(self, chat_messages: list[dict]) -> str:
        """
        Make a chat completion call to Azure OpenAI.
        
        Args:
            chat_messages: List of message dictionaries in OpenAI format
        
        Returns:
            Response text from the model
        """
        response = self.client.chat.completions.create(
            model=self.deployment_name,  # In Azure, this is the deployment name
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        llm_output = response.choices[0].message.content.strip()
        return llm_output
    
    def to_dict(self):
        """Serialize configuration to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'deployment_name': self.deployment_name,
            'api_version': self.api_version,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'azure_endpoint': self.azure_endpoint,
            'api_key': self.api_key,
        })
        return base_dict
