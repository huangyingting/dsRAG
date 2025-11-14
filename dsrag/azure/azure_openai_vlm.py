"""Azure OpenAI VLM (Vision Language Model) implementation for dsRAG."""

import os
from typing import Any, Dict, Optional
import base64

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError(
        "OpenAI package not found. Install with: pip install 'dsrag[openai]'"
    )

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.dsparse.file_parsing.vlm_clients import VLM


class AzureOpenAIVLM(VLM):
    """
    Azure OpenAI VLM client for vision-language models.
    
    Supports Azure OpenAI vision-capable models like GPT-4 Vision (gpt-4-vision-preview,
    gpt-4-turbo, gpt-4o, etc.) for analyzing images and extracting structured data.
    
    Fields
    ------
    - deployment_name: Azure OpenAI deployment name (required)
    - azure_endpoint: Azure OpenAI endpoint URL (optional, falls back to env var)
    - api_key: Azure OpenAI API key (optional, falls back to env var)
    - api_version: Azure API version (default: "2024-02-15-preview")
    
    Behavior
    --------
    - Encodes images as base64 for API submission
    - Supports JSON mode for structured outputs
    - Uses Azure OpenAI's vision capabilities
    """
    
    def __init__(
        self,
        deployment_name: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ) -> None:
        """
        Initialize Azure OpenAI VLM client.
        
        Args:
            deployment_name: Name of the Azure OpenAI deployment (must be a vision model)
            azure_endpoint: Azure OpenAI endpoint URL (falls back to AZURE_OPENAI_ENDPOINT env var)
            api_key: Azure OpenAI API key (falls back to AZURE_OPENAI_API_KEY env var)
            api_version: Azure OpenAI API version
        """
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
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image file as base64 string.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def make_llm_call(
        self,
        image_path: str,
        system_message: str,
        response_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.5,
    ) -> str:
        """
        Perform a VLM call to Azure OpenAI and return the response text.
        
        Args:
            image_path: Path to the image file to analyze
            system_message: System/user message with instructions
            response_schema: Optional JSON schema for structured output
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
        
        Returns:
            Response text from the model
        """
        # Encode the image
        base64_image = self._encode_image(image_path)
        
        # Determine image format from file extension
        image_ext = os.path.splitext(image_path)[1].lower()
        if image_ext in ['.jpg', '.jpeg']:
            media_type = "image/jpeg"
        elif image_ext == '.png':
            media_type = "image/png"
        elif image_ext == '.gif':
            media_type = "image/gif"
        elif image_ext == '.webp':
            media_type = "image/webp"
        else:
            # Default to jpeg if unknown
            media_type = "image/jpeg"
        
        # Build the message content
        message_content = [
            {
                "type": "text",
                "text": system_message
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}"
                }
            }
        ]
        
        # Build messages array
        messages = [
            {
                "role": "user",
                "content": message_content
            }
        ]
        
        # Prepare API call parameters
        api_params = {
            "model": self.deployment_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add response format for JSON mode if schema provided
        if response_schema is not None:
            api_params["response_format"] = {"type": "json_object"}
        
        # Make the API call
        response = self.client.chat.completions.create(**api_params)
        
        return response.choices[0].message.content.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this VLM instance to a dictionary.
        
        Returns:
            Dictionary containing configuration
        """
        return {
            "subclass_name": self.__class__.__name__,
            "deployment_name": self.deployment_name,
            "azure_endpoint": self.azure_endpoint,
            "api_key": self.api_key,
            "api_version": self.api_version,
        }
