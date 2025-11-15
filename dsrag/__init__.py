import logging

# Configure the root dsrag logger with a NullHandler to prevent "No handler found" warnings
# This follows Python best practices for library logging
# Users will need to configure their own handlers if they want to see dsrag logs
logger = logging.getLogger("dsrag")
logger.addHandler(logging.NullHandler())

# Import Azure components to register them in the subclass registry
# This ensures they can be deserialized from saved KB configurations
try:
    from dsrag.azure import (
        AzureBlobStorage,
        AzureOpenAIChatAPI,
        AzureOpenAIEmbedding,
        AzureOpenAIVLM,
    )
except ImportError:
    # Azure dependencies are optional
    pass