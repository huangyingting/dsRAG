# NOTE: this is deprecated but remains for backwards compatibility.
# The new way to specify model names for Chat is {provider}/{model_name} so we don't have to do this lookup (and maintain an up-to-date mapping here).

ANTHROPIC_MODEL_NAMES = [
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
]

GEMINI_MODEL_NAMES = [
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002", 
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-pro-exp",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-pro-preview-03-25", 
]

OPENAI_MODEL_NAMES = [
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "o1",
    "o1-2024-12-17"
    "o1-preview",
    "o1-preview-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
    "chatgpt-4o-latest",
]

AZURE_OPENAI_MODEL_NAMES = [
    "azure/gpt-4o-mini",
    "azure/gpt-4o",
    "azure/o1",
    "azure/o1-preview",
    "azure/o3-mini",
    "azure/o3",
    "azure/o4-mini",
    "azure/gpt-4.5-preview",
    "azure/gpt-4.1-nano",
    "azure/gpt-4.1-mini",
    "azure/gpt-4.1",
]