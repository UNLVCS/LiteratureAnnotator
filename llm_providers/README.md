# LLM Provider Interface

A unified interface for different Large Language Model (LLM) providers using LangChain as the underlying framework. This package allows you to easily switch between different LLM providers while maintaining a consistent API.

## Features

- **Unified Interface**: Single API for multiple LLM providers
- **LangChain Integration**: Built on top of LangChain for reliability and consistency
- **Type Safety**: Full type hints and validation
- **Extensible**: Easy to add new providers
- **Query Validation**: Built-in validation for each provider's requirements

## Supported Providers

- **OpenAI**: GPT-3.5, GPT-4, and other OpenAI models
- **Anthropic**: Claude models (Claude-3, Claude-2, etc.)
- **Hugging Face**: Various open-source models from Hugging Face Hub

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from llm_providers import OpenAIProvider, Query

# Initialize provider
provider = OpenAIProvider(
    api_key="your-openai-api-key",
    model="gpt-3.5-turbo"
)

# Create a query
query = Query(
    prompt="Explain machine learning in simple terms",
    system_message="You are a helpful AI assistant",
    temperature=0.7,
    max_tokens=150
)

# Call the API
response = provider.call_api(query)
print(response.content)
```

## Usage Examples

### Basic Usage

```python
from llm_providers import OpenAIProvider, AnthropicProvider, Query

# OpenAI
openai_provider = OpenAIProvider(api_key="your-key", model="gpt-3.5-turbo")
response = openai_provider.call_api(Query(prompt="Hello, world!"))

# Anthropic
anthropic_provider = AnthropicProvider(api_key="your-key", model="claude-3-sonnet-20240229")
response = anthropic_provider.call_api(Query(prompt="Hello, world!"))
```

### Advanced Query Configuration

```python
query = Query(
    prompt="Write a creative story about a robot",
    system_message="You are a creative writing assistant",
    temperature=0.9,
    max_tokens=500,
    top_p=0.95,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    metadata={"user_id": "123", "session_id": "abc"}
)
```

### Provider-Agnostic Usage

```python
from llm_providers import OpenAIProvider, AnthropicProvider, Query

# List of available providers
providers = [
    OpenAIProvider(api_key="openai-key"),
    AnthropicProvider(api_key="anthropic-key")
]

query = Query(prompt="Explain quantum computing")

# Try each provider
for provider in providers:
    try:
        response = provider.call_api(query)
        print(f"{provider.get_provider_name()}: {response.content}")
    except Exception as e:
        print(f"{provider.get_provider_name()} failed: {e}")
```

## Query Object

The `Query` class provides a standardized way to specify requests:

```python
@dataclass
class Query:
    prompt: str                                    # The main prompt/question
    model: Optional[str] = None                   # Override default model
    temperature: float = 0.7                      # Randomness (0.0 to 2.0)
    max_tokens: Optional[int] = None              # Maximum response length
    top_p: float = 1.0                           # Nucleus sampling
    frequency_penalty: float = 0.0               # Reduce repetition
    presence_penalty: float = 0.0                # Encourage new topics
    stop: Optional[List[str]] = None             # Stop sequences
    system_message: Optional[str] = None         # System/instruction message
    metadata: Optional[Dict[str, Any]] = None    # Custom metadata
```

## Response Object

All providers return a standardized `LLMResponse`:

```python
@dataclass
class LLMResponse:
    content: str                          # The generated text
    model: str                           # Model used
    usage: Dict[str, Any]                # Token usage information
    metadata: Optional[Dict[str, Any]]   # Additional metadata
    finish_reason: Optional[str]         # Why generation stopped
```

## Adding New Providers

To add a new provider, inherit from `BaseLLMProvider`:

```python
from llm_providers.base import BaseLLMProvider, Query, LLMResponse

class MyProvider(BaseLLMProvider):
    def call_api(self, query: Query) -> LLMResponse:
        # Implement API call logic
        pass
    
    def get_available_models(self) -> List[str]:
        # Return list of available models
        pass
    
    def validate_query(self, query: Query) -> bool:
        # Validate query for this provider
        pass
```

## Environment Variables

Set these environment variables for API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export HUGGINGFACE_API_TOKEN="your-hf-token"
```

## Error Handling

The interface provides consistent error handling:

```python
try:
    response = provider.call_api(query)
except ValueError as e:
    print(f"Invalid query: {e}")
except Exception as e:
    print(f"API error: {e}")
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic usage for each provider
- Query validation
- Provider-agnostic usage
- Error handling

## Requirements

- Python 3.8+
- LangChain and provider-specific integrations
- Valid API keys for desired providers

## License

This project is part of the ADBM research project.
