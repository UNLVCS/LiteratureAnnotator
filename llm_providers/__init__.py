"""
LLM Provider Interface Package

This package provides a unified interface for different LLM providers
using LangChain as the underlying framework.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .base import BaseLLMProvider, LLMResponse, Query
from .config_models import LLMModelConfig, LLMProviderConfig, LLMProvidersConfig

# Backends load on first access (PEP 562), so `import llm_providers` does not pull in
# vLLM, torch or every langchain integration.
_LAZY = {
    "OpenAIProvider": ".openai_provider",
    "AnthropicProvider": ".anthropic_provider",
    "HuggingFaceProvider": ".huggingface_provider",
    "OllamaProvider": ".ollama_provider",
    "VLLMProvider": ".vllm_provider",
    "VLLMNativeProvider": ".vllm_native_provider",
}

if TYPE_CHECKING:
    from .anthropic_provider import AnthropicProvider
    from .huggingface_provider import HuggingFaceProvider
    from .ollama_provider import OllamaProvider
    from .openai_provider import OpenAIProvider
    from .vllm_native_provider import VLLMNativeProvider
    from .vllm_provider import VLLMProvider


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BaseLLMProvider',
    'Query',
    'LLMResponse',
    'LLMModelConfig',
    'LLMProviderConfig',
    'LLMProvidersConfig',
    'OpenAIProvider',
    'AnthropicProvider',
    'HuggingFaceProvider',
    'OllamaProvider',
    'VLLMProvider',
    'VLLMNativeProvider',
]
