"""
LLM provider configuration models.
Compatible with load_config_from_yaml_file and load_config_from_json_file.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, RootModel

if TYPE_CHECKING:
    from llm_providers.base import BaseLLMProvider


class LLMProvidersDictMixin:
    """
    Mixin providing get_providers_dict() for any config that has LLM providers.
    Subclasses must implement _get_providers_config() to return the providers dict.
    """

    @abstractmethod
    def _get_providers_config(self) -> "Dict[str, LLMProviderConfig]":
        """Return the provider name -> LLMProviderConfig mapping."""
        ...

    def get_providers_dict(self) -> "Dict[str, BaseLLMProvider]":
        """Instantiate providers and return model name -> provider map."""
        from llm_providers.anthropic_provider import AnthropicProvider
        from llm_providers.base import BaseLLMProvider
        from llm_providers.gemini_provider import GeminiProvider
        from llm_providers.huggingface_provider import HuggingFaceProvider
        from llm_providers.ollama_provider import OllamaProvider
        from llm_providers.openai_provider import OpenAIProvider
        from llm_providers.vllm_native_provider import VLLMNativeProvider
        from llm_providers.vllm_provider import VLLMProvider

        _local_providers = ("ollama", "vllm", "vllm_native")

        providers: Dict[str, BaseLLMProvider] = {}
        for provider_name, provider_config in self._get_providers_config().items():
            for model_cfg, kwargs in provider_config.iter_models():
                if model_cfg.skip:
                    continue

                kwargs = dict(kwargs)
                # Reuse the Google AI Studio / Gemini API key from embeddings when unset.
                if provider_name == "gemini" and not kwargs.get("api_key"):
                    emb = getattr(self, "embeddings", None)
                    if emb is not None and getattr(emb, "api_key", ""):
                        kwargs["api_key"] = emb.api_key

                # Local providers do not require a cloud API key.
                if provider_name not in _local_providers and not kwargs.get("api_key"):
                    print(f"Skipping {model_cfg.model} - no API key")
                    continue

                try:
                    if provider_name == "openai":
                        providers[model_cfg.model] = OpenAIProvider(**kwargs)
                    elif provider_name == "anthropic":
                        providers[model_cfg.model] = AnthropicProvider(**kwargs)
                    elif provider_name == "gemini":
                        providers[model_cfg.model] = GeminiProvider(**kwargs)
                    elif provider_name == "huggingface":
                        providers[model_cfg.model] = HuggingFaceProvider(**kwargs)
                    elif provider_name == "vllm":
                        vk = dict(kwargs)
                        if not vk.get("api_key"):
                            vk["api_key"] = "EMPTY"
                        providers[model_cfg.model] = VLLMProvider(**vk)
                    elif provider_name == "vllm_native":
                        vk = dict(kwargs)
                        vk.pop("api_key", None)
                        providers[model_cfg.model] = VLLMNativeProvider(**vk)
                        print(f"Added {model_cfg.model} (vllm_native)")
                    elif provider_name == "ollama":
                        candidate = OllamaProvider(**kwargs)
                        if candidate.check_server_status():
                            providers[model_cfg.model] = candidate
                            print(f"Added {model_cfg.model} (ollama)")
                        else:
                            print(f"Skipping {model_cfg.model} - ollama server not running")
                    else:
                        print(f"Unknown provider: {provider_name}")
                except Exception as e:
                    print(f"Failed to setup {model_cfg.model}: {e}")

        return providers


class LLMModelConfig(BaseModel):
    """Per-model configuration (one entry per model under a provider)."""

    model_config = ConfigDict(extra="allow")

    model: str
    temperature: float = 0.7
    skip: bool = False
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    base_url: Optional[str] = None
    timeout: Optional[int] = None
    node_name: Optional[str] = None
    node_port: Optional[int] = None
    think: Optional[bool] = None
    options: Optional[Dict[str, Any]] = None

    def to_provider_kwargs(self, provider_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Convert to kwargs for provider constructor, merging provider-level api_key."""
        d = self.model_dump(exclude_none=True, exclude={"skip"})
        if d.get("api_key") is None and provider_api_key is not None:
            d["api_key"] = provider_api_key
        return d


class LLMProviderConfig(BaseModel):
    """Per-provider configuration: api_key plus an array of model configs."""

    model_config = ConfigDict(extra="allow")

    api_key: Optional[str] = None
    models: List[LLMModelConfig]

    def iter_models(self):
        """Yield (model_config, merged_kwargs) for each model, with provider api_key applied."""
        for m in self.models:
            yield m, m.to_provider_kwargs(self.api_key)


class LLMProvidersConfig(LLMProvidersDictMixin, RootModel[Dict[str, LLMProviderConfig]]):
    """
    Wrapper for a config file: mapping provider name -> LLMProviderConfig.
    Use with load_config_from_yaml_file(LLMProvidersConfig, path) or
    load_config_from_json_file(LLMProvidersConfig, path).
    Call .get_providers_dict() to get instantiated providers: Dict[str, BaseLLMProvider].
    """

    def _get_providers_config(self) -> Dict[str, LLMProviderConfig]:
        return self.root

    def get_providers(self) -> Dict[str, LLMProviderConfig]:
        """Return the provider name -> config map (raw config, not instances)."""
        return self.root
