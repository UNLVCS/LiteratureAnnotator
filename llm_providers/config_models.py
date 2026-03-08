"""
Pydantic models for LLM provider configuration.
Compatible with load_config_from_yaml_file and load_config_from_json_file.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, RootModel

if TYPE_CHECKING:
    from .base import BaseLLMProvider


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


class LLMProvidersConfig(RootModel[Dict[str, LLMProviderConfig]]):
    """
    Wrapper for a config file: mapping provider name -> LLMProviderConfig.
    Use with load_config_from_yaml_file(LLMProvidersConfig, path) or
    load_config_from_json_file(LLMProvidersConfig, path).
    Call .to_dict() to get instantiated providers: Dict[str, BaseLLMProvider].
    """

    def get_providers(self) -> Dict[str, LLMProviderConfig]:
        """Return the provider name -> config map (raw config, not instances)."""
        return self.root

    def to_dict(self) -> "Dict[str, BaseLLMProvider]":
        """Instantiate providers and return model name -> provider map."""
        from .anthropic_provider import AnthropicProvider
        from .base import BaseLLMProvider
        from .huggingface_provider import HuggingFaceProvider
        from .ollama_provider import OllamaProvider
        from .openai_provider import OpenAIProvider
        from .vllm_provider import VLLMProvider

        providers: Dict[str, BaseLLMProvider] = {}
        for provider_name, provider_config in self.root.items():
            for model_cfg, kwargs in provider_config.iter_models():
                if model_cfg.skip:
                    continue
                if provider_name != "ollama" and not kwargs.get("api_key"):
                    print(f"Skipping {model_cfg.model} - no API key")
                    continue

                try:
                    if provider_name == "openai":
                        providers[model_cfg.model] = OpenAIProvider(**kwargs)
                    elif provider_name == "anthropic":
                        providers[model_cfg.model] = AnthropicProvider(**kwargs)
                    elif provider_name == "huggingface":
                        providers[model_cfg.model] = HuggingFaceProvider(**kwargs)
                    elif provider_name == "vllm":
                        providers[model_cfg.model] = VLLMProvider(**kwargs)
                    elif provider_name == "ollama":
                        temp = OllamaProvider(**kwargs)
                        if temp.check_server_status():
                            providers[model_cfg.model] = OllamaProvider(**kwargs)
                            print(f"Added {model_cfg.model} (ollama)")
                        else:
                            print(f"Skipping {model_cfg.model} - ollama server not running")
                    else:
                        print(f"Unknown provider: {provider_name}")
                except Exception as e:
                    print(f"Failed to setup {model_cfg.model}: {e}")

        return providers