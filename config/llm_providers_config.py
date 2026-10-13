"""
LLM provider configuration models.
Compatible with load_config_from_yaml_file and load_config_from_json_file.
"""

import importlib
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, RootModel

if TYPE_CHECKING:
    from llm_providers.base import BaseLLMProvider

# Provider type -> (module, class). Backends are imported only when used: vllm_native
# pulls in vLLM/torch and huggingface needs transformers, which most runs never touch.
_PROVIDER_CLASSES: Dict[str, Tuple[str, str]] = {
    "openai": ("llm_providers.openai_provider", "OpenAIProvider"),
    "anthropic": ("llm_providers.anthropic_provider", "AnthropicProvider"),
    "huggingface": ("llm_providers.huggingface_provider", "HuggingFaceProvider"),
    "vllm": ("llm_providers.vllm_provider", "VLLMProvider"),
    "vllm_native": ("llm_providers.vllm_native_provider", "VLLMNativeProvider"),
    "ollama": ("llm_providers.ollama_provider", "OllamaProvider"),
}
_LOCAL_PROVIDERS = ("ollama", "vllm", "vllm_native")


def _provider_class(provider_type: str) -> type:
    module_name, class_name = _PROVIDER_CLASSES[provider_type]
    return getattr(importlib.import_module(module_name), class_name)


class LLMProvidersDictMixin:
    """
    Mixin providing get_providers_dict() for any config that has LLM providers.
    Subclasses must implement _get_providers_config() to return the providers dict.
    """

    @abstractmethod
    def _get_providers_config(self) -> "Dict[str, LLMProviderConfig]":
        """Return the provider name -> LLMProviderConfig mapping."""
        ...

    def get_provider_specs(self) -> "Dict[str, Tuple[str, Dict[str, Any]]]":
        """
        Return ``{model_name: (provider_type, kwargs)}`` for every non-skipped model.

        This is config-only — no GPU/network side-effects — making it safe to
        call in the coordinator process before forking workers.  Each worker can
        then call :func:`instantiate_provider` with its own entry to load only
        its own model.
        """
        specs: Dict[str, Tuple[str, Any]] = {}
        for provider_name, provider_config in self._get_providers_config().items():
            for model_cfg, kwargs in provider_config.iter_models():
                if model_cfg.skip:
                    continue
                specs[model_cfg.model] = (provider_name, kwargs)
        return specs

    def get_providers_dict(self) -> "Dict[str, BaseLLMProvider]":
        """Instantiate providers and return model name -> provider map.

        Only the backends that are configured (and not skipped) are imported, so an
        OpenAI/Anthropic-only config never loads vLLM or torch.
        """
        providers: Dict[str, BaseLLMProvider] = {}
        for provider_name, provider_config in self._get_providers_config().items():
            for model_cfg, kwargs in provider_config.iter_models():
                if model_cfg.skip:
                    continue
                # Local providers do not require a cloud API key.
                if provider_name not in _LOCAL_PROVIDERS and not kwargs.get("api_key"):
                    print(f"Skipping {model_cfg.model} - no API key")
                    continue
                if provider_name not in _PROVIDER_CLASSES:
                    print(f"Unknown provider: {provider_name}")
                    continue
                try:
                    providers[model_cfg.model] = instantiate_provider(provider_name, kwargs)
                    if provider_name in _LOCAL_PROVIDERS:
                        print(f"Added {model_cfg.model} ({provider_name})")
                except Exception as e:
                    print(f"Skipping {model_cfg.model}: {e}")

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


def instantiate_provider(
    provider_type: str, provider_kwargs: Dict[str, Any]
) -> "BaseLLMProvider":
    """
    Instantiate a single provider by type name and kwargs.

    Used by ``labeler_mp.py`` worker processes so each worker loads only its
    own model rather than all models at once.

    Args:
        provider_type: One of ``"openai"``, ``"anthropic"``, ``"huggingface"``,
                       ``"vllm"``, ``"vllm_native"``, ``"ollama"``.
        provider_kwargs: Constructor kwargs as returned by
                         :meth:`LLMProvidersDictMixin.get_provider_specs`.

    Returns:
        Instantiated :class:`~llm_providers.base.BaseLLMProvider`.
    """
    if provider_type not in _PROVIDER_CLASSES:
        raise ValueError(f"Unknown provider type: '{provider_type}'")
    if provider_type not in _LOCAL_PROVIDERS and not provider_kwargs.get("api_key"):
        raise ValueError(f"Provider '{provider_type}' requires an api_key in .env.yaml")

    cls = _provider_class(provider_type)
    kw = dict(provider_kwargs)
    if provider_type == "vllm" and not kw.get("api_key"):
        kw["api_key"] = "EMPTY"
    if provider_type == "vllm_native":
        kw.pop("api_key", None)
    provider = cls(**kw)
    if provider_type == "ollama" and not provider.check_server_status():
        raise RuntimeError(f"Ollama server not reachable at {provider.base_url}")
    return provider


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
