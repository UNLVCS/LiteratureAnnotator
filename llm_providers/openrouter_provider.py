"""OpenRouter's OpenAI-compatible chat completions provider."""

from copy import deepcopy
from typing import Any

import requests

from .base import BaseLLMProvider, LLMResponse, Query, retry_with_backoff


class OpenRouterProvider(BaseLLMProvider):
    base_url = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        provider: dict[str, Any] | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        timeout: int = 300,
        http_client=None,
        **kwargs,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("OpenRouter requires an api_key")
        super().__init__(api_key, model, **kwargs)
        self.routing = deepcopy(provider) if provider is not None else {}
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "X-OpenRouter-Metadata": "enabled",
        }
        if http_referer:
            self.headers["HTTP-Referer"] = http_referer
        if x_title:
            self.headers["X-Title"] = x_title
        self.timeout = timeout
        self.http_client = http_client if http_client is not None else requests

    def get_available_models(self) -> list[str]:
        """Return the configured model without a catalog lookup."""
        return [self.default_model] if self.default_model else []

    def _sampling(self, query: Query) -> dict[str, Any]:
        defaults = {
            "temperature": 0.7,
            "max_tokens": None,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None,
        }
        sampling = {}
        for name, default in defaults.items():
            value = getattr(query, name)
            if value == default:
                value = self.config.get(name, default)
            # Leave neutral optional controls out so require_parameters does not
            # exclude hosts for controls the caller isn't using.
            if value is not None and (name == "temperature" or value != default):
                sampling[name] = value
        return sampling

    def validate_query(self, query: Query) -> bool:
        if not isinstance(query.prompt, str) or not query.prompt or not (query.model or self.default_model):
            return False
        sampling = self._sampling(query)
        return (
            0 <= sampling["temperature"] <= 2
            and 0 <= sampling.get("top_p", 1.0) <= 1
            and sampling.get("max_tokens", 1) >= 1
            and -2 <= sampling.get("frequency_penalty", 0.0) <= 2
            and -2 <= sampling.get("presence_penalty", 0.0) <= 2
        )

    def _complete(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self.http_client.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def call_api(self, query: Query) -> LLMResponse:
        if not self.validate_query(query):
            raise ValueError("Invalid query for OpenRouter provider")

        messages = []
        if query.system_message:
            messages.append({"role": "system", "content": query.system_message})
        messages.append({"role": "user", "content": query.prompt})
        model = query.model or self.default_model
        sampling = self._sampling(query)
        body = {"model": model, "messages": messages, **sampling}
        if self.routing:
            body["provider"] = self.routing

        response = retry_with_backoff(self._complete, body)
        choices = response.get("choices") or []
        if response.get("error") or not choices or choices[0].get("error"):
            raise RuntimeError(f"OpenRouter returned no usable completion (id={response.get('id')})")
        choice = choices[0]
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"OpenRouter returned no text completion (id={response.get('id')})")

        usage = dict(response.get("usage") or {})
        usage["cached_prompt_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        response_model = response.get("model") or model
        routing_metadata = response.get("openrouter_metadata") or {}
        metadata = self._prepare_metadata(query, {"id": response.get("id")})
        metadata["openrouter"] = {
            "host": response.get("provider"),
            "model": response_model,
            "generation_id": response.get("id"),
            "requested_sampling": sampling,
            "requested_routing": self.routing,
            "response_metadata": routing_metadata,
        }
        return LLMResponse(
            content=content,
            model=response_model,
            usage=usage,
            metadata=metadata,
            finish_reason=choice.get("finish_reason"),
        )
