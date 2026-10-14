"""
Anthropic provider implementation using the official ``anthropic`` SDK.

Supports prompt caching: when ``Query.cache_prefix_len`` is set, that many leading
characters of the prompt are sent as a separate content block marked
``cache_control: ephemeral``. Labeling sends the same paper with six different criterion
tasks, so after the first call the paper is read from cache at ~10% of the input price.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import anthropic

from .base import BaseLLMProvider, LLMResponse, Query

# Claude 4.7+ and the 5.x family reject non-default temperature/top_p (400).
_NO_SAMPLING = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(5|4-7|4-8)")


def _anthropic_temperature_top_p(
    temperature: float, top_p: float
) -> tuple[Optional[float], Optional[float]]:
    """
    Anthropic Claude 4+ rejects requests that send both ``temperature`` and ``top_p``.
    Prefer ``temperature`` unless ``top_p`` is explicitly set below 1.0.
    """
    if top_p != 1.0:
        return None, top_p
    return temperature, None


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic provider using the Messages API directly.

    Config (``llm_providers.anthropic.models[]`` in .env.yaml) may set ``max_tokens``,
    ``temperature``, ``top_p``, ``effort`` (``output_config.effort`` on models that
    support it), ``max_concurrency`` (parallel requests per batch, default 8) and
    ``cache_ttl`` (``"5m"`` default or ``"1h"``).
    """

    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        super().__init__(api_key, model or "claude-sonnet-4-6", **kwargs)
        # The SDK retries 408/409/429/5xx and connection errors with backoff.
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=kwargs.get("max_retries", 6))
        self.max_concurrency = int(kwargs.get("max_concurrency", 8))
        self.cache_ttl = kwargs.get("cache_ttl")

    def _request(self, query: Query) -> Dict[str, Any]:
        model = query.model or self.default_model
        if query.cache_prefix_len:
            prefix_ctl: Dict[str, Any] = {"type": "ephemeral"}
            if self.cache_ttl:
                prefix_ctl["ttl"] = self.cache_ttl
            content: Any = [
                {"type": "text", "text": query.prompt[: query.cache_prefix_len], "cache_control": prefix_ctl},
                {"type": "text", "text": query.prompt[query.cache_prefix_len :]},
            ]
        else:
            content = query.prompt

        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": query.max_tokens or self.config.get("max_tokens") or 1000,
            "messages": [{"role": "user", "content": content}],
        }
        if query.system_message:
            params["system"] = query.system_message
        if query.stop:
            params["stop_sequences"] = query.stop
        # anthropic SDK 1.x dropped the sampling keywords; models before 4.7 still honour
        # them, so they travel in extra_body (merged into the request JSON as-is).
        if not _NO_SAMPLING.search(model):
            temperature, top_p = _anthropic_temperature_top_p(query.temperature, query.top_p)
            sampling = {k: v for k, v in (("temperature", temperature), ("top_p", top_p)) if v is not None}
            if sampling:
                params["extra_body"] = sampling
        if self.config.get("effort"):
            params["output_config"] = {"effort": self.config["effort"]}
        return params

    def call_api(self, query: Query) -> LLMResponse:
        if not self.validate_query(query):
            raise ValueError("Invalid query for Anthropic provider")

        params = self._request(query)
        response = self.client.messages.create(**params)

        text = "".join(block.text for block in response.content if block.type == "text")
        u = response.usage
        cache_write = u.cache_creation_input_tokens or 0
        cache_read = u.cache_read_input_tokens or 0
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
            "total_tokens": u.input_tokens + cache_write + cache_read + u.output_tokens,
        }
        metadata = self._prepare_metadata(
            query, {"model_name": response.model, "request_id": response._request_id}
        )
        return LLMResponse(
            content=text,
            model=response.model,
            usage=usage,
            metadata=metadata,
            finish_reason=response.stop_reason,
        )

    def call_api_batch(self, queries: List[Query]) -> List[LLMResponse]:
        """Concurrent requests, returned in input order.

        When the queries share a cacheable prefix, the first runs alone so it writes
        the cache; requests sent in parallel before that would all miss it.
        """
        if not queries:
            return []
        first = queries[0]
        shared_prefix = bool(first.cache_prefix_len) and all(
            q.cache_prefix_len == first.cache_prefix_len
            and q.prompt[: q.cache_prefix_len] == first.prompt[: first.cache_prefix_len]
            for q in queries
        )
        head: List[LLMResponse] = []
        rest = queries
        if shared_prefix and len(queries) > 1:
            head = [self.call_api(first)]
            rest = queries[1:]
        with ThreadPoolExecutor(max_workers=max(1, min(self.max_concurrency, len(rest)))) as pool:
            return head + list(pool.map(self.call_api, rest))

    def get_available_models(self) -> List[str]:
        # Hints only; validate_query does not require models to be in this list.
        return [
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ]

    def validate_query(self, query: Query) -> bool:
        if not query.prompt or not isinstance(query.prompt, str):
            return False
        if query.model is not None and not isinstance(query.model, str):
            return False
        if query.temperature < 0 or query.temperature > 1:
            return False
        if query.max_tokens is not None and query.max_tokens < 1:
            return False
        if query.cache_prefix_len is not None and not 0 < query.cache_prefix_len < len(query.prompt):
            return False
        return True
