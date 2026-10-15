"""OllamaProvider sends the `think` and `options` set in .env.yaml."""

from unittest.mock import MagicMock, patch

from llm_providers.base import Query
from llm_providers.ollama_provider import OllamaProvider


def _payload(provider, query):
    response = MagicMock()
    response.json.return_value = {"response": "ok", "done": True}
    with patch("llm_providers.ollama_provider.requests.post", return_value=response) as post:
        provider.call_api(query)
    return post.call_args.kwargs["json"]


def test_config_options_and_think_are_sent():
    provider = OllamaProvider(model="m", options={"num_ctx": 65536}, think=True)
    payload = _payload(provider, Query(prompt="hi", temperature=0.1, max_tokens=50))
    assert payload["think"] is True
    assert payload["options"]["num_ctx"] == 65536
    assert payload["options"]["temperature"] == 0.1
    assert payload["options"]["num_predict"] == 50


def test_query_values_override_config_options():
    provider = OllamaProvider(model="m", options={"temperature": 0.9, "num_predict": 10})
    payload = _payload(provider, Query(prompt="hi", temperature=0.1, max_tokens=50))
    assert payload["options"]["temperature"] == 0.1
    assert payload["options"]["num_predict"] == 50


def test_defaults_unchanged_without_config():
    payload = _payload(OllamaProvider(model="m"), Query(prompt="hi"))
    assert payload["think"] is False
    assert "num_ctx" not in payload["options"]
