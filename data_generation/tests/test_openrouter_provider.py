"""OpenRouter integration tests with an in-memory HTTP client."""

from unittest.mock import Mock

import yaml

import pytest

from config import app_config, vault_client
from config.llm_providers_config import instantiate_provider
from config.vault_client import VaultSecretsClient
from llm_providers import OpenRouterProvider, Query
from test_labeling import PAPER
from labeling import label_paper


def test_openrouter_config_request_and_saved_metadata(monkeypatch, tmp_path):
    routing = {
        "order": ["test-host"],
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
        "require_parameters": True,
    }
    http = Mock()
    http.post.return_value.json.return_value = {
        "id": "gen-123",
        "model": "vendor/model",
        "provider": "Reported Host",
        "openrouter_metadata": {"sampling": {"temperature": 0.1}},
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 10}, "cost": 0.001},
    }
    vault = VaultSecretsClient()
    monkeypatch.setattr(vault, "login", Mock())
    monkeypatch.setattr(vault, "get_kv_secret", lambda path: {"api_key": "test-key"} if path == "openrouter" else {})
    overlay = vault.fetch_secret_overlay()
    assert overlay == {"llm_providers": {"openrouter": {"api_key": "test-key"}}}
    models = [{
        "model": "vendor/model",
        "top_p": 0.9,
        "provider": routing,
        "http_referer": "https://example.org",
        "x_title": "LiteratureAnnotator",
    }]
    monkeypatch.setattr(vault_client, "fetch_secret_overlay", lambda: overlay.copy())
    monkeypatch.setattr(app_config, "_app_config", None)
    monkeypatch.delenv("LA_CONFIG_OVERRIDE", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    assert app_config.load_app_config(config_path).get_provider_specs() == {}
    config_path.write_text(yaml.safe_dump({"llm_providers": {"openrouter": {"models": models}}}))
    config = app_config.load_app_config(config_path, reload=True)
    kind, kwargs = config.get_provider_specs()["vendor/model"]
    provider = instantiate_provider(kind, {**kwargs, "http_client": http})
    assert isinstance(provider, OpenRouterProvider)
    result = label_paper(PAPER, "vendor/model", provider, temperature=0.1, max_tokens=16000)

    assert http.post.call_count == len(result["criteria_results"]) == 6
    args, request = http.post.call_args
    assert args == ("https://openrouter.ai/api/v1/chat/completions",)
    assert request["headers"] == {
        "Authorization": "Bearer test-key",
        "HTTP-Referer": "https://example.org",
        "X-Title": "LiteratureAnnotator",
        "X-OpenRouter-Metadata": "enabled",
    }
    assert request["json"]["provider"] == routing
    assert [message["role"] for message in request["json"]["messages"]] == ["system", "user"]
    for criterion in result["criteria_results"]:
        metadata = criterion["openrouter"]
        assert metadata["host"] == "Reported Host"
        assert metadata["requested_sampling"] == {"temperature": 0.1, "top_p": 0.9, "max_tokens": 16000}
        assert metadata["response_metadata"] == {"sampling": {"temperature": 0.1}}
        assert criterion["finish_reason"] == "stop"
        assert criterion["usage"]["cached_prompt_tokens"] == 10
        assert criterion["usage"]["cost"] == 0.001


def test_openrouter_requires_key_and_handles_missing_response_metadata():
    with pytest.raises(ValueError, match="api_key"):
        OpenRouterProvider("", "vendor/model")

    http = Mock()
    http.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "answer"}}],
    }
    provider = OpenRouterProvider("test-key", "vendor/model", http_client=http)
    responses = provider.call_api_batch([
        Query("First", model="vendor/other", temperature=0.2, stop=["END"]),
        Query("Second"),
    ])

    first, second = [call.kwargs["json"] for call in http.post.call_args_list]
    assert first["model"] == "vendor/other"
    assert first["temperature"] == 0.2
    assert first["stop"] == ["END"]
    assert second["model"] == "vendor/model"
    assert second["temperature"] == 0.7
    assert "stop" not in second
    assert responses[1].metadata["openrouter"]["host"] is None
