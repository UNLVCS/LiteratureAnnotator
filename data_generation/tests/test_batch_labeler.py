"""Tests for batch_labeler request/result conversion (no network, no MinIO)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "data_generation")]

from batch_labeler import OpenAIBatches, custom_id, parse_custom_id  # noqa: E402
from llm_providers.base import Query  # noqa: E402


def _openai(model="gpt-6-luna", **kwargs):
    backend = OpenAIBatches.__new__(OpenAIBatches)  # no client: these tests never call the API
    backend.model, backend.kwargs = model, kwargs
    return backend


def test_custom_id_round_trip():
    for i in range(6):
        assert parse_custom_id(custom_id("41000001", i)) == ("41000001", i)


def test_openai_request_omits_temperature_for_reasoning_models():
    q = Query(prompt="paper + task", system_message="sys", temperature=0.1, max_tokens=16000)
    body = _openai("gpt-6-luna", reasoning_effort="medium").request("41000001-c1", q, "41000001")["body"]
    assert "temperature" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 16000
    assert body["reasoning_effort"] == "medium"
    assert body["prompt_cache_key"] == "paper-41000001"
    assert _openai("gpt-4o").request("x-c1", q, "x")["body"]["temperature"] == 0.1


def test_openai_outcome_success_and_error():
    ok = {
        "custom_id": "41000001-c2",
        "response": {"status_code": 200, "request_id": "req_1", "body": {
            "model": "gpt-6-luna",
            "choices": [{"message": {"content": '{"criterion_2": {"satisfied": true}}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                      "prompt_tokens_details": {"cached_tokens": 64},
                      "completion_tokens_details": {"reasoning_tokens": 12}},
        }},
        "error": None,
    }
    cid, (resp, err) = _openai()._outcome(ok)
    assert cid == "41000001-c2" and err is None
    assert resp.content.startswith('{"criterion_2"') and resp.finish_reason == "stop"
    assert resp.usage["cached_prompt_tokens"] == 64 and resp.usage["reasoning_tokens"] == 12

    bad = {"custom_id": "41000001-c3", "response": {"status_code": 400, "body": {
        "error": {"code": "invalid_value", "message": "nope"}}}, "error": None}
    cid, (resp, err) = _openai()._outcome(bad)
    assert resp is None and "invalid_value" in err
