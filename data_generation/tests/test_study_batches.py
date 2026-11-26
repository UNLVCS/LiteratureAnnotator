"""Study request, result and guard checks with no service access."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from data_generation.batch_labeler import (
    AnthropicBatches,
    BatchLabeler,
    OpenAIBatches,
    OpenRouterBatches,
    read_pmids,
    usage_cost,
)
from config.app_config import AppConfig
from config.study_config import load_study
from llm_providers.base import Query


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def study():
    return load_study(ROOT / "config/study_models.yaml")


class FakeHTTP:
    def __init__(self):
        self.submitted = None

    def post(self, url, *, json, **kwargs):
        assert url.endswith("/batches")
        self.submitted = json
        return self.response({"id": "batch_1"})

    def get(self, url, *, params, **kwargs):
        assert url.endswith("/batches/batch_1/results")
        if "cursor" in params:
            return self.response({
                "data": [{"custom_id": "123-c2", "error": {"code": "failed"}}],
                "has_more": False,
                "next_cursor": None,
            })
        return self.response({
            "data": [{
                "custom_id": "123-c1",
                "response": {"status_code": 200, "body": {
                    "model": "test/model-2026",
                    "provider": "Together",
                    "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens_details": {"cached_tokens": 20}, "cost": 0.01},
                }},
            }],
            "has_more": True,
            "next_cursor": "next",
        })

    @staticmethod
    def response(payload):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


def test_openrouter_batch_uses_documented_pin_and_paginates():
    client = FakeHTTP()
    batch = OpenRouterBatches("test/model", {"api_key": "fake", "host": "together", "http_client": client})
    query = Query(prompt="paper first, criterion second", temperature=1, max_tokens=16000)
    row = batch.request("123-c1", query, "123")
    assert row["body"]["temperature"] == 1
    assert "provider" not in row["body"]
    assert batch.submit([row]) == "batch_1"
    assert list(client.submitted)[-1] == "requests"
    assert client.submitted["provider"] == {"only": ["together"], "allow_fallbacks": False}
    assert batch.status("batch_1")[0]
    (_, (result, error)), (_, (failed, detail)) = list(batch.results("batch_1"))
    assert error is None and result.model == "test/model-2026"
    assert result.usage["cached_prompt_tokens"] == 20
    assert result.metadata["openrouter"]["host"] == "Together"
    assert result.metadata["openrouter"]["quantization"] is None
    assert failed is None and "failed" in detail


def test_openrouter_refuses_missing_provider_pin():
    client = FakeHTTP()
    batch = OpenRouterBatches("test/model", {
        "api_key": "fake", "http_client": client,
    })
    with pytest.raises(ValueError, match="provider pin"):
        batch.submit([])
    assert client.submitted is None


def test_direct_opus_and_gpt_use_provider_defaults():
    query = Query(prompt="paper and task", cache_prefix_len=5, temperature=1, max_tokens=16000)
    anthropic = AnthropicBatches("claude-opus-5-5", {"model": "claude-opus-5-5", "api_key": "fake"})
    openai = OpenAIBatches("gpt-6.2-sol", {"api_key": "fake"})
    params = anthropic.request("123-c1", query, "123")["params"]
    assert not {"temperature", "top_p", "top_k", "thinking"} & params.keys()
    assert params["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    body = openai.request("123-c1", query, "123")["body"]
    assert "temperature" not in body
    assert body["max_completion_tokens"] == 16000


def _openrouter_body(model, **kwargs):
    batch = OpenRouterBatches(model, {"api_key": "fake", "host": "x", "http_client": FakeHTTP(), **kwargs})
    query = Query(prompt="PAPER|task", cache_prefix_len=6, system_message="sys",
                  temperature=1, max_tokens=16000)
    return batch.request("123-c1", query, "123")["body"]


def test_openrouter_claude_caches_the_paper_and_sends_no_sampling():
    body = _openrouter_body("anthropic/claude-opus-5.5")
    system, user = body["messages"]
    assert system == {"role": "system", "content": "sys"}
    cached, rest = user["content"]
    assert cached == {"type": "text", "text": "PAPER|",
                      "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    assert rest == {"type": "text", "text": "task"}
    assert not {"temperature", "top_p", "top_k", "prompt_cache_key"} & body.keys()
    assert body["max_tokens"] == 16000
    ttl = _openrouter_body("anthropic/claude-opus-5.5", cache_ttl="5m")["messages"][1]["content"][0]
    assert ttl["cache_control"]["ttl"] == "5m"


def test_openrouter_gpt_gets_cache_key_and_open_models_are_unchanged():
    body = _openrouter_body("openai/gpt-6.2-sol")
    assert body["prompt_cache_key"] == "paper-123"
    assert "temperature" not in body
    assert body["messages"][1]["content"] == "PAPER|task"
    body = _openrouter_body("z-ai/glm-5.3")
    assert body["temperature"] == 1 and "prompt_cache_key" not in body
    assert body["messages"][1]["content"] == "PAPER|task"


@pytest.mark.parametrize("model, sent", [
    ("claude-opus-5-5", False), ("anthropic/claude-opus-5.5", False),
    ("anthropic/claude-sonnet-4.7", False), ("anthropic/claude-sonnet-4.5", True),
    ("gpt-6.2-sol", False), ("openai/gpt-6.2-sol", False), ("openai/gpt-4.1", True),
    ("z-ai/glm-5.3", True), ("google/gemini-3.1-pro-preview", True),
])
def test_sends_temperature(model, sent):
    from data_generation.batch_labeler import sends_temperature
    assert sends_temperature(model) is sent


def test_sample_gate_and_usage_cost(study):
    labeler = BatchLabeler.__new__(BatchLabeler)
    labeler.study = study
    labeler.study_model = study.models[0]
    labeler.model = labeler.study_model.id
    labeler.sample_index = 1
    labeler.config = AppConfig()
    labeler.config.minio.output_prefix = study.sample_prefixes[1]
    labeler.gen = SimpleNamespace(result_exists=lambda model, pmid: False)
    with pytest.raises(SystemExit, match="s00"):
        labeler.check_study_submission(["123"])
    assert labeler.config.minio.output_prefix == study.sample_prefixes[1]
    labeler.gen.result_exists = lambda model, pmid: True
    for model in study.models:
        labeler.study_model = model
        labeler.check_study_submission(["123"])
    labeler.study_model = study.models[2].model_copy(update={"host": None})
    with pytest.raises(SystemExit, match="provider pin"):
        labeler.check_study_submission(["123"])
    direct_opus = study.models[0].model_copy(update={"route": "anthropic", "host": None})
    assert usage_cost({
        "input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }, direct_opus) == pytest.approx(16.1)
    # Through OpenRouter, Claude usage arrives in chat-completions form.
    assert usage_cost({
        "prompt_tokens": 3_000_000,
        "cached_prompt_tokens": 1_000_000,
        "prompt_tokens_details": {"cache_write_tokens": 1_000_000},
        "completion_tokens": 1_000_000,
    }, study.models[0]) == pytest.approx(16.1)


def test_manifest_and_offline_dry_run(study, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"papers": {"123": {}}}')
    assert read_pmids(manifest) == ["123"]
    paper = tmp_path / "123.json"
    paper.write_text('{"documents":[{"id":"123","passages":[{"offset":0,"text":"A paper."}]}]}')
    runner = BatchLabeler("gemini", None, study=study, sample_index=5, dry_run=True, papers_dir=tmp_path)
    assert runner.submit(read_pmids(manifest), dry_run=True) == 0
    assert runner.output_prefix == "study_v1/s05"
    assert "6 requests" in capsys.readouterr().out


def test_collection_preserves_result_shape_and_records_metadata(study, tmp_path, monkeypatch):
    import data_generation.batch_labeler as batch_module
    from llm_providers.base import LLMResponse
    from papers import parse_bioc
    from utilities.criteria import CRITERION_NAMES

    labeler = BatchLabeler.__new__(BatchLabeler)
    labeler.study_model = study.models[2]
    labeler.model = labeler.study_model.id
    labeler.sample_index = 0
    labeler.output_prefix = study.sample_prefixes[0]
    labeler.config = AppConfig()
    labeler.config.minio.synthetic_data_bucket = "synthetic-data-v3"
    labeler.state_path = tmp_path / "state.json"
    labeler.out_dir = tmp_path / "results"
    uploaded = []
    labeler.gen = SimpleNamespace(minio=object(), upload=uploaded.append)
    paper = parse_bioc("123", {"documents": [{"id": "123", "passages": [{"text": "A paper."}]}]})
    monkeypatch.setattr(batch_module, "load_paper_from_config", lambda *args, **kwargs: paper)

    def results(batch_id):
        for index, criterion in enumerate(CRITERION_NAMES, 1):
            yield f"123-c{index}", (LLMResponse(
                content=f'{{"{criterion}": {{"satisfied": true, "reason": "Because."}}}}',
                model="exact-version",
                usage={"cost": 0.25, "cached_prompt_tokens": 10},
                metadata={"created": 123, "openrouter": {"host": "Google Vertex", "quantization": "fp8"}},
            ), None)

    labeler.backend = SimpleNamespace(results=results)
    batch = {
        "id": "test", "bucket": "synthetic-data-v3", "output_prefix": labeler.output_prefix,
        "raw_prefix": "papers", "pmids": ["123"],
    }
    labeler.state = {"batches": [batch]}
    labeler.collect_batch(batch, "completed")

    record, = uploaded
    assert record["paper_id"] == "123" and record["provider"] == labeler.model
    assert record["sample_index"] == 0 and record["batch_id"] == "test"
    assert batch["actual_cost_usd"] == 1.5 and batch["collected"]
    for answer in record["criteria_results"]:
        assert answer["response"][answer["criterion"]]["satisfied"] is True
        assert answer["usage"]["cached_prompt_tokens"] == 10
        assert answer["generation"]["model"] == "exact-version"
        assert answer["generation"]["host"] == "Google Vertex"
        assert answer["generation"]["quantization"] == "fp8"
        assert answer["generation"]["response_timestamp"] == 123


@pytest.mark.parametrize("alias", ["opus", "sol", "gemini"])
def test_study_submit_reaches_backend_without_quantization(study, tmp_path, alias):
    runner = BatchLabeler(alias, None, study=study, dry_run=True, papers_dir=tmp_path)
    submitted = []
    runner.gen = SimpleNamespace(raw_prefix=lambda: "raw-papers")
    runner.backend.submit = lambda rows: submitted.extend(rows) or "fake-batch"
    runner.state_path = tmp_path / "state.json"
    runner.state = {"batches": []}
    runner.tokens_in_flight = lambda: 0
    (tmp_path / "123.json").write_text(
        '{"documents":[{"id":"123","passages":[{"offset":0,"text":"A paper."}]}]}'
    )

    assert runner.submit(["123"]) == 0
    assert len(submitted) == 6
    assert runner.state["batches"][0]["id"] == "fake-batch"
    assert runner.state["batches"][0]["output_prefix"] == "study_v1/s00"


def _capped_runner(study, tmp_path, cap):
    runner = BatchLabeler("opus", None, study=study, dry_run=True, papers_dir=tmp_path)
    runner.gen = SimpleNamespace(raw_prefix=lambda: "raw-papers")
    runner.state_path = tmp_path / "state.json"
    runner.state = {"batches": []}
    runner.tokens_in_flight = lambda: 0
    runner.max_batch_usd = cap
    for pmid in ("1", "2", "3"):
        (tmp_path / f"{pmid}.json").write_text(
            '{"documents":[{"id":"%s","passages":[{"offset":0,"text":"A paper."}]}]}' % pmid
        )
    return runner


def test_batch_cost_cap_splits_batches(study, tmp_path):
    # Opus worst case per paper is about 6 x 16,000 output tokens x $10/M = $0.96.
    runner = _capped_runner(study, tmp_path, cap=1.5)
    batches = []
    runner.backend.submit = lambda rows: batches.append(rows) or f"b{len(batches)}"
    assert runner.submit(["1", "2", "3"]) == 0
    assert [len(rows) for rows in batches] == [6, 6, 6]
    assert [b["pmids"] for b in runner.state["batches"]] == [["1"], ["2"], ["3"]]

    runner = _capped_runner(study, tmp_path, cap=None)
    batches = []
    runner.backend.submit = lambda rows: batches.append(rows) or "one"
    assert runner.submit(["1", "2", "3"]) == 0
    assert [len(rows) for rows in batches] == [18]


def test_insufficient_balance_stops_and_reports_unsent_papers(study, tmp_path, capsys):
    from data_generation.batch_labeler import InsufficientBalance

    runner = _capped_runner(study, tmp_path, cap=1.5)
    calls = []

    def submit(rows):
        calls.append(rows)
        if len(calls) == 2:
            raise InsufficientBalance("insufficient credits")
        return "b1"

    runner.backend.submit = submit
    assert runner.submit(["1", "2", "3"]) == 2
    assert [b["pmids"] for b in runner.state["batches"]] == [["1"]]
    assert "insufficient balance" in capsys.readouterr().out


def test_openrouter_maps_402_to_insufficient_balance():
    import requests
    from data_generation.batch_labeler import InsufficientBalance

    class Refusing(FakeHTTP):
        def post(self, url, *, json, **kwargs):
            response = SimpleNamespace(status_code=402, text="Insufficient credits")
            error = requests.HTTPError(response=response)

            def fail():
                raise error
            return SimpleNamespace(raise_for_status=fail, json=lambda: {})

    batch = OpenRouterBatches("test/model", {"api_key": "fake", "host": "x", "http_client": Refusing()})
    with pytest.raises(InsufficientBalance, match="Insufficient credits"):
        batch.submit([{"custom_id": "1-c1", "body": {}}])
