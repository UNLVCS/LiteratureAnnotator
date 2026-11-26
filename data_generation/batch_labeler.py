"""
Whole-paper labeling through the providers' batch APIs (50% of the live price).

OpenAI (Batch API on /v1/chat/completions) and Anthropic (Message Batches) process
requests asynchronously, usually within an hour and at most 24 hours. This script
splits the work into three steps, so nothing has to stay running while a batch is
processed:

    # 1. Submit every paper not yet in minio.synthetic_data_bucket (6 requests per paper)
    python data_generation/batch_labeler.py submit --model claude-sonnet-5-5 --from-prefix pilot_2026_10
    # 2. Check progress
    python data_generation/batch_labeler.py status --model claude-sonnet-5-5
    # 3. Parse finished batches and upload results (--wait polls until all have ended)
    python data_generation/batch_labeler.py collect --model claude-sonnet-5-5 --wait

``--model`` is a model name from the openai/anthropic sections of .env.yaml; its
settings (api_key, max_tokens, effort / reasoning_effort, cache_ttl) are used.
Submitted batch IDs are kept in logs/batches/<model>.json. Papers whose requests
failed are not uploaded; running ``submit`` again resubmits them. ``submit --dry-run``
shows the batch sizes without sending anything.

Results are the same records as labeler.py writes (``{provider}/{pmid}.json``),
built by the same labeling.build_queries / assemble_result code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from config.study_config import StudyModel, load_study
from llm_providers.openrouter_provider import OpenRouterProvider

from labeler import LabelingGenerator
from labeling import SCHEMA_VERSION, SkipPaper, assemble_result, build_queries
from llm_providers.base import LLMResponse, Query
from papers import PaperNotFoundError, load_paper_from_config, parse_bioc
from utilities.criteria import CRITERION_NAMES

REPO = Path(__file__).parent.parent
STATE_DIR = REPO / "logs" / "batches"
POLL_SECONDS = 300
# Conservative estimate for the token budget (request JSON runs ~3.9 chars/token).
CHARS_PER_TOKEN = 3.5

# Reasoning models (o-series, gpt-5+) accept only the default temperature.
_OPENAI_NO_TEMPERATURE = re.compile(r"^(o\d|gpt-[5-9])")
# Claude 4.7+ and the 5.x family reject sampling parameters. OpenRouter IDs use dots
# (anthropic/claude-opus-5.5), Anthropic's use dashes (claude-opus-5-5).
_ANTHROPIC_NO_SAMPLING = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(5|4[-.][78])")


def sends_temperature(model_id: str) -> bool:
    """Whether requests for this model carry temperature (bare or OpenRouter author/model ID)."""
    name = model_id.rpartition("/")[2]
    return not (_OPENAI_NO_TEMPERATURE.match(name) or _ANTHROPIC_NO_SAMPLING.search(name))

Outcome = Tuple[Optional[LLMResponse], Optional[str]]  # (response, error)


class InsufficientBalance(Exception):
    """The provider refused a batch because its worst-case cost exceeds the balance (HTTP 402)."""


def parse_criteria(value: Optional[str]) -> Optional[List[int]]:
    """0-based criterion indices from a 1-based list like "6" or "2,6"; None means all."""
    if not value or not value.strip():
        return None
    indices = sorted({int(part) - 1 for part in value.split(",") if part.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= len(CRITERION_NAMES):
        raise SystemExit(f"LA_CRITERIA must list criteria 1-{len(CRITERION_NAMES)}, got {value!r}")
    return None if len(indices) == len(CRITERION_NAMES) else indices


def custom_id(pmid: str, criterion_index: int) -> str:
    return f"{pmid}-c{criterion_index + 1}"


def parse_custom_id(cid: str) -> Tuple[str, int]:
    pmid, _, c = cid.rpartition("-c")
    return pmid, int(c) - 1


class AnthropicBatches:
    """Message Batches API: up to 100,000 requests / 256 MB per batch."""

    max_requests = 100_000
    max_bytes = 200_000_000
    default_token_budget = None

    def __init__(self, model: str, kwargs: Dict[str, Any]):
        from config.llm_providers_config import instantiate_provider

        self.provider = instantiate_provider("anthropic", kwargs)
        self.client = self.provider.client

    def request(self, cid: str, query: Query, pmid: str) -> Dict[str, Any]:
        params = self.provider._request(query)
        params.update(params.pop("extra_body", {}))  # batch params are plain JSON
        return {"custom_id": cid, "params": params}

    def submit(self, requests: List[Dict[str, Any]]) -> str:
        return self.client.messages.batches.create(requests=requests).id

    def status(self, batch_id: str) -> Tuple[bool, str]:
        b = self.client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        return b.processing_status == "ended", (
            f"{b.processing_status}: processing={c.processing} succeeded={c.succeeded} "
            f"errored={c.errored} canceled={c.canceled} expired={c.expired}"
        )

    def results(self, batch_id: str) -> Iterator[Tuple[str, Outcome]]:
        for entry in self.client.messages.batches.results(batch_id):
            r = entry.result
            if r.type == "succeeded":
                yield entry.custom_id, (self.provider._to_llm_response(Query(prompt="-"), r.message), None)
            elif r.type == "errored":
                err = getattr(r.error, "error", r.error)  # ErrorResponse wraps the error object
                yield entry.custom_id, (None, f"errored: {err.type}: {getattr(err, 'message', '')}")
            else:
                yield entry.custom_id, (None, r.type)


class OpenAIBatches:
    """Batch API on /v1/chat/completions: up to 50,000 requests / 200 MB input file per batch."""

    max_requests = 50_000
    max_bytes = 180_000_000
    # OpenAI rejects a batch whose input tokens, plus those of the organization's other
    # in-progress batches for the model, exceed its enqueued-token limit (20M for
    # gpt-6-luna on our tier). Override with batch_token_budget in .env.yaml.
    default_token_budget = 18_000_000

    def __init__(self, model: str, kwargs: Dict[str, Any]):
        import openai

        self.model = model
        self.kwargs = kwargs
        self.client = openai.OpenAI(api_key=kwargs["api_key"], max_retries=6)

    def request(self, cid: str, query: Query, pmid: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": query.system_message},
                {"role": "user", "content": query.prompt},
            ],
            "max_completion_tokens": query.max_tokens,
            # OpenAI caches long prefixes automatically; the key routes a paper's six
            # requests to the same cache.
            "prompt_cache_key": f"paper-{pmid}",
        }
        if not _OPENAI_NO_TEMPERATURE.match(self.model):
            body["temperature"] = query.temperature
        if self.kwargs.get("reasoning_effort"):
            body["reasoning_effort"] = self.kwargs["reasoning_effort"]
        return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": body}

    def submit(self, requests: List[Dict[str, Any]]) -> str:
        data = "".join(json.dumps(r) + "\n" for r in requests).encode("utf-8")
        f = self.client.files.create(file=("batch.jsonl", data), purpose="batch")
        return self.client.batches.create(
            input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h"
        ).id

    def status(self, batch_id: str) -> Tuple[bool, str]:
        b = self.client.batches.retrieve(batch_id)
        c = b.request_counts
        detail = f"{b.status}: completed={c.completed} failed={c.failed} total={c.total}" if c else b.status
        if b.errors and b.errors.data:
            detail += " errors=" + "; ".join(f"{e.code}: {e.message}" for e in b.errors.data[:3])
        return b.status in ("completed", "failed", "expired", "cancelled"), detail

    def results(self, batch_id: str) -> Iterator[Tuple[str, Outcome]]:
        b = self.client.batches.retrieve(batch_id)
        for file_id in (b.output_file_id, b.error_file_id):
            if not file_id:
                continue
            for line in self.client.files.content(file_id).text.splitlines():
                if line.strip():
                    yield self._outcome(json.loads(line))

    def _outcome(self, row: Dict[str, Any]) -> Tuple[str, Outcome]:
        cid = row["custom_id"]
        resp = row.get("response") or {}
        if row.get("error") or resp.get("status_code") != 200:
            err = row.get("error") or (resp.get("body") or {}).get("error") or {}
            return cid, (None, f"HTTP {resp.get('status_code')}: {err.get('code')}: {err.get('message')}")
        body = resp["body"]
        choices = body.get("choices") or []
        if body.get("error") or not choices or choices[0].get("error"):
            return cid, (None, f"No usable completion: {body.get('error')}")
        choice = choices[0]
        u = body.get("usage") or {}
        usage = {
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "cached_prompt_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        }
        return cid, (
            LLMResponse(
                content=choice["message"].get("content") or "",
                model=body.get("model", self.model),
                usage=usage,
                metadata={"provider": "openai", "request_id": resp.get("request_id"),
                          "created": body.get("created"), "batch": True},
                finish_reason=choice.get("finish_reason"),
            ),
            None,
        )


class OpenRouterBatches(OpenAIBatches):
    """OpenRouter chat completions with one batch-level provider pin."""

    # Bound inline request bodies locally. The service also applies account-specific limits.
    max_requests = 1_000
    max_bytes = 20_000_000
    default_token_budget = None
    terminal = {"completed", "failed", "expired", "cancelled"}

    def __init__(self, model: str, kwargs: Dict[str, Any]):
        self.model = model
        self.kwargs = kwargs
        self.host = kwargs.get("host")
        self.provider = OpenRouterProvider(
            kwargs["api_key"], model, http_client=kwargs.get("http_client"),
        )

    def request(self, cid: str, query: Query, pmid: str) -> Dict[str, Any]:
        content: Any = query.prompt
        if query.cache_prefix_len and self.model.startswith("anthropic/"):
            # Claude caches only at explicit breakpoints. A paper's six requests share the
            # paper text; batch requests run concurrently and in any order, so OpenRouter
            # recommends the 1-hour TTL to keep the prefix warm across the batch.
            content = [
                {"type": "text", "text": query.prompt[: query.cache_prefix_len],
                 "cache_control": {"type": "ephemeral", "ttl": self.kwargs.get("cache_ttl") or "1h"}},
                {"type": "text", "text": query.prompt[query.cache_prefix_len :]},
            ]
        messages = [{"role": "user", "content": content}]
        if query.system_message:
            messages.insert(0, {"role": "system", "content": query.system_message})
        body = {"messages": messages, **self.provider._sampling(query)}
        if not sends_temperature(self.model):
            for name in ("temperature", "top_p", "top_k"):
                body.pop(name, None)
        if self.model.startswith("openai/"):
            # OpenAI caches long prefixes automatically; the key keeps a paper's six
            # requests on the same cache.
            body["prompt_cache_key"] = f"paper-{pmid}"
        if self.kwargs.get("reasoning_effort"):
            body["reasoning"] = {"effort": self.kwargs["reasoning_effort"]}
        return {"custom_id": cid, "body": body}

    def _http(self, method, path, **kwargs):
        response = method(
            f"{self.provider.base_url}/batches{path}",
            headers=self.provider.headers, timeout=self.provider.timeout, **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def submit(self, rows: List[Dict[str, Any]]) -> str:
        if not self.host or not self.host.strip():
            raise ValueError("OpenRouter batch requires a provider pin")
        try:
            data = self._http(self.provider.http_client.post, "", json={
                "endpoint": "/v1/chat/completions",
                "model": self.model,
                "provider": {"only": [self.host], "allow_fallbacks": False},
                "requests": rows,
            })
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 402:
                raise InsufficientBalance(response.text[:300]) from exc
            raise
        return data["id"]

    def status(self, batch_id: str) -> Tuple[bool, str]:
        client = self.provider.http_client
        try:
            page = self._http(client.get, f"/{batch_id}/results", params={"limit": 1})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 409:
                return False, "processing"
            raise
        if page["data"]:
            return True, "results available"
        batch = self._http(client.get, f"/{batch_id}")
        detail = f"{batch['status']}: {batch.get('request_counts')}"
        error = (batch.get("error") or {}).get("message")
        if error:
            detail += f" error={error}"
        return batch["status"] in self.terminal, detail

    def results(self, batch_id: str) -> Iterator[Tuple[str, Outcome]]:
        client = self.provider.http_client
        params = {"limit": 100}
        while True:
            page = self._http(client.get, f"/{batch_id}/results", params=params)
            for row in page["data"]:
                cid, outcome = self._outcome(row)
                response, error = outcome
                if response is not None:
                    body = row["response"]["body"]
                    response.usage.update(body.get("usage") or {})
                    response.metadata["provider"] = "openrouter"
                    response.metadata["created"] = body.get("created")
                    response.metadata["openrouter"] = {
                        "host": body.get("provider"),
                        "quantization": (body.get("openrouter_metadata") or {}).get("quantization"),
                        "model": body.get("model"),
                        "generation_id": body.get("id"),
                        "response_metadata": body.get("openrouter_metadata"),
                    }
                yield cid, (response, error)
            if not page["has_more"]:
                return
            cursor = page.get("next_cursor")
            if not cursor or cursor == params.get("cursor"):
                raise ValueError(f"Invalid pagination cursor for batch {batch_id}")
            params["cursor"] = cursor


BACKENDS = {"anthropic": AnthropicBatches, "openai": OpenAIBatches, "openrouter": OpenRouterBatches}


class BatchLabeler:
    def __init__(self, model: str, raw_prefix: Optional[str], *,
                 study=None, sample_index: int = 0, dry_run: bool = False,
                 papers_dir: Path | None = None):
        self.papers_dir = papers_dir
        if papers_dir is None:
            self.gen = LabelingGenerator(raw_prefix=raw_prefix, load_providers=False)
            self.config = self.gen.config
        else:
            from config.app_config import AppConfig

            self.config = AppConfig()
            self.gen = None
        self.study = study
        self.sample_index = sample_index
        self.study_model = None
        self.model = model
        if study:
            self.study_model = next((spec for spec in study.models if model in (spec.id, spec.alias)), None)
            if self.study_model is None:
                raise SystemExit(f"Unknown study model {model}")
            spec = self.study_model
            self.model = model = spec.id
            provider_type = spec.route
            # Study settings come entirely from the study config. Only the credential
            # comes from the standard provider section.
            kwargs = {
                "model": spec.id, "host": spec.host,
                "max_tokens": spec.max_tokens,
                "temperature": spec.temperature if spec.temperature is not None else 1.0,
            }
            if spec.reasoning_mode not in ("default", "always_on"):
                key = "effort" if spec.route == "anthropic" else "reasoning_effort"
                kwargs[key] = spec.reasoning_mode
            if dry_run:
                kwargs["api_key"] = "offline-dry-run"
            else:
                route_config = self.config.llm_providers.get(spec.route)
                kwargs["api_key"] = route_config.api_key if route_config else None
            self.config.minio.output_prefix = study.sample_prefixes[sample_index]
            self.config.minio.synthetic_data_bucket = "synthetic-data-v3"
        else:
            specs = self.config.get_provider_specs()
            if model not in specs:
                raise SystemExit(f"{model} is not an enabled model in .env.yaml (have: {', '.join(specs)})")
            provider_type, kwargs = specs[model]
        if provider_type not in BACKENDS:
            raise SystemExit(f"{model} is a {provider_type} model; batch labeling supports {', '.join(BACKENDS)}")
        if not kwargs.get("api_key"):
            raise SystemExit(f"{model}: no api_key in .env.yaml")
        self.kwargs = kwargs
        self.backend = BACKENDS[provider_type](model, kwargs)
        self.token_budget = kwargs.get("batch_token_budget") or self.backend.default_token_budget
        # OpenRouter refuses (402) a batch whose worst case, every request using its full
        # max_tokens, exceeds the account balance. LA_MAX_BATCH_USD caps that worst case
        # per batch; a 402 stops submission cleanly so `run` retries after a collect.
        self.max_batch_usd = float(os.environ.get("LA_MAX_BATCH_USD") or 0) or None
        # LA_CRITERIA (e.g. "6") asks only those criteria (1-based). Collection then
        # replaces them in the paper's stored record, so a revised criterion is re-run
        # without paying for the other five.
        self.criteria = parse_criteria(os.environ.get("LA_CRITERIA"))
        # One state file per (model, output_prefix), so runs into different prefixes
        # (e.g. entropy resamples entropy/s01..s05) track their batches separately.
        self.output_prefix = (self.config.minio.output_prefix or "").strip("/")
        self.model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        slug = self.model_slug + (f"__{re.sub(r'[^A-Za-z0-9._-]', '_', self.output_prefix)}" if self.output_prefix else "")
        self.state_path = STATE_DIR / f"{slug}.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"batches": []}
        self.out_dir = REPO / "logs" / f"results_batch_{slug}"

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def tokens_in_flight(self) -> int:
        """Estimated tokens of this model's uncollected batches across every output prefix.

        The provider's enqueued-token limit is per model, not per run, so concurrent runs
        into different prefixes share one budget.
        """
        total = 0
        for path in STATE_DIR.glob(f"{self.model_slug}*.json"):
            if path.stem != self.model_slug and not path.stem.startswith(f"{self.model_slug}__"):
                continue  # a different model whose name merely starts the same way
            state = self.state if path == self.state_path else json.loads(path.read_text())
            total += sum(b.get("est_tokens", 0) for b in state["batches"] if not b.get("collected"))
        return total

    def pending(self) -> List[Dict[str, Any]]:
        return [b for b in self.state["batches"] if not b.get("collected")]

    def asked(self) -> List[int]:
        """0-based indices of the criteria this run asks."""
        criteria = getattr(self, "criteria", None)
        return list(range(len(CRITERION_NAMES))) if criteria is None else criteria

    def needs_labeling(self, pmid: str) -> bool:
        """Whether this paper still needs the asked criteria at the current schema version."""
        if getattr(self, "criteria", None) is None:
            return not self.gen.result_exists(self.model, pmid)
        record = self.gen.fetch_result(self.model, pmid)
        if record is None:
            return True
        current = {r["criterion"]: r.get("schema_version", record.get("schema_version"))
                   for r in record.get("criteria_results", [])}
        return any(current.get(CRITERION_NAMES[i]) != SCHEMA_VERSION for i in self.asked())

    def build(self, pmid: str) -> Optional[List[Dict[str, Any]]]:
        """The batch requests for one paper (one per asked criterion), or None if it cannot be labeled."""
        if getattr(self, "papers_dir", None) is not None:
            path = self.papers_dir / f"{pmid}.json"
            if not path.is_file():
                raise PaperNotFoundError(str(path))
            paper = parse_bioc(pmid, json.loads(path.read_text()), self.config.papers.exclude_section_types)
        else:
            paper = load_paper_from_config(pmid, self.config, client=self.gen.minio)
        _, queries = build_queries(
            paper,
            self.model,
            max_chars=self.config.papers.max_chars,
            temperature=self.kwargs.get("temperature", 0.1),
            max_tokens=self.kwargs.get("max_tokens") or 16000,
        )
        return [self.backend.request(custom_id(pmid, i), queries[i], pmid) for i in self.asked()]

    def check_study_submission(self, pmids: List[str]) -> None:
        spec = self.study_model
        if spec is None:
            return
        if self.sample_index:
            original_prefix = self.config.minio.output_prefix
            try:
                self.config.minio.output_prefix = self.study.sample_prefixes[0]
                missing = sum(not self.gen.result_exists(self.model, pmid) for pmid in pmids)
            finally:
                self.config.minio.output_prefix = original_prefix
            if missing:
                raise SystemExit(f"Collect sample s00 for all requested papers first ({missing} missing)")
        if spec.route == "openrouter" and not spec.host:
            raise SystemExit(f"{self.model}: OpenRouter study submission requires a provider pin")

    def submit(self, pmids: List[str], dry_run: bool = False) -> int:
        """Submit unlabeled papers within the token budget; returns how many are left over."""
        spec = getattr(self, "study_model", None)
        if spec and not dry_run:
            self.check_study_submission(pmids)
        if spec and not spec.batch_prices:
            raise SystemExit(f"{self.model}: configure batch prices before estimating cost")
        in_flight = {p for b in self.pending() for p in b["pmids"]}
        todo = (pmids if getattr(self, "papers_dir", None) is not None else
                [p for p in pmids if p not in in_flight and self.needs_labeling(p)])
        if getattr(self, "criteria", None) is not None:
            print(f"  asking only {', '.join(CRITERION_NAMES[i] for i in self.asked())}; "
                  f"collection replaces them in each paper's stored record")
        print(f"{self.model}: {len(pmids)} papers, {len(pmids) - len(todo)} already labeled or in a "
              f"pending batch, {len(todo)} to submit")
        tokens_in_flight = 0 if spec and dry_run else self.tokens_in_flight()
        chunk: List[Dict[str, Any]] = []
        chunk_pmids: List[str] = []
        size = tokens = 0
        chunk_usd = 0.0
        estimated_requests = estimated_input = estimated_output = 0
        max_batch_usd = getattr(self, "max_batch_usd", None)

        def flush() -> None:
            nonlocal chunk, chunk_pmids, size, tokens, tokens_in_flight, chunk_usd
            if not chunk:
                return
            desc = f"{len(chunk_pmids)} papers, {len(chunk)} requests, {size / 1e6:.1f} MB, ~{tokens / 1e6:.1f}M tokens"
            if max_batch_usd:
                desc += f", worst case ${chunk_usd:.2f}"
            if dry_run:
                print(f"  [dry run] batch of {desc}")
            else:
                batch_id = self.backend.submit(chunk)
                self.state["batches"].append({
                    "id": batch_id,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "raw_prefix": self.gen.raw_prefix(),
                    "bucket": self.config.minio.synthetic_data_bucket,
                    "output_prefix": self.output_prefix,
                    "pmids": chunk_pmids,
                    "requests": len(chunk),
                    "megabytes": round(size / 1e6, 1),
                    "est_tokens": tokens,
                    "criteria": [i + 1 for i in self.asked()],
                    "collected": False,
                })
                self.save_state()
                print(f"  submitted {batch_id}: {desc}")
            tokens_in_flight += tokens
            chunk, chunk_pmids, size, tokens, chunk_usd = [], [], 0, 0, 0.0

        def out_of_balance(exc: InsufficientBalance, i: int) -> int:
            # Papers in the refused chunk plus every paper not yet built.
            left = len(chunk_pmids) + len(todo) - i + 1
            print(f"  provider refused the batch for insufficient balance ({exc}); {left} papers not "
                  f"submitted. Collect finished batches, lower LA_MAX_BATCH_USD or add credit, then submit again")
            return left

        for i, pmid in enumerate(todo, 1):
            try:
                requests = self.build(pmid)
            except (PaperNotFoundError, SkipPaper) as e:
                print(f"  [{i}/{len(todo)}] {pmid} skipped: {e}")
                continue
            paper_bytes = sum(len(json.dumps(r)) for r in requests)
            paper_tokens = int(paper_bytes / CHARS_PER_TOKEN)
            paper_usd = 0.0
            if max_batch_usd and spec and spec.batch_prices:
                paper_usd = (paper_tokens * spec.batch_prices.input
                             + len(requests) * spec.max_tokens * spec.batch_prices.output) / 1e6
                if chunk and chunk_usd + paper_usd > max_batch_usd:
                    try:
                        flush()
                    except InsufficientBalance as exc:
                        return out_of_balance(exc, i)
                if paper_usd > max_batch_usd:
                    print(f"  {pmid}: worst case ${paper_usd:.2f} alone exceeds LA_MAX_BATCH_USD; sent on its own")
            if spec and dry_run:
                estimated_requests += len(requests)
                estimated_input += paper_tokens
                estimated_output += len(requests) * spec.max_tokens
            if self.token_budget and not (spec and dry_run) and tokens_in_flight + tokens + paper_tokens > self.token_budget:
                try:
                    flush()
                except InsufficientBalance as exc:
                    return out_of_balance(exc, i)
                if tokens_in_flight + paper_tokens > self.token_budget:
                    left = len(todo) - i + 1
                    print(f"  token budget reached (~{tokens_in_flight / 1e6:.1f}M of {self.token_budget / 1e6:.0f}M "
                          f"in flight); {left} papers wait for a batch to finish")
                    return left
            if chunk and (size + paper_bytes > self.backend.max_bytes
                          or len(chunk) + len(requests) > self.backend.max_requests):
                try:
                    flush()
                except InsufficientBalance as exc:
                    return out_of_balance(exc, i)
            chunk += requests
            chunk_pmids.append(pmid)
            size += paper_bytes
            tokens += paper_tokens
            chunk_usd += paper_usd
            if i % 100 == 0:
                print(f"  built {i}/{len(todo)}", flush=True)
        try:
            flush()
        except InsufficientBalance as exc:
            return out_of_balance(exc, len(todo) + 1)
        if spec and dry_run:
            prices = spec.batch_prices
            claude = spec.route == "anthropic" or spec.id.startswith("anthropic/")
            input_rate = prices.cache_write if claude and prices.cache_write else prices.input
            cost = (estimated_input * input_rate + estimated_output * prices.output) / 1e6
            print(f"{self.model} {self.output_prefix}: {estimated_requests} requests, "
                  f"~{estimated_input:,} input tokens, up to {estimated_output:,} output tokens, "
                  f"${cost:.6f} estimated without cache reads")
        return 0

    def run(self, pmids: List[str]) -> None:
        """Collect finished batches and submit more until every paper is labeled or has failed twice."""
        attempts: Dict[str, int] = {}
        while True:
            self.collect()
            for b in self.state["batches"]:
                for p in [] if b.get("batch_failed") else b.get("failed") or []:
                    attempts[p] = attempts.get(p, 0) + 1
            retry = [p for p in pmids if attempts.get(p, 0) < 2]
            attempts.clear()
            left = self.submit(retry)
            if not self.pending():
                gave_up = [p for p in pmids if p not in retry]
                print(f"done: no batches pending, {left} papers not submitted"
                      + (f"; gave up on {len(gave_up)} papers that failed twice: {gave_up}" if gave_up else ""))
                return
            print(f"  {len(self.pending())} batch(es) processing; next check in {POLL_SECONDS // 60} min", flush=True)
            time.sleep(POLL_SECONDS)

    def status(self) -> bool:
        """Print each pending batch's status; True when all have ended."""
        done_all = True
        for b in self.pending():
            done, detail = self.backend.status(b["id"])
            done_all &= done
            print(f"  {b['id']} ({len(b['pmids'])} papers): {detail}")
        if not self.pending():
            print(f"  no pending batches for {self.model}")
        return done_all

    def collect(self, wait: bool = False) -> None:
        while True:
            for b in self.pending():
                done, detail = self.backend.status(b["id"])
                if done:
                    self.collect_batch(b, detail)
            if not wait or not self.pending():
                break
            print(f"  {len(self.pending())} batch(es) still processing; checking again in {POLL_SECONDS // 60} min",
                  flush=True)
            time.sleep(POLL_SECONDS)

    def merge_criteria(self, pmid: str, fresh: Dict[str, Any], batch_id: str) -> Dict[str, Any]:
        """Replace the freshly asked criteria in the paper's stored record.

        With no stored record (e.g. a criterion-only test into its own prefix) the fresh,
        partial record is kept as is and marked ``partial``.
        """
        stored = self.gen.fetch_result(self.model, pmid)
        replaced = [r["criterion"] for r in fresh["criteria_results"]]
        if stored is None:
            fresh["partial"] = True
            return fresh
        new = {r["criterion"]: r for r in fresh["criteria_results"]}
        stored["criteria_results"] = [new.get(r["criterion"], r) for r in stored["criteria_results"]]
        missing = [name for name in replaced if name not in {r["criterion"] for r in stored["criteria_results"]}]
        stored["criteria_results"] += [new[name] for name in missing]
        order = {name: i for i, name in enumerate(CRITERION_NAMES)}
        stored["criteria_results"].sort(key=lambda r: order.get(r["criterion"], len(order)))
        stored.setdefault("revisions", []).append({
            "criteria": replaced, "batch_id": batch_id, "schema_version": SCHEMA_VERSION,
            "previous_schema_version": stored.get("schema_version"), "at": fresh["labeled_at"],
        })
        stored["schema_version"] = SCHEMA_VERSION
        return stored

    def collect_batch(self, b: Dict[str, Any], detail: str) -> None:
        print(f"collecting {b['id']} ({detail})", flush=True)
        if b["bucket"] != self.config.minio.synthetic_data_bucket:
            raise SystemExit(f"{b['id']} was submitted for bucket {b['bucket']}, but .env.yaml now names "
                             f"{self.config.minio.synthetic_data_bucket}; restore it to collect")
        if b.get("output_prefix", "") != self.output_prefix:
            raise SystemExit(f"{b['id']} was submitted for output_prefix {b.get('output_prefix', '')!r}, but the "
                             f"config now has {self.output_prefix!r}")
        # Papers are reloaded to resolve evidence quotes; use the folder they were submitted from.
        self.config.papers.raw_prefix = b["raw_prefix"]
        outcomes: Dict[str, Dict[int, Outcome]] = {}
        for cid, outcome in self.backend.results(b["id"]):
            pmid, idx = parse_custom_id(cid)
            outcomes.setdefault(pmid, {})[idx] = outcome
        spec = getattr(self, "study_model", None)
        if spec:
            usage = [response.usage for got in outcomes.values() for response, _ in got.values() if response]
            cost = sum(usage_cost(tokens, spec) for tokens in usage)
            b["actual_cost_usd"] = cost
            print(f"  {self.model} {self.output_prefix}: ${cost:.6f} from reported token usage")
        # Batches from before criterion subsets existed asked all six.
        asked = [c - 1 for c in b.get("criteria") or range(1, len(CRITERION_NAMES) + 1)]
        subset = len(asked) < len(CRITERION_NAMES)
        uploaded, failed = 0, []
        for pmid in b["pmids"]:
            got = outcomes.get(pmid, {})
            errors = [f"{CRITERION_NAMES[i]}: {got[i][1] if i in got else 'no result'}"
                      for i in asked if i not in got or got[i][0] is None]
            if errors:
                failed.append(pmid)
                if outcomes:  # a batch rejected as a whole is reported once, below
                    print(f"  {pmid}: not uploaded ({'; '.join(errors[:2])})")
                continue
            paper = load_paper_from_config(pmid, self.config, client=self.gen.minio)
            result, _ = build_queries(paper, self.model, max_chars=self.config.papers.max_chars)
            result = assemble_result(paper, result, [got[i][0] for i in asked], criteria=asked)
            result["batch_id"] = b["id"]
            for record in result["criteria_results"]:
                record["schema_version"] = SCHEMA_VERSION
            if spec:
                result["sample_index"] = self.sample_index
                for i, record in zip(asked, result["criteria_results"]):
                    response = got[i][0]
                    routing = (response.metadata or {}).get("openrouter", {})
                    record["generation"] = {
                        "model": response.model,
                        "requested_model": spec.id,
                        "host": routing.get("host"),
                        "quantization": routing.get("quantization"),
                        "temperature": spec.temperature,
                        "temperature_sent": sends_temperature(spec.id),
                        "reasoning_mode": spec.reasoning_mode,
                        "max_tokens": spec.max_tokens,
                        "batch_id": b["id"],
                        "response_timestamp": (response.metadata or {}).get("created"),
                        "collected_at": result["labeled_at"],
                        "sample_index": self.sample_index,
                    }
            if subset:
                result = self.merge_criteria(pmid, result, b["id"])
            path = self.out_dir / f"{pmid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2))
            self.gen.upload(result)
            uploaded += 1
        # A batch rejected as a whole (e.g. over the enqueued-token limit) says nothing about its papers.
        b.update(collected=True, collected_at=datetime.now(timezone.utc).isoformat(),
                 uploaded=uploaded, failed=failed, batch_failed=not outcomes, detail=detail)
        self.save_state()
        bucket = self.config.minio.synthetic_data_bucket
        print(f"  uploaded {uploaded} to {bucket}/{self.model}/, {len(failed)} failed"
              + (" (run submit again to retry them)" if failed else ""), flush=True)


def usage_cost(usage: dict, spec: StudyModel) -> float:
    if usage.get("cost") is not None:
        return float(usage["cost"])
    prices = spec.batch_prices
    if spec.route == "anthropic":
        uncached = usage.get("input_tokens", 0)
        cached = usage.get("cache_read_input_tokens", 0)
        written = usage.get("cache_creation_input_tokens", 0)
    else:
        cached = usage.get("cached_prompt_tokens", 0)
        details = usage.get("prompt_tokens_details") or {}
        written = details.get("cache_write_tokens", 0)
        uncached = max(0, usage.get("prompt_tokens", 0) - cached - written)
    output = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return (uncached * prices.input + cached * prices.cached_input + output * prices.output
            + written * (prices.cache_write if prices.cache_write is not None else prices.input)) / 1e6


def read_pmids(path: Path) -> list[str]:
    data = path.read_text()
    if path.suffix.lower() == ".json":
        manifest = json.loads(data)
        entries = manifest["papers"] if isinstance(manifest, dict) else manifest
        return [str(pmid) for pmid in entries]
    return [line.strip() for line in data.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["submit", "status", "collect", "run"])
    parser.add_argument("--model", required=True, help="Model name from .env.yaml (openai or anthropic).")
    parser.add_argument("--from-prefix", help="submit: every paper in this MinIO raw-articles folder.")
    parser.add_argument("--pmids-file", help="submit: file with one PMID per line.")
    parser.add_argument("--raw-prefix", help="submit: MinIO folder the papers are read from (default: --from-prefix).")
    parser.add_argument("--limit", type=int, help="submit: only the first N papers.")
    parser.add_argument("--dry-run", action="store_true", help="submit: build and size the batches, send nothing.")
    parser.add_argument("--wait", action="store_true", help="collect: poll until every batch has ended.")
    parser.add_argument("--study-config", type=Path)
    parser.add_argument("--papers-dir", type=Path, help="Local BioC JSON directory for an offline study dry run.")
    parser.add_argument("--sample", type=int, choices=range(6), default=0)
    args = parser.parse_args()
    study = load_study(args.study_config) if args.study_config else None
    if study and args.command in ("submit", "run") and not args.pmids_file:
        parser.error("study submissions require --pmids-file")

    if args.dry_run and args.command != "submit":
        parser.error("--dry-run applies only to submit")
    if args.papers_dir and not (study and args.dry_run):
        parser.error("--papers-dir requires a study dry run")
    labeler = BatchLabeler(
        args.model, raw_prefix=args.raw_prefix or args.from_prefix,
        study=study, sample_index=args.sample, dry_run=args.dry_run, papers_dir=args.papers_dir,
    )
    if args.command in ("submit", "run"):
        pmids: List[str] = []
        if args.pmids_file:
            pmids += read_pmids(Path(args.pmids_file))
        if args.from_prefix and not study:
            pmids += labeler.gen.list_prefix_pmids()
        if not pmids:
            parser.error(f"{args.command} needs --from-prefix or --pmids-file")
        if study and any(not re.fullmatch(r"[0-9]+", pmid) for pmid in pmids):
            parser.error("Study manifest must contain numeric PMIDs")
        pmids = list(dict.fromkeys(pmids))[: args.limit] if args.limit else list(dict.fromkeys(pmids))
        if args.command == "run":
            labeler.run(pmids)
        elif labeler.submit(pmids, dry_run=args.dry_run):
            print("run submit again after a batch finishes, or use the run command")
    elif args.command == "status":
        labeler.status()
    else:
        labeler.collect(wait=args.wait)


if __name__ == "__main__":
    main()
