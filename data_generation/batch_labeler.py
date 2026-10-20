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
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from labeler import LabelingGenerator
from labeling import SkipPaper, assemble_result, build_queries
from llm_providers.base import LLMResponse, Query
from papers import PaperNotFoundError, load_paper_from_config
from utilities.criteria import CRITERION_NAMES

REPO = Path(__file__).parent.parent
STATE_DIR = REPO / "logs" / "batches"
POLL_SECONDS = 300
# Conservative estimate for the token budget (request JSON runs ~3.9 chars/token).
CHARS_PER_TOKEN = 3.5

# Reasoning models (o-series, gpt-5+) accept only the default temperature.
_OPENAI_NO_TEMPERATURE = re.compile(r"^(o\d|gpt-[5-9])")

Outcome = Tuple[Optional[LLMResponse], Optional[str]]  # (response, error)


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
        choice = body["choices"][0]
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
                metadata={"provider": "openai", "request_id": resp.get("request_id"), "batch": True},
                finish_reason=choice.get("finish_reason"),
            ),
            None,
        )


BACKENDS = {"anthropic": AnthropicBatches, "openai": OpenAIBatches}


class BatchLabeler:
    def __init__(self, model: str, raw_prefix: Optional[str]):
        self.gen = LabelingGenerator(raw_prefix=raw_prefix, load_providers=False)
        specs = self.gen.config.get_provider_specs()
        if model not in specs:
            raise SystemExit(f"{model} is not an enabled model in .env.yaml (have: {', '.join(specs)})")
        provider_type, kwargs = specs[model]
        if provider_type not in BACKENDS:
            raise SystemExit(f"{model} is a {provider_type} model; batch labeling supports {', '.join(BACKENDS)}")
        if not kwargs.get("api_key"):
            raise SystemExit(f"{model}: no api_key in .env.yaml")
        self.model = model
        self.kwargs = kwargs
        self.backend = BACKENDS[provider_type](model, kwargs)
        self.token_budget = kwargs.get("batch_token_budget") or self.backend.default_token_budget
        self.state_path = STATE_DIR / f"{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"batches": []}
        self.out_dir = REPO / "logs" / f"results_batch_{re.sub(r'[^A-Za-z0-9._-]', '_', model)}"

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def pending(self) -> List[Dict[str, Any]]:
        return [b for b in self.state["batches"] if not b.get("collected")]

    def build(self, pmid: str) -> Optional[List[Dict[str, Any]]]:
        """The six batch requests for one paper, or None if it cannot be labeled."""
        paper = load_paper_from_config(pmid, self.gen.config, client=self.gen.minio)
        _, queries = build_queries(
            paper,
            self.model,
            max_chars=self.gen.config.papers.max_chars,
            temperature=self.kwargs.get("temperature", 0.1),
            max_tokens=self.kwargs.get("max_tokens") or 16000,
        )
        return [self.backend.request(custom_id(pmid, i), q, pmid) for i, q in enumerate(queries)]

    def submit(self, pmids: List[str], dry_run: bool = False) -> int:
        """Submit unlabeled papers within the token budget; returns how many are left over."""
        in_flight = {p for b in self.pending() for p in b["pmids"]}
        todo = [p for p in pmids if p not in in_flight and not self.gen.result_exists(self.model, p)]
        print(f"{self.model}: {len(pmids)} papers, {len(pmids) - len(todo)} already labeled or in a "
              f"pending batch, {len(todo)} to submit")
        tokens_in_flight = sum(b.get("est_tokens", 0) for b in self.pending())
        chunk: List[Dict[str, Any]] = []
        chunk_pmids: List[str] = []
        size = tokens = 0

        def flush() -> None:
            nonlocal chunk, chunk_pmids, size, tokens, tokens_in_flight
            if not chunk:
                return
            desc = f"{len(chunk_pmids)} papers, {len(chunk)} requests, {size / 1e6:.1f} MB, ~{tokens / 1e6:.1f}M tokens"
            if dry_run:
                print(f"  [dry run] batch of {desc}")
            else:
                batch_id = self.backend.submit(chunk)
                self.state["batches"].append({
                    "id": batch_id,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "raw_prefix": self.gen.raw_prefix(),
                    "bucket": self.gen.config.minio.synthetic_data_bucket,
                    "pmids": chunk_pmids,
                    "requests": len(chunk),
                    "megabytes": round(size / 1e6, 1),
                    "est_tokens": tokens,
                    "collected": False,
                })
                self.save_state()
                print(f"  submitted {batch_id}: {desc}")
            tokens_in_flight += tokens
            chunk, chunk_pmids, size, tokens = [], [], 0, 0

        for i, pmid in enumerate(todo, 1):
            try:
                requests = self.build(pmid)
            except (PaperNotFoundError, SkipPaper) as e:
                print(f"  [{i}/{len(todo)}] {pmid} skipped: {e}")
                continue
            paper_bytes = sum(len(json.dumps(r)) for r in requests)
            paper_tokens = int(paper_bytes / CHARS_PER_TOKEN)
            if self.token_budget and tokens_in_flight + tokens + paper_tokens > self.token_budget:
                flush()
                if tokens_in_flight + paper_tokens > self.token_budget:
                    left = len(todo) - i + 1
                    print(f"  token budget reached (~{tokens_in_flight / 1e6:.1f}M of {self.token_budget / 1e6:.0f}M "
                          f"in flight); {left} papers wait for a batch to finish")
                    return left
            if chunk and (size + paper_bytes > self.backend.max_bytes
                          or len(chunk) + len(requests) > self.backend.max_requests):
                flush()
            chunk += requests
            chunk_pmids.append(pmid)
            size += paper_bytes
            tokens += paper_tokens
            if i % 100 == 0:
                print(f"  built {i}/{len(todo)}", flush=True)
        flush()
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

    def collect_batch(self, b: Dict[str, Any], detail: str) -> None:
        print(f"collecting {b['id']} ({detail})", flush=True)
        if b["bucket"] != self.gen.config.minio.synthetic_data_bucket:
            raise SystemExit(f"{b['id']} was submitted for bucket {b['bucket']}, but .env.yaml now names "
                             f"{self.gen.config.minio.synthetic_data_bucket}; restore it to collect")
        # Papers are reloaded to resolve evidence quotes; use the folder they were submitted from.
        self.gen.config.papers.raw_prefix = b["raw_prefix"]
        outcomes: Dict[str, Dict[int, Outcome]] = {}
        for cid, outcome in self.backend.results(b["id"]):
            pmid, idx = parse_custom_id(cid)
            outcomes.setdefault(pmid, {})[idx] = outcome
        uploaded, failed = 0, []
        for pmid in b["pmids"]:
            got = outcomes.get(pmid, {})
            errors = [f"{CRITERION_NAMES[i]}: {got[i][1] if i in got else 'no result'}"
                      for i in range(len(CRITERION_NAMES)) if i not in got or got[i][0] is None]
            if errors:
                failed.append(pmid)
                if outcomes:  # a batch rejected as a whole is reported once, below
                    print(f"  {pmid}: not uploaded ({'; '.join(errors[:2])})")
                continue
            paper = load_paper_from_config(pmid, self.gen.config, client=self.gen.minio)
            result, _ = build_queries(paper, self.model, max_chars=self.gen.config.papers.max_chars)
            result = assemble_result(paper, result, [got[i][0] for i in range(len(CRITERION_NAMES))])
            result["batch_id"] = b["id"]
            path = self.out_dir / f"{pmid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2))
            self.gen.upload(result)
            uploaded += 1
        # A batch rejected as a whole (e.g. over the enqueued-token limit) says nothing about its papers.
        b.update(collected=True, collected_at=datetime.now(timezone.utc).isoformat(),
                 uploaded=uploaded, failed=failed, batch_failed=not outcomes, detail=detail)
        self.save_state()
        bucket = self.gen.config.minio.synthetic_data_bucket
        print(f"  uploaded {uploaded} to {bucket}/{self.model}/, {len(failed)} failed"
              + (" (run submit again to retry them)" if failed else ""), flush=True)


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
    args = parser.parse_args()

    labeler = BatchLabeler(args.model, raw_prefix=args.raw_prefix or args.from_prefix)
    if args.command in ("submit", "run"):
        pmids: List[str] = []
        if args.pmids_file:
            pmids += [line.strip() for line in Path(args.pmids_file).read_text().splitlines() if line.strip()]
        if args.from_prefix:
            pmids += labeler.gen.list_prefix_pmids()
        if not pmids:
            parser.error(f"{args.command} needs --from-prefix or --pmids-file")
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
