#!/usr/bin/env bash
# Live (non-batch) smoke test: one paper, one study model, results kept local.
#
#   ./study_v1/run/sync.sh <alias> [criteria] [pmid]
#
# alias     a model alias from models.yaml with route anthropic or openrouter
# criteria  how many of the six criterion requests to send (default 6)
# pmid      paper to label (default: first line of pmids.txt)
#
# Requests are built exactly as the batch path builds them (same prompts and
# study settings; OpenRouter calls are pinned to the study host with no
# fallback). This is billed at live prices, not batch prices. Nothing is
# uploaded to MinIO; responses go to logs/study_v1/sync/<alias>_<pmid>.json.
set -euo pipefail

if (( $# < 1 )) || [[ $1 == --help ]]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

cd "$(dirname "$0")/../.."
project="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"  # main checkout: .env.yaml, Vault files
export LA_MAIN_ROOT="$project"
export VAULT_ADDR="${VAULT_ADDR:-https://localhost:8200}"
export VAULT_CACERT="${VAULT_CACERT:-$project/vault/tls/vault-cert.pem}"
export VAULT_SECRET_ID_FILE="${VAULT_SECRET_ID_FILE:-$project/.vault-secret-id}"
export VAULT_ROLE_ID="${VAULT_ROLE_ID:-$(sed -n 's/^VAULT_ROLE_ID=//p' "$project/label_api/.env")}"
export LA_CONFIG_OVERRIDE="$PWD/study_v1/run/runtime-overlay.yaml"

if [[ -z ${UV_CACHE_DIR:-} && -d /tmp/la-t03-uv ]]; then
  export UV_CACHE_DIR=/tmp/la-t03-uv
fi

uv run --offline --no-project --python 3.12 \
  --with pydantic --with pyyaml --with requests --with anthropic --with openai \
  --with minio --with hvac python -u - "$@" <<'PY'
import json
import sys
import time
from pathlib import Path
import os

from config.app_config import load_app_config
from config.study_config import load_study
from data_generation.batch_labeler import BatchLabeler, read_pmids
from labeling import build_queries
from llm_providers.openrouter_provider import OpenRouterProvider
from papers import load_paper_from_config

smoke = Path("study_v1/run")
alias = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 6
study = load_study(smoke / "smoke_models.yaml")
pmid = sys.argv[3] if len(sys.argv) > 3 else read_pmids(Path("study_v1/selection/smoke_pmids.txt"))[0]
try:
    load_app_config(Path(os.environ["LA_MAIN_ROOT"]) / ".env.yaml")
except Exception as exc:
    raise SystemExit(f"App configuration failed: {type(exc).__name__}") from None

runner = BatchLabeler(alias, "pilot_2026_10", study=study)
spec = runner.study_model
if spec.route == "anthropic":
    provider = runner.backend.provider  # same AnthropicProvider the batch requests come from
elif spec.route == "openrouter":
    provider = OpenRouterProvider(runner.kwargs["api_key"], spec.id,
                                  provider={"only": [spec.host], "allow_fallbacks": False})
else:
    raise SystemExit(f"{alias}: route {spec.route} has no live test here (anthropic or openrouter only)")

paper = load_paper_from_config(pmid, runner.config, client=runner.gen.minio)
_, queries = build_queries(paper, spec.id, max_chars=runner.config.papers.max_chars,
                           temperature=runner.kwargs["temperature"], max_tokens=runner.kwargs["max_tokens"])
queries = queries[:count]
print(f"{alias}: model={spec.id} route={spec.route} host={spec.host} pmid={pmid} requests={len(queries)}")


def cost(usage):
    if usage.get("cost") is not None:  # OpenRouter reports the charged amount
        return float(usage["cost"])
    p = spec.batch_prices  # batch prices are about half of live, so double them
    written = usage.get("cache_creation_input_tokens", 0)
    return 2 * (usage.get("input_tokens", 0) * p.input
                + usage.get("cache_read_input_tokens", 0) * p.cached_input
                + written * (p.cache_write if p.cache_write is not None else p.input)
                + usage.get("output_tokens", 0) * p.output) / 1e6


out, total = [], 0.0
for i, query in enumerate(queries, 1):
    start = time.time()
    try:
        response = provider.call_api(query)
    except Exception as exc:
        resp = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(resp, "status_code", None)
        body = getattr(exc, "message", None) or (resp.text[:800] if resp is not None else str(exc))
        print(f"  c{i}: FAILED {type(exc).__name__} HTTP={status} {body}")
        out.append({"criterion": i, "error": f"{type(exc).__name__} HTTP={status}: {body}"})
        if status in (400, 401, 402, 403, 404):
            print("  stopping: the remaining requests would fail the same way")
            break
        continue
    usage = dict(response.usage or {})
    c = cost(usage)
    total += c
    host = ((response.metadata or {}).get("openrouter") or {}).get("host", spec.route)
    print(f"  c{i}: OK {time.time() - start:.0f}s host={host} finish={response.finish_reason} "
          f"usage={ {k: v for k, v in usage.items() if isinstance(v, (int, float))} } ~${c:.4f}")
    print(f"      {response.content[:160]!r}")
    out.append({"criterion": i, "content": response.content, "model": response.model,
                "finish_reason": response.finish_reason, "usage": usage, "metadata": response.metadata})

dest = Path("logs/study_v1/sync") / f"{alias}_{pmid}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2, default=str))
print(f"~${total:.4f} at live prices; saved {dest}")
PY
