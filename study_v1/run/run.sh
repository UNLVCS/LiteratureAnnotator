#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )) || [[ $1 == --help ]]; then
  printf 'Usage: %s {submit|status|collect} [model aliases...]\n' "$0"
  printf 'Use explicit aliases when submitting. Status and collect default to all models.\n'
  exit 0
fi

case "$1" in
  submit) if (( $# < 2 )); then exit 2; fi ;;
  status|collect) ;;
  *) printf 'Unknown command %s\n' "$1" >&2; exit 2 ;;
esac

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
from pathlib import Path
import os
import sys

from config.app_config import load_app_config
from config.study_config import load_study
from data_generation.batch_labeler import BatchLabeler, read_pmids

smoke = Path("study_v1/run")
# Defaults run the smoke test; override for the pilot or the production run, e.g.
# LA_STUDY_CONFIG=config/study_models.yaml LA_PMIDS_FILE=... LA_SAMPLE=0
study = load_study(Path(os.environ.get("LA_STUDY_CONFIG") or smoke / "smoke_models.yaml"))
pmids = read_pmids(Path(os.environ.get("LA_PMIDS_FILE") or Path("study_v1/selection/smoke_pmids.txt")))
sample = int(os.environ.get("LA_SAMPLE") or 0)
print(f"study={os.environ.get('LA_STUDY_CONFIG') or smoke / 'smoke_models.yaml'} "
      f"prefix={study.sample_prefixes[sample]} papers={len(pmids)} "
      f"max_batch_usd={os.environ.get('LA_MAX_BATCH_USD') or 'none'}")
command, *aliases = sys.argv[1:]
aliases = aliases or [model.alias for model in study.models]
try:
    load_app_config(Path(os.environ["LA_MAIN_ROOT"]) / ".env.yaml")
except Exception as exc:
    raise SystemExit(f"App configuration failed: {type(exc).__name__}") from None

failed = []
for alias in aliases:
    try:
        runner = BatchLabeler(alias, "pilot_2026_10", study=study, sample_index=sample)
        if command == "submit":
            runner.submit(pmids)
        elif command == "status":
            runner.status()
        else:
            runner.collect()
    except (Exception, SystemExit) as exc:
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        detail = str(exc).split(" | ", 1)[1][:300] if " | " in str(exc) else ""
        print(f"{alias}: {type(exc).__name__} HTTP={status} code={getattr(exc, 'code', None)} {detail}")
        failed.append(alias)
if failed:
    raise SystemExit(f"Failed: {', '.join(failed)}")
PY
