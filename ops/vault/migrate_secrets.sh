#!/usr/bin/env bash
# One-time migration: copy the secret values currently living in .env.yaml
# (merged with .env.docker-override.yaml) into Vault's KV v2 store.
#
# This only COPIES. It does not modify the YAML files and does not rotate
# anything, so it is safe to run against a live stack — nothing reads from
# Vault until the app is wired up and VAULT_ADDR is set.
#
# Values are piped straight into Vault and never echoed to the terminal.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_unsealed_vault
vault_login_rotate

# Emits "field<TAB>value" lines for one logical secret, read out of the merged
# YAML config. Uses the same base+override merge order as load_app_config().
extract() {
  python3 - "$REPO_ROOT" "$1" <<'PY'
import sys, pathlib, yaml

repo, which = pathlib.Path(sys.argv[1]), sys.argv[2]

def load(p):
    return yaml.safe_load(p.read_text()) if p.exists() else {}

def merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out

cfg = merge(load(repo / ".env.yaml"), load(repo / ".env.docker-override.yaml"))

def emit(field, value):
    if value:
        print(f"{field}\t{value}")

if which == "redis":
    # Vault stores the bare password; host/port stay in YAML since they
    # differ per environment (docker service DNS vs. the vostok hostname).
    url = cfg.get("redis", {}).get("url", "")
    from urllib.parse import urlsplit
    emit("password", urlsplit(url).password or "")
elif which == "minio":
    m = cfg.get("minio", {})
    emit("access_key", m.get("access_key"))
    emit("secret_key", m.get("secret_key"))
elif which == "pinecone":
    emit("api_key", cfg.get("pinecone", {}).get("api_key"))
elif which == "label-studio":
    emit("api_key", cfg.get("label_studio", {}).get("api_key"))
elif which == "openai":
    emit("api_key", cfg.get("embeddings", {}).get("api_key"))
elif which == "ncbi":
    emit("api_key", cfg.get("bioc_download", {}).get("ncbi_api_key"))
PY
}

for name in redis minio pinecone label-studio openai ncbi; do
  mapfile -t pairs < <(extract "$name")
  if [ "${#pairs[@]}" -eq 0 ]; then
    echo "==> $name: no value found in YAML, skipping"
    continue
  fi

  args=()
  for line in "${pairs[@]}"; do
    field="${line%%$'\t'*}"
    value="${line#*$'\t'}"
    args+=("$field=$value")
  done

  vault_cmd kv put "secret/literature-annotator/$name" "${args[@]}" >/dev/null
  echo "==> $name: stored ${#args[@]} field(s)"
done

echo
echo "Migration complete. Verify with:"
echo "  vault kv list secret/literature-annotator"
