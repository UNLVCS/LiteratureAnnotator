#!/usr/bin/env bash
# Fetches secrets from Vault on the SLURM *login* node and writes them to a
# local override file, so batch jobs never need to reach Vault themselves.
#
# Compute nodes allocated by SLURM often have no route to vostok:8200. Rather
# than assume they do, this runs before `sbatch` and writes
# <repo>/override.env.yaml — which load_app_config() already merges on top of
# .env.yaml, so data_generation/labeler_mp.py needs no code change.
#
# Uses curl rather than the vault CLI or docker, since the login node has
# neither. Requires the read-only AppRole credentials and the CA cert to be
# present on that machine.
#
#   VAULT_ADDR=https://vostok.cs.unlv.edu:8200 \
#   VAULT_CACERT=/path/to/vault-cert.pem \
#   VAULT_ROLE_ID=... VAULT_SECRET_ID_FILE=~/.vault-secret-id \
#     ./ops/vault/prepare_slurm_secrets.sh
#
# Delete the file when the job finishes: rm -f <repo>/override.env.yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_FILE="$REPO_ROOT/override.env.yaml"

: "${VAULT_ADDR:?set VAULT_ADDR, e.g. https://vostok.cs.unlv.edu:8200}"
: "${VAULT_ROLE_ID:?set VAULT_ROLE_ID}"
: "${VAULT_SECRET_ID_FILE:?set VAULT_SECRET_ID_FILE}"

CURL_OPTS=(--fail --silent --show-error)
if [ -n "${VAULT_CACERT:-}" ]; then
  CURL_OPTS+=(--cacert "$VAULT_CACERT")
fi

SECRET_ID="$(tr -d '\n' < "$VAULT_SECRET_ID_FILE")"

TOKEN="$(curl "${CURL_OPTS[@]}" -X POST \
  -d "{\"role_id\":\"$VAULT_ROLE_ID\",\"secret_id\":\"$SECRET_ID\"}" \
  "$VAULT_ADDR/v1/auth/approle/login" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["auth"]["client_token"])')"

kv() {
  curl "${CURL_OPTS[@]}" -H "X-Vault-Token: $TOKEN" \
    "$VAULT_ADDR/v1/secret/data/literature-annotator/$1" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['data'].get('$2',''))"
}

umask 077
python3 - "$OUT_FILE" \
  "$(kv redis password)" "$(kv minio access_key)" "$(kv minio secret_key)" \
  "$(kv pinecone api_key)" "$(kv label-studio api_key)" \
  "$(kv openai api_key)" "$(kv ncbi api_key)" <<'PY'
import sys, yaml

out, redis_pw, mk, sk, pine, ls, oai, ncbi = sys.argv[1:9]

# Only secrets. Structural config (hosts, buckets, model names) stays in
# .env.yaml; this file is merged on top of it.
doc = {
    "minio": {"access_key": mk, "secret_key": sk},
    "pinecone": {"api_key": pine},
    "label_studio": {"api_key": ls},
    "embeddings": {"api_key": oai},
    "bioc_download": {"ncbi_api_key": ncbi},
}
if redis_pw:
    # The compute node reaches Redis over the network, not docker DNS.
    doc["redis"] = {"url": f"redis://:{redis_pw}@vostok.cs.unlv.edu:6379/0"}

with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(doc, f)
PY

chmod 600 "$OUT_FILE"
echo "Wrote $OUT_FILE (mode 600)."
echo "Submit the job now, and delete this file when it finishes."
