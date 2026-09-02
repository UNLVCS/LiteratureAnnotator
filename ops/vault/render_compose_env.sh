#!/usr/bin/env bash
# Populates label_api/.env from Vault before `docker compose up`.
#
# Compose resolves ${REDIS_PASSWORD} and friends at `up` time — before any
# container, including Vault, is running — and has no way to call Vault
# itself. So this script does the fetch and writes the values Compose needs.
#
# Standard startup procedure on vostok:
#   ./ops/vault/render_compose_env.sh && (cd label_api && docker compose up -d)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_unsealed_vault
vault_login_rotate

set_env_key REDIS_PASSWORD    "$(get_kv_field redis password)"
set_env_key MINIO_ACCESS_KEY  "$(get_kv_field minio access_key)"
set_env_key MINIO_SECRET_KEY  "$(get_kv_field minio secret_key)"

echo "Wrote REDIS_PASSWORD, MINIO_ACCESS_KEY, MINIO_SECRET_KEY to $ENV_FILE"
echo "VAULT_ROLE_ID is left untouched (set once by bootstrap.sh)."
