#!/usr/bin/env bash
# Shared helpers for the Vault ops scripts. Source this, don't execute it.
#
# There is no vault binary on the host, so every CLI call is routed through
# the running Vault container.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VAULT_CONTAINER="${VAULT_CONTAINER:-annotation-vault}"
COMPOSE_DIR="$REPO_ROOT/label_api"
ENV_FILE="$COMPOSE_DIR/.env"

vault_cmd() {
  docker exec -i \
    -e VAULT_ADDR=https://127.0.0.1:8200 \
    -e VAULT_CACERT=/vault/tls/vault-cert.pem \
    -e VAULT_TOKEN="${VAULT_TOKEN:-}" \
    "$VAULT_CONTAINER" vault "$@"
}

# Fails loudly if Vault is sealed — every script here needs it unsealed, and
# the error Vault returns otherwise is not obvious.
require_unsealed_vault() {
  local sealed
  sealed="$(VAULT_TOKEN= vault_cmd status -format=json 2>/dev/null | grep -o '"sealed": *[a-z]*' | awk '{print $2}')"
  if [ "$sealed" != "false" ]; then
    echo "ERROR: Vault is sealed or unreachable. Unseal it first:" >&2
    echo "  docker exec -e VAULT_ADDR=https://127.0.0.1:8200 -e VAULT_CACERT=/vault/tls/vault-cert.pem \\" >&2
    echo "    $VAULT_CONTAINER vault operator unseal <unseal-key>" >&2
    return 1
  fi
}

# Logs in with the rotation AppRole and exports VAULT_TOKEN for later calls.
vault_login_rotate() {
  local role_id secret_id
  role_id="$(cat "$REPO_ROOT/ops/vault/.rotate-role-id")"
  secret_id="$(cat "$REPO_ROOT/ops/vault/.rotate-secret-id")"
  VAULT_TOKEN="$(VAULT_TOKEN= vault_cmd write -field=token auth/approle/login \
    role_id="$role_id" secret_id="$secret_id")"
  export VAULT_TOKEN
}

# Read-modify-write a single KEY=VALUE line in an env file, leaving every
# other key untouched. Filter-and-append rather than sed, so values
# containing /, +, = and other regex/replacement metacharacters are safe.
set_env_key() {
  local key="$1" value="$2" file="${3:-$ENV_FILE}"
  local tmp
  touch "$file"
  tmp="$(mktemp)"
  grep -v "^${key}=" "$file" > "$tmp" || true
  printf '%s=%s\n' "$key" "$value" >> "$tmp"
  mv "$tmp" "$file"
  chmod 600 "$file"
}

get_kv_field() {
  local path="$1" field="$2"
  vault_cmd kv get -field="$field" "secret/literature-annotator/$path"
}
