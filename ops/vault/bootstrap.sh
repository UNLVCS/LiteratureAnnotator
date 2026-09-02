#!/usr/bin/env bash
# One-time Vault setup: KV v2 engine, policies, and the two AppRole identities.
# Re-running is safe (already-enabled mounts are skipped), but it issues fresh
# secret_ids each time, which invalidates nothing but does add credentials.
#
# Requires the root token (from `vault operator init`) in the environment:
#   VAULT_TOKEN=hvs.xxxx ./ops/vault/bootstrap.sh
#
# The Postgres database secrets engine is NOT set up here — see
# ops/vault/bootstrap_postgres_engine.sh, which is run later at cutover time
# because it modifies the live database.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ -z "${VAULT_TOKEN:-}" ]; then
  echo "ERROR: VAULT_TOKEN must be set to the Vault root token." >&2
  exit 1
fi

require_unsealed_vault

echo "==> Enabling KV v2 at secret/"
if vault_cmd secrets list -format=json | grep -q '"secret/"'; then
  echo "    already enabled, skipping"
else
  vault_cmd secrets enable -path=secret kv-v2
fi

echo "==> Writing policies"
vault_cmd policy write literature-annotator-read - \
  < "$REPO_ROOT/vault/policies/literature-annotator-read.hcl"
vault_cmd policy write literature-annotator-rotate - \
  < "$REPO_ROOT/vault/policies/literature-annotator-rotate.hcl"

echo "==> Enabling AppRole auth"
if vault_cmd auth list -format=json | grep -q '"approle/"'; then
  echo "    already enabled, skipping"
else
  vault_cmd auth enable approle
fi

echo "==> Creating AppRole roles"
vault_cmd write auth/approle/role/literature-annotator \
  token_policies="literature-annotator-read" \
  token_ttl=1h token_max_ttl=4h

# token_ttl MUST exceed the Postgres credential TTL in
# bootstrap_postgres_engine.sh. A dynamic secret read with this token becomes a
# CHILD lease of it, and Vault cascades revocation when the parent expires — so
# a short token here silently caps how long Label Studio's database credential
# actually lives, no matter what TTL the database role advertises.
vault_cmd write auth/approle/role/literature-annotator-rotate \
  token_policies="literature-annotator-rotate" \
  token_ttl=6h token_max_ttl=12h

echo "==> Issuing credentials"
APP_ROLE_ID="$(vault_cmd read -field=role_id auth/approle/role/literature-annotator/role-id)"
APP_SECRET_ID="$(vault_cmd write -f -field=secret_id auth/approle/role/literature-annotator/secret-id)"
ROTATE_ROLE_ID="$(vault_cmd read -field=role_id auth/approle/role/literature-annotator-rotate/role-id)"
ROTATE_SECRET_ID="$(vault_cmd write -f -field=secret_id auth/approle/role/literature-annotator-rotate/secret-id)"

# role_id is not secret; secret_id is. Both stay out of git (see .gitignore).
printf '%s' "$APP_SECRET_ID" > "$REPO_ROOT/.vault-secret-id"
chmod 600 "$REPO_ROOT/.vault-secret-id"

printf '%s' "$ROTATE_ROLE_ID" > "$REPO_ROOT/ops/vault/.rotate-role-id"
chmod 600 "$REPO_ROOT/ops/vault/.rotate-role-id"
printf '%s' "$ROTATE_SECRET_ID" > "$REPO_ROOT/ops/vault/.rotate-secret-id"
chmod 600 "$REPO_ROOT/ops/vault/.rotate-secret-id"

# The app's role_id rides in label_api/.env so Compose can pass it through.
set_env_key VAULT_ROLE_ID "$APP_ROLE_ID"

echo
echo "Bootstrap complete."
echo "  app role_id      -> label_api/.env (VAULT_ROLE_ID)"
echo "  app secret_id    -> .vault-secret-id"
echo "  rotate role_id   -> ops/vault/.rotate-role-id"
echo "  rotate secret_id -> ops/vault/.rotate-secret-id"
