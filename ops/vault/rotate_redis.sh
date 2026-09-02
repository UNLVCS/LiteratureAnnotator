#!/usr/bin/env bash
# Rotates the Redis password: new value -> Vault -> label_api/.env -> restart.
#
# Redis takes --requirepass as a startup argument and there is no redis.conf
# backing it, so a container recreate is the mechanism (not CONFIG SET).
# fastapi-backend is recreated too, because it reads secrets from Vault only
# at process start.
#
# WARNING: any consumer outside this compose stack that connects to Redis
# (host scripts already running, the cross-network app) loses access until it
# picks up the new password.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_unsealed_vault
vault_login_rotate

# Alphanumeric only: this value travels inside a redis:// URL and through
# Compose variable substitution, both of which treat / + @ = as syntax.
NEW_PASSWORD="$(openssl rand -base64 32 | tr -dc 'A-Za-z0-9' | cut -c1-32)"

echo "==> Writing new password to Vault"
vault_cmd kv put secret/literature-annotator/redis password="$NEW_PASSWORD" >/dev/null

echo "==> Updating $ENV_FILE"
set_env_key REDIS_PASSWORD "$NEW_PASSWORD"

echo "==> Recreating redis and fastapi-backend"
(cd "$COMPOSE_DIR" && docker compose up -d --force-recreate redis fastapi-backend)

echo "==> Verifying"
sleep 5
if docker exec annotation-redis redis-cli -a "$NEW_PASSWORD" ping 2>/dev/null | grep -q PONG; then
  echo "    redis accepts the new password"
else
  echo "    ERROR: redis did not accept the new password" >&2
  exit 1
fi

echo "Redis password rotated."
echo "Remember: external consumers need the new value from Vault."
