#!/usr/bin/env bash
# Requests fresh dynamic Postgres credentials from Vault and points Label
# Studio at them.
#
# Label Studio is an unmodified third-party image with a persistent connection
# pool — it cannot re-fetch credentials mid-lease. So this runs on a schedule
# comfortably inside the lease TTL (hourly against a 24h TTL) and recreates the
# container to pick up the new values.
#
# Vault revokes the previous lease's Postgres role on its own; nothing here
# needs to drop it.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_unsealed_vault
vault_login_rotate

echo "==> Requesting dynamic credentials"
CREDS_JSON="$(vault_cmd read -format=json database/creds/label-studio-role)"
PG_USER="$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["username"])')"
PG_PASS="$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["password"])')"
LEASE="$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lease_duration"])')"

echo "    issued user $PG_USER (lease ${LEASE}s)"

echo "==> Updating $ENV_FILE"
set_env_key POSTGRE_USER "$PG_USER"
set_env_key POSTGRE_PASSWORD "$PG_PASS"

echo "==> Recreating label-studio"
(cd "$COMPOSE_DIR" && docker compose up -d --force-recreate label-studio)

echo "==> Waiting for Label Studio to become healthy"
for i in $(seq 1 30); do
  status="$(docker inspect -f '{{.State.Health.Status}}' annotation-label-studio 2>/dev/null || echo unknown)"
  if [ "$status" = "healthy" ]; then
    echo "    healthy after ${i}0s"
    exit 0
  fi
  sleep 10
done

echo "ERROR: Label Studio did not become healthy. Check:" >&2
echo "  docker logs annotation-label-studio --tail 50" >&2
exit 1
