#!/usr/bin/env bash
# Requests fresh dynamic Postgres credentials from Vault and points Label
# Studio at them.
#
# Label Studio is an unmodified third-party image with a persistent connection
# pool — it cannot re-fetch credentials mid-lease. So this runs on a schedule
# comfortably inside the lease TTL (hourly against a 24h TTL) and recreates the
# container to pick up the new values.
#
# The previous lease is revoked explicitly once the new credentials are live.
# Leaving it to expire on its own would pile up a stale Postgres role on every
# run — hourly refreshes against a 24h TTL means ~24 of them.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

LEASE_FILE="$REPO_ROOT/ops/vault/.postgres-lease-id"

require_unsealed_vault
vault_login_rotate

PREVIOUS_LEASE=""
[ -f "$LEASE_FILE" ] && PREVIOUS_LEASE="$(cat "$LEASE_FILE")"

echo "==> Requesting dynamic credentials"
CREDS_JSON="$(vault_cmd read -format=json database/creds/label-studio-role)"
PG_USER="$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["username"])')"
PG_PASS="$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["password"])')"
LEASE="$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lease_duration"])')"
LEASE_ID="$(echo "$CREDS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lease_id"])')"

echo "    issued user $PG_USER (lease ${LEASE}s)"
printf '%s' "$LEASE_ID" > "$LEASE_FILE"
chmod 600 "$LEASE_FILE"

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

    # Only now that the new credentials are proven working — revoking earlier
    # would cut the connection Label Studio is still using if this run fails.
    if [ -n "$PREVIOUS_LEASE" ]; then
      echo "==> Revoking the previous lease"
      if vault_cmd write sys/leases/revoke lease_id="$PREVIOUS_LEASE" >/dev/null 2>&1; then
        echo "    revoked"
      else
        echo "    WARNING: could not revoke $PREVIOUS_LEASE (it may have already expired)"
      fi
    fi
    exit 0
  fi
  sleep 10
done

echo "ERROR: Label Studio did not become healthy. Check:" >&2
echo "  docker logs annotation-label-studio --tail 50" >&2
echo "Previous credentials were left active, so the old lease still works." >&2
exit 1
