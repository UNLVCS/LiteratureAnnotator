#!/usr/bin/env bash
# Rotates the MinIO root credentials: new values -> Vault -> .env -> restart.
#
# MinIO reads MINIO_ROOT_USER/MINIO_ROOT_PASSWORD from its environment at
# startup, so this is a recreate, not a live update. Bucket data is unaffected
# (credentials are env-driven, not stored in the data directory).
#
# WARNING: wider blast radius than the Redis rotation. Anything reading these
# buckets breaks until it picks up the new keys — host scripts
# (data_vectorize/, utilities/seed_queue_from_bucket.py), the SLURM job, and
# any cross-network consumer.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_unsealed_vault
vault_login_rotate

NEW_ACCESS_KEY="$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | cut -c1-20)"
NEW_SECRET_KEY="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | cut -c1-40)"

echo "==> Writing new credentials to Vault"
vault_cmd kv put secret/literature-annotator/minio \
  access_key="$NEW_ACCESS_KEY" secret_key="$NEW_SECRET_KEY" >/dev/null

echo "==> Updating $ENV_FILE"
set_env_key MINIO_ACCESS_KEY "$NEW_ACCESS_KEY"
set_env_key MINIO_SECRET_KEY "$NEW_SECRET_KEY"

echo "==> Recreating minio and fastapi-backend"
(cd "$COMPOSE_DIR" && docker compose up -d --force-recreate minio fastapi-backend)

echo "==> Verifying"
sleep 8
if docker exec annotation-minio mc alias set verify http://localhost:9000 \
     "$NEW_ACCESS_KEY" "$NEW_SECRET_KEY" >/dev/null 2>&1; then
  echo "    minio accepts the new credentials"
  docker exec annotation-minio mc alias remove verify >/dev/null 2>&1 || true
else
  echo "    NOTE: could not verify via mc (not present in image is normal)."
  echo "    Check the console at :9001 with the new credentials instead."
fi

echo "MinIO credentials rotated."
echo "Remember: host scripts and any external consumer need the new values."
