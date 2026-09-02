#!/usr/bin/env bash
# Records a rotated third-party API key in Vault.
#
# Semi-manual by necessity: none of these providers expose a self-service
# rotation API, so you generate the new key in their dashboard first, then
# run this to store it and restart the consumer.
#
# Usage: ./ops/vault/rotate_saas_key.sh <pinecone|label-studio|openai|ncbi>
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PROVIDER="${1:-}"
case "$PROVIDER" in
  pinecone|label-studio|openai|ncbi) ;;
  *)
    echo "Usage: $0 <pinecone|label-studio|openai|ncbi>" >&2
    exit 1
    ;;
esac

case "$PROVIDER" in
  pinecone)     WHERE="Pinecone console -> API Keys" ;;
  label-studio) WHERE="Label Studio -> Account & Settings -> Access Token" ;;
  openai)       WHERE="platform.openai.com -> API keys" ;;
  ncbi)         WHERE="NCBI account -> API Key Management" ;;
esac

echo "Generate the replacement key first: $WHERE"
echo "Then paste it below (input is hidden)."
read -r -s -p "New $PROVIDER key: " NEW_KEY
echo

if [ -z "$NEW_KEY" ]; then
  echo "ERROR: empty key, aborting." >&2
  exit 1
fi

require_unsealed_vault
vault_login_rotate

vault_cmd kv put "secret/literature-annotator/$PROVIDER" api_key="$NEW_KEY" >/dev/null
echo "==> Stored new $PROVIDER key in Vault"

echo "==> Recreating fastapi-backend"
(cd "$COMPOSE_DIR" && docker compose up -d --force-recreate fastapi-backend)

echo
echo "$PROVIDER key rotated. Revoke the OLD key in the provider's dashboard now."
echo "Host-side scripts pick up the new value on their next run."
