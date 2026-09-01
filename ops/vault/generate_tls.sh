#!/usr/bin/env bash
# Generates a self-signed TLS cert/key for Vault's HTTP API listener.
# Re-run only when the cert needs regenerating (10-year expiry by design).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TLS_DIR="$REPO_ROOT/vault/tls"
mkdir -p "$TLS_DIR"

openssl req -x509 -newkey rsa:4096 -days 3650 -nodes \
  -keyout "$TLS_DIR/vault-key.pem" -out "$TLS_DIR/vault-cert.pem" \
  -subj "/CN=vault" \
  -addext "subjectAltName=DNS:vault,DNS:localhost,DNS:vostok.cs.unlv.edu,IP:127.0.0.1"

# 644 (not 600) because the official hashicorp/vault image drops privileges
# to a non-root "vault" user (uid 100) that must be able to read this via a
# bind mount from the host, where the host UID won't generally match.
# Acceptable at single-host lab scale; the container is the only reader that
# matters and the host directory itself isn't world-accessible.
chmod 644 "$TLS_DIR/vault-key.pem"
chmod 644 "$TLS_DIR/vault-cert.pem"

echo "Generated $TLS_DIR/vault-cert.pem and vault-key.pem"
echo "Copy vault-cert.pem to any machine (host scripts, SLURM login node) that needs VAULT_CACERT."
