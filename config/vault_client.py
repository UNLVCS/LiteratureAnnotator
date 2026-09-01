"""
Vault-backed secret loading for AppConfig.

Vault is opt-in: with VAULT_ADDR unset, nothing here runs and configuration
comes entirely from the YAML files, exactly as it did before Vault existed.

Environment:
    VAULT_ADDR            https://vault:8200 (unset disables Vault entirely)
    VAULT_CACERT          path to the self-signed CA cert used to verify Vault
    VAULT_ROLE_ID         AppRole role_id (not secret)
    VAULT_SECRET_ID_FILE  path to a chmod-600 file holding the AppRole secret_id

Usage:
    from config.vault_client import fetch_secret_overlay
    overlay = fetch_secret_overlay()  # {} when disabled or unreachable
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

KV_MOUNT = "secret"
KV_PREFIX = "literature-annotator"
POSTGRES_ROLE = "label-studio-role"

# Vault KV path -> (AppConfig section, {vault field: config field})
_SECRET_MAP = {
    "redis": ("redis", {"password": "password"}),
    "minio": ("minio", {"access_key": "access_key", "secret_key": "secret_key"}),
    "pinecone": ("pinecone", {"api_key": "api_key"}),
    "label-studio": ("label_studio", {"api_key": "api_key"}),
    "openai": ("embeddings", {"api_key": "api_key"}),
    "ncbi": ("bioc_download", {"api_key": "ncbi_api_key"}),
}


class VaultSecretsClient:
    """Thin hvac wrapper doing AppRole login and KV/database reads."""

    def __init__(
        self,
        addr: Optional[str] = None,
        cacert: Optional[str] = None,
        role_id: Optional[str] = None,
        secret_id_file: Optional[str] = None,
    ):
        self.addr = addr or os.environ.get("VAULT_ADDR", "")
        self.cacert = cacert or os.environ.get("VAULT_CACERT") or None
        self.role_id = role_id or os.environ.get("VAULT_ROLE_ID", "")
        self.secret_id_file = secret_id_file or os.environ.get("VAULT_SECRET_ID_FILE", "")
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.addr)

    def _read_secret_id(self) -> str:
        if not self.secret_id_file:
            raise ValueError("VAULT_SECRET_ID_FILE is not set")
        secret_id = Path(self.secret_id_file).read_text(encoding="utf-8").strip()
        if not secret_id:
            raise ValueError(f"{self.secret_id_file} is empty")
        return secret_id

    def login(self):
        import hvac

        if self._client is not None:
            return self._client

        if not self.role_id:
            raise ValueError("VAULT_ROLE_ID is not set")

        client = hvac.Client(url=self.addr, verify=self.cacert or True)
        client.auth.approle.login(role_id=self.role_id, secret_id=self._read_secret_id())
        if not client.is_authenticated():
            raise RuntimeError("Vault AppRole login did not yield an authenticated client")

        self._client = client
        return client

    def get_kv_secret(self, path: str) -> Dict[str, Any]:
        client = self.login()
        resp = client.secrets.kv.v2.read_secret_version(
            path=f"{KV_PREFIX}/{path}", mount_point=KV_MOUNT, raise_on_deleted_version=True
        )
        return resp["data"]["data"]

    def get_dynamic_postgres_creds(self, role: str = POSTGRES_ROLE) -> Dict[str, Any]:
        """Request short-lived Postgres credentials from the database engine.

        Used by the ops refresh script; no Python code here talks to Postgres.
        """
        client = self.login()
        resp = client.read(f"database/creds/{role}")
        return {
            "username": resp["data"]["username"],
            "password": resp["data"]["password"],
            "lease_id": resp["lease_id"],
            "lease_duration": resp["lease_duration"],
        }

    def fetch_secret_overlay(self) -> Dict[str, Any]:
        """Build a dict shaped for deep-merging over the YAML config data."""
        # Authenticate once up front so an unreachable/misconfigured Vault
        # fails immediately, instead of once per secret path below.
        self.login()

        overlay: Dict[str, Any] = {}
        for vault_path, (section, fields) in _SECRET_MAP.items():
            try:
                secret = self.get_kv_secret(vault_path)
            except Exception as e:
                logger.warning("Vault: could not read %s (%s)", vault_path, e)
                continue
            values = {
                cfg_field: secret[vault_field]
                for vault_field, cfg_field in fields.items()
                if secret.get(vault_field)
            }
            if values:
                overlay.setdefault(section, {}).update(values)
        return overlay


def fetch_secret_overlay() -> Dict[str, Any]:
    """Vault-sourced config overlay, or {} if Vault is disabled/unreachable."""
    client = VaultSecretsClient()
    if not client.enabled:
        return {}
    try:
        overlay = client.fetch_secret_overlay()
    except Exception as e:
        # print(), not logger: uvicorn does not surface INFO/WARNING from
        # module loggers, and operators must be able to tell from the logs
        # whether secrets came from Vault or silently fell back to YAML.
        print(f"[Vault] UNAVAILABLE - falling back to YAML config: {e}")
        return {}
    if overlay:
        print(f"[Vault] loaded secrets for: {', '.join(sorted(overlay))}")
    return overlay
