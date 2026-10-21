#!/bin/bash
# Usage: weak_supervision/ws.sh <script.py> [args]   (script path relative to weak_supervision/)
#        weak_supervision/ws.sh test [pytest args]
# Runs in a throwaway uv env on Python 3.12 with Vault-backed MinIO/NCBI credentials.
# Deliberately not `uv sync`: the workspace pulls torch/CUDA and vostok's disk is small.
cd "$(dirname "$0")/.."
export VAULT_ADDR=https://localhost:8200 VAULT_CACERT=$PWD/vault/tls/vault-cert.pem VAULT_SECRET_ID_FILE=$PWD/.vault-secret-id
export VAULT_ROLE_ID=$(grep -oP '^VAULT_ROLE_ID=\K.*' label_api/.env)
export PYTHONPATH=$PWD:$PWD/data_generation:$PWD/weak_supervision PYTHONUNBUFFERED=1
if [ "$1" = test ]; then
  shift; set -- -m pytest -q -p no:cacheprovider weak_supervision/tests "$@"
else
  script=weak_supervision/$1; shift; set -- "$script" "$@"
fi
exec uv run -q --no-project --python /usr/bin/python3.12 \
  --with minio --with pyyaml --with pydantic --with pydantic-settings --with hvac --with requests \
  --with numpy --with pandas --with pyarrow --with tabulate --with pytest \
  python "$@"
