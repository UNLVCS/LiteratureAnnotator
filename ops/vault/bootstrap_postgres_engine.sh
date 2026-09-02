#!/usr/bin/env bash
# One-time setup of Vault's database secrets engine for Postgres.
#
# THIS MODIFIES THE LIVE DATABASE. It:
#   1. rotates the Postgres superuser password away from the hardcoded default
#   2. creates a labelstudio_app role and transfers ownership of existing
#      objects to it, so Vault's ephemeral users inherit access to data that
#      already exists
#   3. configures Vault to mint short-lived credentials in that role
#
# Between step 1 and the first refresh_postgres_creds.sh run, Label Studio
# cannot reach the database — this script runs that refresh itself at the end.
#
# Requires the root token:
#   VAULT_TOKEN=hvs.xxxx ./ops/vault/bootstrap_postgres_engine.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ -z "${VAULT_TOKEN:-}" ]; then
  echo "ERROR: VAULT_TOKEN must be set to the Vault root token." >&2
  exit 1
fi

require_unsealed_vault

PG_CONTAINER=annotation-postgres
PG_SUPERUSER=labelstudio
PG_DB=labelstudio

psql_super() {
  docker exec -i -e PGPASSWORD="$1" "$PG_CONTAINER" \
    psql -U "$PG_SUPERUSER" -d "$PG_DB" -v ON_ERROR_STOP=1 "${@:2}"
}

# The current superuser password: the hardcoded default on first run, or the
# already-rotated value from Vault on a re-run.
CURRENT_PASSWORD="$(vault_cmd kv get -field=password \
  secret/literature-annotator/postgres-admin 2>/dev/null || echo "labelstudio")"

echo "==> Rotating the Postgres superuser password"
NEW_ADMIN_PASSWORD="$(openssl rand -base64 32 | tr -dc 'A-Za-z0-9' | cut -c1-32)"
psql_super "$CURRENT_PASSWORD" -c \
  "ALTER USER $PG_SUPERUSER WITH PASSWORD '$NEW_ADMIN_PASSWORD';" >/dev/null
vault_cmd kv put secret/literature-annotator/postgres-admin \
  username="$PG_SUPERUSER" password="$NEW_ADMIN_PASSWORD" >/dev/null
echo "    rotated and stored in Vault (secret/literature-annotator/postgres-admin)"

echo "==> Creating the labelstudio_app role and transferring ownership"
psql_super "$NEW_ADMIN_PASSWORD" <<SQL >/dev/null
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'labelstudio_app') THEN
    CREATE ROLE labelstudio_app NOLOGIN;
  END IF;
END
\$\$;

GRANT ALL PRIVILEGES ON DATABASE $PG_DB TO labelstudio_app;
GRANT ALL ON SCHEMA public TO labelstudio_app;
GRANT ALL ON ALL TABLES IN SCHEMA public TO labelstudio_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO labelstudio_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO labelstudio_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO labelstudio_app;

-- Transfer ownership object by object. A blanket REASSIGN OWNED is rejected
-- here because the superuser also owns the database itself, which Postgres
-- treats as required by the database system. Ownership (not just GRANT) is
-- what lets Label Studio's migrations run ALTER/DROP on existing tables.
DO \$\$
DECLARE r record;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP
    EXECUTE format('ALTER TABLE public.%I OWNER TO labelstudio_app', r.tablename);
  END LOOP;
  FOR r IN SELECT sequencename FROM pg_sequences WHERE schemaname = 'public' LOOP
    EXECUTE format('ALTER SEQUENCE public.%I OWNER TO labelstudio_app', r.sequencename);
  END LOOP;
  FOR r IN SELECT viewname FROM pg_views WHERE schemaname = 'public' LOOP
    EXECUTE format('ALTER VIEW public.%I OWNER TO labelstudio_app', r.viewname);
  END LOOP;
END
\$\$;
SQL
echo "    labelstudio_app owns the existing objects"

echo "==> Enabling the database secrets engine"
if vault_cmd secrets list -format=json | grep -q '"database/"'; then
  echo "    already enabled, skipping"
else
  vault_cmd secrets enable database
fi

echo "==> Configuring the Postgres connection"
vault_cmd write database/config/postgres \
  plugin_name=postgresql-database-plugin \
  allowed_roles="label-studio-role" \
  connection_url="postgresql://{{username}}:{{password}}@postgres:5432/$PG_DB?sslmode=disable" \
  username="$PG_SUPERUSER" \
  password="$NEW_ADMIN_PASSWORD" >/dev/null

echo "==> Creating the dynamic role"
# default_ttl must stay BELOW the rotation AppRole's token_ttl (6h, set in
# bootstrap.sh): credentials issued with that token are child leases and die
# with it. 4h against an hourly refresh still tolerates three consecutive
# failed refreshes before Label Studio loses its database access.
vault_cmd write database/roles/label-studio-role \
  db_name=postgres \
  creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}' IN ROLE labelstudio_app INHERIT;" \
  default_ttl=4h \
  max_ttl=24h >/dev/null

echo "==> Issuing the first set of credentials for Label Studio"
"$REPO_ROOT/ops/vault/refresh_postgres_creds.sh"

echo
echo "Postgres is now on Vault-issued dynamic credentials."
echo "Schedule ops/vault/refresh_postgres_creds.sh hourly (see vault/README.md)."
