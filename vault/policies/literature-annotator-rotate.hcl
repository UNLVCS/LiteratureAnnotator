# Read+write identity used only by the ops/vault/*.sh rotation scripts.
# Kept separate from the app's read-only role so a compromised app container
# cannot overwrite secrets.

path "secret/data/literature-annotator/*" {
  capabilities = ["read", "create", "update"]
}

path "secret/metadata/literature-annotator/*" {
  capabilities = ["read", "list"]
}

path "database/creds/label-studio-role" {
  capabilities = ["read"]
}

# Revoke a dynamic Postgres lease early (used when refreshing credentials).
path "sys/leases/revoke" {
  capabilities = ["update"]
}
