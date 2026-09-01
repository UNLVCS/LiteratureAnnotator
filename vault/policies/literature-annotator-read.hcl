# Read-only identity used by fastapi-backend and the host-side CLI scripts.
# Scoped to this project's secret paths only.

path "secret/data/literature-annotator/*" {
  capabilities = ["read"]
}

path "secret/metadata/literature-annotator/*" {
  capabilities = ["read", "list"]
}

# Dynamic Postgres credentials (used at the Postgres cutover; harmless until
# the database secrets engine is configured).
path "database/creds/label-studio-role" {
  capabilities = ["read"]
}
