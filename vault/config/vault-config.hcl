storage "raft" {
  path    = "/vault/data"
  node_id = "vostok-node1"
}

listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/vault-cert.pem"
  tls_key_file  = "/vault/tls/vault-key.pem"
}

api_addr     = "https://vault:8200"
cluster_addr = "https://vault:8201"
ui           = true
disable_mlock = false
