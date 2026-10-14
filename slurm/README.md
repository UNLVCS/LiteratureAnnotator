# Labeling on the GPU cluster

`label_vllm.sbatch` labels papers with an open model (default `openai/gpt-oss-120b`)
on cluster GPUs. Papers and results stay in MinIO on vostok.

The cluster can reach vostok only over SSH (port 22). MinIO (9000) and Vault (8200)
are blocked. So the job opens its own SSH tunnel:

```
cluster compute node                                  vostok
labeler.py ──► localhost:<p1> ══ SSH tunnel ══►  localhost:9000  MinIO
config     ──► localhost:<p2> ══ SSH tunnel ══►  localhost:8200  Vault
labeler.py ──► localhost:<p3>  vllm serve (job's GPUs)
```

The job then:

1. Fetches the MinIO keys from Vault through the tunnel.
2. Checks that MinIO works, before it loads the model.
3. Serves the model with `vllm serve`.
4. Runs `data_generation/labeler.py --upload --skip-existing`.

Results land at `synthetic-data/{output_prefix}/{model}/{pmid}.json`. Papers that
already have a result are skipped, so after a timeout or preemption you can resubmit
the same job.

## One-time setup (on the cluster login node)

1. **Code and tools**

   ```bash
   git clone git@github.com:UNLVCS/LiteratureAnnotator.git && cd LiteratureAnnotator
   git checkout retire-rag
   command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
   cp slurm/cluster.env.yaml.example .env.yaml
   cp slurm/cluster.env.example slurm/cluster.env   # then edit; gitignored
   ```

2. **Tunnel key.** Use a dedicated key with no passphrase, because jobs run unattended.

   ```bash
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/vostok_tunnel -C cluster-tunnel
   cat ~/.ssh/vostok_tunnel.pub
   ```

   On vostok, add the public key to `~/.ssh/authorized_keys` with these restrictions.
   The key can only forward to MinIO and Vault, and cannot open a shell or run
   commands:

   ```
   restrict,port-forwarding,permitopen="localhost:9000",permitopen="localhost:8200",command="/bin/false" ssh-ed25519 AAAA... cluster-tunnel
   ```

   Back on the login node, connect once to record vostok's host key. Jobs run in
   `BatchMode` and would fail on an unknown host. Then test the tunnel:

   ```bash
   ssh -i ~/.ssh/vostok_tunnel -N -L 29000:localhost:9000 abreham@vostok.cs.unlv.edu &
   curl -s -o /dev/null -w "%{http_code}\n" http://localhost:29000/minio/health/live   # 200
   kill %1
   ```

3. **Vault credentials.** Copy these from vostok into the paths named in
   `slurm/cluster.env`, with `chmod 600` on the secret:
   - `vault/tls/vault-cert.pem`, the CA certificate. Vault's certificate covers
     `localhost`, so it verifies through the tunnel.
   - `.vault-secret-id`, the read-only app AppRole secret.
   - `VAULT_ROLE_ID` from `label_api/.env`.

   Vault must be unsealed. Otherwise the MinIO check in step 2 of the job fails, and
   it fails before any GPU time is spent.

4. **Partition and GPUs.** Edit `#SBATCH --partition` and, if needed, `--gres`,
   `--mem` and `--time` in `label_vllm.sbatch`. `TP` (tensor parallel size) must equal
   the number of GPUs requested.

## Running

```bash
# Stratified 33-paper pilot (3 per stratum)
PMIDS_FILE=data_download/strata/pilot_2026_10_sample33.txt sbatch slurm/label_vllm.sbatch

# Everything in the pilot corpus
sbatch slurm/label_vllm.sbatch

# Another model or GPU count
MODEL=meta-llama/Llama-3.1-70B-Instruct TP=4 MAX_TOKENS=1000 sbatch slurm/label_vllm.sbatch
```

The job writes these logs:
- `logs/label_<job>.out`: progress, one line per paper.
- `logs/vllm_<job>.log`: the model server.
- `logs/results_<job>/`: local copies of each result.

The first run downloads the model weights into `$HF_HOME`, about 65 GB for
gpt-oss-120b, so point it at scratch storage.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `openai/gpt-oss-120b` | Hugging Face model id served by vLLM |
| `TP` | `4` | GPUs to shard the model across |
| `MAX_TOKENS` | `8000` | Output budget per answer. Reasoning models need room to think before the JSON |
| `MAX_MODEL_LEN` | `65536` | Context length; the longest pilot papers are about 20k tokens |
| `RAW_PREFIX` | `pilot_2026_10` | MinIO folder of papers |
| `PMIDS_FILE` / `LIMIT` | unset | Label only the listed papers / only the first N |

## API models

GPT-4o and Claude need no GPU. Run them from vostok, or from anywhere that can reach
MinIO:

```bash
python data_generation/labeler.py --pmids-file data_download/strata/pilot_2026_10_sample33.txt \
    --raw-prefix pilot_2026_10 --providers gpt-4o,claude-sonnet-4-6 --upload --skip-existing
```

To land next to the cluster results, use the same `minio.synthetic_data_bucket` and
`minio.output_prefix` as `cluster.env.yaml.example`, i.e. `synthetic-data` and
`pilot_2026_10`. Set them in `.env.yaml` or in a file passed via `$LA_CONFIG_OVERRIDE`.
