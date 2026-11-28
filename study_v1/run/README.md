# Step 0 smoke test replay

Run from the existing worktree. The wrapper loads the root application config and
uses the configured Vault AppRole. It uses the selected PMIDs and smoke-only prefixes
from this directory. It keeps credentials out of command output.

```bash
bash study_v1/run/run.sh status
bash study_v1/run/run.sh collect qwen
```

The collection command checks once. Run it again if the batch is still pending.
Collection can write results to MinIO but does not submit paid requests. The saved
batch IDs and local result records are under `logs/batches/` and
`logs/results_batch_*/` in this worktree. The smoke bucket prefix is
`synthetic-data-v3/study_v1_smoke/s00/`.

To repeat the original submission command, invoke the wrapper with explicit aliases.
Submissions are paid, and a previously failed model will be retried. Pending or
already collected paper requests are skipped. Use a new smoke namespace for a fresh
run that should also repeat successful models.

```bash
bash study_v1/run/run.sh submit opus sol gemini deepseek qwen glm
```

The original run first verified the three selected BioC files in MinIO, then called
`BatchLabeler.submit(pmids)` for each alias. The follow-up retried the Anthropic
rejection once to obtain its billing error, and retried the two Together batches
once after their first attempts failed with zero usage. The collection loop called
`BatchLabeler.collect()` on pending batches every 30 seconds.

Gemini completed and uploaded three paper records at a reported cost of $0.251982.
Qwen was still pending when the collector stopped. Anthropic reported insufficient
credits, OpenAI reported a billing hard limit, and OpenRouter reported that it
could not submit either DeepSeek or GLM to Together.
