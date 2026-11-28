# Historical record: the original smoke-test submission, as run on 2026-11-26 from
# logs/step0-smoke in the step0-generation worktree. Kept for provenance; use run.sh instead.
VAULT_ADDR=https://localhost:8200 VAULT_CACERT="$PWD/../../vault/tls/vault-cert.pem" VAULT_SECRET_ID_FILE="$PWD/../../.vault-secret-id" VAULT_ROLE_ID="$(sed -n 's/^VAULT_ROLE_ID=//p' ../../label_api/.env)" UV_CACHE_DIR=/tmp/la-t03-uv uv run --offline --no-project --python 3.12 --with pydantic --with pyyaml --with requests --with anthropic --with openai --with minio --with hvac python -u - <<'PY'
import os
from pathlib import Path
from config.app_config import load_app_config
from config.study_config import load_study
from data_generation.batch_labeler import BatchLabeler, read_pmids
from papers import load_paper_from_config
from data_generation.batch_labeler import LabelingGenerator

smoke = Path("logs/step0-smoke").resolve()
overlay = smoke / "runtime-overlay.yaml"
overlay.write_text("llm_providers:\n  openrouter:\n    models: []\n")
os.environ["LA_CONFIG_OVERRIDE"] = str(overlay)
try:
    config = load_app_config(Path("../../.env.yaml").resolve())
except Exception as exc:
    raise SystemExit(f"App configuration failed: {type(exc).__name__}") from None
study = load_study(smoke / "models.yaml")
pmids = read_pmids(smoke / "pmids.txt")
raw_prefix = "pilot_2026_10"
gen = LabelingGenerator(raw_prefix=raw_prefix, load_providers=False)
for pmid in pmids:
    load_paper_from_config(pmid, gen.config, client=gen.minio)
print(f"Verified all {len(pmids)} selected papers in storage")
failures = []
for model in study.models:
    try:
        runner = BatchLabeler(model.alias, raw_prefix, study=study, sample_index=0)
        runner.submit(pmids)
    except (Exception, SystemExit) as exc:
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        code = getattr(exc, "code", None)
        print(f"{model.alias} failed: {type(exc).__name__} HTTP={status} code={code}", flush=True)
        failures.append(model.alias)
if failures:
    raise SystemExit(f"Submission failed for: {', '.join(failures)}")
PY
