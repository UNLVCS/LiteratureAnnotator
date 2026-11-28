"""Re-parse stored answers that failed to parse, with the current parser. No API calls.

Keeps each answer's usage, finish reason and generation metadata; records the change
in the record's revisions; updates the local copy and the MinIO object.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.app_config import load_app_config
from study_v1.paths import main_checkout
from config.study_config import load_study
from data_generation.batch_labeler import BatchLabeler
from labeling import assemble_result, build_queries
from llm_providers.base import LLMResponse
from papers import load_paper_from_config
from utilities.criteria import CRITERION_NAMES

load_app_config(main_checkout() / ".env.yaml")
study = load_study(Path("config/study_models.yaml"))
apply = "--apply" in sys.argv
changed = 0
for spec in study.models:
    for k in range(6):
        runner = BatchLabeler(spec.alias, "pilot_2026_10", study=study, sample_index=k)
        for path in sorted(runner.out_dir.glob("*.json")):
            record = json.loads(path.read_text())
            for i, old in enumerate(record["criteria_results"]):
                if not (old.get("error") and (old.get("raw_response") or "").strip()):
                    continue
                paper = load_paper_from_config(record["paper_id"], runner.config, client=runner.gen.minio)
                fresh, _ = build_queries(paper, spec.id, max_chars=runner.config.papers.max_chars)
                response = LLMResponse(content=old["raw_response"], model=spec.id, usage=old.get("usage") or {},
                                       metadata={}, finish_reason=old.get("finish_reason"))
                index = CRITERION_NAMES.index(old["criterion"])
                new = assemble_result(paper, fresh, [response], criteria=[index])["criteria_results"][0]
                if new.get("error"):
                    print(f"  still fails: {spec.alias} s0{k} {record['paper_id']} {old['criterion']}")
                    continue
                for key in ("generation", "openrouter", "schema_version", "usage", "finish_reason"):
                    if key in old:
                        new[key] = old[key]
                record["criteria_results"][i] = new
                record.setdefault("revisions", []).append({
                    "criteria": [old["criterion"]], "reparsed": True,
                    "reason": "parser accepts control characters in strings (strict=False)",
                    "at": datetime.now(timezone.utc).isoformat()})
                changed += 1
                print(f"  reparsed: {spec.alias} s0{k} {record['paper_id']} {old['criterion']} -> "
                      f"{new['response'][old['criterion']]['satisfied']}")
                if apply:
                    path.write_text(json.dumps(record, indent=2))
                    print("    uploaded", runner.gen.upload(record))
print(f"{changed} answer(s) {'updated' if apply else 'would change (dry run; pass --apply)'}")
