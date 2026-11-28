"""Integrity check of the study_v1 records (local copies written at collection)."""
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
read = lambda p: [l.strip() for l in open(ROOT / p) if l.strip()]
S00, RES = read("study_v1/selection/study_v1_s00_pmids.txt"), read("study_v1/selection/study_v1_resample_pmids.txt")
MODELS = {"anthropic_claude-opus-5.5": "opus", "openai_gpt-6.2-sol": "sol", "google_gemini-4-carbon": "gemini",
          "deepseek_deepseek-v4.1-flash-1017": "deepseek", "qwen_qwen3.8-max-0902": "qwen",
          "moonshotai_kimi-k3.1-pro": "kimi"}
NAMES = [f"criterion_{i}" for i in range(1, 7)]

missing, structure, problems = [], [], []
finish = collections.defaultdict(collections.Counter)
hosts = collections.defaultdict(collections.Counter)
verdicts = collections.defaultdict(collections.Counter)
total = 0
for slug, alias in MODELS.items():
    for k in range(6):
        for pmid in (S00 if k == 0 else RES):
            path = ROOT / f"logs/results_batch_{slug}__study_v1_s0{k}/{pmid}.json"
            if not path.exists():
                missing.append((alias, k, pmid)); continue
            r = json.loads(path.read_text()); total += 1
            names = [c["criterion"] for c in r["criteria_results"]]
            if names != NAMES or r.get("partial") or str(r.get("paper_id")) != pmid:
                structure.append((alias, k, pmid, names, r.get("partial")))
            if r.get("sample_index", k) != k:
                structure.append((alias, k, pmid, "sample_index", r.get("sample_index")))
            c6 = r["criteria_results"][5] if len(r["criteria_results"]) == 6 else {}
            if c6.get("schema_version", r.get("schema_version")) != "whole-paper-v3":
                structure.append((alias, k, pmid, "C6 schema", c6.get("schema_version")))
            for c in r["criteria_results"]:
                fr = c.get("finish_reason"); finish[alias][fr] += 1
                hosts[alias][(c.get("generation") or {}).get("host")] += 1
                v = ((c.get("response") or {}).get(c["criterion"]) or {}).get("satisfied")
                verdicts[(alias, c["criterion"])][str(v)] += 1
                if c.get("error") or fr != "stop":
                    problems.append((alias, f"s0{k}", pmid, c["criterion"], fr, (c.get("error") or "")[:40]))

print(f"records found: {total} of {len(MODELS) * (len(S00) + 5 * len(RES))}; missing: {len(missing)} {missing[:5]}")
print(f"structure problems: {len(structure)} {structure[:5]}")
print("\nfinish reasons and hosts per model:")
for a in MODELS.values():
    print(f"  {a:<9} {dict(finish[a])}  hosts {dict(hosts[a])}")
print(f"\nproblem answers ({len(problems)} of {total * 6}):")
for p in problems: print("  ", p)
bad_c6 = {k: v for k, v in verdicts.items() if k[1] == "criterion_6" and set(v) - {"True", "False"}}
bad_na = {k: v for k, v in verdicts.items() if k[1] not in ("criterion_3", "criterion_6") and set(v) - {"True", "False", "None"}}
print("\nC6 answers outside true/false:", bad_c6 or "none")
print("not_applicable outside C3:", bad_na or "none")
