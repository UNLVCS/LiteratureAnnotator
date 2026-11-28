"""One line per sample: papers collected per model, pending batches, failures and cost so far."""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S00 = {l.strip() for l in open(ROOT / "study_v1/selection/study_v1_s00_pmids.txt") if l.strip()}
RES = {l.strip() for l in open(ROOT / "study_v1/selection/study_v1_resample_pmids.txt") if l.strip()}
SHORT = {"anthropic": "opus", "openai": "sol", "google": "gemini", "deepseek": "deepseek", "qwen": "qwen", "z-ai": "glm", "moonshotai": "kimi"}

for k in range(6):
    target = S00 if k == 0 else RES
    parts, cost, pending, failed = [], 0.0, 0, 0
    for f in sorted(glob.glob(str(ROOT / f"logs/batches/*__study_v1_s0{k}.json"))):
        name = SHORT.get(Path(f).name.split("_")[0], Path(f).name[:8])
        batches = [b for b in json.load(open(f))["batches"] if (b.get("criteria") or [1, 2, 3, 4, 5, 6]) != [6]]
        # A paper is done once any batch returned it without failure (a retry may succeed).
        done = {p for b in batches if b.get("collected") for p in b["pmids"] if p not in (b.get("failed") or [])}
        bad = {p for b in batches if b.get("collected") for p in (b.get("failed") or [])}
        pending += sum(1 for b in batches if not b.get("collected"))
        failed += len(bad - done)
        cost += sum(b.get("actual_cost_usd") or 0 for b in batches)
        parts.append(f"{name} {len(done & target)}")
    if parts:
        print(f"s0{k} /{len(target)}: " + ", ".join(parts)
              + f" | pending batches {pending} | failed papers {failed} | ${cost:.2f}")
