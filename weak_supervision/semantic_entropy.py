"""
Semantic entropy over reason texts (HANDOFF §5a level (b), Farquhar et al. 2024). Runs on the GPU
cluster via slurm/semantic_entropy.sbatch; needs torch + transformers, so not on vostok.

For each (pmid, criterion, provider) the original answer and its resamples
(entropy/s01..s05) are clustered greedily by bidirectional entailment: an answer joins a
cluster if it has the same verdict as the cluster's first member and the NLI model says
each entails the other. Discrete semantic entropy is the Shannon entropy (nats) of the
cluster sizes. The table goes to MinIO as OUTPUT_OBJECT; entropy.load_semantic_entropy
reads it on vostok.

Columns: sample_id, n (answers, original included), n_clusters, semantic_entropy,
orig_in_largest (the original's cluster is a largest one), largest_share,
n_truncated (answers longer than the NLI model's input).
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "weak_supervision")]

from entropy import PREFIXES  # noqa: E402
from samples import fetch_records, minio_client, record_rows  # noqa: E402

NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"  # the model Farquhar et al. used
OUTPUT_OBJECT = "entropy/semantic_entropy.parquet"
MAX_LEN = 512

Answer = Tuple[str, str]  # (verdict, reason)


def answer_text(criterion: str, verdict: str, reason: str) -> str:
    # The verdict is in the text so "blood is not a biomarker" under true and under false
    # never read as the same claim.
    return f"{criterion.replace('_', ' ').capitalize()}. Answer: {verdict}. Reason: {reason}"


def cluster(verdicts: Sequence[str], entails: Callable[[int, int], bool]) -> List[int]:
    """Greedy bidirectional-entailment clustering; returns a cluster id per answer.

    Each answer is compared with the first member of each existing cluster, in order.
    """
    reps: List[int] = []
    ids: List[int] = []
    for i, v in enumerate(verdicts):
        for c, r in enumerate(reps):
            if verdicts[r] == v and entails(i, r) and entails(r, i):
                ids.append(c)
                break
        else:
            reps.append(i)
            ids.append(len(reps) - 1)
    return ids


def summarize(ids: List[int]) -> Dict:
    """Entropy stats from cluster ids; ids[0] is the original answer."""
    sizes = Counter(ids)
    n = len(ids)
    largest = max(sizes.values())
    return {
        "n": n,
        "n_clusters": len(sizes),
        "semantic_entropy": -sum(s / n * math.log(s / n) for s in sizes.values()),
        "orig_in_largest": sizes[ids[0]] == largest,
        "largest_share": largest / n,
    }


class NLI:
    def __init__(self, model_name: str = NLI_MODEL, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_id = labels["entailment"]
        self.batch_size = batch_size

    def entailment(self, pairs: List[Tuple[str, str]]) -> List[bool]:
        """Argmax == entailment for each (premise, hypothesis)."""
        out: List[bool] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                           max_length=MAX_LEN, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                pred = self.model(**enc).logits.argmax(-1)
            out += (pred == self.entail_id).tolist()
        return out

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text)["input_ids"])


def group_answers(rows: List[Dict]) -> Dict[str, List[Tuple[str, str, str]]]:
    """sample_id -> [(source, verdict, reason)], original first; unparsed answers dropped."""
    groups: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for r in rows:
        if r["verdict"] is None or not r["reason"]:
            continue
        sid = f"{r['pmid']}|{r['criterion']}|{r['provider']}"
        groups[sid].append((r["source"], r["verdict"], r["reason"]))
    for answers in groups.values():
        answers.sort(key=lambda a: (a[0] != "original", a[0]))
    return {sid: a for sid, a in groups.items() if a[0][0] == "original" and len(a) >= 2}


def run(nli: NLI, groups: Dict[str, List[Tuple[str, str, str]]], log_every: int = 1000):
    import pandas as pd

    out = []
    for k, (sid, answers) in enumerate(groups.items(), 1):
        criterion = sid.split("|")[1]
        texts = [answer_text(criterion, v, r) for _, v, r in answers]
        verdicts = [v for _, v, _ in answers]
        # All same-verdict ordered pairs in one batch; the greedy pass then reads the cache.
        pairs = [(i, j) for i in range(len(texts)) for j in range(len(texts)) if i != j and verdicts[i] == verdicts[j]]
        cache = dict(zip(pairs, nli.entailment([(texts[i], texts[j]) for i, j in pairs]))) if pairs else {}
        ids = cluster(verdicts, lambda i, j: cache[(i, j)])
        stats = summarize(ids)
        stats["n_truncated"] = sum(nli.n_tokens(t) > MAX_LEN // 2 for t in texts)  # a pair holds two answers
        out.append({"sample_id": sid, **stats})
        if k % log_every == 0:
            print(f"{k}/{len(groups)} samples", flush=True)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=NLI_MODEL)
    parser.add_argument("--limit", type=int, help="Only the first N samples (smoke test).")
    parser.add_argument("--no-upload", action="store_true", help="Write locally only.")
    args = parser.parse_args()

    from config.app_config import load_app_config

    app_config = load_app_config()
    client = minio_client(app_config)
    bucket = app_config.minio.synthetic_data_bucket
    rows = []
    for source, prefix in [("original", ""), *[(p.split("/")[-1], p) for p in PREFIXES]]:
        records = fetch_records(client, bucket, prefix=prefix)
        print(f"{source}: {len(records)} records", flush=True)
        rows += [{**r, "source": source} for rec in records for r in record_rows(rec)]
    groups = group_answers(rows)
    if args.limit:
        groups = dict(list(groups.items())[: args.limit])
    print(f"{len(groups)} samples with >= 1 resample", flush=True)

    table = run(NLI(args.model), groups)
    local = ROOT / "weak_supervision" / "data" / "semantic_entropy.parquet"
    local.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(local, index=False)
    print(table.describe().to_string())
    if not args.no_upload:
        data = io.BytesIO()
        table.to_parquet(data, index=False)
        client.put_object(bucket, OUTPUT_OBJECT, io.BytesIO(data.getvalue()), len(data.getvalue()))
        print(f"uploaded {bucket}/{OUTPUT_OBJECT}")


if __name__ == "__main__":
    main()
