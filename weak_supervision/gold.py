"""
Gold-set hook: score LFs and the label model against SME VALID/INVALID judgements.

    weak_supervision/ws.sh run.py --gold path/to/gold.csv

The gold file (CSV, JSON lines, or parquet) needs columns ``pmid, criterion, provider,
label``. ``label`` may be VALID/INVALID, valid/invalid, 1/0 or true/false; ``criterion``
may be "criterion_3" or just 3; ``provider`` is the object prefix (e.g. "qwen3.8:27b").
Rows that match no sample are reported and dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from lfs import ABSTAIN, INVALID, VALID

LABELS = {"valid": VALID, "invalid": INVALID, "1": VALID, "0": INVALID, "true": VALID, "false": INVALID}


def load_gold(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        df = pd.read_csv(path, dtype=str)
    missing = {"pmid", "criterion", "provider", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df.astype({"pmid": str, "criterion": str, "provider": str})
    df["criterion"] = df["criterion"].map(lambda c: c if c.startswith("criterion_") else f"criterion_{c}")
    labels = df["label"].astype(str).str.strip().str.lower().map(LABELS)
    if labels.isna().any():
        bad = df.loc[labels.isna(), "label"].unique()[:5]
        raise ValueError(f"{path}: unrecognised labels {list(bad)}")
    df["gold"] = labels.astype(int)
    df["sample_id"] = df["pmid"] + "|" + df["criterion"] + "|" + df["provider"]
    return df[["sample_id", "gold"]].drop_duplicates("sample_id", keep="last")


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Accuracy plus precision/recall/F1 for INVALID, the class we act on (drop)."""
    tp = int(((pred == INVALID) & (y == INVALID)).sum())
    fp = int(((pred == INVALID) & (y == VALID)).sum())
    fn = int(((pred == VALID) & (y == INVALID)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"n": len(y), "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
            "invalid_precision": prec, "invalid_recall": rec, "invalid_f1": f1}


def calibration(y: np.ndarray, p_valid: np.ndarray, bins: int = 10) -> Dict[str, float]:
    """Brier score and expected calibration error of P(VALID)."""
    y = (y == VALID).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p_valid, edges) - 1, 0, bins - 1)
    ece = sum(abs(p_valid[which == b].mean() - y[which == b].mean()) * (which == b).mean()
              for b in range(bins) if (which == b).any())
    return {"brier": float(((p_valid - y) ** 2).mean()), "ece": float(ece)}


def evaluate(samples: pd.DataFrame, L: pd.DataFrame, scores: pd.DataFrame, gold: pd.DataFrame,
             threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
    """Per-LF and label-model metrics on the gold samples, overall and per provider/criterion.

    ``samples``, ``L`` and ``scores`` share an index; ``samples`` has ``sample_id``.
    """
    joined = samples[["sample_id", "provider", "criterion"]].join(L).join(scores).merge(gold, on="sample_id")
    unmatched = len(gold) - len(joined)
    y = joined["gold"].to_numpy()

    lf_rows = []
    for lf in L.columns:
        votes = joined[lf].to_numpy()
        voted = votes != ABSTAIN
        lf_rows.append({"lf": lf, "coverage": voted.mean() if len(votes) else float("nan"),
                        **binary_metrics(y[voted], votes[voted])})
    lf_table = pd.DataFrame(lf_rows).set_index("lf")

    pred = np.where(joined["p_valid"] >= threshold, VALID, INVALID)
    overall = {**binary_metrics(y, pred), **calibration(y, joined["p_valid"].to_numpy()),
               "gold_rows_without_sample": unmatched}
    by = {}
    for key in ("provider", "criterion"):
        by[key] = pd.DataFrame({g: binary_metrics(y[idx], pred[idx])
                                for g, idx in joined.groupby(key).indices.items()}).T
    return {"lfs": lf_table, "label_model": pd.DataFrame([overall]), **{f"by_{k}": v for k, v in by.items()}}
