"""Tests for the sample table, label model, entropy and gold hook (synthetic data, no MinIO)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "weak_supervision")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from entropy import entropy_table, verdict_entropy  # noqa: E402
from gold import evaluate, load_gold  # noqa: E402
from label_model import LabelModel, fit_facets  # noqa: E402
from lfs import ABSTAIN, INVALID, VALID  # noqa: E402
from samples import blood_mentions, build_table, record_rows  # noqa: E402


def record(pmid, provider, verdicts, matched=True):
    results = []
    for i, v in enumerate(verdicts, 1):
        name = f"criterion_{i}"
        results.append({
            "criterion": name,
            "response": None if v is None else {name: {"satisfied": v, "reason": "r", "evidence": []}},
            "error": "Failed to parse JSON" if v is None else None,
            "evidence": [{"quote": "q", "cited_section": "A", "matched": matched, "section": "A", "section_agrees": True}],
            "finish_reason": "stop", "usage": {"output_tokens": 5},
        })
    return {"paper_id": pmid, "provider": provider, "paper": {"is_full_text": True}, "criteria_results": results,
            "errors": []}


MANIFEST = {
    "strata": [{"name": "core_positive", "expect": {"criterion_1": True}},
               {"name": "c1_review", "expect": {"criterion_1": False}}],
    "papers": {"1": {"sampled_from": "core_positive", "in_strata": ["core_positive"]},
               "2": {"sampled_from": "c1_review", "in_strata": ["c1_review", "core_positive"]}},
}


def test_record_rows_and_table():
    recs = [record("1", "claude-sonnet-5-5", [True, "not_applicable"]),
            record("1", "qwen3.8:27b", [False, None], matched=False),
            record("2", "claude-sonnet-5-5", [True, True])]
    rows = record_rows(recs[1])
    assert rows[0]["verdict"] == "false" and rows[0]["n_unmatched"] == 1
    assert rows[1]["verdict"] is None and rows[1]["error"] == "Failed to parse JSON"

    feats = {"1": {"blood_mentions": 0, "blood_sample_mentions": 0}, "2": {"blood_mentions": 1, "blood_sample_mentions": 1}}
    df = build_table(recs, MANIFEST, feats)
    assert len(df) == 6
    r = df[(df.pmid == "1") & (df.criterion == "criterion_1") & (df.provider == "claude-sonnet-5-5")].iloc[0]
    assert r["v_qwen"] == "false" and r["v_claude"] == "true" and pd.isna(r["v_luna"])
    assert r["prior"] == True  # noqa: E712
    # Paper 2 is in two strata with conflicting C1 expectations: no prior.
    assert df[(df.pmid == "2") & (df.criterion == "criterion_1")]["prior"].isna().all()
    assert r["sample_id"] == "1|criterion_1|claude-sonnet-5-5"


def test_blood_mentions():
    text = ("Plasma Aβ42 was measured in serum. The blood-brain barrier and blood pressure were not. "
            "Cells were grown in fetal bovine serum.")
    assert blood_mentions(text) == {"blood_mentions": 5, "blood_sample_mentions": 2}
    assert blood_mentions("No such words.") == {"blood_mentions": 0, "blood_sample_mentions": 0}


def _simulate(n=4000, seed=0):
    """Three verdict LFs with known accuracies over a 10% INVALID population."""
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.9, VALID, INVALID)
    cols = {}
    for name, acc, cov in [("a", 0.95, 1.0), ("b", 0.85, 0.6), ("c", 0.8, 0.5)]:
        correct = rng.random(n) < acc
        vote = np.where(correct, y, 1 - y)
        cols[name] = np.where(rng.random(n) < cov, vote, ABSTAIN)
    return y, pd.DataFrame(cols)


def test_label_model_recovers_accuracies():
    y, L = _simulate()
    m = LabelModel(prior_strength=1.0).fit(L.to_numpy())
    assert abs(m.prior_ - 0.9) < 0.03
    assert np.allclose(m.accuracy_, [0.95, 0.85, 0.8], atol=0.04)
    pred = np.where(m.predict_proba(L.to_numpy()) >= 0.5, VALID, INVALID)
    assert (pred == y).mean() > 0.93


def test_fit_facets_combines_and_hard_invalid():
    y, L = _simulate()
    L = L.rename(columns={"a": "agrees_with_majority", "b": "stratum_prior", "c": "pubmed_pubtype"})
    L["quote_match"] = VALID
    L.loc[:9, "quote_match"] = INVALID
    L["parse_error"] = ABSTAIN
    L.loc[10, "parse_error"] = INVALID
    scores, models = fit_facets(L, pd.Series("criterion_1", index=L.index))
    assert (scores.loc[:9, "p_evidence"] < 0.5).all()
    assert scores.loc[10, "p_valid"] == 0.0
    assert np.allclose(scores["p_valid"], (scores["p_verdict"] * scores["p_evidence"]).where(scores.index != 10, 0))
    assert models[("evidence", "all")][0].learn is False


def test_small_group_falls_back_to_fixed_parameters():
    # One unipolar LF: EM would drive the class prior to 0 and call everything INVALID.
    L = pd.DataFrame({"reason_verdict_mismatch": [INVALID] + [ABSTAIN] * 99})
    scores, models = fit_facets(L, pd.Series("criterion_3", index=L.index))
    assert (scores["p_verdict"].iloc[1:] > 0.9).all()
    assert models[("verdict", "criterion_3")][0].learn is False


def test_verdict_entropy():
    e = verdict_entropy(["true"] * 5)
    assert e["n"] == 5 and e["entropy"] == 0 and e["mode"] == "true"
    e = verdict_entropy(["true", "true", "false", None])
    assert e["n"] == 3 and abs(e["entropy"] - 0.6365) < 1e-3 and json.loads(e["counts"]) == {"true": 2, "false": 1}
    t = entropy_table([{"pmid": "1", "criterion": "criterion_1", "provider": "p", "verdict": v}
                       for v in ["true", "false", "true"]])
    assert t.iloc[0]["sample_id"] == "1|criterion_1|p" and t.iloc[0]["mode"] == "true"


def test_gold_hook(tmp_path):
    samples = pd.DataFrame({
        "sample_id": ["1|criterion_1|p", "2|criterion_1|p", "3|criterion_1|p"],
        "provider": "p", "criterion": "criterion_1",
    })
    L = pd.DataFrame({"lf": [VALID, INVALID, ABSTAIN]})
    scores = pd.DataFrame({"p_valid": [0.9, 0.2, 0.4]})
    path = tmp_path / "gold.csv"
    pd.DataFrame({"pmid": ["1", "2", "3", "9"], "criterion": ["1", "criterion_1", "1", "1"], "provider": "p",
                  "label": ["VALID", "invalid", "1", "0"]}).to_csv(path, index=False)
    gold = load_gold(path)
    res = evaluate(samples, L, scores, gold)
    lm = res["label_model"].iloc[0]
    assert lm["n"] == 3 and lm["gold_rows_without_sample"] == 1
    assert abs(lm["accuracy"] - 2 / 3) < 1e-9  # sample 3 is VALID in gold but scored 0.4
    assert res["lfs"].loc["lf", "accuracy"] == 1.0 and abs(res["lfs"].loc["lf", "coverage"] - 2 / 3) < 1e-9


def test_semantic_entropy_clustering():
    from semantic_entropy import cluster, group_answers, summarize

    # 0,1,2 mutually entail; 3 entails 0 one way only; 4 has another verdict.
    verdicts = ["true", "true", "true", "true", "false"]
    one_way = {(3, 0)}

    def entails(i, j):
        return (i < 3 and j < 3) or (i, j) in one_way

    ids = cluster(verdicts, entails)
    assert ids == [0, 0, 0, 1, 2]
    s = summarize(ids)
    assert s["n_clusters"] == 3 and s["orig_in_largest"] and abs(s["largest_share"] - 0.6) < 1e-9

    rows = [{"pmid": "1", "criterion": "criterion_1", "provider": "p", "verdict": v, "reason": r, "source": src}
            for src, v, r in [("s01", "true", "a"), ("original", "true", "b"), ("s02", None, None)]]
    rows.append({**rows[0], "pmid": "2"})  # no original: dropped
    g = group_answers(rows)
    assert list(g) == ["1|criterion_1|p"] and g["1|criterion_1|p"][0][0] == "original"
