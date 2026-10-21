"""Unit tests for the labeling functions (pure functions over a row dict)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "weak_supervision")]

import pandas as pd  # noqa: E402

from lfs import (  # noqa: E402
    ABSTAIN, INVALID, VALID, agrees_with_majority, apply_lfs, lf_summary, na_consistency, parse_error,
    pubmed_mesh_animals, pubmed_pubtype, quote_match, reason_verdict_mismatch, section_mismatch,
    semantic_entropy, stratum_prior,
)


def row(**kw):
    base = {
        "pmid": "1", "criterion": "criterion_1", "provider": "claude-sonnet-5-5", "sample_id": "1|criterion_1|c",
        "verdict": "true", "reason": "Original study.", "error": None, "has_response": True,
        "n_quotes": 1, "n_matched": 1, "n_unmatched": 0, "n_section_mismatch": 0,
        "v_claude": "true", "v_luna": "true", "v_gptoss": "true", "v_qwen": "true",
        "prior": None, "blood_mentions": 0, "blood_sample_mentions": 0, "pubmed": None, "entropy": None,
    }
    base.update(kw)
    return base


def test_parse_error():
    assert parse_error(row()) == ABSTAIN
    assert parse_error(row(error="Failed to parse JSON", verdict=None)) == INVALID
    assert parse_error(row(has_response=False)) == INVALID
    assert parse_error(row(verdict=float("nan"))) == INVALID  # parquet round-trip of None


def test_quote_match():
    assert quote_match(row()) == VALID
    assert quote_match(row(n_quotes=3, n_matched=2, n_unmatched=1)) == INVALID
    # Empty evidence is allowed for absence decisions.
    assert quote_match(row(n_quotes=0, n_matched=0)) == ABSTAIN


def test_section_mismatch():
    assert section_mismatch(row()) == ABSTAIN
    assert section_mismatch(row(n_section_mismatch=1)) == INVALID


def test_agrees_with_majority_excludes_own_provider():
    assert agrees_with_majority(row()) == VALID
    # Own column (v_claude) disagrees with the verdict but must not count.
    assert agrees_with_majority(row(verdict="false", v_claude="false")) == INVALID
    assert agrees_with_majority(row(verdict="false", v_claude="false", v_luna="false")) == INVALID
    assert agrees_with_majority(row(verdict="false", v_claude="false", v_luna="false", v_gptoss="false")) == VALID


def test_agrees_with_majority_ties_and_missing():
    # Two others available, split 1-1: tie.
    assert agrees_with_majority(row(v_luna="false", v_gptoss=None)) == ABSTAIN
    # Three-way split on a conditional criterion: 1 agrees, 2 disagree (with each other too).
    r = row(criterion="criterion_6", verdict="true", v_luna="true", v_gptoss="false", v_qwen="not_applicable")
    assert agrees_with_majority(r) == INVALID
    assert agrees_with_majority(row(verdict=None)) == ABSTAIN


def test_stratum_prior():
    assert stratum_prior(row(criterion="criterion_6", prior=True)) == VALID
    assert stratum_prior(row(criterion="criterion_6", prior=True, verdict="not_applicable")) == INVALID
    assert stratum_prior(row(criterion="criterion_6", prior=False, verdict="not_applicable")) == VALID
    assert stratum_prior(row(criterion="criterion_6", prior=None)) == ABSTAIN
    # C1/C5 strata are PubMed-selected; the PubMed LFs vote there instead.
    assert stratum_prior(row(criterion="criterion_1", prior=False)) == ABSTAIN


def test_pubmed_pubtype():
    review = {"pub_types": ["Journal Article", "Review"], "mesh": []}
    trial = {"pub_types": ["Journal Article", "Randomized Controlled Trial"], "mesh": []}
    plain = {"pub_types": ["Journal Article"], "mesh": []}
    assert pubmed_pubtype(row(pubmed=review)) == INVALID
    assert pubmed_pubtype(row(pubmed=review, verdict="false")) == VALID
    assert pubmed_pubtype(row(pubmed=trial)) == VALID
    assert pubmed_pubtype(row(pubmed=trial, verdict="false")) == INVALID
    assert pubmed_pubtype(row(pubmed=plain, verdict="false")) == ABSTAIN
    assert pubmed_pubtype(row(pubmed=review, criterion="criterion_2")) == ABSTAIN
    assert pubmed_pubtype(row(pubmed=None)) == ABSTAIN


def test_pubmed_mesh_animals():
    c5 = {"criterion": "criterion_5"}
    animals = {"pub_types": [], "mesh": ["Animals", "Mice"]}
    humans = {"pub_types": [], "mesh": ["Humans"]}
    both = {"pub_types": [], "mesh": ["Animals", "Humans"]}
    assert pubmed_mesh_animals(row(**c5, pubmed=animals)) == INVALID
    assert pubmed_mesh_animals(row(**c5, pubmed=animals, verdict="false")) == VALID
    assert pubmed_mesh_animals(row(**c5, pubmed=humans)) == VALID
    assert pubmed_mesh_animals(row(**c5, pubmed=humans, verdict="false")) == INVALID
    assert pubmed_mesh_animals(row(**c5, pubmed=both)) == ABSTAIN


def test_na_consistency_c6():
    c6 = {"criterion": "criterion_6"}
    assert na_consistency(row(**c6, verdict="not_applicable")) == VALID
    assert na_consistency(row(**c6, verdict="not_applicable", blood_mentions=5, blood_sample_mentions=3)) == INVALID
    # Only blood-brain-barrier style mentions: the open SME question, abstain.
    assert na_consistency(row(**c6, verdict="not_applicable", blood_mentions=2, blood_sample_mentions=0)) == ABSTAIN
    assert na_consistency(row(**c6, verdict="false")) == INVALID  # never mentioned -> should be N/A
    assert na_consistency(row(**c6, verdict="false", blood_mentions=4, blood_sample_mentions=4)) == ABSTAIN
    assert na_consistency(row(verdict="not_applicable")) == ABSTAIN  # not C6


def test_reason_verdict_mismatch_generic_and_c1():
    assert reason_verdict_mismatch(row(reason="It is original research, so the criterion is met.")) == ABSTAIN
    assert reason_verdict_mismatch(row(verdict="false", reason="Data and stats, so the criterion is met.")) == INVALID
    assert reason_verdict_mismatch(row(reason="The paper is a narrative review of plasma markers.")) == INVALID
    # Negated or hedged mentions do not fire.
    assert reason_verdict_mismatch(row(reason="It is not a review; nothing marks it as a review.")) == ABSTAIN


def test_reason_verdict_mismatch_c3_c5_c6():
    c3 = {"criterion": "criterion_3"}
    assert reason_verdict_mismatch(row(**c3, verdict="not_applicable",
                                       reason="The study enrolled 120 patients with MCI.")) == INVALID
    assert reason_verdict_mismatch(row(**c3, verdict="not_applicable",
                                       reason="Mouse study, 12 mice per group; no human participants.")) == ABSTAIN
    assert reason_verdict_mismatch(row(**c3, verdict="false",
                                       reason="No human participants, so the criterion does not apply.")) == INVALID
    c5 = {"criterion": "criterion_5"}
    assert reason_verdict_mismatch(row(**c5, reason="The study uses APP/PS1 transgenic mice.")) == INVALID
    assert reason_verdict_mismatch(row(**c5, reason="It does not use animal models; human data only.")) == ABSTAIN
    assert reason_verdict_mismatch(row(**c5, verdict="false", reason="Human data only, no animal models.")) == INVALID
    assert reason_verdict_mismatch(row(**c5, reason="Uses a PC-12 rat cell line in vitro.")) == ABSTAIN
    c6 = {"criterion": "criterion_6"}
    assert reason_verdict_mismatch(row(**c6, reason="Plasma is not used as an AD biomarker.")) == INVALID
    assert reason_verdict_mismatch(row(**c6, verdict="false",
                                       reason="Blood, serum and plasma are never mentioned.")) == INVALID
    assert reason_verdict_mismatch(row(**c6, verdict="false",
                                       reason="The word 'blood' never appears; serum only as FBS.")) == ABSTAIN


def test_semantic_entropy():
    def ent(h, mode, n=5):
        return {"n": n, "entropy": h, "mode": mode}

    assert semantic_entropy(row()) == ABSTAIN  # no resamples yet
    assert semantic_entropy(row(entropy=ent(0.0, "true"))) == VALID
    assert semantic_entropy(row(entropy=ent(0.0, "false"))) == INVALID  # confident mode disagrees
    assert semantic_entropy(row(entropy=ent(0.67, "true"))) == ABSTAIN  # 3-2 split
    assert semantic_entropy(row(entropy=ent(1.05, "true"))) == INVALID  # scattered
    assert semantic_entropy(row(entropy=ent(0.0, "true", n=2))) == ABSTAIN  # too few


def test_apply_lfs_and_summary():
    df = pd.DataFrame([row(), row(pmid="2", sample_id="2|criterion_1|c", n_unmatched=1, verdict="false")])
    df = df.drop(columns=["pubmed", "entropy"])
    L = apply_lfs(df, pubmed={"1": {"pub_types": ["Review"], "mesh": []}})
    assert L.loc[0, "pubmed_pubtype"] == INVALID and L.loc[1, "pubmed_pubtype"] == ABSTAIN
    assert L.loc[1, "quote_match"] == INVALID
    s = lf_summary(L)
    assert s.loc["quote_match", "coverage"] == 1.0
    assert s.loc["semantic_entropy", "coverage"] == 0.0


def test_semantic_entropy_reasons():
    from lfs import semantic_entropy_reasons as lf

    def ent(h, orig, share, n=6):
        return {"n": n, "semantic_entropy": h, "orig_in_largest": orig, "largest_share": share}

    assert lf(row()) == ABSTAIN
    assert lf(row(sem_entropy=ent(0.0, True, 1.0))) == VALID
    assert lf(row(sem_entropy=ent(0.45, False, 5 / 6))) == INVALID  # original is the odd one out
    assert lf(row(sem_entropy=ent(0.69, True, 0.5))) == ABSTAIN
    assert lf(row(sem_entropy=ent(1.01, True, 0.5))) == INVALID
    assert lf(row(sem_entropy=ent(0.0, True, 1.0, n=3))) == ABSTAIN
