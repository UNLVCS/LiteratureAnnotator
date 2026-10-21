"""
Labeling functions over sample rows (one LLM output for one (pmid, criterion, provider)).

Every LF votes on whether the *sample* is a trustworthy training example, not on
whether the paper satisfies the criterion:

    VALID = 1, INVALID = 0, ABSTAIN = -1   (Snorkel's convention)

An LF takes a row as a mapping (a ``samples.parquet`` row as a dict, plus ``pubmed``
metadata and ``entropy`` stats joined in by ``apply_lfs``) and returns one vote.
Thresholds and keyword lists are first guesses; tune them against the gold set.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

VALID, INVALID, ABSTAIN = 1, 0, -1
TRUE, FALSE, NA = "true", "false", "not_applicable"
OTHER_VERDICTS = ("v_claude", "v_luna", "v_gptoss", "v_qwen")
PROVIDER_COLUMN = {"claude-sonnet-5-5": "v_claude", "gpt-6-luna": "v_luna",
                   "openai/gpt-oss-120b": "v_gptoss", "qwen3.8:27b": "v_qwen"}


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# ---------------------------------------------------------------------------
# Output-format and citation LFs
# ---------------------------------------------------------------------------


def parse_error(row: Mapping) -> int:
    """INVALID if the answer failed to parse or validate."""
    if not _missing(row.get("error")) or not row.get("has_response") or _missing(row.get("verdict")):
        return INVALID
    return ABSTAIN


def quote_match(row: Mapping) -> int:
    """INVALID if any quote is not verbatim in the paper; VALID if >= 1 quote and all match.

    This is the handoff's ``unmatched_quote`` and ``quotes_all_matched`` merged into one
    LF: the two are mutually exclusive functions of the same evidence, and as separate
    LFs the label model would count that evidence twice. Empty evidence abstains,
    because the rules allow it for decisions based on absence.
    """
    if row["n_quotes"] == 0:
        return ABSTAIN
    return INVALID if row["n_unmatched"] > 0 else VALID


def section_mismatch(row: Mapping) -> int:
    """INVALID if a verbatim quote sits in a different section than the model cited."""
    return INVALID if row["n_section_mismatch"] > 0 else ABSTAIN


# ---------------------------------------------------------------------------
# Agreement and prior LFs
# ---------------------------------------------------------------------------


def agrees_with_majority(row: Mapping) -> int:
    """VALID/INVALID if a strict majority of the other providers' answers agree/disagree."""
    verdict = row.get("verdict")
    if _missing(verdict):
        return ABSTAIN
    own = PROVIDER_COLUMN.get(row["provider"])
    others = [row.get(c) for c in OTHER_VERDICTS if c != own and not _missing(row.get(c))]
    agree = sum(v == verdict for v in others)
    disagree = len(others) - agree
    if agree > len(others) / 2:
        return VALID
    if disagree > len(others) / 2:
        return INVALID
    return ABSTAIN


# C1 and C5 strata were selected by PubMed publication type / MeSH (review[pt],
# animals[MeSH]), which pubmed_pubtype and pubmed_mesh_animals read directly. Voting
# here too would count that evidence twice.
STRATUM_PRIOR_CRITERIA = frozenset({"criterion_2", "criterion_4", "criterion_6"})


def stratum_prior(row: Mapping) -> int:
    """Agreement with the sampling stratum's ``expect`` (weak: strata come from keyword search).

    ``prior=False`` accepts both false and not_applicable (e.g. the C6 csf/pet-only stratum,
    where blood may never be mentioned).
    """
    prior, verdict = row.get("prior"), row.get("verdict")
    if row["criterion"] not in STRATUM_PRIOR_CRITERIA or _missing(prior) or _missing(verdict):
        return ABSTAIN
    return VALID if (verdict == TRUE) == bool(prior) else INVALID


# ---------------------------------------------------------------------------
# PubMed metadata LFs (row["pubmed"] = {"pub_types": [...], "mesh": [...]} or None)
# ---------------------------------------------------------------------------

NON_ORIGINAL_TYPES = frozenset({
    "Review", "Systematic Review", "Editorial", "Comment", "Preprint", "News", "Newspaper Article",
    "Practice Guideline", "Guideline", "Consensus Statement", "Introductory Journal Article",
    "Published Erratum", "Biography", "Interview", "Lecture",
})
ORIGINAL_TYPES = frozenset({
    "Randomized Controlled Trial", "Clinical Trial", "Clinical Trial, Phase I", "Clinical Trial, Phase II",
    "Clinical Trial, Phase III", "Clinical Trial, Phase IV", "Controlled Clinical Trial", "Observational Study",
    "Multicenter Study", "Comparative Study", "Validation Study", "Evaluation Study", "Clinical Study",
})


def pubmed_pubtype(row: Mapping) -> int:
    """C1: INVALID if the verdict contradicts the PubMed publication type, VALID if it matches.

    Non-original types (review, editorial, comment, ...) decide both directions. A study
    type (trial, observational, ...) only supports true; a plain "Journal Article" abstains,
    since perspectives are often filed under it.
    """
    meta, verdict = row.get("pubmed"), row.get("verdict")
    if row["criterion"] != "criterion_1" or not meta or _missing(verdict):
        return ABSTAIN
    types = set(meta.get("pub_types") or [])
    non_original = bool(types & NON_ORIGINAL_TYPES)
    study = bool(types & ORIGINAL_TYPES)
    if non_original and not study:
        return INVALID if verdict == TRUE else VALID
    if study and not non_original:
        return VALID if verdict == TRUE else INVALID
    return ABSTAIN


def pubmed_mesh_animals(row: Mapping) -> int:
    """C5 (true = no animal models): check the verdict against the MeSH Animals/Humans tags.

    Animals without Humans contradicts true; Humans without Animals contradicts false.
    Both or neither tag abstains.
    """
    meta, verdict = row.get("pubmed"), row.get("verdict")
    if row["criterion"] != "criterion_5" or not meta or _missing(verdict):
        return ABSTAIN
    mesh = set(meta.get("mesh") or [])
    animals, humans = "Animals" in mesh, "Humans" in mesh
    if animals == humans:
        return ABSTAIN
    return VALID if (verdict == TRUE) == humans else INVALID


# ---------------------------------------------------------------------------
# Text heuristics
# ---------------------------------------------------------------------------


def na_consistency(row: Mapping) -> int:
    """C6: check not_applicable against blood/serum/plasma mentions in the paper text.

    The criterion says N/A only when blood, serum and plasma are never mentioned.
    - N/A with sample-like mentions (not blood pressure, blood-brain barrier, FBS, ...) -> INVALID
    - N/A with no mention at all -> VALID; true/false with no mention at all -> INVALID
    - N/A where every mention is a non-sample use: ABSTAIN. Whether that is N/A or false
      is the open SME question (HANDOFF §8).
    C3 has no text feature yet; its N/A checks live in reason_verdict_mismatch.
    """
    verdict = row.get("verdict")
    if row["criterion"] != "criterion_6" or _missing(verdict) or _missing(row.get("blood_mentions")):
        return ABSTAIN
    total, sample_like = row["blood_mentions"], row["blood_sample_mentions"]
    if verdict == NA:
        if total == 0:
            return VALID
        return INVALID if sample_like > 0 else ABSTAIN
    return INVALID if total == 0 else ABSTAIN


NEGATION = re.compile(r"\b(not|no|nothing|never|without|nor|neither|rather than|instead of|n't|absence of|lack of)\b", re.I)


def _asserts(text: str, pattern: str, window: int = 60) -> bool:
    """``pattern`` occurs somewhere in ``text`` without a negation just before it in its sentence."""
    for m in re.finditer(pattern, text, re.I):
        start = max(text.rfind(".", 0, m.start()) + 1, m.start() - window)
        if not NEGATION.search(text[start:m.start()]) and not NEGATION.search(m.group(0)):
            return True
    return False


def _mentions(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I) is not None


# Reason phrases that say the criterion is met / not met / does not apply.
SAYS_MET = r"\b(criterion (is )?(met|satisfied)|meets (this|the) criterion|satisfies (this|the) criterion)\b"
SAYS_NOT_MET = (r"\b((does not|doesn't|do not|fails to) (meet|satisfy|fulfil+) (this|the)( \w+)? criterion"
                r"|criterion is not (met|satisfied))\b")
SAYS_NOT_APPLICABLE = (r"\b(criterion|sample[- ]size requirement|it) (does not|doesn't) apply\b"
                       r"|\b(is|are|therefore|thus|so) not[ _]applicable\b")

ANIMAL = r"(mice|mouse|rats?|rodents?|animal models?|primates?|marmosets?|zebrafish|drosophila|monkeys?|dogs?|pigs?)"
HUMANS_ONLY = r"\b(human (data|subjects|participants|samples|tissue)? ?only|only human|exclusively (uses? )?human)"
NO_ANIMALS = r"\b(no animal|(not|n't) (use|involve|include|perform|conduct)s? (any )?animal)"
HUMAN_COUNT = re.compile(
    r"\b\d[\d,]*\s+(participants|patients|subjects|individuals|cases|controls|people|persons|adults|"
    r"women|men|volunteers|donors)\b", re.I)


def reason_verdict_mismatch(row: Mapping) -> int:
    """INVALID if the reason text contradicts the verdict (keyword rules, no LLM)."""
    verdict, reason, crit = row.get("verdict"), row.get("reason"), row["criterion"]
    if _missing(verdict) or not isinstance(reason, str) or not reason:
        return ABSTAIN
    r = reason

    contradicts = (
        (verdict == TRUE and _mentions(r, SAYS_NOT_MET))
        or (verdict == FALSE and _asserts(r, SAYS_MET))
        or (verdict in (TRUE, FALSE) and crit in ("criterion_3", "criterion_6") and _asserts(r, SAYS_NOT_APPLICABLE))
    )
    if crit == "criterion_1":
        non_original = r"\bis an? (narrative |systematic |scoping |literature |mini[- ])?(review|editorial|commentary|perspective|preprint|poster)\b"
        contradicts |= verdict == TRUE and _asserts(r, non_original)
        contradicts |= verdict == FALSE and _asserts(r, r"\bis an? original research (article|study|paper)\b")
    elif crit == "criterion_2":
        contradicts |= verdict == TRUE and _mentions(r, r"\b(is|was) not (the )?(main|primary|central) focus|\bnot focused on (AD|Alzheimer)")
        contradicts |= verdict == FALSE and _asserts(r, r"\b(AD|Alzheimer'?s?( disease)?) is the (main|primary|central) focus")
    elif crit == "criterion_3":
        # N/A means no human participants: a reason quoting participant counts contradicts it.
        # Counts of animals, or of other studies' cohorts cited by a review, are not participants.
        cites_others = r"\b(review\w*|meta-analys\w*|summar\w*|commentary|editorial|other stud\w+|prior|previous\w*)\b"
        contradicts |= (verdict == NA and HUMAN_COUNT.search(r) is not None
                        and not _mentions(r, rf"{cites_others}|\b{ANIMAL}\b"))
        contradicts |= verdict in (TRUE, FALSE) and _mentions(r, r"\bno human (participants|subjects)|does not (include|involve|report) (any )?human")
    elif crit == "criterion_5":
        # "uses ... animal models" but not "as an alternative to" them or an animal-derived cell line.
        uses_animals = _asserts(r, rf"\b(uses?|used|utiliz\w+|employ\w*)\b(?![^.;]{{0,40}}alternative)[^.;]{{0,40}}"
                                   rf"\b{ANIMAL}\b(?! (cell|neuron|hepatocyte|embryonic|primary))")
        humans_only = _asserts(r, HUMANS_ONLY) or _mentions(r, NO_ANIMALS)
        contradicts |= verdict == TRUE and uses_animals and not humans_only
        contradicts |= verdict == FALSE and humans_only and not uses_animals
    elif crit == "criterion_6":
        # Only claims about all three words; "'blood' never appears" alone is common and fine.
        none_mentioned = (r"\b(blood|serum|plasma),? (serum|plasma),? (and|or|nor) (serum|plasma) (are|were) "
                          r"(never|not) (mentioned|appear)")
        not_biomarker = r"\bnot (used |measured |analy[sz]ed )?as an? (AD |Alzheimer'?s? )?biomarker"
        contradicts |= verdict == TRUE and (_mentions(r, not_biomarker) or _mentions(r, none_mentioned))
        contradicts |= verdict == FALSE and _mentions(r, none_mentioned)
    return INVALID if contradicts else ABSTAIN


# ---------------------------------------------------------------------------
# Semantic entropy over resamples (row["entropy"] from entropy.verdict_entropy)
# ---------------------------------------------------------------------------

# Placeholders until the gold set exists. With N=5 resamples: 5-0 split H=0, 4-1 H=0.50,
# 3-2 H=0.67, 3-1-1 H=0.95, 2-2-1 H=1.05 (nats).
ENTROPY_LOW = 0.51
ENTROPY_HIGH = 0.90
MIN_RESAMPLES = 3


def semantic_entropy(row: Mapping) -> int:
    """Verdict entropy of the same model's resamples (level (a) of HANDOFF §5a).

    VALID if the resamples are consistent (H <= ENTROPY_LOW) and their mode equals this
    sample's verdict; INVALID if they scatter (H >= ENTROPY_HIGH) or a confident mode
    disagrees with this sample; ABSTAIN in between or without enough resamples.
    """
    ent, verdict = row.get("entropy"), row.get("verdict")
    if not ent or _missing(verdict) or ent["n"] < MIN_RESAMPLES:
        return ABSTAIN
    if ent["entropy"] >= ENTROPY_HIGH:
        return INVALID
    if ent["entropy"] <= ENTROPY_LOW:
        return VALID if ent["mode"] == verdict else INVALID
    return ABSTAIN


# Reason-level semantic entropy (row["sem_entropy"] from semantic_entropy.py on the cluster).
# Over up to 6 answers (original + 5): 5-1 split H=0.45, 4-2 0.64, 3-3 0.69, 4-1-1 0.87,
# 3-2-1 1.01. Placeholders until the gold set exists.
SEM_ENTROPY_LOW = 0.5
SEM_ENTROPY_HIGH = 1.0
SEM_MIN_ANSWERS = 4


def semantic_entropy_reasons(row: Mapping) -> int:
    """VALID if the answers mostly mean the same thing and the original is in the largest
    meaning cluster; INVALID if they scatter, or a clear majority means something else."""
    ent = row.get("sem_entropy")
    if not ent or ent["n"] < SEM_MIN_ANSWERS:
        return ABSTAIN
    h = ent["semantic_entropy"]
    if h >= SEM_ENTROPY_HIGH or (not ent["orig_in_largest"] and ent["largest_share"] >= 2 / 3):
        return INVALID
    if h <= SEM_ENTROPY_LOW and ent["orig_in_largest"]:
        return VALID
    return ABSTAIN


# ---------------------------------------------------------------------------
# Applying LFs
# ---------------------------------------------------------------------------

LFS: List[Callable[[Mapping], int]] = [
    parse_error,
    quote_match,
    section_mismatch,
    agrees_with_majority,
    stratum_prior,
    pubmed_pubtype,
    pubmed_mesh_animals,
    na_consistency,
    reason_verdict_mismatch,
    semantic_entropy,
    semantic_entropy_reasons,
]


def apply_lfs(df: pd.DataFrame, lfs: List[Callable] = LFS,
              pubmed: Optional[Dict[str, Dict]] = None,
              entropy: Optional[Dict[str, Dict]] = None,
              sem_entropy: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """LF matrix (rows aligned with ``df``, one column per LF) of VALID/INVALID/ABSTAIN."""
    pubmed, entropy, sem_entropy = pubmed or {}, entropy or {}, sem_entropy or {}
    out = np.full((len(df), len(lfs)), ABSTAIN, dtype=np.int8)
    for i, row in enumerate(df.to_dict("records")):
        row["pubmed"] = pubmed.get(row["pmid"])
        row["entropy"] = entropy.get(row["sample_id"])
        row["sem_entropy"] = sem_entropy.get(row["sample_id"])
        for j, lf in enumerate(lfs):
            out[i, j] = lf(row)
    return pd.DataFrame(out, columns=[lf.__name__ for lf in lfs], index=df.index)


def lf_summary(L: pd.DataFrame) -> pd.DataFrame:
    """Snorkel LFAnalysis-style table: polarity, coverage, overlaps, conflicts per LF."""
    M = L.to_numpy()
    voted = M != ABSTAIN
    n_voting = voted.sum(axis=1)
    rows = []
    for j, name in enumerate(L.columns):
        col, v = M[:, j], voted[:, j]
        others = np.delete(M, j, axis=1)
        conflict = v & ((others != ABSTAIN) & (others != col[:, None])).any(axis=1)
        rows.append({
            "lf": name,
            "polarity": sorted({int(x) for x in col[v]}),
            "coverage": v.mean(),
            "overlaps": (v & (n_voting > 1)).mean(),
            "conflicts": conflict.mean(),
            "n_valid": int((col == VALID).sum()),
            "n_invalid": int((col == INVALID).sum()),
        })
    return pd.DataFrame(rows).set_index("lf")
