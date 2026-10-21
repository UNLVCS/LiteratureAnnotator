"""
Label model: P(VALID) per sample from the LF matrix, in numpy (replaces Snorkel's torch LabelModel).

A sample can be invalid in two unrelated ways, and the LFs split along them:
- **verdict**: the answer is wrong (agreement, stratum prior, PubMed, text heuristics,
  entropy). These LFs' INVALID votes co-occur 2-28x more than chance.
- **evidence**: the citations are unfaithful (quote_match, section_mismatch). Their INVALID
  votes are independent of the verdict LFs' (lift ~1).
A single latent "valid" class cannot explain two independent failure modes: fitted
jointly, EM concluded that every INVALID vote was noise (class prior -> 0.999). So each
facet gets its own model and

    P(valid) = P(verdict ok) * P(evidence ok),   and 0 if parse_error fires (no answer at all).

Each facet model is a two-class, one-accuracy-per-LF ("one-coin" Dawid-Skene) model:

This replaces Snorkel's torch LabelModel. The model:

    P(y = VALID) = prior
    P(LF j votes y | LF j votes) = a_j,   P(LF j votes the other class | votes) = 1 - a_j
    whether LF j votes at all does not depend on y

LFs are assumed conditionally independent given y within a facet (see the HANDOFF caution;
the two quote LFs were merged in lfs.py). A unipolar LF (one that only ever says INVALID)
still counts: each of its votes moves the posterior by logit(a_j). The verdict facet is
fitted per criterion: which LFs can vote depends on the criterion, and agreement is far
less reliable on C6 than on C1. The evidence facet has only two LFs, too few to identify
EM's parameters, so it uses fixed accuracies (FIXED_FACETS) until the gold set measures them.

Why not the full Dawid-Skene with abstain as a third output? It was tried first. Which
LFs can vote depends on the verdict and the criterion (na_consistency fires mostly on
C6 not_applicable answers, pubmed_pubtype only on C1). EM then spent the latent class on
those coverage patterns: on C6 the "INVALID" class became "answered not_applicable",
and unmatched-quote samples came out *less* often INVALID than matched ones.

EM starts from the LF majority vote, which fixes which latent class is VALID.

Each accuracy gets a Beta prior (mean ``prior_accuracy``, ``prior_strength`` pseudo-votes),
so an LF that rarely overlaps the others is not pushed to accuracy 0 or 1 by a handful of
votes. Priors are modelling assumptions, not data; the gold set should replace them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from lfs import ABSTAIN, INVALID, VALID

EPS = 1e-4

FACETS = {
    "verdict": ["agrees_with_majority", "stratum_prior", "pubmed_pubtype", "pubmed_mesh_animals",
                "na_consistency", "reason_verdict_mismatch", "semantic_entropy", "semantic_entropy_reasons"],
    "evidence": ["quote_match", "section_mismatch"],
}
HARD_INVALID = ["parse_error"]  # any INVALID vote sets P(valid) = 0
# Facets fitted without EM: {facet: P(ok)}; the LF accuracies come from PRIOR_ACCURACY.
# An unmatched quote outweighs the 5% base rate (the handoff's strongest per-sample signal);
# a section mismatch alone does not outweigh a fully matched quote set.
FIXED_FACETS = {"evidence": 0.95}
# EM needs overlapping LFs that vote both ways; a group with fewer than 2 active LFs, or
# with no LF that ever votes VALID, is scored with the prior accuracies and this class
# prior instead (left alone, EM sends its class prior to 0 or 1). With exactly 2 LFs EM
# runs, but the accuracies lean on their Beta priors; 3+ (e.g. once semantic_entropy has
# resamples) makes it identifiable.
DEFAULT_CLASS_PRIOR = 0.95
MIN_LFS_FOR_EM = 2

# Prior mean accuracy per LF (DEFAULT_PRIOR_ACCURACY otherwise).
PRIOR_ACCURACY = {"quote_match": 0.98, "section_mismatch": 0.7}
DEFAULT_PRIOR_ACCURACY = 0.75


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def majority_vote(L: np.ndarray, tie: float = 0.5) -> np.ndarray:
    """P(VALID) from the plain LF vote count; ``tie`` where votes are tied or absent."""
    v = (L == VALID).sum(axis=1)
    i = (L == INVALID).sum(axis=1)
    return np.where(v > i, 1.0, np.where(i > v, 0.0, tie))


@dataclass
class LabelModel:
    max_iter: int = 500
    tol: float = 1e-8
    prior_accuracy: Optional[Sequence[float]] = None  # per LF; None = DEFAULT_PRIOR_ACCURACY
    prior_strength: float = 20.0  # pseudo-votes behind each prior accuracy
    class_prior: Optional[float] = None  # fixed P(VALID); None = learn it
    learn: bool = True  # False: use prior_accuracy and class_prior as they are (no EM)

    def fit(self, L: np.ndarray) -> "LabelModel":
        L = np.asarray(L)
        a0 = np.full(L.shape[1], DEFAULT_PRIOR_ACCURACY) if self.prior_accuracy is None else np.asarray(self.prior_accuracy)
        self._pseudo_ok, self._pseudo_n = self.prior_strength * a0, self.prior_strength
        if not self.learn:
            if self.class_prior is None:
                raise ValueError("learn=False needs a class_prior")
            self.accuracy_, self.prior_ = a0.astype(float), self.class_prior
            self.n_votes_ = (L != ABSTAIN).sum(axis=0)
            self.log_likelihood_ = self._e_step(L)[1]
            return self
        mv = majority_vote(L)
        # Ties and rows without votes start at the majority class.
        q = np.where(mv == 0.5, float((mv == 1).sum() >= (mv == 0).sum()), mv)
        prev = -np.inf
        for _ in range(self.max_iter):
            self._m_step(L, q)
            q, ll = self._e_step(L)
            if ll - prev < self.tol:
                break
            prev = ll
        self.log_likelihood_ = ll
        return self

    def _m_step(self, L: np.ndarray, q: np.ndarray) -> None:
        correct = ((L == VALID) * q[:, None] + (L == INVALID) * (1 - q[:, None])).sum(axis=0)
        n_votes = (L != ABSTAIN).sum(axis=0)
        self.accuracy_ = (correct + self._pseudo_ok) / (n_votes + self._pseudo_n)
        self.n_votes_ = n_votes
        self.prior_ = self.class_prior if self.class_prior is not None else float(np.clip(q.mean(), EPS, 1 - EPS))

    def _log_odds(self, L: np.ndarray) -> np.ndarray:
        w = _logit(self.accuracy_)
        return _logit(self.prior_) + ((L == VALID) * w).sum(axis=1) - ((L == INVALID) * w).sum(axis=1)

    def _e_step(self, L: np.ndarray):
        z = self._log_odds(L)
        q = 1 / (1 + np.exp(-z))
        # Marginal log-likelihood: log sum_y P(y) prod_j P(lambda_j | y).
        w_ok, w_bad = np.log(self.accuracy_), np.log1p(-self.accuracy_)
        agree_v = ((L == VALID) * w_ok + (L == INVALID) * w_bad).sum(axis=1) + np.log(self.prior_)
        agree_i = ((L == INVALID) * w_ok + (L == VALID) * w_bad).sum(axis=1) + np.log1p(-self.prior_)
        return q, float(np.logaddexp(agree_v, agree_i).sum())

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        """P(VALID | LF votes) per row."""
        return self._e_step(np.asarray(L))[0]

    def lf_params(self, names) -> pd.DataFrame:
        return pd.DataFrame({"accuracy": self.accuracy_, "weight": _logit(self.accuracy_), "n_votes": self.n_votes_},
                            index=list(names))


def fit_grouped(L: pd.DataFrame, groups: Optional[pd.Series] = None,
                prior_accuracy: Mapping[str, float] = PRIOR_ACCURACY, **kwargs):
    """Fit one model per group (e.g. criterion); LFs that never vote in a group are dropped.

    Returns (p_valid aligned with L.index, {group: (model, lf_names)}).
    """
    if groups is None:
        groups = pd.Series("all", index=L.index)
    p_valid = pd.Series(np.nan, index=L.index)
    models: Dict[str, tuple] = {}
    for g, idx in groups.groupby(groups).groups.items():
        sub = L.loc[idx]
        active = [c for c in sub.columns if (sub[c] != ABSTAIN).any()]
        a0 = [prior_accuracy.get(c, DEFAULT_PRIOR_ACCURACY) for c in active]
        M = sub[active].to_numpy()
        if kwargs.get("learn", True) and (len(active) < MIN_LFS_FOR_EM or not (M == VALID).any()):
            kwargs_g = {**kwargs, "learn": False, "class_prior": kwargs.get("class_prior") or DEFAULT_CLASS_PRIOR}
        else:
            kwargs_g = kwargs
        model = LabelModel(prior_accuracy=a0, **kwargs_g).fit(M)
        p_valid.loc[idx] = model.predict_proba(sub[active].to_numpy())
        models[g] = (model, active)
    return p_valid, models


def fit_facets(L: pd.DataFrame, groups: Optional[pd.Series] = None,
               facets: Mapping[str, Sequence[str]] = FACETS,
               fixed: Mapping[str, float] = FIXED_FACETS,
               hard_invalid: Sequence[str] = HARD_INVALID, **kwargs):
    """P(ok) per facet and the combined P(valid), one column each, aligned with L.index.

    Only LFs present in ``L`` are used, so a leave-one-out refit just drops a column.
    Facets in ``fixed`` are not fitted; groups too small for EM fall back to fixed
    parameters (see MIN_LFS_FOR_EM). Returns (scores DataFrame, {(facet, group): (model, lf_names)}).
    """
    scores = pd.DataFrame(index=L.index)
    models: Dict[tuple, tuple] = {}
    for facet, names in facets.items():
        cols = [c for c in names if c in L.columns]
        if not cols:
            scores[f"p_{facet}"] = 1.0
            continue
        if facet in fixed:
            p, ms = fit_grouped(L[cols], None, learn=False, class_prior=fixed[facet], **kwargs)
        else:
            p, ms = fit_grouped(L[cols], groups, **kwargs)
        scores[f"p_{facet}"] = p
        models.update({(facet, g): m for g, m in ms.items()})
    hard = [c for c in hard_invalid if c in L.columns]
    no_answer = (L[hard] == INVALID).any(axis=1) if hard else pd.Series(False, index=L.index)
    scores["p_valid"] = scores[[f"p_{f}" for f in facets]].prod(axis=1).where(~no_answer, 0.0)
    return scores, models


def lf_params_table(models: Dict) -> pd.DataFrame:
    """Learned LF parameters for every (facet, group), stacked as index levels."""
    frames = {k: m.lf_params(names).assign(class_prior=m.prior_, fitted=m.learn) for k, (m, names) in models.items()}
    return pd.concat(frames, names=["facet", "group", "lf"] if frames and isinstance(next(iter(frames)), tuple)
                     else ["group", "lf"])
