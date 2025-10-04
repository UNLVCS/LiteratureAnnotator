from typing import Iterable
from snorkel.labeling import LabelingFunction
from cross_ref import cross_ref, judge_me
from heuristics import heuristics
from human_scores import researchers_scores
from typing import Dict, Any, Optional, List
from llm_providers import (
    BaseLLMProvider,
    OpenAIProvider, 
    AnthropicProvider, 
    HuggingFaceProvider,
    Query,
    LLMResponse
)


# Values that a LabelFunction should spit out according to Snorkel
ABSTAIN = -1, INCORRECT = 0, CORRECT = 1

def Rater():
    @property
    def name(self) -> str: ...
    def score_pair(self, generated_label: str, justificiation: str, rubric: Dict[str, Any]) -> Dict[str, float]:
        # Call LLM adapters here 
        pass

def cross_ref_judge(model, rater: Rater, target: LLMResponse, rubric: Dict[str, Any]) -> int:
    """
        Returns CORRECT/INCORRECT/ABSTAIN for 'rater judges target' on row x.
            Assumes x has attributes: model_name, inference, justification.
        Note that this is the logic only for the cross_referncing label functions that use LLMs to judge each other's responses
        Logic for other label functions is encapsulated elsewhere
    """

    if model.get_provider_name == target.model:
        return ABSTAIN
    
    try:
        out = rater.score_pair(model.generated_label, model.justification, rubric)
    except:
        return ABSTAIN
    
    if not out:
        return ABSTAIN

    score = float(out.score, 0.0)
    conf = float(out.confidence, 0.0)


    min_conf = float(rubic_num(rubric, "min_confidence", 0.5))
    thresh   = float(rubic_num(rubric, "pass_threshold", 0.6))

    if conf < min_conf:
        return ABSTAIN
    return CORRECT if score >= thresh else INCORRECT

def rubic_num(rubric: Dict[str, Any], key: str, default: float) -> float:
    v = rubric.get(key, default)
    try:
        return float(v)
    except Exception:
        return default

def make_pair_lf(rater: Rater, target: LLMResponse, rubric: Dict[str, any]) -> LabelingFunction:
    """
        Creates one LF named 'lf_<rater>_judges_<target>' and injects dependencies
    """
    lf_name = f"lf_{rater.name}_judges_{target.model}"
    resources = {"rater": rater, "target": target, "rubric": dict(rubric)}
    return LabelingFunction(name = lf_name, f = cross_ref_judge, resources = resources)

def build_pairwise_lfs(raters: Iterable[any], rubric: Dict[str, Any]) -> List[LabelingFunction]:
    lfs: List[LabelingFunction] = []
    raters = list(raters)
    targets = list(targets)

    for r in raters:
        for t in targets:
            if r.name == t:
                continue
            lfs.append(make_pair_lf(r, t, rubric))

@labeling_function('CrossReferencer', cross_ref, {})
def cross_ref_lf():
    pass

@labeling_function('Heuristics', heuristics, {})
def heuristics_lf():
    pass

@labeling_function('ResearchersScores', researchers_scores, {})
def researcher_scores():
    pass