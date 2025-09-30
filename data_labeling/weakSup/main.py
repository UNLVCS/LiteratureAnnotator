from snorkel.labeling import labeling_function
from cross_ref import cross_ref
from heuristics import heuristics
from human_scores import researchers_scores

@labeling_function('CrossReferencer', cross_ref, {})
def cross_ref_lf():
    pass

@labeling_function('Heuristics', heuristics, {})
def heuristics_lf():
    pass

@labeling_function('ResearchersScores', researchers_scores, {})
def researcher_scores():
    pass