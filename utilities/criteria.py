"""
Shared criteria prompts for model labeling and human review.
"""

# Criterion instructions shown to models (with OUTPUT_FORMAT appended) and to human labelers.
CRITERIA_INSTRUCTIONS = [
    # 1) Original research
    """Criterion 1 – Original Research
    Decide if the paper is an original research article (not a review, perspective, poster, or preprint).
    - Positive signals: data collection + statistical analysis (often in Methods).
    - Negative signals: clear mentions of review, perspective, poster, preprint.""",

    # 2) AD focus
    """Criterion 2 – AD Focus
    Decide if Alzheimer's Disease (AD) is the main focus (diagnosis, treatment, biomarkers, pathology; AD patients incl. MCI/at risk).
    - Include AD biomarkers: amyloid-beta, tau.
    - Exclude if focus is general neurodegeneration markers without AD specificity.""",

    # 3) Sample size >= 50 (leniency note)
    """Criterion 3 – Sample Size
    If human study: determine if sample size n >= 50.
    - If stated n >= 50 → satisfied=true.
    - If < 50 → satisfied=false (note: can be relaxed later if other criteria are very strong).""",

    # 4) Proteins as biomarkers (exclude gene/RNA/transcript/fragment focus)
    """Criterion 4 – Protein Biomarkers
    Decide if the study's biomarker focus is on proteins (e.g., protein, amyloid, tau; beta-amyloid).
    - Satisfied if protein focus is central and recurrent.
    - Not satisfied if focus is genes/RNA/transcripts/fragments.""",

    # 5) Animal models exclusion (use human; flag patient cell-cultures)
    """Criterion 5 – Animal Models Exclusion
    Determine if the study uses animal models.
    - If animal models are used → satisfied=false.
    - If human data only → satisfied=true.
    - If using patient-derived cell cultures (not animals), note that explicitly.""",

    # 6) Blood as AD biomarker (not blood pressure)
    """Criterion 6 – Blood as AD Biomarker
    If 'blood' appears, decide if it is used as an AD biomarker (e.g., serum/plasma for amyloid/tau).
    - Exclude circulatory measures (e.g., blood pressure, hypertension, vascular health).""",
]

# Criterion names for display and matching
CRITERION_NAMES = [f"criterion_{i+1}" for i in range(len(CRITERIA_INSTRUCTIONS))]

# Models see the whole paper and must cite where their answer comes from. Quotes are
# resolved back to the source text (papers.evidence); a quote that cannot be found is
# recorded as unmatched, so copying verbatim matters more than quoting a lot.
OUTPUT_FORMAT = """
    Evidence rules:
    - Cite 1-3 passages that support your decision. Each quote must be copied VERBATIM
      from the paper (one sentence or less; do not paraphrase, do not merge passages).
    - "section" is the section heading exactly as it appears after "##" in the paper.
    - If the decision rests on the ABSENCE of something (e.g. no animal model is ever
      mentioned), return an empty evidence list and say so in the reason.
    Return JSON only:
    {{"{name}": {{"satisfied": true/false, "reason": "<brief reason>", "evidence": [{{"section": "<section heading>", "quote": "<verbatim text>"}}]}}}}"""


def build_criterion_prompt(index: int) -> str:
    """Full model prompt for one criterion: instructions + shared output format."""
    return CRITERIA_INSTRUCTIONS[index] + OUTPUT_FORMAT.format(name=CRITERION_NAMES[index])


CRITERIA_PROMPTS = [build_criterion_prompt(i) for i in range(len(CRITERIA_INSTRUCTIONS))]

