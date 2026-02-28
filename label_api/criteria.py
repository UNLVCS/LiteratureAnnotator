"""
Shared criteria prompts for RAG and Human labeling projects.
Extracted from data_generation/rag_labeling_script.py.
"""

# First 6 criteria (no final aggregation) - used for both RAG and Human labeling
CRITERIA_PROMPTS = [
    # 1) Original research
    """Criterion 1 – Original Research
    Decide if the paper is an original research article (not a review, perspective, poster, or preprint).
    - Positive signals: data collection + statistical analysis (often in Methods).
    - Negative signals: clear mentions of review, perspective, poster, preprint.
    Return JSON only:
    {"criterion_1": {"satisfied": true/false, "reason": "<brief reason>"}}""",

    # 2) AD focus
    """Criterion 2 – AD Focus
    Decide if Alzheimer's Disease (AD) is the main focus (diagnosis, treatment, biomarkers, pathology; AD patients incl. MCI/at risk).
    - Include AD biomarkers: amyloid-beta, tau.
    - Exclude if focus is general neurodegeneration markers without AD specificity.
    Return JSON only:
    {"criterion_2": {"satisfied": true/false, "reason": "<brief reason>"}}""",

    # 3) Sample size >= 50 (leniency note)
    """Criterion 3 – Sample Size
    If human study: determine if sample size n >= 50.
    - If stated n >= 50 → satisfied=true.
    - If < 50 → satisfied=false (note: can be relaxed later if other criteria are very strong).
    Return JSON only:
    {"criterion_3": {"satisfied": true/false, "reason": "<brief reason; include n if found>"}}""",

    # 4) Proteins as biomarkers (exclude gene/RNA/transcript/fragment focus)
    """Criterion 4 – Protein Biomarkers
    Decide if the study's biomarker focus is on proteins (e.g., protein, amyloid, tau; beta-amyloid).
    - Satisfied if protein focus is central and recurrent.
    - Not satisfied if focus is genes/RNA/transcripts/fragments.
    Return JSON only:
    {"criterion_4": {"satisfied": true/false, "reason": "<brief reason>"}}""",

    # 5) Animal models exclusion (use human; flag patient cell-cultures)
    """Criterion 5 – Animal Models Exclusion
    Determine if the study uses animal models.
    - If animal models are used → satisfied=false.
    - If human data only → satisfied=true.
    - If using patient-derived cell cultures (not animals), note that explicitly.
    Return JSON only:
    {"criterion_5": {"satisfied": true/false, "reason": "<brief reason; note 'patient cell cultures' if applicable>"}}""",

    # 6) Blood as AD biomarker (not blood pressure)
    """Criterion 6 – Blood as AD Biomarker
    If 'blood' appears, decide if it is used as an AD biomarker (e.g., serum/plasma for amyloid/tau).
    - Exclude circulatory measures (e.g., blood pressure, hypertension, vascular health).
    Return JSON only:
    {"criterion_6": {"satisfied": true/false, "reason": "<brief reason>"}}""",
]

# Criterion names for display and matching
CRITERION_NAMES = [f"criterion_{i+1}" for i in range(6)]
