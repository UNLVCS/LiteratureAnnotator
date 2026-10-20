"""
Whole-paper criterion labeling shared by labeler.py (sequential) and labeler_mp.py.

Each criterion is evaluated in its own LLM call over the full paper text, so one
criterion's reasoning cannot leak into another's. Models cite verbatim quotes as
evidence; quotes are resolved against the source passages and stored with offsets
and a ``matched`` flag (unmatched = hallucinated or paraphrased citation).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from llm_providers.base import BaseLLMProvider, LLMResponse, Query
from papers import Paper, render_paper, resolve_evidence
from response_standardizer import standardize_llm_response
from utilities.criteria import CRITERIA_PROMPTS, CRITERION_NAMES, NOT_APPLICABLE, allowed_answers

SCHEMA_VERSION = "whole-paper-v2"  # v2: "not_applicable" answer for conditional criteria



class SkipPaper(Exception):
    """The paper can never be labeled (e.g. empty after parsing); ack it, don't requeue."""


SYSTEM_MESSAGE = (
    "You are an expert research analyst screening biomedical papers. "
    "Read the full paper and respond with valid JSON only."
)


def paper_prefix(paper_text: str) -> str:
    """Shared start of every criterion prompt for one paper (cacheable)."""
    return f"""Full text of the research paper (references removed):

{paper_text}

---

"""


def create_inference_query(paper_text: str, criteria_prompt: str) -> str:
    """Paper first, task last: keeps the long shared prefix cacheable across criteria."""
    return f"{paper_prefix(paper_text)}Task: {criteria_prompt}\n"


def label_paper(
    paper: Paper,
    provider_name: str,
    provider: BaseLLMProvider,
    max_chars: Optional[int] = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """Evaluate every criterion for one paper with one provider.

    Returns the result record written to MinIO as ``{provider}/{paper_id}.json``.
    ``errors`` holds failures that should requeue the paper (the LLM call failed);
    a malformed answer is recorded only on its criterion.

    Raises:
        SkipPaper: the paper has no text to label.
    """
    result, queries = build_queries(paper, provider_name, max_chars, temperature, max_tokens)
    try:
        responses = provider.call_api_batch(queries)
    except Exception as e:
        result["errors"].append(f"LLM batch call failed: {e}")
        result["criteria_results"] = [
            {"criterion": name, "prompt": prompt, "response": None, "error": str(e), "evidence": []}
            for name, prompt in zip(CRITERION_NAMES, CRITERIA_PROMPTS)
        ]
        return result
    return assemble_result(paper, result, responses)


def build_queries(
    paper: Paper,
    provider_name: str,
    max_chars: Optional[int] = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> Tuple[Dict[str, Any], List[Query]]:
    """The result record (without answers) and one query per criterion, in CRITERION_NAMES order.

    Raises:
        SkipPaper: the paper has no text to label.
    """
    rendered = render_paper(paper, max_chars=max_chars)
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper.pmid,
        "provider": provider_name,
        "title": paper.title,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "source": paper.source,
            "source_id": paper.source_id,
            "is_full_text": paper.is_full_text,
            "rendered_chars": len(rendered.text),
            "truncated": rendered.truncated,
            "passages_included": rendered.passages_included,
            "passages_total": rendered.passages_total,
        },
        "criteria_results": [],
        "errors": [],
    }
    if rendered.passages_total == 0:
        raise SkipPaper(f"{paper.pmid}: no text after parsing")

    prefix_len = len(paper_prefix(rendered.text))
    queries = [
        Query(
            prompt=create_inference_query(rendered.text, prompt),
            cache_prefix_len=prefix_len,
            system_message=SYSTEM_MESSAGE,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for prompt in CRITERIA_PROMPTS
    ]
    return result, queries


def assemble_result(paper: Paper, result: Dict[str, Any], responses: List[LLMResponse]) -> Dict[str, Any]:
    """Parse one response per criterion (CRITERION_NAMES order) into ``result``."""
    for name, prompt, response in zip(CRITERION_NAMES, CRITERIA_PROMPTS, responses):
        parsed, cleaned, success = standardize_llm_response(response.content)
        answer = (parsed or {}).get(name) if success else None
        error = None
        if not success or not parsed:
            error = "Failed to parse JSON"
        elif not isinstance(answer, dict) or not _is_allowed(name, _normalize_satisfied(answer.get("satisfied"))):
            error = f"Missing or malformed '{name}.satisfied'"
        else:
            answer["satisfied"] = _normalize_satisfied(answer["satisfied"])
        evidence = resolve_evidence(paper, answer.get("evidence")) if isinstance(answer, dict) else []
        # Per-criterion parse errors are kept in the record, not in result["errors"]:
        # the result is still saved, and a malformed answer is a quality signal itself.
        result["criteria_results"].append(
            {
                "criterion": name,
                "prompt": prompt,
                "response": parsed if success else None,
                "raw_response": response.content,
                "cleaned_response": cleaned,
                "error": error,
                "evidence": [e.to_dict() for e in evidence],
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            }
        )
    return result


def _normalize_satisfied(value: Any) -> Any:
    """Map spelling variants of not-applicable ("N/A", "not applicable") to NOT_APPLICABLE."""
    if isinstance(value, str) and value.strip().lower().replace("/", "").replace(" ", "_").replace("-", "_") in (
        "not_applicable", "na",
    ):
        return NOT_APPLICABLE
    return value


def _is_allowed(name: str, value: Any) -> bool:
    # Checked by type, not ``in``: 1 == True in Python, and 1 is not a valid answer.
    return isinstance(value, bool) or (value == NOT_APPLICABLE and NOT_APPLICABLE in allowed_answers(name))


def result_object_name(result: Dict[str, Any], output_prefix: str = "") -> str:
    prefix = output_prefix.strip("/")
    name = f"{result['provider']}/{result['paper_id']}.json"
    return f"{prefix}/{name}" if prefix else name
