"""Tests for whole-paper labeling with a fake provider (no network, no MinIO)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "data_generation")]

from labeling import SkipPaper, label_paper, result_object_name  # noqa: E402
from llm_providers.base import LLMResponse  # noqa: E402
from papers import parse_bioc  # noqa: E402
from utilities.criteria import CRITERION_NAMES  # noqa: E402


def _p(offset, section_type, ptype, text):
    return {"offset": offset, "infons": {"section_type": section_type, "type": ptype}, "text": text}


PAPER = parse_bioc("41000001", [{"source": "PMC", "documents": [{"id": "PMC1", "passages": [
    _p(0, "TITLE", "front", "Plasma p-tau217 in a memory clinic cohort"),
    _p(43, "ABSTRACT", "abstract", "We measured plasma p-tau217 in 312 participants with cognitive complaints."),
    _p(120, "METHODS", "title_1", "Methods"),
    _p(128, "METHODS", "paragraph", "A total of 312 patients were recruited from the memory clinic."),
    _p(200, "REF", "ref", "Tau pathology in transgenic mice."),
]}]}])


class FakeProvider:
    """Answers each criterion from a canned map; records the prompts it was sent."""

    def __init__(self, answers):
        self.answers = answers
        self.prompts = []

    def call_api_batch(self, queries):
        self.prompts.extend(q.prompt for q in queries)
        out = []
        for q in queries:
            name = next(n for n in CRITERION_NAMES if f'{{"{n}":' in q.prompt)
            out.append(LLMResponse(content=self.answers.get(name, "not json"), model="fake", usage={}))
        return out


def _answer(name, satisfied, quote, section="Methods"):
    return "```json\n" + json.dumps({name: {
        "satisfied": satisfied, "reason": "r", "evidence": [{"section": section, "quote": quote}]}}) + "\n```"


def test_labels_every_criterion_over_full_text_and_resolves_evidence():
    provider = FakeProvider({
        "criterion_1": _answer("criterion_1", True, "A total of 312 patients were recruited"),
        "criterion_3": _answer("criterion_3", True, "n = 500 participants were enrolled"),
    })
    result = label_paper(PAPER, "fake-model", provider)

    assert result["schema_version"] == "whole-paper-v1"
    assert result["paper_id"] == "41000001" and result["errors"] == []
    assert [c["criterion"] for c in result["criteria_results"]] == CRITERION_NAMES
    # One call per criterion, each carrying the whole paper but not the references.
    assert len(provider.prompts) == len(CRITERION_NAMES)
    assert all("memory clinic" in p and "transgenic mice" not in p for p in provider.prompts)

    c1, c2, c3 = result["criteria_results"][:3]
    assert c1["error"] is None and c1["evidence"][0]["matched"] and c1["evidence"][0]["start"] == 128
    assert c3["evidence"][0]["matched"] is False  # quote not in the paper
    assert c2["error"] == "Failed to parse JSON" and c2["response"] is None and c2["evidence"] == []


def test_malformed_answer_is_recorded_on_the_criterion_only():
    provider = FakeProvider({"criterion_1": json.dumps({"criterion_1": {"satisfied": "yes"}})})
    result = label_paper(PAPER, "fake-model", provider)
    assert result["errors"] == []
    assert "malformed" in result["criteria_results"][0]["error"]


def test_provider_failure_is_a_paper_level_error():
    class Broken:
        def call_api_batch(self, queries):
            raise RuntimeError("rate limited")

    result = label_paper(PAPER, "fake-model", Broken())
    assert result["errors"] and "rate limited" in result["errors"][0]
    assert all(c["error"] == "rate limited" for c in result["criteria_results"])


def test_empty_paper_is_skipped():
    with pytest.raises(SkipPaper):
        label_paper(parse_bioc("1", []), "fake-model", FakeProvider({}))


def test_truncation_is_recorded():
    result = label_paper(PAPER, "fake-model", FakeProvider({}), max_chars=80)
    assert result["paper"]["truncated"] is True


def test_result_object_name():
    r = {"provider": "openai/gpt-oss-120b", "paper_id": "41000001"}
    assert result_object_name(r) == "openai/gpt-oss-120b/41000001.json"
    assert result_object_name(r, "/run_v1/") == "run_v1/openai/gpt-oss-120b/41000001.json"
