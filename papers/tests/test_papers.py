"""Tests for BioC parsing, rendering and quote resolution (no MinIO needed)."""

from papers import parse_bioc, render_paper, resolve_evidence, resolve_quote


def _p(offset, section_type, ptype, text):
    return {"offset": offset, "infons": {"section_type": section_type, "type": ptype}, "text": text}


# Shape mirrors a PMC-OA BioC_json response: list of one collection, generic INTRO
# tagging for body passages, headings as title_N passages, figures floated inline.
BIOC = [
    {
        "source": "PMC",
        "documents": [
            {
                "id": "PMC0000001",
                "passages": [
                    _p(0, "TITLE", "front", "Plasma p-tau217 in a memory clinic cohort"),
                    _p(43, "ABSTRACT", "abstract", "We measured plasma p-tau217 in 312 participants."),
                    _p(94, "INTRO", "title_1", "Methods"),
                    _p(102, "INTRO", "title_2", "Participants"),
                    _p(115, "INTRO", "paragraph", "A total of 312 patients (n = 312) were recruited—no animal models were used."),
                    _p(190, "FIG", "fig_caption", "Flow chart of participant selection."),
                    _p(228, "INTRO", "title_2", "Assays"),
                    _p(235, "INTRO", "paragraph", "Plasma was collected by venipuncture and analysed by Simoa."),
                    _p(300, "RESULTS", "title_1", "Results"),
                    _p(308, "RESULTS", "paragraph", "p-tau217 discriminated amyloid-positive participants."),
                    _p(370, "REF", "title", "References"),
                    _p(381, "REF", "ref", "Tau pathology in transgenic mice: a review."),
                    _p(430, "COMP_INT", "paragraph", "The authors declare no competing interests."),
                ],
            }
        ],
    }
]


def test_parse_drops_references_and_keeps_offsets():
    paper = parse_bioc("123", BIOC)
    assert paper.title == "Plasma p-tau217 in a memory clinic cohort"
    assert paper.source == "PMC" and paper.source_id == "PMC0000001"
    texts = " ".join(p.text for p in paper.passages)
    assert "transgenic mice" not in texts  # REF excluded
    assert "competing interests" not in texts  # COMP_INT excluded
    assert paper.passages[0].offset == 0
    assert paper.is_full_text


def test_sections_follow_headings_and_figures_do_not_reset_them():
    paper = parse_bioc("123", BIOC)
    by_text = {p.text[:20]: p.section for p in paper.passages}
    assert by_text["A total of 312 patie"] == "Methods > Participants"
    assert by_text["Flow chart of partic"] == "Methods > Participants > Figure"
    assert by_text["Plasma was collected"] == "Methods > Assays"
    assert by_text["p-tau217 discriminat"] == "Results"
    assert by_text["Methods"] == "Introduction"  # top heading has no parent


def test_abstract_only_paper():
    data = {"source": "PubMed", "documents": [{"id": "123", "passages": [
        _p(0, "TITLE", "front", "A title"), _p(8, "ABSTRACT", "abstract", "An abstract.")]}]}
    paper = parse_bioc("123", data)
    assert not paper.is_full_text
    assert render_paper(paper).text == "Title: A title\n\n## Abstract\nAn abstract."


def test_empty_payload():
    paper = parse_bioc("123", [])
    assert paper.passages == [] and paper.title == "123"


def test_render_has_section_headers_and_truncates_at_passage_boundary():
    paper = parse_bioc("123", BIOC)
    full = render_paper(paper)
    assert "## Methods > Participants" in full.text
    assert "## Methods > Assays" in full.text
    assert not full.truncated and full.passages_included == full.passages_total
    short = render_paper(paper, max_chars=200)
    assert short.truncated and short.passages_included < short.passages_total
    assert short.text.endswith("[... remainder of paper truncated ...]")


def test_quote_matches_despite_whitespace_case_and_dash_differences():
    paper = parse_bioc("123", BIOC)
    ev = resolve_quote(paper, "A total of 312 patients  (n = 312) were RECRUITED - no animal", "Participants")
    assert ev.matched and ev.section == "Methods > Participants" and ev.section_agrees
    assert ev.start == 115
    source = "A total of 312 patients (n = 312) were recruited—no animal"
    assert ev.end == 115 + len(source)


def test_hallucinated_or_too_short_quotes_do_not_match():
    paper = parse_bioc("123", BIOC)
    assert not resolve_quote(paper, "Forty transgenic mice were used in this study").matched
    assert not resolve_quote(paper, "312").matched
    assert not resolve_quote(paper, "").matched


def test_quote_from_reference_list_is_not_evidence():
    paper = parse_bioc("123", BIOC)
    assert not resolve_quote(paper, "Tau pathology in transgenic mice").matched


def test_ellipsis_fragments_must_appear_in_order():
    paper = parse_bioc("123", BIOC)
    assert resolve_quote(paper, "A total of 312 patients ... no animal models were used").matched
    assert not resolve_quote(paper, "no animal models were used ... A total of 312 patients").matched


def _markers_paper():
    return parse_bioc("124", [{"source": "PMC", "documents": [{"id": "PMC2", "passages": [
        _p(0, "RESULTS", "paragraph",
           "Levels were higher in AD than controls (p < 0.05) (Fig. 2A & B). "
           "This agrees with earlier cohorts [12, 14-16] and (see Table 3) holds after adjustment."),
    ]}]}])


def test_quote_matches_without_in_text_figure_and_citation_markers():
    paper = _markers_paper()
    # Model closes the sentence early and drops the figure pointer.
    ev = resolve_quote(paper, "Levels were higher in AD than controls (p < 0.05).")
    assert ev.matched and ev.start == 0
    # Model drops numeric citations and an inline table pointer mid-quote.
    assert resolve_quote(paper, "This agrees with earlier cohorts and holds after adjustment").matched
    # The marker itself is not matchable evidence, and paraphrase still fails.
    assert not resolve_quote(paper, "Levels were lower in AD than controls").matched


def test_section_disagreement_is_reported():
    paper = parse_bioc("123", BIOC)
    ev = resolve_quote(paper, "Plasma was collected by venipuncture", "Results")
    assert ev.matched and ev.section_agrees is False


def test_resolve_evidence_is_lenient_about_shape():
    paper = parse_bioc("123", BIOC)
    out = resolve_evidence(paper, [
        {"section": "Methods", "quote": "analysed by Simoa"},
        "p-tau217 discriminated amyloid-positive",
        {"nonsense": True},
        42,
    ])
    assert [e.matched for e in out] == [True, True, False]
    assert resolve_evidence(paper, "not a list") == []


def test_html_escapes_and_highlights_cited_spans():
    from papers import render_paper_html

    paper = parse_bioc("123", BIOC)
    ev = resolve_quote(paper, "analysed by Simoa", "Assays")
    page = render_paper_html(paper, [(ev.start, ev.end, "C4"), (None, None, "unmatched")])
    assert '<mark style="background:#fff3a3;padding:0 2px;border-radius:3px;" title="C4">analysed by Simoa</mark>' in page
    assert page.count("<mark") == 1
    assert "Methods &gt; Assays" in page  # section header escaped
    assert "transgenic mice" not in page


def test_html_overlapping_highlights_do_not_duplicate_text():
    from papers import render_paper_html

    paper = parse_bioc("123", BIOC)
    a = resolve_quote(paper, "A total of 312 patients (n = 312)")
    b = resolve_quote(paper, "312 patients (n = 312) were recruited")
    page = render_paper_html(paper, [(a.start, a.end, "C1"), (b.start, b.end, "C3")])
    assert page.count("312 patients") == 1


def test_html_merges_labels_for_identical_spans():
    from papers import render_paper_html

    paper = parse_bioc("123", BIOC)
    ev = resolve_quote(paper, "analysed by Simoa")
    page = render_paper_html(paper, [(ev.start, ev.end, "C2"), (ev.start, ev.end, "C4"), (ev.start, ev.end, "C2")])
    assert page.count("<mark") == 1 and 'title="C2 C4"' in page
