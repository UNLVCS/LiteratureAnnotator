# papers

Whole-paper access for the pipeline. Reads the raw BioC JSON that `data_download` stores
in MinIO, turns it into a `Paper` with section labels and character offsets, renders it
for a model (plain text) or a reviewer (HTML), and resolves model-cited quotes back to
exact locations. It replaces the retired `data_vectorize` + Pinecone path.

Installable workspace member: `import papers`.

## API

```python
from papers import load_paper_from_config, render_paper, render_paper_html, resolve_evidence

paper = load_paper_from_config("41646005", app_config)   # MinIO → Paper
paper.title, paper.is_full_text, paper.sections

rendered = render_paper(paper, max_chars=None)            # text for the LLM prompt
rendered.text, rendered.truncated

evidence = resolve_evidence(paper, [{"section": "Methods", "quote": "312 patients were recruited"}])
evidence[0].matched, evidence[0].start, evidence[0].end  # BioC document offsets

html = render_paper_html(paper, [(ev.start, ev.end, "C3") for ev in evidence if ev.matched])
```

| Module | Contents |
|---|---|
| `bioc.py` | `parse_bioc()` → `Paper` / `Passage`. Drops excluded section types (default: References, author contributions, competing interests, funding). |
| `render.py` | `render_paper()` → text with `## <section>` headers; optional `max_chars` cut at a passage boundary. |
| `evidence.py` | `resolve_quote()` / `resolve_evidence()` → `ResolvedEvidence` (matched flag, offsets, located section). |
| `html.py` | `render_paper_html()` → escaped HTML with `<mark>`ed citations, used by `label_api`. |
| `store.py` | `load_paper()` / `load_paper_from_config()` from MinIO; `PaperNotFoundError`. |

## Section labels

PMC full text often tags every body passage `INTRO`, so `section_type` alone is not
usable. Labels come from the heading passages (`title_1`, `title_2`, ...), e.g.
`Methods > Participants`, falling back to the mapped `section_type` (`Abstract`,
`Results`, ...). Figure captions and tables keep the surrounding heading path and add
`> Figure` / `> Table`. The same labels appear after `##` in the rendered text, which is
what the model is asked to cite.

## Quote matching

Matching is exact after normalization (whitespace collapsed, case folded, typographic
quotes/dashes mapped to ASCII, spacing around dashes ignored). `...` splits a quote into
fragments that must occur in order within one passage. Quotes whose longest fragment is
under 12 characters never match, and nothing in excluded sections can match, so a
citation of the reference list counts as unmatched.

There is deliberately no fuzzy matching: an unmatched quote is the signal that a model
paraphrased or invented its evidence.

## Tests

```bash
uv run pytest papers/tests
```
