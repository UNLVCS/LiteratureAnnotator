from data_generation.labeler_mp import initialize_shared_resources, return_relevant_chunks, get_paper_chunks
from utilities.criteria import CRITERIA_PROMPTS

initialize_shared_resources()

paper_id = "41491101"  # example

# All chunks the “broad” path can see (k=20, empty query)
docs = get_paper_chunks(paper_id)
print("get_paper_chunks count:", len(docs))
for i, d in enumerate(docs):
    print(i, d.metadata, d.page_content[:400].replace("\n", " "), "...")

# Same as labeling: top-k per criterion
for idx, prompt in enumerate(CRITERIA_PROMPTS[:]):  # or all
    rel = return_relevant_chunks(paper_id, prompt, k=5)
    print(f"\n=== criterion {idx+1} k={len(rel)} ===")
    for j, d in enumerate(rel):
        print(j, d.metadata.get("title"), "|", d.page_content[:300].replace("\n", " "), "...")