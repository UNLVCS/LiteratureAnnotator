## LabelStudio tool for human annotations

This directory contains the code that fetches synthetic data and provides for human annotation and scoring.

### Projects

- **RAG Annotation Project**: Review and validate LLM-generated classifications (CORRECT/INCORRECT/ABSTAIN).
- **Human Labeling Project**: Direct human classification from chunks (SATISFIED/NOT_SATISFIED/ABSTAIN) for fine-tuning. Uses Pinecone for chunk retrieval.

### Environment variables (Human Labeling)

| Variable | Purpose |
|----------|---------|
| `HUMAN_PAPER_QUEUE` | Redis key for human paper queue (default: `q:papers:human:v1`) |
| `PINECONE_HUMAN_NAMESPACE` | Pinecone namespace for chunks (e.g. `article_upload_test_2`) |
| `HUMAN_ANNOTATIONS_BUCKET` | MinIO bucket for completed human annotations (default: `human-annotations`) |

Seed the human queue: `python utilities/seed_human_queue.py` (uses `human_papers.txt` or `HUMAN_PAPERS_FILE`).