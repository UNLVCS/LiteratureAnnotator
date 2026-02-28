"""
Seed the human labeling paper queue from a file of paper IDs.
Paper IDs must exist in Pinecone (chunked via data_label.py).
"""

import os

from queue_helpers import enqueue_paper_id_human

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
print(f"Redis URL: {REDIS_URL}")


def seed_human_queue_from_file(file_path: str) -> None:
    """Read paper IDs from file and enqueue for human labeling."""
    with open(file_path, "r") as f:
        for line in f:
            paper_id = line.strip()
            if paper_id:
                success = enqueue_paper_id_human(paper_id)
                if success:
                    print(f"Enqueued: {paper_id}")
                else:
                    print(f"Duplicate or failed: {paper_id}")


if __name__ == "__main__":
    # Default: human_papers.txt or fallback to test_papers.txt
    file_path = os.getenv("HUMAN_PAPERS_FILE", "human_papers.txt")
    if not os.path.exists(file_path):
        file_path = "test_papers.txt"
    if not os.path.exists(file_path):
        print(f"No file found. Create {file_path} or HUMAN_PAPERS_FILE with paper IDs (one per line).")
    else:
        seed_human_queue_from_file(file_path)
