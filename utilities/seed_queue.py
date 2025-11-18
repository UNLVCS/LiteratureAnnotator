import os
import redis

# Import the enqueue_paper_id function from queue_helpers
from queue_helpers import enqueue_paper_id

# Set up Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def seed_queue_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            paper_id = line.strip()
            if paper_id:  # Ensure it's not an empty line
                success = enqueue_paper_id(paper_id)
                if success:
                    print(f"Enqueued: {paper_id}")
                else:
                    print(f"Duplicate or failed to enqueue: {paper_id}")

# Path to your text file containing paper IDs
# file_path = 'test_papers.txt'
file_path = 'labeled_paper_ids.txt'
seed_queue_from_file(file_path)
