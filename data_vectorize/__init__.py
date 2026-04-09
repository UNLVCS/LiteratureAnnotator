"""
data_vectorize — vector database retrieval and MinIO ingestion package.

Public API
----------
VectorDb
    Pinecone wrapper.  Use ``query()`` to retrieve chunks relevant to a
    natural-language query; use ``upsert()`` to store pre-built records.

Ingester
    End-to-end pipeline: fetches JSON articles from a MinIO bucket,
    splits them into overlapping chunks, generates OpenAI embeddings,
    and upserts the results into Pinecone.

Chunker
    Low-level text splitter used internally by Ingester.  Expose here
    so callers can chunk arbitrary article dicts without a full ingest.

Example — retrieval (data_generation, label_api)::

    from data_vectorize import VectorDb

    vdb = VectorDb()
    matches = vdb.query(namespace="project-x", query_text="cancer biomarkers", top_k=5)
    for m in matches:
        print(m.score, m.metadata["text"])

Example — ingestion (populate the vector DB from MinIO)::

    from data_vectorize import Ingester

    Ingester().run()
"""

from data_vectorize.chnker import Chunker
from data_vectorize.ingester import Ingester
from data_vectorize.vector_db import VectorDb

__all__ = [
    "Chunker",
    "Ingester",
    "VectorDb",
]
