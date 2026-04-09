"""
Ingester: MinIO bucket → chunk → embed → Pinecone upsert pipeline.
"""

import json
from typing import Optional, TYPE_CHECKING

from minio import Minio

from data_vectorize.chnker import Chunker
from data_vectorize.vector_db import VectorDb

if TYPE_CHECKING:
    from config.app_config import EmbeddingsConfig, MinioConfig, PineconeConfig


class Ingester:
    """
    Orchestrates loading articles from MinIO, chunking, embedding, and
    upserting into the vector database.

    Typical usage::

        ingester = Ingester()
        ingester.run()                          # whole bucket
        ingester.ingest_article({pmid: data})   # single article
    """

    def __init__(
        self,
        minio_config: Optional["MinioConfig"] = None,
        pinecone_config: Optional["PineconeConfig"] = None,
        embeddings_config: Optional["EmbeddingsConfig"] = None,
        namespace: Optional[str] = None,
    ):
        app_config = None
        if any(c is None for c in (minio_config, pinecone_config, embeddings_config)):
            from config.app_config import load_app_config
            app_config = load_app_config()

        minio_config = minio_config or app_config.minio
        pinecone_config = pinecone_config or app_config.pinecone
        embeddings_config = embeddings_config or app_config.embeddings

        self._bucket = minio_config.raw_articles_bucket
        self._namespace = namespace or pinecone_config.namespace

        self._minio = Minio(
            minio_config.url,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.secure,
        )
        self._vdb = VectorDb(
            pinecone_config=pinecone_config,
            embeddings_config=embeddings_config,
        )
        self._chunker = Chunker()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest_article(self, article_dict: dict) -> None:
        """
        Chunk, embed, and upsert a single article.

        Args:
            article_dict: ``{pmid: {section_name: content, ...}}``
        """
        self._chunker.set_chunk(article_dict)
        chunked = self._chunker.get_chunked_article()

        vectors = []
        for i, chunk in enumerate(chunked["chunks"]):
            vectors.append({
                "id": f"{chunked['id']}-chunk{i}",
                "values": self._vdb._embed(chunk),
                "metadata": {
                    "text": chunk,
                    "doc": chunked["id"],
                    "title": chunked["title"],
                    "chunk": i,
                },
            })

        if vectors:
            self._vdb.upsert(self._namespace, vectors)
            print(f"Upserted {len(vectors)} chunks for {chunked['id']}")

    def run(self) -> None:
        """Load every article from the MinIO bucket and ingest it."""
        articles = self._load_from_minio()
        print(f"Ingesting {len(articles)} articles…")
        for article in articles:
            article_id = next(iter(article), "unknown")
            try:
                self.ingest_article(article)
            except Exception as exc:
                print(f"Failed to ingest {article_id}: {exc}")
        print("Ingestion complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_minio(self) -> list[dict]:
        """Return all JSON articles from the configured MinIO bucket."""
        articles = []
        for obj in self._minio.list_objects(self._bucket):
            try:
                response = self._minio.get_object(
                    bucket_name=self._bucket,
                    object_name=obj.object_name,
                )
                data = json.loads(response.read().decode("utf-8"))
                article_id = obj.object_name.split("/")[-1].split(".")[0]
                articles.append({article_id: data})
                print(f"Loaded {obj.object_name}")
            except Exception as exc:
                print(f"Skipping {obj.object_name}: {exc}")
        return articles
