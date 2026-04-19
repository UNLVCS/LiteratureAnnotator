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

    By default the ingester reads ``bioc_download.object_prefix`` from
    ``.env.yaml`` and uses it as both the MinIO object prefix filter **and**
    the Pinecone namespace.  This means each download batch gets its own
    isolated namespace automatically.  When the prefix is empty the fallback
    namespace is ``pinecone.namespace``.

    Typical usage::

        # Auto-scoped to bioc_download.object_prefix / its namespace:
        Ingester().run()

        # Override the prefix (and therefore the namespace):
        Ingester(prefix="run_2026_04").run()

        # Explicit namespace, ignoring the prefix-derived one:
        Ingester(namespace="my-ns").run()
    """

    def __init__(
        self,
        minio_config: Optional["MinioConfig"] = None,
        pinecone_config: Optional["PineconeConfig"] = None,
        embeddings_config: Optional["EmbeddingsConfig"] = None,
        namespace: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        """
        Args:
            minio_config:      MinIO connection settings. Loaded from .env.yaml if omitted.
            pinecone_config:   Pinecone settings. Loaded from .env.yaml if omitted.
            embeddings_config: OpenAI embeddings settings. Loaded from .env.yaml if omitted.
            prefix:            MinIO object prefix to filter on and use as the Pinecone
                               namespace.  Defaults to ``bioc_download.object_prefix`` from
                               .env.yaml.  Pass ``""`` explicitly to process the whole bucket
                               and fall back to ``pinecone.namespace``.
            namespace:         Override the Pinecone namespace independently of the prefix.
                               Rarely needed — by default the prefix becomes the namespace.
        """
        from config.app_config import load_app_config
        app_config = load_app_config()

        minio_config = minio_config or app_config.minio
        pinecone_config = pinecone_config or app_config.pinecone
        embeddings_config = embeddings_config or app_config.embeddings

        # Resolve prefix: explicit arg → bioc_download.object_prefix → ""
        self._prefix = prefix if prefix is not None else (app_config.bioc_download.object_prefix or "")

        # Namespace: explicit arg → prefix (if non-empty) → pinecone.namespace
        self._namespace = namespace or self._prefix or pinecone_config.namespace

        self._bucket = minio_config.raw_articles_bucket

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
                "values": self._vdb._embed(chunk["text"]),
                "metadata": {
                    "text": chunk["text"],
                    "parent_text": chunk["parent_text"],
                    "section": chunk["section"],
                    "doc": chunked["id"],
                    "title": chunked["title"],
                    "chunk": i,
                },
            })

        if vectors:
            self._vdb.upsert(self._namespace, vectors)
            print(f"Upserted {len(vectors)} chunks for {chunked['id']}")

    def run(self) -> None:
        """Load articles from the configured MinIO bucket (and prefix) and ingest them."""
        scope = f"{self._bucket}/{self._prefix}" if self._prefix else self._bucket
        print(f"Ingesting from  : {scope}")
        print(f"Pinecone ns     : {self._namespace}")
        articles = self._load_from_minio()
        print(f"Articles found  : {len(articles)}")
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
        """Return JSON articles from the configured MinIO bucket, filtered by prefix if set."""
        articles = []
        for obj in self._minio.list_objects(self._bucket, prefix=self._prefix or None, recursive=True):
            try:
                response = self._minio.get_object(
                    bucket_name=self._bucket,
                    object_name=obj.object_name,
                )
                data = json.loads(response.read().decode("utf-8"))
                article_id = obj.object_name.split("/")[-1].split(".")[0]
                articles.append(self._parse_bioc(article_id, data))
                print(f"Loaded {obj.object_name}")
            except Exception as exc:
                print(f"Skipping {obj.object_name}: {exc}")
        return articles

    @staticmethod
    def _parse_bioc(pmid: str, bioc_data: dict) -> dict:
        """Convert raw BioC JSON from the NCBI API into the flat section dict
        expected by ``Chunker``.

        BioC passages carry an ``infons.section_type`` field (TITLE, ABSTRACT,
        INTRO, METHODS, RESULTS, DISCUSS, CONCL, REF).  Multiple passages with
        the same section type are grouped into a list.  The Title is kept as a
        plain string because ``Chunker`` reads it separately.

        Returns ``{pmid: {"Title": str, "Abstract": [...], ...}}``.
        """
        SECTION_MAP = {
            "TITLE":    "Title",
            "ABSTRACT": "Abstract",
            "INTRO":    "Introduction",
            "METHODS":  "Methodology",
            "RESULTS":  "Results",
            "DISCUSS":  "Discussion",
            "CONCL":    "Conclusion",
            "REF":      "References",
        }

        sections: dict[str, list[str]] = {}
        # The BioC API returns a JSON array of collection objects; unwrap the first.
        if isinstance(bioc_data, list):
            bioc_data = bioc_data[0] if bioc_data else {}
        documents = bioc_data.get("documents", [])
        if not documents:
            return {pmid: {}}

        for passage in documents[0].get("passages", []):
            text = passage.get("text", "").strip()
            if not text:
                continue
            raw_type = passage.get("infons", {}).get("section_type", "").upper()
            key = SECTION_MAP.get(raw_type) or raw_type.title() or "Body"
            sections.setdefault(key, []).append(text)

        # Title should be a plain string for Chunker's .article_title field.
        # Fall back to the PMID if no title passage was present.
        flat: dict = {}
        for key, values in sections.items():
            flat[key] = values[0] if key == "Title" else values
        flat.setdefault("Title", pmid)

        return {pmid: flat}
