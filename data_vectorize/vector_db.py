from typing import Optional, TYPE_CHECKING

import numpy as np

import openai
from pinecone import Pinecone, ServerlessSpec, PineconeApiException

if TYPE_CHECKING:
    from config.app_config import EmbeddingsConfig, PineconeConfig


class VectorDb:
    """
    Pinecone vector database wrapper with integrated embedding generation.

    Handles both sides of the RAG loop:
      - upsert(): store pre-built vector records (called by Ingester)
      - query():  retrieve the top-k chunks most similar to a text query
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        pinecone_config: Optional["PineconeConfig"] = None,
        embeddings_config: Optional["EmbeddingsConfig"] = None,
    ):
        app_config = None
        if pinecone_config is None and api_key is None:
            from config.app_config import load_app_config
            app_config = load_app_config()
            pinecone_config = app_config.pinecone

        if embeddings_config is None:
            if app_config is None:
                from config.app_config import load_app_config
                app_config = load_app_config()
            embeddings_config = app_config.embeddings

        if pinecone_config is not None:
            api_key = pinecone_config.api_key
            index_name = index_name or pinecone_config.index_name

        index_name = index_name or "adbm"
        dimensions = embeddings_config.dimensions if embeddings_config and embeddings_config.model != "text-embedding-ada-002" else None 

        self._embeddings_config = embeddings_config
        self._openai = openai.OpenAI(api_key=embeddings_config.api_key)

        self.pc = Pinecone(api_key=api_key)

        try:
            self.pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except PineconeApiException as e:
            if e.status == 409:
                pass  # index already exists
            else:
                raise

        self.index = self.pc.Index(index_name)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Generate a single embedding vector for *text*."""
        kwargs: dict = {
            "model": self._embeddings_config.model,
            "input": text,
            "encoding_format": "float",
        }
        # text-embedding-ada-002 does not accept a dimensions parameter
        if self._embeddings_config.model != "text-embedding-ada-002":
            kwargs["dimensions"] = self._embeddings_config.dimensions
        response = self._openai.embeddings.create(**kwargs)
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, namespace: str, vectors: list[dict]) -> None:
        """
        Upsert pre-built vector records into *namespace*.

        Each record must follow the Pinecone format::

            {"id": str, "values": list[float], "metadata": dict}
        """
        self.index.upsert(namespace=namespace, vectors=vectors)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        namespace: str,
        query_text: str,
        top_k: int = 5,
        include_metadata: bool = True,
    ) -> list:
        """
        Return the *top_k* chunks most semantically similar to *query_text*.

        Args:
            namespace:        Logical partition — typically a user / project ID.
            query_text:       Natural-language query; embedded internally.
            top_k:            Number of results to return.
            include_metadata: When True, each match includes its stored metadata
                              (raw text, doc ID, title, chunk index).

        Returns:
            List of Pinecone ``ScoredVector`` objects, each with ``id``,
            ``score``, and (if requested) ``metadata``.
        """
        embedding = self._embed(query_text)
        result = self.index.query(
            namespace=namespace,
            vector=embedding,
            top_k=top_k,
            include_metadata=include_metadata,
        )
        return result.matches

    def query_mmr(
        self,
        namespace: str,
        query_text: str,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list:
        """
        Return top_k chunks using Maximal Marginal Relevance to reduce redundancy.

        Fetches fetch_k candidates by cosine similarity, then iteratively selects
        results that balance relevance (sim to query) against redundancy (sim to
        already-selected results).  lambda_mult=1.0 is pure similarity; 0.0 is
        pure diversity; 0.5 is a balanced default.
        """
        embedding = self._embed(query_text)
        candidates = self.index.query(
            namespace=namespace,
            vector=embedding,
            top_k=fetch_k,
            include_metadata=True,
            include_values=True,
        ).matches

        if not candidates:
            return []

        # Unit-normalize so dot product == cosine similarity.
        # Pinecone normalizes at upsert for cosine indexes, but we normalize
        # defensively in case of floating-point drift.
        vecs = np.array([c.values for c in candidates], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1.0, norms)

        selected: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break
            if not selected:
                best_idx = remaining[0]  # Pinecone already sorted by similarity
            else:
                sel_vecs = vecs[selected]
                best_score, best_idx = -float("inf"), remaining[0]
                for idx in remaining:
                    relevance = candidates[idx].score
                    redundancy = float(np.max(vecs[idx] @ sel_vecs.T))
                    score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                    if score > best_score:
                        best_score, best_idx = score, idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i] for i in selected]
