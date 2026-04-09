from typing import Optional, TYPE_CHECKING

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
