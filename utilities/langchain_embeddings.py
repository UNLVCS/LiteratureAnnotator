"""
LangChain ``Embeddings`` factory driven by ``EmbeddingsConfig``.

Claude (Anthropic) does not expose a text-embedding API. For RAG alongside
``llm_providers.anthropic``, use ``provider: voyage`` (Anthropic’s recommended
embedding partner) with a `Voyage API key <https://www.voyageai.com/>`__ — not
``llm_providers.anthropic.api_key``. Alternatively ``provider: gemini`` uses
Google’s embedding models (often convenient on the Gemini API free tier / AI Studio).
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config.app_config import EmbeddingsConfig

_VOYAGE_BATCH = 64


class VoyageLangChainEmbeddings(Embeddings):
    """Thin LangChain wrapper around the Voyage AI ``embed`` API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        output_dimension: Optional[int] = None,
    ) -> None:
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "embeddings.provider is 'voyage' but package 'voyageai' is not installed. "
                "Add it to your environment (e.g. `uv sync` after workspace deps update)."
            ) from e
        if not api_key:
            raise ValueError(
                "embeddings.api_key is empty. Voyage embeddings require a Voyage API key "
                "(not the Anthropic chat API key)."
            )
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._output_dimension = output_dimension

    def _embed(self, texts: List[str], *, input_type: str) -> List[List[float]]:
        kwargs = {"model": self._model, "input_type": input_type}
        if self._output_dimension is not None:
            kwargs["output_dimension"] = self._output_dimension
        out: List[List[float]] = []
        for i in range(0, len(texts), _VOYAGE_BATCH):
            batch = texts[i : i + _VOYAGE_BATCH]
            resp = self._client.embed(batch, **kwargs)
            out.extend(resp.embeddings)
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], input_type="query")[0]


def build_langchain_embeddings(cfg: EmbeddingsConfig) -> Embeddings:
    """
    Build a LangChain ``Embeddings`` instance from config.

    * ``voyage`` — Voyage AI (default; pairs with Claude for retrieval).
    * ``openai`` — ``langchain_openai.OpenAIEmbeddings``.
    * ``gemini`` — ``langchain_google_genai.GoogleGenerativeAIEmbeddings``.
    """
    provider = (cfg.provider or "voyage").strip().lower()
    if provider in ("anthropic", "claude"):
        raise ValueError(
            "embeddings.provider cannot be 'anthropic' or 'claude': those APIs are for chat, not vectors. "
            "Use 'voyage' (recommended with Claude), 'openai', or 'gemini'. "
            "For Voyage, set embeddings.api_key to a Voyage key from https://www.voyageai.com/ "
            "(not llm_providers.anthropic.api_key)."
        )
    if provider == "openai":
        dims: Optional[int] = None if cfg.model == "text-embedding-ada-002" else cfg.dimensions
        return OpenAIEmbeddings(
            model=cfg.model,
            api_key=cfg.api_key or None,
            dimensions=dims,
        )
    if provider == "voyage":
        return VoyageLangChainEmbeddings(
            api_key=cfg.api_key,
            model=cfg.model,
            output_dimension=cfg.dimensions if cfg.dimensions > 0 else None,
        )
    if provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "embeddings.provider is 'gemini' but package 'langchain-google-genai' is not installed. "
                "Add it to your environment (e.g. uv sync in the workspace root)."
            ) from e
        return GoogleGenerativeAIEmbeddings(
            model=cfg.model,
            google_api_key=cfg.api_key or None,
            output_dimensionality=cfg.dimensions if cfg.dimensions > 0 else None,
        )
    raise ValueError(
        f"Unknown embeddings.provider {cfg.provider!r}; use 'voyage', 'openai', or 'gemini'."
    )
