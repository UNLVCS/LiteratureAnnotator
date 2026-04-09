"""
Base classes for LLM provider interface
"""

import random
import time
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional, List, TypeVar
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for transient errors that should trigger a retry (rate limits, server overload)."""
    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()
    retryable_type_hints = ("ratelimit", "rate_limit", "overloaded", "timeout", "connection")
    retryable_msg_hints = (
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "quota exceeded",
        "insufficient_quota",
        "overloaded",
        "503",
        "502",
        "temporarily unavailable",
        "retry after",
        "capacity",
        "server error",
    )
    return (
        any(h in exc_type for h in retryable_type_hints)
        or any(h in exc_msg for h in retryable_msg_hints)
    )


def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    **kwargs,
) -> T:
    """
    Call func(*args, **kwargs) and retry with exponential backoff + jitter on transient errors.

    Only retries on errors detected as rate-limit / server-overload by `_is_retryable_error`.
    Non-transient errors (auth failures, bad requests, etc.) are re-raised immediately.

    Args:
        func: Callable to invoke.
        max_retries: Maximum number of retry attempts after the first failure.
        base_delay: Initial backoff delay in seconds (doubles on each attempt).
        max_delay: Maximum delay cap in seconds.
    """
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                "Retryable error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            print(
                f"[retry] Attempt {attempt + 1}/{max_retries} failed ({type(exc).__name__}): "
                f"{exc}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    raise last_exc


class ModelType(Enum):
    """Enum for different types of models"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


@dataclass
class Query:
    """Standardized query structure for LLM providers"""
    prompt: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    system_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None




@dataclass
class LLMResponse:
    """Standardized response structure from LLM providers"""
    content: str
    model: str
    usage: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers
    
    This class defines the common interface that all LLM providers must implement.
    It uses LangChain as the underlying framework for consistency.
    """
    
    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        """
        Initialize the LLM provider
        
        Args:
            api_key: API key for the provider
            model: Default model to use (can be overridden in queries)
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.default_model = model
        self.config = kwargs
        
    @abstractmethod
    def call_api(self, query: Query) -> LLMResponse:
        """
        Call the LLM API with the provided query
        
        Args:
            query: Query object containing the prompt and parameters
            
        Returns:
            LLMResponse object containing the response and metadata
        """
        pass

    def call_api_batch(self, queries: List[Query]) -> List[LLMResponse]:
        """
        Call the LLM API with a list of queries and return responses in the same order.

        Default implementation calls call_api() sequentially. Override in subclasses
        for true parallelism (e.g. GPU batch inference or concurrent HTTP requests).

        Args:
            queries: List of Query objects to process.

        Returns:
            List of LLMResponse objects, one per query, in the same order.
        """
        return [self.call_api(q) for q in queries]
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    def validate_query(self, query: Query) -> bool:
        """
        Validate if the query is compatible with this provider
        
        Args:
            query: Query to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def get_provider_name(self) -> str:
        """
        Get the name of this provider
        
        Returns:
            Provider name
        """
        return self.__class__.__name__.replace('Provider', '').lower()
    
    def _prepare_metadata(self, query: Query, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for the response
        
        Args:
            query: Original query
            response_data: Raw response data from the provider
            
        Returns:
            Dictionary containing metadata
        """
        return {
            'provider': self.get_provider_name(),
            'model': query.model or self.default_model,
            'query_metadata': query.metadata or {},
            'raw_response': response_data
        }
