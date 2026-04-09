"""
Anthropic provider implementation using LangChain
"""

from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from .base import BaseLLMProvider, Query, LLMResponse, retry_with_backoff


def _anthropic_temperature_top_p(
    temperature: float, top_p: float
) -> tuple[Optional[float], Optional[float]]:
    """
    Anthropic Claude 4+ rejects requests that send both ``temperature`` and ``top_p``.
    Prefer ``temperature`` unless ``top_p`` is explicitly set below 1.0.
    """
    if top_p != 1.0:
        return None, top_p
    return temperature, None


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic provider implementation using LangChain's ChatAnthropic
    """
    
    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic provider
        
        Args:
            api_key: Anthropic API key
            model: Default model (e.g. ``claude-sonnet-4-6``, ``claude-opus-4-6`` — see Anthropic models docs)
            **kwargs: Additional Anthropic-specific configuration
        """
        super().__init__(api_key, model, **kwargs)
        temp = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)
        t_llm, p_llm = _anthropic_temperature_top_p(temp, top_p)
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name=model or "claude-sonnet-4-6",
            temperature=t_llm,
            max_tokens=kwargs.get("max_tokens", 1000),
            top_p=p_llm,
        )
    
    def call_api(self, query: Query) -> LLMResponse:
        """
        Call Anthropic API with the provided query
        
        Args:
            query: Query object containing the prompt and parameters
            
        Returns:
            LLMResponse object containing the response and metadata
        """
        if not self.validate_query(query):
            raise ValueError("Invalid query for Anthropic provider")
        
        # Prepare messages
        messages = []
        if query.system_message:
            messages.append(SystemMessage(content=query.system_message))
        messages.append(HumanMessage(content=query.prompt))
        
        # Update model if specified in query
        model_name = query.model or self.default_model
        if query.model and query.model != self.default_model:
            self.llm.model_name = query.model
        
        t_llm, p_llm = _anthropic_temperature_top_p(query.temperature, query.top_p)
        self.llm.temperature = t_llm
        self.llm.top_p = p_llm
        if query.max_tokens:
            self.llm.max_tokens = query.max_tokens

        # Make the API call with exponential backoff for rate-limit / transient errors
        response = retry_with_backoff(self.llm.invoke, messages)
        
        # Extract usage information
        usage = {
            'input_tokens': getattr(response, 'input_tokens', 0),
            'output_tokens': getattr(response, 'output_tokens', 0),
            'total_tokens': getattr(response, 'input_tokens', 0) + getattr(response, 'output_tokens', 0)
        }
        
        # Prepare metadata
        metadata = self._prepare_metadata(query, {
            'response_object': str(response),
            'model_name': model_name
        })
        
        return LLMResponse(
            content=response.content,
            model=model_name,
            usage=usage,
            metadata=metadata,
            finish_reason=getattr(response, 'stop_reason', None)
        )
    
    def call_api_batch(self, queries: List[Query]) -> List[LLMResponse]:
        """
        Concurrent batch using LangChain's batch() method, which fires all
        requests in parallel threads and returns results in the same order.
        """
        if not queries:
            return []

        messages_list = []
        for query in queries:
            msgs = []
            if query.system_message:
                msgs.append(SystemMessage(content=query.system_message))
            msgs.append(HumanMessage(content=query.prompt))
            messages_list.append(msgs)

        # Batch call with exponential backoff for rate-limit / transient errors
        raw_responses = retry_with_backoff(self.llm.batch, messages_list)

        results = []
        for query, response in zip(queries, raw_responses):
            model_name = query.model or self.default_model
            usage = {
                "input_tokens": getattr(response, "input_tokens", 0),
                "output_tokens": getattr(response, "output_tokens", 0),
                "total_tokens": getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0),
            }
            metadata = self._prepare_metadata(
                query, {"response_object": str(response), "model_name": model_name}
            )
            results.append(LLMResponse(
                content=response.content,
                model=model_name,
                usage=usage,
                metadata=metadata,
                finish_reason=getattr(response, "stop_reason", None),
            ))
        return results

    def get_available_models(self) -> List[str]:
        """
        Get list of available Anthropic models
        
        Returns:
            List of model names
        """
        # Hints only; validate_query does not require models to be in this list.
        return [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
        ]
    
    def validate_query(self, query: Query) -> bool:
        """
        Validate if the query is compatible with Anthropic provider
        
        Args:
            query: Query to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not query.prompt or not isinstance(query.prompt, str):
            return False
        
        if query.model is not None and not isinstance(query.model, str):
            return False

        if query.temperature < 0 or query.temperature > 1:
            return False
        
        if query.max_tokens and (
            query.max_tokens < 1 or query.max_tokens > 4096
        ):
            return False

        return True
