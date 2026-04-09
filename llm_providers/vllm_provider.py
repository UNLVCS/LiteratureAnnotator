"""
vLLM provider implementation using LangChain's ChatOpenAI interface
"""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .base import BaseLLMProvider, Query, LLMResponse


class VLLMProvider(BaseLLMProvider):
    """
    vLLM provider implementation using LangChain's ChatOpenAI client.
    Connects to a vLLM server running at the specified base_url.
    """
    
    def __init__(self, api_key: str = "EMPTY", model: Optional[str] = None, base_url: str = "http://localhost:8000/v1", **kwargs):
        """
        Initialize vLLM provider
        
        Args:
            api_key: API key (usually "EMPTY" for vLLM)
            model: Default model to use
            base_url: URL of the vLLM server (e.g., "http://node05:8000/v1")
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url
        self.default_model = model
        
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=base_url,
            model_name=model,
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens'),
            top_p=kwargs.get('top_p', 1.0),
            frequency_penalty=kwargs.get('frequency_penalty', 0.0),
            presence_penalty=kwargs.get('presence_penalty', 0.0),
        )
    
    def call_api(self, query: Query) -> LLMResponse:
        """
        Call vLLM API with the provided query
        """
        if not self.validate_query(query):
            raise ValueError("Invalid query for vLLM provider")
        
        # Prepare messages
        messages = []
        if query.system_message:
            messages.append(SystemMessage(content=query.system_message))
        messages.append(HumanMessage(content=query.prompt))
        
        # Update model if specified in query
        model_name = query.model or self.default_model
        if query.model and query.model != self.default_model:
            self.llm.model_name = query.model
        
        # Update parameters if specified in query
        if query.temperature != 0.7:
            self.llm.temperature = query.temperature
        if query.max_tokens:
            self.llm.max_tokens = query.max_tokens
        if query.top_p != 1.0:
            self.llm.top_p = query.top_p
        
        # Make the API call
        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            raise RuntimeError(f"vLLM API call failed: {str(e)}")
        
        # Extract usage information
        usage = {
            'prompt_tokens': getattr(response, 'prompt_tokens', 0),
            'completion_tokens': getattr(response, 'completion_tokens', 0),
            'total_tokens': getattr(response, 'total_tokens', 0)
        }
        
        # Prepare metadata
        metadata = self._prepare_metadata(query, {
            'response_object': str(response),
            'model_name': model_name,
            'base_url': self.base_url
        })
        
        return LLMResponse(
            content=response.content,
            model=model_name,
            usage=usage,
            metadata=metadata,
            finish_reason=getattr(response, 'finish_reason', None)
        )
    
    def call_api_batch(self, queries: List[Query]) -> List[LLMResponse]:
        """
        Concurrent batch using LangChain's batch() method, which fires all
        requests to the vLLM HTTP server in parallel and returns results in order.
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

        try:
            raw_responses = self.llm.batch(messages_list)
        except Exception as e:
            raise RuntimeError(f"vLLM batch API call failed: {str(e)}")

        results = []
        for query, response in zip(queries, raw_responses):
            model_name = query.model or self.default_model
            usage = {
                "prompt_tokens": getattr(response, "prompt_tokens", 0),
                "completion_tokens": getattr(response, "completion_tokens", 0),
                "total_tokens": getattr(response, "total_tokens", 0),
            }
            metadata = self._prepare_metadata(query, {
                "response_object": str(response),
                "model_name": model_name,
                "base_url": self.base_url,
            })
            results.append(LLMResponse(
                content=response.content,
                model=model_name,
                usage=usage,
                metadata=metadata,
                finish_reason=getattr(response, "finish_reason", None),
            ))
        return results

    def get_available_models(self) -> List[str]:
        """
        Get list of available models from vLLM server (optional implementation)
        """
        # We could query /v1/models here, but for now just return the configured model
        return [self.default_model] if self.default_model else []
    
    def validate_query(self, query: Query) -> bool:
        """
        Validate if the query is compatible with vLLM provider
        """
        if not query.prompt or not isinstance(query.prompt, str):
            return False
        
        # vLLM supports 0-2 temperature range generally
        if query.temperature < 0 or query.temperature > 2:
            return False
        
        return True
