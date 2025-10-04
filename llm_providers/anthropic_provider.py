"""
Anthropic provider implementation using LangChain
"""

from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from .base import BaseLLMProvider, Query, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic provider implementation using LangChain's ChatAnthropic
    """
    
    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        """
        Initialize Anthropic provider
        
        Args:
            api_key: Anthropic API key
            model: Default model (e.g., 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307')
            **kwargs: Additional Anthropic-specific configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model_name=model or 'claude-3-sonnet-20240229',
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000),
            top_p=kwargs.get('top_p', 1.0),
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
        
        # Update parameters if specified in query
        if query.temperature != 0.7:
            self.llm.temperature = query.temperature
        if query.max_tokens:
            self.llm.max_tokens = query.max_tokens
        if query.top_p != 1.0:
            self.llm.top_p = query.top_p
        
        # Make the API call
        response = self.llm.invoke(messages)
        
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
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Anthropic models
        
        Returns:
            List of model names
        """
        return [
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307',
            'claude-2.1',
            'claude-2.0',
            'claude-instant-1.2'
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
        
        if query.model and query.model not in self.get_available_models():
            return False
        
        if query.temperature < 0 or query.temperature > 1:
            return False
        
        if query.max_tokens and (query.max_tokens < 1 or query.max_tokens > 4096):
            return False
        
        return True
