"""
OpenAI provider implementation using LangChain
"""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage, SystemMessage
from .base import BaseLLMProvider, Query, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation using LangChain's ChatOpenAI
    """
    
    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key
            model: Default model (e.g., 'gpt-3.5-turbo', 'gpt-4')
            **kwargs: Additional OpenAI-specific configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model or 'gpt-3.5-turbo',
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens'),
            top_p=kwargs.get('top_p', 1.0),
            frequency_penalty=kwargs.get('frequency_penalty', 0.0),
            presence_penalty=kwargs.get('presence_penalty', 0.0),
        )
    
    def call_api(self, query: Query) -> LLMResponse:
        """
        Call OpenAI API with the provided query
        
        Args:
            query: Query object containing the prompt and parameters
            
        Returns:
            LLMResponse object containing the response and metadata
        """
        if not self.validate_query(query):
            raise ValueError("Invalid query for OpenAI provider")
        
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
        if query.frequency_penalty != 0.0:
            self.llm.frequency_penalty = query.frequency_penalty
        if query.presence_penalty != 0.0:
            self.llm.presence_penalty = query.presence_penalty
        
        # Make the API call
        response = self.llm.invoke(messages)
        
        # Extract usage information
        usage = {
            'prompt_tokens': getattr(response, 'prompt_tokens', 0),
            'completion_tokens': getattr(response, 'completion_tokens', 0),
            'total_tokens': getattr(response, 'total_tokens', 0)
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
            finish_reason=getattr(response, 'finish_reason', None)
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models
        
        Returns:
            List of model names
        """
        return [
            'gpt-4',
            'gpt-4-turbo-preview',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k',
            'text-davinci-003',
            'text-davinci-002',
            'text-curie-001',
            'text-babbage-001',
            'text-ada-001'
        ]
    
    def validate_query(self, query: Query) -> bool:
        """
        Validate if the query is compatible with OpenAI provider
        
        Args:
            query: Query to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not query.prompt or not isinstance(query.prompt, str):
            return False
        
        if query.model and query.model not in self.get_available_models():
            return False
        
        if query.temperature < 0 or query.temperature > 2:
            return False
        
        if query.max_tokens and (query.max_tokens < 1 or query.max_tokens > 4096):
            return False
        
        return True
