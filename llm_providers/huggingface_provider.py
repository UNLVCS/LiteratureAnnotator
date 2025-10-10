"""
Hugging Face provider implementation using LangChain
"""

from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.schema import HumanMessage, SystemMessage
from .base import BaseLLMProvider, Query, LLMResponse


class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face provider implementation using LangChain's HuggingFaceHub
    """
    
    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs):
        """
        Initialize Hugging Face provider
        
        Args:
            api_key: Hugging Face API key
            model: Default model (e.g., 'microsoft/DialoGPT-medium', 'google/flan-t5-large')
            **kwargs: Additional HuggingFace-specific configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.llm = ChatHuggingFace(
            huggingfacehub_api_token=api_key,
            repo_id=model or 'microsoft/DialoGPT-medium',
            model_kwargs={
                'temperature': kwargs.get('temperature', 0.7),
                'max_length': kwargs.get('max_tokens', 1000),
                'top_p': kwargs.get('top_p', 1.0),
                'do_sample': True,
            }
        )
    
    def call_api(self, query: Query) -> LLMResponse:
        """
        Call Hugging Face API with the provided query
        
        Args:
            query: Query object containing the prompt and parameters
            
        Returns:
            LLMResponse object containing the response and metadata
        """
        if not self.validate_query(query):
            raise ValueError("Invalid query for Hugging Face provider")
        
        # Update model if specified in query
        model_name = query.model or self.default_model
        if query.model and query.model != self.default_model:
            self.llm.repo_id = query.model
        
        # Update parameters if specified in query
        if query.temperature != 0.7:
            self.llm.model_kwargs['temperature'] = query.temperature
        if query.max_tokens:
            self.llm.model_kwargs['max_length'] = query.max_tokens
        if query.top_p != 1.0:
            self.llm.model_kwargs['top_p'] = query.top_p
        
        # Prepare the prompt with system message if provided
        full_prompt = query.prompt
        if query.system_message:
            full_prompt = f"System: {query.system_message}\n\nHuman: {query.prompt}"
        
        # Make the API call
        response = self.llm.invoke(full_prompt)
        
        # Extract usage information (HuggingFace doesn't provide detailed usage)
        usage = {
            'input_tokens': len(full_prompt.split()),
            'output_tokens': len(response.split()) if isinstance(response, str) else 0,
            'total_tokens': len(full_prompt.split()) + (len(response.split()) if isinstance(response, str) else 0)
        }
        
        # Prepare metadata
        metadata = self._prepare_metadata(query, {
            'response_object': str(response),
            'model_name': model_name,
            'repo_id': model_name
        })
        
        return LLMResponse(
            content=response if isinstance(response, str) else str(response),
            model=model_name,
            usage=usage,
            metadata=metadata,
            finish_reason='stop'
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Hugging Face models
        
        Returns:
            List of model names
        """
        return [
            'microsoft/DialoGPT-medium',
            'microsoft/DialoGPT-large',
            'google/flan-t5-large',
            'google/flan-t5-xl',
            'facebook/blenderbot-400M-distill',
            'facebook/blenderbot-1B-distill',
            'microsoft/DialoGPT-small',
            'EleutherAI/gpt-neo-2.7B',
            'EleutherAI/gpt-j-6B',
            'bigscience/bloom-560m',
            'bigscience/bloom-1b7',
            'bigscience/bloom-3b',
            'bigscience/bloom-7b1'
        ]
    
    def validate_query(self, query: Query) -> bool:
        """
        Validate if the query is compatible with Hugging Face provider
        
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
        
        if query.max_tokens and (query.max_tokens < 1 or query.max_tokens > 1024):
            return False
        
        return True
