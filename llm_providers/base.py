"""
Base classes for LLM provider interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Enum for different types of models"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


@dataclass
class Query:
    """Standardized query structure for LLM providers"""
    prompt: Dict[str]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    system_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def stringify_prompt(self):
        self.prompt = str(self.prompt)



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
