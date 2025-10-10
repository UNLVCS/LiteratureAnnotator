"""
OLLAMA provider implementation using direct API calls
"""

import requests
import json
from typing import List, Dict, Any, Optional
from .base import BaseLLMProvider, Query, LLMResponse


class OllamaProvider(BaseLLMProvider):
    """
    OLLAMA provider implementation using direct API calls to local OLLAMA server
    """
    
    def __init__(self, api_key: str = "dummy", model: Optional[str] = None, base_url: str = "http://localhost:11434", **kwargs):
        """
        Initialize OLLAMA provider
        
        Args:
            api_key: Not used for OLLAMA (local server), but required by base class
            model: Default model (e.g., 'llama3.1', 'mistral', 'codellama')
            base_url: OLLAMA server URL (default: http://localhost:11434)
            **kwargs: Additional OLLAMA-specific configuration
        """
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.default_model = model or 'llama3.1'
        
    def call_api(self, query: Query) -> LLMResponse:
        """
        Call OLLAMA API with the provided query
        
        Args:
            query: Query object containing the prompt and parameters
            
        Returns:
            LLMResponse object containing the response and metadata
        """
        if not self.validate_query(query):
            raise ValueError("Invalid query for OLLAMA provider")
        
        # Prepare the request payload
        model_name = query.model or self.default_model
        
        payload = {
            "model": model_name,
            "prompt": query.prompt,
            "stream": False,
            "options": {
                "temperature": query.temperature,
                "top_p": query.top_p,
                "num_predict": query.max_tokens or 4000,  # Default to 4000 if not specified
                "stop": query.stop if query.stop else []
            }
        }
        
        # Add system message if provided
        if query.system_message:
            payload["system"] = query.system_message
        
        # Make the API call
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.get('timeout', 120)
            )
            response.raise_for_status()
            
            response_data = response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OLLAMA API request failed: {str(e)}")
        
        # Extract response content
        content = response_data.get("response", "")
        
        # Extract usage information
        usage = {
            'prompt_tokens': response_data.get("prompt_eval_count", 0),
            'completion_tokens': response_data.get("eval_count", 0),
            'total_tokens': response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
        }
        
        # Prepare metadata
        metadata = self._prepare_metadata(query, {
            'response_data': response_data,
            'model_name': model_name,
            'ollama_server': self.base_url
        })
        
        return LLMResponse(
            content=content,
            model=model_name,
            usage=usage,
            metadata=metadata,
            finish_reason=response_data.get("done", True) and "stop" or "length"
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OLLAMA models
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
            
        except requests.exceptions.RequestException:
            # Return common OLLAMA models if API call fails
            return [
                'llama3.1',
                'llama3.1:latest',
                'llama3',
                'llama3:latest',
                'mistral',
                'mistral:latest',
                'codellama',
                'codellama:latest',
                'phi3',
                'phi3:latest',
                'gemma',
                'gemma:latest',
                'qwen',
                'qwen:latest'
            ]
    
    def validate_query(self, query: Query) -> bool:
        """
        Validate if the query is compatible with OLLAMA provider
        
        Args:
            query: Query to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not query.prompt or not isinstance(query.prompt, str):
            return False
        
        if query.temperature < 0 or query.temperature > 2:
            return False
        
        if query.max_tokens and (query.max_tokens < 1 or query.max_tokens > 32768):
            return False
        
        if query.top_p < 0 or query.top_p > 1:
            return False
        
        return True
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from OLLAMA registry
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minutes timeout for model pulling
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException:
            return False
    
    def check_server_status(self) -> bool:
        """
        Check if OLLAMA server is running
        
        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
