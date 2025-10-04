"""
Example usage of the LLM Provider Interface

This script demonstrates how to use different LLM providers
with the unified interface.
"""

import os
from llm_providers import (
    OpenAIProvider, 
    AnthropicProvider, 
    HuggingFaceProvider,
    Query
)


def main():
    """
    Example usage of different LLM providers
    """
    
    # Example 1: OpenAI Provider
    print("=== OpenAI Provider Example ===")
    try:
        # Initialize OpenAI provider (you'll need to set OPENAI_API_KEY environment variable)
        openai_provider = OpenAIProvider(
            api_key=os.getenv('OPENAI_API_KEY', 'your-openai-key-here'),
            model='gpt-3.5-turbo',
            temperature=0.7
        )
        
        # Create a query
        query = Query(
            prompt="Explain the concept of machine learning in simple terms.",
            system_message="You are a helpful AI assistant that explains complex topics simply.",
            temperature=0.5,
            max_tokens=150
        )
        
        # Call the API
        response = openai_provider.call_api(query)
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print()
        
    except Exception as e:
        print(f"OpenAI example failed: {e}")
        print()
    
    # Example 2: Anthropic Provider
    print("=== Anthropic Provider Example ===")
    try:
        # Initialize Anthropic provider (you'll need to set ANTHROPIC_API_KEY environment variable)
        anthropic_provider = AnthropicProvider(
            api_key=os.getenv('ANTHROPIC_API_KEY', 'your-anthropic-key-here'),
            model='claude-3-sonnet-20240229',
            temperature=0.7
        )
        
        # Create a query
        query = Query(
            prompt="What are the key differences between supervised and unsupervised learning?",
            system_message="You are an expert in machine learning. Provide clear, concise explanations.",
            temperature=0.3,
            max_tokens=200
        )
        
        # Call the API
        response = anthropic_provider.call_api(query)
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print()
        
    except Exception as e:
        print(f"Anthropic example failed: {e}")
        print()
    
    # Example 3: Hugging Face Provider
    print("=== Hugging Face Provider Example ===")
    try:
        # Initialize Hugging Face provider (you'll need to set HUGGINGFACE_API_TOKEN environment variable)
        hf_provider = HuggingFaceProvider(
            api_key=os.getenv('HUGGINGFACE_API_TOKEN', 'your-hf-token-here'),
            model='microsoft/DialoGPT-medium',
            temperature=0.7
        )
        
        # Create a query
        query = Query(
            prompt="Hello, how are you today?",
            temperature=0.8,
            max_tokens=50
        )
        
        # Call the API
        response = hf_provider.call_api(query)
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print()
        
    except Exception as e:
        print(f"Hugging Face example failed: {e}")
        print()
    
    # Example 4: Provider-agnostic usage
    print("=== Provider-Agnostic Usage Example ===")
    
    # List of providers to try
    providers = []
    
    if os.getenv('OPENAI_API_KEY'):
        providers.append(OpenAIProvider(os.getenv('OPENAI_API_KEY')))
    if os.getenv('ANTHROPIC_API_KEY'):
        providers.append(AnthropicProvider(os.getenv('ANTHROPIC_API_KEY')))
    if os.getenv('HUGGINGFACE_API_TOKEN'):
        providers.append(HuggingFaceProvider(os.getenv('HUGGINGFACE_API_TOKEN')))
    
    if not providers:
        print("No API keys found. Please set environment variables:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY") 
        print("- HUGGINGFACE_API_TOKEN")
        return
    
    # Create a query
    query = Query(
        prompt="Write a short poem about artificial intelligence.",
        temperature=0.9,
        max_tokens=100
    )
    
    # Try each provider
    for provider in providers:
        try:
            print(f"Trying {provider.get_provider_name()}...")
            response = provider.call_api(query)
            print(f"Provider: {provider.get_provider_name()}")
            print(f"Model: {response.model}")
            print(f"Response: {response.content}")
            print(f"Usage: {response.usage}")
            print("-" * 50)
        except Exception as e:
            print(f"Provider {provider.get_provider_name()} failed: {e}")
            print("-" * 50)


def demonstrate_query_validation():
    """
    Demonstrate query validation across different providers
    """
    print("\n=== Query Validation Example ===")
    
    # Create providers
    openai_provider = OpenAIProvider(api_key="dummy-key")
    anthropic_provider = AnthropicProvider(api_key="dummy-key")
    hf_provider = HuggingFaceProvider(api_key="dummy-key")
    
    # Test queries
    test_queries = [
        Query(prompt="Valid query", temperature=0.5),
        Query(prompt="", temperature=0.5),  # Invalid: empty prompt
        Query(prompt="Valid query", temperature=2.5),  # Invalid: temperature too high for Anthropic
        Query(prompt="Valid query", model="invalid-model"),  # Invalid: unknown model
    ]
    
    providers = [
        ("OpenAI", openai_provider),
        ("Anthropic", anthropic_provider),
        ("Hugging Face", hf_provider)
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\nTest Query {i+1}: {query.prompt or 'Empty prompt'}")
        for name, provider in providers:
            is_valid = provider.validate_query(query)
            print(f"  {name}: {'✓ Valid' if is_valid else '✗ Invalid'}")


if __name__ == "__main__":
    main()
    demonstrate_query_validation()
