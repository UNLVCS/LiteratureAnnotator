"""
Simple test script to verify the LLM provider interface works correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base import BaseLLMProvider, Query, LLMResponse
from openai_provider import OpenAIProvider
from anthropic_provider import AnthropicProvider
from huggingface_provider import HuggingFaceProvider


def test_query_creation():
    """Test Query object creation and validation"""
    print("Testing Query creation...")
    
    # Test basic query
    query1 = Query(prompt="Hello, world!")
    assert query1.prompt == "Hello, world!"
    assert query1.temperature == 0.7
    assert query1.max_tokens is None
    
    # Test query with all parameters
    query2 = Query(
        prompt="Test prompt",
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=100,
        system_message="You are a helpful assistant"
    )
    assert query2.prompt == "Test prompt"
    assert query2.model == "gpt-3.5-turbo"
    assert query2.temperature == 0.5
    assert query2.max_tokens == 100
    assert query2.system_message == "You are a helpful assistant"
    
    print("✓ Query creation tests passed")


def test_provider_initialization():
    """Test provider initialization"""
    print("Testing provider initialization...")
    
    # Test OpenAI provider
    openai_provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
    assert openai_provider.api_key == "test-key"
    assert openai_provider.default_model == "gpt-3.5-turbo"
    assert openai_provider.get_provider_name() == "openai"
    
    # Test Anthropic provider
    anthropic_provider = AnthropicProvider(api_key="test-key", model="claude-3-sonnet-20240229")
    assert anthropic_provider.api_key == "test-key"
    assert anthropic_provider.default_model == "claude-3-sonnet-20240229"
    assert anthropic_provider.get_provider_name() == "anthropic"
    
    # Test Hugging Face provider
    hf_provider = HuggingFaceProvider(api_key="test-key", model="microsoft/DialoGPT-medium")
    assert hf_provider.api_key == "test-key"
    assert hf_provider.default_model == "microsoft/DialoGPT-medium"
    assert hf_provider.get_provider_name() == "huggingface"
    
    print("✓ Provider initialization tests passed")


def test_query_validation():
    """Test query validation across providers"""
    print("Testing query validation...")
    
    # Create providers
    openai_provider = OpenAIProvider(api_key="test-key")
    anthropic_provider = AnthropicProvider(api_key="test-key")
    hf_provider = HuggingFaceProvider(api_key="test-key")
    
    # Valid query
    valid_query = Query(prompt="Valid prompt", temperature=0.5)
    assert openai_provider.validate_query(valid_query) == True
    assert anthropic_provider.validate_query(valid_query) == True
    assert hf_provider.validate_query(valid_query) == True
    
    # Invalid query - empty prompt
    invalid_query1 = Query(prompt="", temperature=0.5)
    assert openai_provider.validate_query(invalid_query1) == False
    assert anthropic_provider.validate_query(invalid_query1) == False
    assert hf_provider.validate_query(invalid_query1) == False
    
    # Invalid query - temperature too high for Anthropic
    invalid_query2 = Query(prompt="Valid prompt", temperature=1.5)
    assert openai_provider.validate_query(invalid_query2) == True  # OpenAI allows up to 2.0
    assert anthropic_provider.validate_query(invalid_query2) == False  # Anthropic allows up to 1.0
    assert hf_provider.validate_query(invalid_query2) == True  # HF allows up to 2.0
    
    # Invalid query - unknown model
    invalid_query3 = Query(prompt="Valid prompt", model="unknown-model")
    assert openai_provider.validate_query(invalid_query3) == False
    assert anthropic_provider.validate_query(invalid_query3) == False
    assert hf_provider.validate_query(invalid_query3) == False
    
    print("✓ Query validation tests passed")


def test_available_models():
    """Test getting available models"""
    print("Testing available models...")
    
    openai_provider = OpenAIProvider(api_key="test-key")
    anthropic_provider = AnthropicProvider(api_key="test-key")
    hf_provider = HuggingFaceProvider(api_key="test-key")
    
    # Test that we get lists of models
    openai_models = openai_provider.get_available_models()
    anthropic_models = anthropic_provider.get_available_models()
    hf_models = hf_provider.get_available_models()
    
    assert isinstance(openai_models, list)
    assert isinstance(anthropic_models, list)
    assert isinstance(hf_models, list)
    
    # Test that expected models are present
    assert "gpt-3.5-turbo" in openai_models
    assert "gpt-4" in openai_models
    assert "claude-3-sonnet-20240229" in anthropic_models
    assert "microsoft/DialoGPT-medium" in hf_models
    
    print("✓ Available models tests passed")


def test_metadata_preparation():
    """Test metadata preparation"""
    print("Testing metadata preparation...")
    
    provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
    query = Query(prompt="Test", metadata={"test": "value"})
    response_data = {"test_response": "data"}
    
    metadata = provider._prepare_metadata(query, response_data)
    
    assert metadata["provider"] == "openai"
    assert metadata["model"] == "gpt-3.5-turbo"
    assert metadata["query_metadata"]["test"] == "value"
    assert metadata["raw_response"]["test_response"] == "data"
    
    print("✓ Metadata preparation tests passed")


def main():
    """Run all tests"""
    print("Running LLM Provider Interface Tests")
    print("=" * 40)
    
    try:
        test_query_creation()
        test_provider_initialization()
        test_query_validation()
        test_available_models()
        test_metadata_preparation()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! The interface is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
