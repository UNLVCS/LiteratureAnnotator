#!/usr/bin/env python3
"""
Example usage of RAG labeling script with OLLAMA provider
"""

import os
import sys
from pathlib import Path

# Add the llm_providers to the path
sys.path.append(str(Path(__file__).parent.parent / "llm_providers"))

from llm_providers import OllamaProvider, Query

def test_ollama_connection():
    """Test OLLAMA connection and available models"""
    print("Testing OLLAMA connection...")
    
    # Initialize OLLAMA provider
    provider = OllamaProvider(
        api_key="dummy",
        model="llama3.1",
        base_url="http://localhost:11434"
    )
    
    # Check server status
    if not provider.check_server_status():
        print("❌ OLLAMA server is not running!")
        print("Please start it with: ollama serve")
        return False
    
    print("✅ OLLAMA server is running")
    
    # Get available models
    models = provider.get_available_models()
    print(f"Available models: {models[:5]}...")  # Show first 5
    
    # Test a simple query
    query = Query(
        prompt="Hello! Can you tell me a short joke?",
        temperature=0.7,
        max_tokens=100
    )
    
    try:
        response = provider.call_api(query)
        print(f"✅ Test successful! Response: {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function to test OLLAMA setup"""
    print("OLLAMA RAG Labeling Setup Test")
    print("=" * 40)
    
    if test_ollama_connection():
        print("\n🎉 OLLAMA is ready for RAG labeling!")
        print("\nTo use OLLAMA in the RAG labeling script:")
        print("1. Make sure OLLAMA server is running: ollama serve")
        print("2. Pull a model: ollama pull llama3.1")
        print("3. Run the RAG labeling script - it will automatically detect OLLAMA")
    else:
        print("\n❌ OLLAMA setup incomplete")
        print("Please ensure:")
        print("- OLLAMA is installed")
        print("- OLLAMA server is running (ollama serve)")
        print("- At least one model is pulled (ollama pull llama3.1)")

if __name__ == "__main__":
    main()
