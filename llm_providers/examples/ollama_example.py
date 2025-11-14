"""
Example usage of OLLAMA provider
"""

from ollama_provider import OllamaProvider
from base import Query
import json
import os

def main():
    # Initialize OLLAMA provider
    # Note: OLLAMA server must be running on localhost:11434

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "llm_params.json")) as f:
        params = json.load(f)

    provider_config = params["ollama"]

    provider = OllamaProvider(**provider_config)
    
    # Check if OLLAMA server is running
    if not provider.check_server_status():
        print("Error: OLLAMA server is not running. Please start it with 'ollama serve'")
        return
    
    # Get available models
    print("Available models:")
    models = provider.get_available_models()
    for model in models[:5]:  # Show first 5 models
        print(f"  - {model}")
    
    # Create a query
    query = Query(
        prompt="Hello! Can you tell me a short joke?",
        model=provider_config["model"],  # Optional: specify model
        temperature=0.7,
        max_tokens=10000,
        system_message="You are a helpful assistant that tells clean, family-friendly jokes."
    )
    
    try:
        # Make API call
        print("\nMaking API call...")
        response = provider.call_api(query)
        
        print(f"\nResponse: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Finish reason: {response.finish_reason}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OLLAMA server is running and the model is available.")
        print(f"You can pull a model with: ollama pull {provider_config['model']}")

if __name__ == "__main__":
    main()
