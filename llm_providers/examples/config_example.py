"""
Example: Load typed LLM config from YAML and run a query on each model.

Copy llm_params_example.yaml.example to llm_params_example.yaml and add your API keys.
Then: load_config_from_yaml_file(LLMProvidersConfig, path).to_dict() -> Dict[str, BaseLLMProvider]
"""

import sys
from dataclasses import replace
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config import load_config_from_yaml_file
from llm_providers.base import Query
from llm_providers.config_models import LLMProvidersConfig


def main():
    config_path = Path(__file__).parent / "llm_params_example.yaml"
    if not config_path.exists():
        print("Copy llm_params_example.yaml.example to llm_params_example.yaml and add your API keys.")
        return
    providers = load_config_from_yaml_file(LLMProvidersConfig, config_path).to_dict()

    if not providers:
        print("No providers available. Check config and API keys.")
        return

    query = Query(
        prompt="Say 'hello' in one word.",
        temperature=0.3,
        max_tokens=50,
    )

    for model_name, provider in providers.items():
        try:
            query_with_model = replace(query, model=model_name)
            response = provider.call_api(query_with_model)
            print(f"[{model_name}] {response.content.strip()}")
        except Exception as e:
            print(f"[{model_name}] Error: {e}")


if __name__ == "__main__":
    main()
