"""
Example: Load typed LLM config from YAML and run a query on each model.

Copy env.yaml.example to env.yaml and add your API keys.
Then: load_config_from_yaml_file(LLMProvidersConfig, path).to_dict() -> Dict[str, BaseLLMProvider]
"""

import sys
from pathlib import Path
from typing import Dict

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import load_config_from_yaml_file
from llm_providers.base import BaseLLMProvider, Query
from llm_providers.config_models import LLMProvidersConfig


def main():
    config_path = Path(__file__).parent / "env.yaml"
    if not config_path.exists():
        print("Copy env.yaml.example to env.yaml and add your API keys.")
        return
    providers: Dict[str, BaseLLMProvider] = load_config_from_yaml_file(
        LLMProvidersConfig, config_path
    ).get_providers_dict()

    if not providers:
        print("No providers available. Check config and API keys.")
        return

    for model_name, provider in providers.items():
        try:
            query = Query(
                prompt="Say 'hello' in one word.",
                model=model_name,
                temperature=0.3,
                max_tokens=50,
            )
            response = provider.call_api(query)
            print(f"[{model_name}] {response.content.strip()}")
        except Exception as e:
            print(f"[{model_name}] Error: {e}")


if __name__ == "__main__":
    main()
