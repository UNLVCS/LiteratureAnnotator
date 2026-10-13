"""Importing llm_providers or building an API provider must not load local-model backends."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAVY = ("vllm", "torch", "transformers", "langchain_huggingface")


def _loaded_after(code: str) -> set[str]:
    """Run ``code`` in a fresh interpreter and return which heavy modules it loaded."""
    probe = f"{code}\nimport sys\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return set(filter(None, out.splitlines()[-1].split(","))) if out else set()


def test_package_import_is_light():
    assert _loaded_after("import llm_providers") == set()


def test_api_provider_does_not_load_local_backends():
    code = (
        "from config.llm_providers_config import instantiate_provider\n"
        "instantiate_provider('openai', {'api_key': 'sk-test', 'model': 'gpt-4o'})"
    )
    assert _loaded_after(code) == set()


def test_attribute_access_still_works():
    code = "import llm_providers\nassert llm_providers.OpenAIProvider.__name__ == 'OpenAIProvider'"
    assert _loaded_after(code) == set()
