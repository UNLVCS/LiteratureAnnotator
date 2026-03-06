#!/usr/bin/env python3
"""
Smoke test for VLLMNativeProvider.

Loads the vllm_native config from llm_params3.json, initialises the provider,
sends a simple "say hello" prompt, and prints the response.

Usage:
    # from repo root
    python llm_providers/test_vllm_native.py

    # override model and tensor parallel size via env
    MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
    TENSOR_PARALLEL_SIZE=2 \
    python llm_providers/test_vllm_native.py
"""

import os
import re
import json
import sys
from pathlib import Path

# Make sure repo root is on the path so relative imports work
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from llm_providers.vllm_native_provider import VLLMNativeProvider
from llm_providers.base import Query


# ── helpers ──────────────────────────────────────────────────────────────────

def load_params(path: Path) -> dict:
    """Load JSON params file, substituting ${VAR} placeholders from env."""
    raw = path.read_text()
    resolved = re.sub(
        r'\$\{(\w+)\}',
        lambda m: os.getenv(m.group(1), m.group(0)),
        raw
    )
    return json.loads(resolved)


def coerce_types(config: dict) -> dict:
    """Cast fields that must not be strings."""
    if "tensor_parallel_size" in config:
        config["tensor_parallel_size"] = int(config["tensor_parallel_size"])
    if "gpu_memory_utilization" in config:
        config["gpu_memory_utilization"] = float(config["gpu_memory_utilization"])
    return config


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    params_path = repo_root / "llm_params" / "llm_params3.json"
    all_params = load_params(params_path)

    vllm_native_configs = all_params.get("vllm_native", [])
    active = [c for c in vllm_native_configs if not c.get("skip", False)]

    if not active:
        print("No active vllm_native config found in llm_params3.json "
              "(all entries have skip=true or section is missing).")
        sys.exit(1)

    config = coerce_types(active[0])
    model_name = config.get("model", "")

    if not model_name:
        print(
            "ERROR: 'model' is empty in the vllm_native config.\n"
            "Set it in llm_params3.json or export MODEL_NAME=<hf-model-id> "
            "before running."
        )
        sys.exit(1)

    print(f"Model          : {model_name}")
    print(f"Tensor parallel: {config.get('tensor_parallel_size', 1)}")
    print(f"dtype          : {config.get('dtype', 'auto')}")
    print(f"GPU mem util   : {config.get('gpu_memory_utilization', 0.9)}")
    print("-" * 60)

    provider = VLLMNativeProvider(**config)

    query = Query(
        prompt="Say hello and introduce yourself in one sentence.",
        system_message="You are a helpful assistant. Be concise.",
        temperature=0.7,
        max_tokens=64,
    )

    print("Sending prompt …")
    response = provider.call_api(query)

    print("\n── Response ──────────────────────────────────────────────")
    print(response.content)
    print("\n── Usage ─────────────────────────────────────────────────")
    print(f"  prompt tokens    : {response.usage['prompt_tokens']}")
    print(f"  completion tokens: {response.usage['completion_tokens']}")
    print(f"  total tokens     : {response.usage['total_tokens']}")
    print(f"  finish reason    : {response.finish_reason}")
    print("──────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
