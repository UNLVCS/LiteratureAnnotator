from ast import List, Str, Tuple
import os
import json
from typing import Dict, Any, Optional, List
from llm_providers import (
    BaseLLMProvider,
    OpenAIProvider, 
    AnthropicProvider, 
    HuggingFaceProvider,
    Query
)

# def cross_ref(llm: BaseLLMProvider, Candidate: Dict[any]):

#     judge_me(llm)
#     for candidate in Candidate:
#         for llm_judge in LLMs:
#             if candidate['model'] == llm_judge.get_provider_name():
#                 continue
#             judge_me(llm_judge, candidate['data'])
#     pass


    

def init_query(query_path: str) -> Query:
    with open(query_path, 'r') as f:
        query_dict = json.load(f)
    return Query(**query_dict)

def init_llms(params_path: str) -> List:
    """
    Initialize all LLM providers using the given parameter dictionary.

    Args:
        params (Dict[any]): Dictionary containing parameters for each provider.
            Example:
                {
                    "openai": {"api_key": "...", "model": "...", ...},
                    "anthropic": {"api_key": "...", "model": "...", ...},
                    "huggingface": {"api_key": "...", "model": "...", ...}
                }

    Returns:
        List: List of initialized LLM provider instances.
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
        
    llms = []
    if "openai" in params:
        openai_params = params["openai"]
        llms.append(OpenAIProvider(**openai_params))
    if "anthropic" in params:
        anthropic_params = params["anthropic"]
        llms.append(AnthropicProvider(**anthropic_params))
    if "huggingface" in params:
        hf_params = params["huggingface"]
        llms.append(HuggingFaceProvider(**hf_params))
    return llms

# def construct_query(query_path) -> Query:

def judge_me(judge_llm: BaseLLMProvider, defendant: Tuple) -> int:
    
    pass


def main():
    LLMs = init_llms(params_path = './')
    query = init_query(query_path= './')