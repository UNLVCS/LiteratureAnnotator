"""
vLLM provider implementation using the native vLLM Python SDK.

Unlike vllm_provider.py (which proxies through LangChain's ChatOpenAI to a
running vLLM HTTP server), this provider loads the model directly into GPU
memory via the vLLM `LLM` class and calls `llm.generate()` offline.

Requirements:
    pip install vllm
"""

from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from .base import BaseLLMProvider, Query, LLMResponse


class VLLMNativeProvider(BaseLLMProvider):
    """
    vLLM provider using the native vLLM SDK (offline inference).

    The model is loaded once into GPU memory on construction and reused for
    every subsequent `call_api` call. SamplingParams are rebuilt per-call so
    that per-query overrides (temperature, max_tokens, etc.) are respected.
    """

    def __init__(
        self,
        model: str,
        api_key: str = "EMPTY",           # kept for interface compatibility
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        quantization: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model: HuggingFace model ID or local path (e.g. "meta-llama/Llama-3-8B-Instruct").
            api_key: Unused; kept so this provider can be swapped in anywhere
                     BaseLLMProvider is expected.
            tensor_parallel_size: Number of GPUs to shard the model across.
            dtype: Weight dtype — "auto", "float16", "bfloat16", "float32".
            gpu_memory_utilization: Fraction of GPU memory vLLM may use (0–1).
            max_model_len: Override the model's max context length (tokens).
            trust_remote_code: Allow executing custom model code from HF Hub.
            quantization: Quantization method, e.g. "awq", "gptq", "squeezellm".
            **kwargs: Forwarded to BaseLLMProvider for bookkeeping.
        """
        super().__init__(api_key=api_key, model=model, **kwargs)

        llm_kwargs: Dict[str, Any] = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if quantization is not None:
            llm_kwargs["quantization"] = quantization

        print(f"Loading model '{model}' via vLLM native SDK …")
        self.llm = LLM(**llm_kwargs)
        print(f"Model '{model}' loaded.")

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def call_api(self, query: Query) -> LLMResponse:
        """
        Generate a single response for `query` using the native vLLM SDK.

        The prompt is assembled as:
            [system_message\\n\\n] <user prompt>

        so that instruction-tuned models receive the right context even though
        vLLM's offline `generate()` works at the raw-text level.
        """
        if not self.validate_query(query):
            raise ValueError(f"Invalid query: prompt must be a non-empty string "
                             f"and temperature must be in [0, 2].")

        # Build the full prompt text
        prompt = self._build_prompt(query)

        sampling_params = SamplingParams(
            temperature=query.temperature,
            top_p=query.top_p,
            max_tokens=query.max_tokens or 512,
            frequency_penalty=query.frequency_penalty,
            presence_penalty=query.presence_penalty,
            stop=query.stop or [],
            n=1,  # one completion per call; sampling loop is handled externally
        )

        try:
            outputs = self.llm.generate([prompt], sampling_params)
        except Exception as e:
            raise RuntimeError(f"vLLM native generate() failed: {e}") from e

        request_output = outputs[0]
        completion = request_output.outputs[0]

        usage = {
            "prompt_tokens": len(request_output.prompt_token_ids),
            "completion_tokens": len(completion.token_ids),
            "total_tokens": len(request_output.prompt_token_ids) + len(completion.token_ids),
        }

        model_name = query.model or self.default_model
        metadata = self._prepare_metadata(query, {
            "finish_reason": completion.finish_reason,
            "model": model_name,
        })

        return LLMResponse(
            content=completion.text,
            model=model_name,
            usage=usage,
            metadata=metadata,
            finish_reason=completion.finish_reason,
        )

    def call_api_batch(self, queries: List[Query]) -> List[LLMResponse]:
        """
        True GPU batch: all prompts are submitted to llm.generate() in a single
        call so the engine can maximally fill its KV-cache budget across the batch.

        All queries must share the same sampling parameters (temperature, top_p,
        max_tokens, etc.). The values from the first query are used as the
        representative for SamplingParams, which is the normal case when all
        criteria are evaluated at temperature=0.1 / max_tokens=500.
        """
        if not queries:
            return []

        prompts = [self._build_prompt(q) for q in queries]
        ref = queries[0]
        sampling_params = SamplingParams(
            temperature=ref.temperature,
            top_p=ref.top_p,
            max_tokens=ref.max_tokens or 512,
            frequency_penalty=ref.frequency_penalty,
            presence_penalty=ref.presence_penalty,
            stop=ref.stop or [],
            n=1,
        )

        try:
            outputs = self.llm.generate(prompts, sampling_params)
        except Exception as e:
            raise RuntimeError(f"vLLM native batch generate() failed: {e}") from e

        responses = []
        for query, request_output in zip(queries, outputs):
            completion = request_output.outputs[0]
            usage = {
                "prompt_tokens": len(request_output.prompt_token_ids),
                "completion_tokens": len(completion.token_ids),
                "total_tokens": len(request_output.prompt_token_ids) + len(completion.token_ids),
            }
            model_name = query.model or self.default_model
            metadata = self._prepare_metadata(query, {
                "finish_reason": completion.finish_reason,
                "model": model_name,
            })
            responses.append(LLMResponse(
                content=completion.text,
                model=model_name,
                usage=usage,
                metadata=metadata,
                finish_reason=completion.finish_reason,
            ))
        return responses

    def get_available_models(self) -> List[str]:
        """Returns the single model loaded by this instance."""
        return [self.default_model] if self.default_model else []

    def validate_query(self, query: Query) -> bool:
        if not query.prompt or not isinstance(query.prompt, str):
            return False
        if not (0.0 <= query.temperature <= 2.0):
            return False
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: Query) -> str:
        """
        Combine an optional system message with the user prompt into a single
        string. For instruction-tuned models that expect a chat template you
        may want to override this or pre-format the prompt before passing it in.
        """
        if query.system_message:
            return f"{query.system_message}\n\n{query.prompt}"
        return query.prompt
