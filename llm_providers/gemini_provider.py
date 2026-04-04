"""
Google Gemini provider (chat) via LangChain ``ChatGoogleGenerativeAI``.
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import BaseLLMProvider, LLMResponse, Query


# Default chat model: fast, widely available on the Gemini API / AI Studio.
_DEFAULT_MODEL = "gemini-2.0-flash"


def _message_content_to_str(content: Any) -> str:
    """Gemini / LangChain may return ``AIMessage.content`` as str or a list of blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(_message_content_to_str(text))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _usage_from_response(response: Any) -> Dict[str, Any]:
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if isinstance(um, dict):
        inp = int(um.get("input_tokens") or 0)
        out = int(um.get("output_tokens") or 0)
        tot = int(um.get("total_tokens") or (inp + out))
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}
    inp = int(getattr(um, "input_tokens", 0) or 0)
    out = int(getattr(um, "output_tokens", 0) or 0)
    tot = int(getattr(um, "total_tokens", 0) or (inp + out))
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


class GeminiProvider(BaseLLMProvider):
    """Gemini chat using ``langchain_google_genai.ChatGoogleGenerativeAI``."""

    def __init__(self, api_key: str, model: Optional[str] = None, **kwargs: Any):
        if not (api_key and str(api_key).strip()):
            raise ValueError(
                "GeminiProvider requires a non-empty api_key "
                "(set llm_providers.gemini.api_key or per-model api_key in your config)."
            )
        super().__init__(api_key, model, **kwargs)
        self.llm = ChatGoogleGenerativeAI(
            model=model or _DEFAULT_MODEL,
            google_api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p", 1.0),
        )

    def call_api(self, query: Query) -> LLMResponse:
        if not self.validate_query(query):
            raise ValueError("Invalid query for Gemini provider")

        messages: List[Any] = []
        if query.system_message:
            messages.append(SystemMessage(content=query.system_message))
        messages.append(HumanMessage(content=query.prompt))

        model_name = query.model or self.default_model or _DEFAULT_MODEL
        self.llm.model = model_name

        self.llm.temperature = query.temperature
        if query.max_tokens is not None:
            self.llm.max_output_tokens = query.max_tokens
        if query.top_p != 1.0:
            self.llm.top_p = query.top_p

        response = self.llm.invoke(messages)
        text = _message_content_to_str(getattr(response, "content", ""))

        usage = _usage_from_response(response)

        metadata = self._prepare_metadata(
            query,
            {"response_object": str(response), "model_name": model_name},
        )

        finish = None
        rm = getattr(response, "response_metadata", None) or {}
        if isinstance(rm, dict):
            finish = rm.get("finish_reason")

        return LLMResponse(
            content=text,
            model=model_name,
            usage=usage,
            metadata=metadata,
            finish_reason=finish,
        )

    def get_available_models(self) -> List[str]:
        return [
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.5-flash-preview-05-20",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

    def validate_query(self, query: Query) -> bool:
        if not query.prompt or not isinstance(query.prompt, str):
            return False
        if query.model is not None and not isinstance(query.model, str):
            return False
        if query.temperature < 0 or query.temperature > 2:
            return False
        if query.max_tokens is not None and query.max_tokens < 1:
            return False
        return True
