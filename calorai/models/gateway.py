"""Model gateway.

Turns a `ModelSpec` (from `config.py`) into a ready LangChain chat model. This
is the only module that imports provider SDKs, so the rest of the app never
cares which provider is in play.

Providers:
  * openai / openai_compatible -> ChatOpenAI (Sarvam and Groq are OpenAI-shaped;
    Sarvam authenticates with an `api-subscription-key` header instead of a
    bearer token, which is passed through `default_headers`).
  * anthropic -> ChatAnthropic
  * google    -> ChatGoogleGenerativeAI
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from calorai.config import ModelSpec, get_settings

# A non-empty placeholder for providers whose key travels in a header, not the
# Authorization bearer slot. The OpenAI client rejects an empty api_key.
_HEADER_AUTH_PLACEHOLDER = "header-auth"


def build_chat_model(
    spec: ModelSpec,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    settings = get_settings()
    timeout = settings.request_timeout_seconds

    if spec.kind in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI

        uses_header_auth = spec.auth == "header"
        return ChatOpenAI(
            model=spec.model,
            api_key=_HEADER_AUTH_PLACEHOLDER if uses_header_auth else spec.api_key,
            base_url=spec.base_url,
            default_headers=spec.extra_headers or None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=2,
            streaming=streaming,
        )

    if spec.kind == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=spec.model,
            api_key=spec.api_key,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
            timeout=timeout,
            max_retries=2,
            streaming=streaming,
        )

    if spec.kind == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=spec.model,
            google_api_key=spec.api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=timeout,
            max_retries=2,
        )

    raise ValueError(f"Unsupported model kind: {spec.kind!r}")


@lru_cache(maxsize=8)
def _cached_model(role: str, temperature: float, max_tokens: int | None, streaming: bool) -> BaseChatModel:
    settings = get_settings()
    spec = {
        "text": settings.text_model,
        "vision": settings.vision_model,
        "worker": settings.worker_model,
    }[role]
    return build_chat_model(spec, temperature=temperature, max_tokens=max_tokens, streaming=streaming)


def get_text_model(*, temperature: float = 0.2, max_tokens: int = 1400, streaming: bool = False) -> BaseChatModel:
    """The conversational agent brain (tool calling)."""
    return _cached_model("text", temperature, max_tokens, streaming)


def get_vision_model(*, temperature: float = 0.0, max_tokens: int = 900) -> BaseChatModel:
    """The photo -> structured-items extractor. Separate model from the text path."""
    return _cached_model("vision", temperature, max_tokens, False)


def get_worker_model(*, temperature: float = 0.0, max_tokens: int = 700) -> BaseChatModel:
    """Background work: the reflection pass and nutrition-estimate fallback."""
    return _cached_model("worker", temperature, max_tokens, False)
