"""Model gateway construction tests (no live API calls)."""

from __future__ import annotations

import pytest

from calorai.config import ModelSpec
from calorai.models.gateway import build_chat_model


def _spec(kind: str, *, provider: str, auth: str = "bearer", base_url: str | None = None, headers=None) -> ModelSpec:
    return ModelSpec(
        role="text",
        provider=provider,
        model="test-model",
        api_key="secret-key-1234",
        kind=kind,
        base_url=base_url,
        auth=auth,
        auth_header="api-subscription-key" if auth == "header" else None,
        extra_headers=headers or ({"api-subscription-key": "secret-key-1234"} if auth == "header" else {}),
    )


def test_openai_compatible_sarvam_uses_header_auth():
    from langchain_openai import ChatOpenAI

    model = build_chat_model(
        _spec("openai_compatible", provider="sarvam", auth="header", base_url="https://api.sarvam.ai/v2")
    )
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "test-model"
    assert str(model.openai_api_base) == "https://api.sarvam.ai/v2"
    # real key travels in the header, not the bearer slot
    assert model.default_headers == {"api-subscription-key": "secret-key-1234"}
    assert model.openai_api_key.get_secret_value() == "header-auth"


def test_openai_uses_bearer_key():
    from langchain_openai import ChatOpenAI

    model = build_chat_model(_spec("openai", provider="openai"))
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_key.get_secret_value() == "secret-key-1234"
    assert not model.default_headers


def test_groq_is_openai_compatible_with_base_url():
    from langchain_openai import ChatOpenAI

    model = build_chat_model(
        _spec("openai_compatible", provider="groq", auth="bearer", base_url="https://api.groq.com/openai/v1")
    )
    assert isinstance(model, ChatOpenAI)
    assert str(model.openai_api_base) == "https://api.groq.com/openai/v1"
    assert model.openai_api_key.get_secret_value() == "secret-key-1234"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_chat_model(_spec("mystery", provider="???"))


def test_as_structured_builds_runnable():
    from pydantic import BaseModel

    from calorai.models.gateway import as_structured

    class Shape(BaseModel):
        n: int

    model = build_chat_model(_spec("openai", provider="openai"))
    runnable = as_structured(model, Shape)
    assert hasattr(runnable, "invoke")  # built without an API call
