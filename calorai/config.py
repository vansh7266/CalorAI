"""Application configuration.

This is the only module that reads environment variables. Everything else asks
`get_settings()` for what it needs.

The `MODEL_PROFILE` env var picks a provider (sarvam / openai / anthropic /
google / groq). Sarvam ships working model names because that is what the
project was built and tested against; for the other providers you set
`TEXT_MODEL` and `VISION_MODEL` yourself, since hosted model names change often
and a stale default would only fail in a confusing way later.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# --- Paths ---

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _os_user_data_dir() -> Path:
    """A per-user, writable data directory for the current OS."""
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        return Path(base) / "CalorAI" if base else Path.home() / "AppData" / "Local" / "CalorAI"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "CalorAI"
    xdg = os.getenv("XDG_DATA_HOME")
    return Path(xdg) / "calorai" if xdg else Path.home() / ".local" / "share" / "calorai"


def _default_data_dir() -> Path:
    """Where to keep the database and session file when nothing is configured.

    A source checkout (the repo, the eval harness, a clean clone) keeps its data
    beside the code under ./data so everything stays self-contained. An installed
    package must not write into site-packages - which may be read-only and is the
    wrong place for user data - so it falls back to the OS user-data directory.
    """
    if (PROJECT_ROOT / "pyproject.toml").is_file():
        return PROJECT_ROOT / "data"
    return _os_user_data_dir()


DATA_DIR = Path(os.getenv("CALORAI_DATA_DIR") or _default_data_dir()).expanduser()
DB_PATH = Path(os.getenv("CALORAI_DB_PATH") or (DATA_DIR / "calorai.db")).expanduser()
SESSION_FILE = DATA_DIR / ".session"  # remembers the last-used user id

# LangGraph's checkpointer keeps its own long-lived connection and writes graph
# state after every node. Keeping that in a separate file from the application
# tables means those writes never contend with meal / memory writes.
CHECKPOINT_PATH = Path(
    os.getenv("CALORAI_CHECKPOINT_PATH") or (DB_PATH.parent / "checkpoints.db")
).expanduser()


def ensure_data_dir() -> None:
    """Create the data directory (and the DB / checkpoint parents, if set elsewhere)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


# --- Providers ---

# How each provider is reached. "auth" is one of:
#   bearer  -> Authorization: Bearer <key>       (OpenAI, Groq, OpenAI-compatible)
#   header  -> a custom header carries the key   (Sarvam)
#   native  -> the provider's LangChain package handles auth (Anthropic, Google)
PROVIDERS = {
    "sarvam": {
        "kind": "openai_compatible",
        "base_url": "https://api.sarvam.ai/v2",
        "api_key_env": "SARVAM_API_KEY",
        "auth": "header",
        "auth_header": "api-subscription-key",
    },
    "openai": {
        "kind": "openai",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "auth": "bearer",
        "auth_header": None,
    },
    "groq": {
        "kind": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "auth": "bearer",
        "auth_header": None,
    },
    "anthropic": {
        "kind": "anthropic",
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "auth": "native",
        "auth_header": None,
    },
    "google": {
        "kind": "google",
        "base_url": None,
        "api_key_env": "GOOGLE_API_KEY",
        "auth": "native",
        "auth_header": None,
    },
}

# Default (provider, model) per profile and role. A None model means the user
# must supply TEXT_MODEL / VISION_MODEL in .env for that profile.
PROFILE_DEFAULTS = {
    "sarvam": {
        "text": ("sarvam", "glm5.2"),
        "vision": ("sarvam", "gemma4"),
    },
    "openai": {
        "text": ("openai", None),
        "vision": ("openai", None),
    },
    "anthropic": {
        "text": ("anthropic", None),
        "vision": ("anthropic", None),
    },
    "google": {
        "text": ("google", None),
        "vision": ("google", None),
    },
    "groq": {
        "text": ("groq", None),
        "vision": ("groq", None),
    },
}

_TRUTHY = {"1", "true", "yes", "on"}


class ConfigError(RuntimeError):
    """Configuration is missing or inconsistent. Raised early with a clear message."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything the gateway needs to build one chat model."""

    role: str  # "text" | "vision" | "worker"
    provider: str
    model: str
    api_key: str
    kind: str
    base_url: str | None = None
    auth: str = "bearer"
    auth_header: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def api_key_env(self) -> str:
        return PROVIDERS[self.provider]["api_key_env"]

    def require_usable(self) -> None:
        """Raise if this spec can't actually reach a model. Called at build time."""
        if not self.model:
            raise ConfigError(
                f"No model set for the {self.role} path. Set {self.role.upper()}_MODEL in your "
                f".env (the '{self.provider}' provider has no built-in default)."
            )
        if not self.api_key:
            raise ConfigError(
                f"Missing {self.api_key_env} for the '{self.provider}' provider "
                f"(needed for the {self.role} model). Add it to your .env file."
            )

    def describe(self) -> str:
        state = "ok" if (self.model and self.api_key) else "not configured"
        return f"{self.role} = {self.provider}/{self.model or '?'} [{state}]"


def _read(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _make_spec(role: str, provider: str, model: str | None) -> ModelSpec:
    """Assemble a ModelSpec. A missing model name or API key is NOT fatal here -
    the spec carries an empty value and the error is raised at model-build time
    (`models.gateway.build_chat_model`). This keeps `get_settings()` usable in
    tests and tooling that never actually call a model."""
    if provider not in PROVIDERS:
        valid = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"Unknown provider '{provider}' for the {role} model. Use one of: {valid}.")

    provider_info = PROVIDERS[provider]
    api_key = _read(provider_info["api_key_env"]) or ""

    headers: dict[str, str] = {}
    if provider_info["auth"] == "header" and provider_info["auth_header"] and api_key:
        headers[provider_info["auth_header"]] = api_key

    return ModelSpec(
        role=role,
        provider=provider,
        model=model or "",
        api_key=api_key,
        kind=provider_info["kind"],
        base_url=provider_info["base_url"],
        auth=provider_info["auth"],
        auth_header=provider_info["auth_header"],
        extra_headers=headers,
    )


def _resolve_role(role: str, profile: str, fallback: tuple[str, str] | None = None) -> ModelSpec:
    """Resolve one model role from env overrides, the profile, then an optional fallback."""
    if fallback is not None:
        default_provider, default_model = fallback
    else:
        if profile not in PROFILE_DEFAULTS:
            valid = ", ".join(sorted(PROFILE_DEFAULTS))
            raise ConfigError(f"Unknown MODEL_PROFILE '{profile}'. Use one of: {valid}.")
        default_provider, default_model = PROFILE_DEFAULTS[profile][role]

    provider = _read(f"{role.upper()}_PROVIDER", default_provider)
    model = _read(f"{role.upper()}_MODEL", default_model)
    return _make_spec(role, provider, model)


@dataclass(frozen=True)
class Settings:
    profile: str
    text_model: ModelSpec
    vision_model: ModelSpec
    worker_model: ModelSpec  # background work: reflection pass, nutrition fallback

    agent_max_loops: int
    request_timeout_seconds: int

    langsmith_tracing: bool
    langsmith_project: str

    def summary(self) -> str:
        lines = [
            f"profile: {self.profile}",
            f"  {self.text_model.describe()}",
            f"  {self.vision_model.describe()}",
            f"  {self.worker_model.describe()}",
            f"agent max loops: {self.agent_max_loops}",
            f"request timeout: {self.request_timeout_seconds}s",
            f"langsmith tracing: {self.langsmith_tracing}",
        ]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    profile = _read("MODEL_PROFILE", "sarvam")

    text_model = _resolve_role("text", profile)
    vision_model = _resolve_role("vision", profile)
    worker_model = _resolve_role("worker", profile, fallback=(text_model.provider, text_model.model))

    tracing = (_read("LANGSMITH_TRACING", "false") or "false").lower() in _TRUTHY
    if tracing:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGSMITH_TRACING"] = "true"

    try:
        max_loops = int(_read("AGENT_MAX_LOOPS", "4"))
        timeout = int(_read("REQUEST_TIMEOUT_SECONDS", "60"))
    except ValueError:
        raise ConfigError("AGENT_MAX_LOOPS and REQUEST_TIMEOUT_SECONDS must be whole numbers.")

    if max_loops < 1:
        raise ConfigError("AGENT_MAX_LOOPS must be at least 1.")

    return Settings(
        profile=profile,
        text_model=text_model,
        vision_model=vision_model,
        worker_model=worker_model,
        agent_max_loops=max_loops,
        request_timeout_seconds=timeout,
        langsmith_tracing=tracing,
        langsmith_project=_read("LANGSMITH_PROJECT", "calorai-logging-agent"),
    )
