"""Public LM facade wrapping the existing `reps.llm.*` providers.

This is the v1 surface — sync `__call__` / `generate` only. Async variants
and ensemble support are deferred to v1.5. The class builds an
`LLMModelConfig` from its kwargs, picks the right provider class via
`provider_of_model` (or an explicit `<provider>/<id>` prefix), and delegates
each call to the provider's async `generate`. Sync wrapping uses
`asyncio.run` so users can call it from notebooks without thinking about
event loops.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from reps.config import LLMModelConfig
from reps.llm.anthropic import AnthropicLLM
from reps.llm.base import LLMInterface
from reps.llm.openai_compatible import OpenAICompatibleLLM
from reps.llm.provider_of import provider_of_model

# Map of <provider> → (env var, default api_base). `None` means the SDK
# already defaults correctly (OpenAI Direct).
_PROVIDER_ENV: dict[str, tuple[str, Optional[str]]] = {
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "openai": ("OPENAI_API_KEY", None),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
}


def _split_provider(model: str) -> tuple[Optional[str], str]:
    """Return (provider, model_id) from `"<provider>/<id>"` or `(None, name)`.

    Only treats the leading segment as a provider when it matches a known
    provider key. Otherwise the slash is part of a vendor-prefixed model id
    (e.g. `"google/gemini-2.5-flash"`) and we return it intact for
    heuristic inference downstream.
    """
    if "/" in model:
        head, _, tail = model.partition("/")
        if head.lower() in _PROVIDER_ENV:
            return head.lower(), tail
    return None, model


def _resolve_provider(model: str) -> str:
    """Pick a provider name for `model` — explicit prefix wins, else infer."""
    explicit, _ = _split_provider(model)
    if explicit is not None:
        return explicit
    # provider_of_model returns "anthropic" or "openai"; never "openrouter"
    # since we can't infer that without an explicit prefix or api_base.
    return provider_of_model(model)


class LM:
    """Sync LLM wrapper around `reps.llm.{anthropic,openai_compatible}`.

    Construct with a model id and optional credentials/generation knobs;
    call as a function or via `.generate()` to get a string completion.
    Async access is deferred to v1.5 (`agenerate`).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        timeout: int = 600,
        retries: int = 2,
        retry_delay: int = 5,
        extended_thinking: Optional[str] = None,
        **provider_kwargs: Any,
    ) -> None:
        if not model or not isinstance(model, str):
            raise ValueError(f"reps.LM: `model` must be a non-empty string, got {model!r}")

        provider = _resolve_provider(model)
        _, raw_model_id = _split_provider(model)

        env_var, default_api_base = _PROVIDER_ENV[provider]

        # Resolve api_key — explicit kwarg wins, else env, else fail loudly.
        resolved_key = api_key if api_key is not None else os.environ.get(env_var)
        if not resolved_key:
            raise ValueError(
                f"reps.LM: no API key for provider {provider!r}. Pass `api_key=...` "
                f"or set ${env_var}."
            )

        resolved_api_base = api_base if api_base is not None else default_api_base

        self.model = model
        self.provider = provider
        self._raw_model_id = raw_model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.extended_thinking = extended_thinking
        self.provider_kwargs = dict(provider_kwargs)

        # All provider SDKs receive the un-prefixed model id. Vendor-routed
        # ids that contain a slash (e.g. "google/gemini-2.5-flash" on
        # OpenRouter) are preserved by `_split_provider` because "google" is
        # not a known provider key — `raw_model_id` already carries them
        # intact.
        client_model_name = raw_model_id

        # `effort` validation lives in the Anthropic provider class; pass
        # through verbatim so we surface the same error.
        reasoning_effort = extended_thinking if extended_thinking != "off" else None

        self._model_cfg = LLMModelConfig(
            name=client_model_name,
            api_key=resolved_key,
            api_base=resolved_api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            reasoning_effort=reasoning_effort,
            provider=provider,
        )

        self._client: LLMInterface = self._build_client()

    def _build_client(self) -> LLMInterface:
        if self.provider == "anthropic":
            return AnthropicLLM(self._model_cfg)
        # openai + openrouter both ride the OpenAI-compatible client.
        return OpenAICompatibleLLM(self._model_cfg)

    # --- sync API -------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Sync wrapper around the underlying provider's async `generate`.

        Raises if called from inside a running event loop — there's no safe
        way to nest `asyncio.run`. Async support lands in v1.5
        (`agenerate`); for now, callers in async contexts should use the
        underlying client directly via `lm._client.generate(...)`.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._client.generate(prompt, **kwargs))
        raise RuntimeError(
            "reps.LM.generate() cannot be called from a running event loop. "
            "Use `await lm._client.generate(...)` directly, or wait for "
            "v1.5's `agenerate()`."
        )

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    # --- introspection helpers (used by reps.REPS) ----------------------

    def _to_model_config(self) -> LLMModelConfig:
        """Return the underlying LLMModelConfig for use by `reps.REPS`.

        A fresh copy is returned so the optimizer can mutate the result
        (e.g. attaching a system_message) without affecting this LM
        instance's client.
        """
        cfg = self._model_cfg
        return LLMModelConfig(
            name=cfg.name,
            api_key=cfg.api_key,
            api_base=cfg.api_base,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            retries=cfg.retries,
            retry_delay=cfg.retry_delay,
            reasoning_effort=cfg.reasoning_effort,
            provider=cfg.provider,
            weight=cfg.weight,
            top_p=cfg.top_p,
            random_seed=cfg.random_seed,
            system_message=cfg.system_message,
        )
