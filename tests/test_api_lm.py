"""Unit tests for `reps.LM` (Phase A of the v1 Python API).

The LM is a thin facade — these tests verify the kwargs → LLMModelConfig
mapping, provider routing (anthropic vs openai vs openrouter), env-var
fallback for api_key, the sync `generate` / `__call__` shortcut wiring,
and the running-loop guard. Provider clients are mocked at the SDK level
so no network or real keys are needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import reps
from reps.api.lm import LM


# ---------------------------------------------------------------------------
# provider routing
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_prefix_routes_to_anthropic_provider(mock_anth):
    lm = LM("anthropic/claude-sonnet-4.6", api_key="k")
    assert lm.provider == "anthropic"
    # Anthropic SDK rejects "anthropic/<id>"; the prefix must be stripped.
    assert lm._model_cfg.name == "claude-sonnet-4.6"
    mock_anth.assert_called_once()


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_inferred_from_bare_claude_name(mock_anth):
    lm = LM("claude-sonnet-4.6", api_key="k")
    assert lm.provider == "anthropic"
    assert lm._model_cfg.name == "claude-sonnet-4.6"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openai_prefix_routes_to_openai_compat(mock_oai):
    lm = LM("openai/gpt-4o", api_key="k")
    assert lm.provider == "openai"
    # For OpenAI Direct we strip the prefix — the SDK doesn't want it.
    assert lm._model_cfg.name == "gpt-4o"
    mock_oai.assert_called_once()


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openrouter_prefix_routes_to_openai_compat_with_default_base(mock_oai):
    lm = LM("openrouter/google/gemini-2.5-flash", api_key="k")
    assert lm.provider == "openrouter"
    # OpenRouter routes on vendor-prefixed ids, so we keep the prefix intact.
    assert lm._model_cfg.name == "google/gemini-2.5-flash"
    assert lm._model_cfg.api_base == "https://openrouter.ai/api/v1"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openrouter_explicit_api_base_overrides_default(mock_oai):
    lm = LM(
        "openrouter/google/gemini-2.5-flash",
        api_key="k",
        api_base="https://custom.example.com/v1",
    )
    assert lm._model_cfg.api_base == "https://custom.example.com/v1"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openai_inferred_from_bare_gpt_name(mock_oai):
    lm = LM("gpt-4o", api_key="k")
    assert lm.provider == "openai"


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="cannot infer provider"):
        LM("totally-fake-model", api_key="k")


def test_empty_model_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        LM("", api_key="k")


# ---------------------------------------------------------------------------
# api_key resolution
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_api_key_from_env_var(mock_anth, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    lm = LM("anthropic/claude-sonnet-4.6")
    assert lm._model_cfg.api_key == "env-key"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_api_key_kwarg_wins_over_env(mock_oai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    lm = LM("openai/gpt-4o", api_key="explicit-key")
    assert lm._model_cfg.api_key == "explicit-key"


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        LM("anthropic/claude-sonnet-4.6")


def test_missing_openrouter_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LM("openrouter/google/gemini-2.5-flash")


# ---------------------------------------------------------------------------
# kwargs propagate into LLMModelConfig
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generation_kwargs_propagate(mock_anth):
    lm = LM(
        "anthropic/claude-sonnet-4.6",
        api_key="k",
        temperature=0.4,
        max_tokens=2048,
        timeout=120,
        retries=5,
        retry_delay=2,
    )
    cfg = lm._model_cfg
    assert cfg.temperature == 0.4
    assert cfg.max_tokens == 2048
    assert cfg.timeout == 120
    assert cfg.retries == 5
    assert cfg.retry_delay == 2


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_extended_thinking_maps_to_reasoning_effort(mock_anth):
    lm = LM("anthropic/claude-opus-4-7", api_key="k", extended_thinking="high")
    assert lm._model_cfg.reasoning_effort == "high"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_extended_thinking_off_clears_reasoning_effort(mock_anth):
    lm = LM("anthropic/claude-opus-4-7", api_key="k", extended_thinking="off")
    assert lm._model_cfg.reasoning_effort is None


# ---------------------------------------------------------------------------
# generate / __call__ wiring
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generate_calls_underlying_client(mock_anth):
    lm = LM("anthropic/claude-sonnet-4.6", api_key="k")
    # Replace the client with an AsyncMock so generate() resolves to a string.
    lm._client = AsyncMock()
    lm._client.generate.return_value = "the answer"

    out = lm.generate("hello")
    assert out == "the answer"
    lm._client.generate.assert_awaited_once_with("hello")


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_call_shortcut_delegates_to_generate(mock_anth):
    lm = LM("anthropic/claude-sonnet-4.6", api_key="k")
    lm._client = AsyncMock()
    lm._client.generate.return_value = "ok"
    assert lm("hi") == "ok"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generate_passes_kwargs(mock_anth):
    lm = LM("anthropic/claude-sonnet-4.6", api_key="k")
    lm._client = AsyncMock()
    lm._client.generate.return_value = "ok"
    lm.generate("hi", temperature=0.1, max_tokens=10)
    lm._client.generate.assert_awaited_once_with("hi", temperature=0.1, max_tokens=10)


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generate_in_running_loop_raises(mock_anth):
    lm = LM("anthropic/claude-sonnet-4.6", api_key="k")

    async def run():
        with pytest.raises(RuntimeError, match="running event loop"):
            lm.generate("hi")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# top-level re-export
# ---------------------------------------------------------------------------


def test_lm_reexported_at_top_level():
    assert reps.LM is LM


# ---------------------------------------------------------------------------
# _to_model_config (used by reps.REPS in Phase B)
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_to_model_config_returns_independent_copy(mock_anth):
    lm = LM("anthropic/claude-sonnet-4.6", api_key="k", temperature=0.6)
    cfg = lm._to_model_config()
    assert cfg.name == "claude-sonnet-4.6"
    assert cfg.temperature == 0.6
    # Mutating the returned copy does not affect the LM's stored cfg.
    cfg.temperature = 9.9
    assert lm._model_cfg.temperature == 0.6
