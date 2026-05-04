"""Unit tests for `reps.Model` (Phase A of the v1 Python API).

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
from reps.api.model import Model, ModelKwargs


# ---------------------------------------------------------------------------
# provider routing
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_prefix_routes_to_anthropic_provider(mock_anth):
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")
    assert model.provider == "anthropic"
    # Anthropic SDK rejects "anthropic/<id>"; the prefix must be stripped.
    assert model._model_cfg.name == "claude-sonnet-4.6"
    mock_anth.assert_called_once()


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_inferred_from_bare_claude_name(mock_anth):
    model = Model("claude-sonnet-4.6", api_key="k")
    assert model.provider == "anthropic"
    assert model._model_cfg.name == "claude-sonnet-4.6"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openai_prefix_routes_to_openai_compat(mock_oai):
    model = Model("openai/gpt-4o", api_key="k")
    assert model.provider == "openai"
    # For OpenAI Direct we strip the prefix — the SDK doesn't want it.
    assert model._model_cfg.name == "gpt-4o"
    mock_oai.assert_called_once()


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openrouter_prefix_routes_to_openai_compat_with_default_base(mock_oai):
    model = Model("openrouter/google/gemini-2.5-flash", api_key="k")
    assert model.provider == "openrouter"
    # OpenRouter routes on vendor-prefixed ids, so we keep the prefix intact.
    assert model._model_cfg.name == "google/gemini-2.5-flash"
    assert model._model_cfg.api_base == "https://openrouter.ai/api/v1"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openrouter_explicit_api_base_overrides_default(mock_oai):
    model = Model(
        "openrouter/google/gemini-2.5-flash",
        api_key="k",
        api_base="https://custom.example.com/v1",
    )
    assert model._model_cfg.api_base == "https://custom.example.com/v1"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openai_inferred_from_bare_gpt_name(mock_oai):
    model = Model("gpt-4o", api_key="k")
    assert model.provider == "openai"


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="cannot infer provider"):
        Model("totally-fake-model", api_key="k")


def test_empty_model_raises():
    with pytest.raises(ValueError, match="non-empty string"):
        Model("", api_key="k")


# ---------------------------------------------------------------------------
# api_key resolution
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_api_key_from_env_var(mock_anth, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    model = Model("anthropic/claude-sonnet-4.6")
    assert model._model_cfg.api_key == "env-key"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_api_key_kwarg_wins_over_env(mock_oai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    model = Model("openai/gpt-4o", api_key="explicit-key")
    assert model._model_cfg.api_key == "explicit-key"


def test_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Model("anthropic/claude-sonnet-4.6")


def test_missing_openrouter_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Model("openrouter/google/gemini-2.5-flash")


# ---------------------------------------------------------------------------
# kwargs propagate into LLMModelConfig
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generation_kwargs_propagate(mock_anth):
    model = Model(
        "anthropic/claude-sonnet-4.6",
        api_key="k",
        temperature=0.4,
        max_tokens=2048,
        timeout=120,
        retries=5,
        retry_delay=2,
    )
    cfg = model._model_cfg
    assert cfg.temperature == 0.4
    assert cfg.max_tokens == 2048
    assert cfg.timeout == 120
    assert cfg.retries == 5
    assert cfg.retry_delay == 2


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_extended_thinking_maps_to_reasoning_effort(mock_anth):
    model = Model("anthropic/claude-opus-4-7", api_key="k", extended_thinking="high")
    assert model._model_cfg.reasoning_effort == "high"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_extended_thinking_off_clears_reasoning_effort(mock_anth):
    model = Model("anthropic/claude-opus-4-7", api_key="k", extended_thinking="off")
    assert model._model_cfg.reasoning_effort is None


# ---------------------------------------------------------------------------
# generate / __call__ wiring
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generate_calls_underlying_client(mock_anth):
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")
    # Replace the client with an AsyncMock so generate() resolves to a string.
    model._client = AsyncMock()
    model._client.generate.return_value = "the answer"

    out = model.generate("hello")
    assert out == "the answer"
    model._client.generate.assert_awaited_once_with("hello")


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_call_shortcut_delegates_to_generate(mock_anth):
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")
    model._client = AsyncMock()
    model._client.generate.return_value = "ok"
    assert model("hi") == "ok"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generate_passes_kwargs(mock_anth):
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")
    model._client = AsyncMock()
    model._client.generate.return_value = "ok"
    model.generate("hi", temperature=0.1, max_tokens=10)
    model._client.generate.assert_awaited_once_with("hi", temperature=0.1, max_tokens=10)


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_generate_in_running_loop_raises(mock_anth):
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")

    async def run():
        with pytest.raises(RuntimeError, match="running event loop"):
            model.generate("hi")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# top-level re-export
# ---------------------------------------------------------------------------


def test_model_reexported_at_top_level():
    assert reps.Model is Model


# ---------------------------------------------------------------------------
# _to_model_config (used by reps.Optimizer in Phase B)
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_to_model_config_returns_independent_copy(mock_anth):
    model = Model("anthropic/claude-sonnet-4.6", api_key="k", temperature=0.6)
    cfg = model._to_model_config()
    assert cfg.name == "claude-sonnet-4.6"
    assert cfg.temperature == 0.6
    # Mutating the returned copy does not affect the LM's stored cfg.
    cfg.temperature = 9.9
    assert model._model_cfg.temperature == 0.6


# ---------------------------------------------------------------------------
# ADVERSARIAL — provider-prefix corner cases
# ---------------------------------------------------------------------------


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_unknown_prefix_treated_as_vendor_route_then_inferred(mock_oai):
    """`badprovider/gpt-4o` — `badprovider` isn't a known provider, so the
    full string is handed to provider_of_model. The bare-id heuristic
    matches `gpt-` in the final segment and infers openai. This is the
    behavior the implementation chose; we pin it so refactors don't break
    legacy openrouter-style ids like `google/gemini-2.5-flash` that
    `_split_provider` deliberately leaves intact."""
    model = Model("badprovider/gpt-4o", api_key="k")
    assert model.provider == "openai"
    # The whole string (including the unknown prefix) is treated as the
    # raw model id since `badprovider` isn't a recognized provider key.
    assert model._model_cfg.name == "badprovider/gpt-4o"


def test_unknown_prefix_with_unknown_model_raises():
    """If the prefix is unknown AND the bare-id heuristic can't infer,
    construction must fail loudly — not silently route to a default
    provider."""
    with pytest.raises(ValueError, match="cannot infer provider"):
        Model("badprovider/totally-fake-model", api_key="k")


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_provider_prefix_case_insensitive(mock_anth):
    """Mixed-case prefix `Anthropic/...` should still resolve correctly
    — the `_PROVIDER_ENV` key lookup lowercases the head."""
    model = Model("Anthropic/claude-sonnet-4.6", api_key="k")
    assert model.provider == "anthropic"


def test_model_must_be_string():
    with pytest.raises(ValueError, match="non-empty string"):
        Model(None, api_key="k")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        Model(123, api_key="k")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ADVERSARIAL — env var fallback edge cases
# ---------------------------------------------------------------------------


def test_empty_string_api_key_kwarg_falls_back_to_env(monkeypatch):
    """`api_key=""` is falsy; the implementation accepts this only via
    the env-var fallback. With no env var, fail loudly."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # api_key="" is treated identically to None: the `if not resolved_key`
    # branch fires when env is also unset.
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        Model("anthropic/claude-sonnet-4.6", api_key="")


def test_missing_openai_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Model("openai/gpt-4o")


# ---------------------------------------------------------------------------
# ADVERSARIAL — SDK client constructor receives the right kwargs
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_sdk_receives_api_key_and_timeout(mock_anth):
    """Verify the Anthropic SDK's __init__ actually receives the
    api_key and timeout we set."""
    Model("anthropic/claude-sonnet-4.6", api_key="my-key", timeout=42)
    # call_args.kwargs has the keyword args passed to anthropic.Anthropic(...)
    kwargs = mock_anth.call_args.kwargs
    assert kwargs.get("api_key") == "my-key"
    assert kwargs.get("timeout") == 42


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openrouter_sdk_receives_api_base(mock_oai):
    Model(
        "openrouter/google/gemini-2.5-flash",
        api_key="rk",
        api_base="https://my-gateway.example.com/v1",
    )
    kwargs = mock_oai.call_args.kwargs
    assert kwargs.get("api_key") == "rk"
    assert kwargs.get("base_url") == "https://my-gateway.example.com/v1"


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_openrouter_sdk_receives_default_api_base_when_unset(mock_oai):
    Model("openrouter/google/gemini-2.5-flash", api_key="rk")
    kwargs = mock_oai.call_args.kwargs
    assert kwargs.get("base_url") == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# ADVERSARIAL — provider_kwargs storage
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_provider_kwargs_stored_on_instance(mock_anth):
    """**provider_kwargs are stashed on `self.provider_kwargs` for
    downstream consumers (today: nobody uses them; tomorrow: power
    users). Verify they're at least preserved verbatim, not silently
    dropped."""
    model = Model(
        "anthropic/claude-sonnet-4.6",
        api_key="k",
        my_custom_flag=True,
        custom_value=42,
    )
    assert model.provider_kwargs == {"my_custom_flag": True, "custom_value": 42}


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_no_provider_kwargs_yields_empty_dict(mock_anth):
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")
    assert model.provider_kwargs == {}


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_provider_kwargs_forwarded_to_anthropic_sdk_client(mock_anth):
    """Spec contract: kwargs are 'passed verbatim to the underlying client
    constructor'. The wrapper rebuilds the SDK client when provider_kwargs
    are non-empty so the forwarded kwargs land in the final
    anthropic.Anthropic(...) call. Storage-only is not enough."""
    Model(
        "anthropic/claude-sonnet-4.6",
        api_key="k",
        default_headers={"X-Foo": "bar"},
        cache_control="test-flag",
    )
    # The wrapper's __init__ calls Anthropic(...) once with the standard
    # args; our forwarding rebuild calls it a second time with the extra
    # kwargs merged in. Inspect the LAST call.
    final_kwargs = mock_anth.call_args.kwargs
    assert final_kwargs.get("default_headers") == {"X-Foo": "bar"}
    assert final_kwargs.get("cache_control") == "test-flag"
    # Standard fields preserved.
    assert final_kwargs.get("api_key") == "k"
    assert final_kwargs.get("max_retries") == 0


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_provider_kwargs_forwarded_to_openai_sdk_client(mock_oai):
    Model(
        "openai/gpt-4o",
        api_key="k",
        default_headers={"X-Org": "test"},
        organization="org-123",
    )
    final_kwargs = mock_oai.call_args.kwargs
    assert final_kwargs.get("default_headers") == {"X-Org": "test"}
    assert final_kwargs.get("organization") == "org-123"
    assert final_kwargs.get("api_key") == "k"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_no_provider_kwargs_does_not_rebuild_sdk_client(mock_anth):
    """When the caller passes nothing extra, the wrapper's own
    Anthropic(...) construction is the only call — we should not re-invoke
    the SDK constructor unnecessarily."""
    Model("anthropic/claude-sonnet-4.6", api_key="k")
    assert mock_anth.call_count == 1


# ---------------------------------------------------------------------------
# ADVERSARIAL — extended_thinking edge values
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_extended_thinking_none_leaves_reasoning_unset(mock_anth):
    """No `extended_thinking` arg => reasoning_effort stays None."""
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")
    assert model._model_cfg.reasoning_effort is None


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_extended_thinking_low_medium_max_propagate(mock_anth):
    for level in ("low", "medium", "high", "xhigh", "max"):
        model = Model("anthropic/claude-sonnet-4.6", api_key="k", extended_thinking=level)
        assert model._model_cfg.reasoning_effort == level


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_extended_thinking_propagates_through_model_config_for_openai(mock_oai):
    """`extended_thinking` must flow through to the OpenAI-compatible
    config too — the OpenAI client reads `reasoning_effort` for o-series
    reasoning models."""
    model = Model("openai/gpt-4o", api_key="k", extended_thinking="medium")
    assert model._model_cfg.reasoning_effort == "medium"


# ---------------------------------------------------------------------------
# ADVERSARIAL — _to_model_config preserves all fields
# ---------------------------------------------------------------------------


@patch("reps.llm.openai_compatible.openai.OpenAI")
def test_to_model_config_carries_provider_for_openrouter(mock_oai):
    """`reps.Optimizer` reads `_to_model_config().provider` to set
    `cfg.provider`. Ensure the openrouter case round-trips."""
    model = Model("openrouter/google/gemini-2.5-flash", api_key="k")
    cfg = model._to_model_config()
    assert cfg.provider == "openrouter"
    assert cfg.api_base == "https://openrouter.ai/api/v1"
    assert cfg.name == "google/gemini-2.5-flash"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_to_model_config_returns_fresh_instance_not_same_object(mock_anth):
    """`_to_model_config()` is documented as returning an independent
    copy. `cfg is model._model_cfg` must be False so Optimizer can attach a
    `system_message` without mutating the LM's client config."""
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")
    cfg_a = model._to_model_config()
    cfg_b = model._to_model_config()
    assert cfg_a is not model._model_cfg
    assert cfg_a is not cfg_b


# ---------------------------------------------------------------------------
# ADVERSARIAL — running-loop guard for __call__ shortcut
# ---------------------------------------------------------------------------


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_call_in_running_loop_raises_too(mock_anth):
    """The `__call__` shortcut delegates to `generate()` — make sure
    the loop guard fires even via the shortcut path."""
    model = Model("anthropic/claude-sonnet-4.6", api_key="k")

    async def run():
        with pytest.raises(RuntimeError, match="running event loop"):
            model("hi")

    asyncio.run(run())
