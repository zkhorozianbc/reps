"""
Tests for the REPS LLM layer: base interface, OpenRouter provider, Anthropic provider, and ensemble.
"""

import asyncio
from unittest.mock import patch, MagicMock

import pytest
from reps.config import LLMModelConfig

from reps.llm.base import LLMInterface
from reps.llm.openrouter import OpenRouterLLM
from reps.llm.anthropic import AnthropicLLM
from reps.llm.ensemble import LLMEnsemble


def _make_model_cfg(**overrides) -> LLMModelConfig:
    """Create an LLMModelConfig with sensible defaults for testing."""
    defaults = dict(
        name="openai/gpt-4o-mini",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        system_message="You are a helpful assistant.",
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        timeout=30,
        retries=2,
        retry_delay=1,
        weight=1.0,
    )
    defaults.update(overrides)
    return LLMModelConfig(**defaults)


# ---------------------------------------------------------------------------
# OpenRouterLLM
# ---------------------------------------------------------------------------

@patch("reps.llm.openrouter.openai.OpenAI")
def test_openrouter_provider_instantiation(mock_openai_cls):
    """OpenRouterLLM can be created with model config."""
    cfg = _make_model_cfg()
    provider = OpenRouterLLM(cfg)

    assert provider.model == "openai/gpt-4o-mini"
    assert provider.api_base == "https://openrouter.ai/api/v1"
    assert provider.temperature == 0.7
    assert provider.retries == 2
    assert provider.last_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # Verify the OpenAI client was constructed
    mock_openai_cls.assert_called_once()


@patch("reps.llm.openrouter.openai.OpenAI")
def test_openrouter_implements_interface(mock_openai_cls):
    """OpenRouterLLM is a subclass of LLMInterface."""
    cfg = _make_model_cfg()
    provider = OpenRouterLLM(cfg)

    assert isinstance(provider, LLMInterface)
    # Verify the abstract methods exist
    assert hasattr(provider, "generate")
    assert hasattr(provider, "generate_with_context")
    assert callable(provider.generate)
    assert callable(provider.generate_with_context)


# ---------------------------------------------------------------------------
# LLMEnsemble
# ---------------------------------------------------------------------------

@patch("reps.llm.openrouter.openai.OpenAI")
def test_ensemble_instantiation(mock_openai_cls):
    """LLMEnsemble can be created with model configs."""
    cfgs = [
        _make_model_cfg(name="openai/gpt-4o-mini", weight=2.0),
        _make_model_cfg(name="openai/gpt-4o", weight=1.0),
    ]
    ensemble = LLMEnsemble(cfgs)

    assert len(ensemble.models) == 2
    assert len(ensemble.weights) == 2
    # Weights should be normalized
    assert abs(ensemble.weights[0] - 2.0 / 3.0) < 1e-9
    assert abs(ensemble.weights[1] - 1.0 / 3.0) < 1e-9
    # last_usage should start at zeros
    assert ensemble.last_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@patch("reps.llm.openrouter.openai.OpenAI")
def test_ensemble_model_override_selection(mock_openai_cls):
    """Ensemble selects the right model on override."""
    cfgs = [
        _make_model_cfg(name="openai/gpt-4o-mini", weight=1.0),
        _make_model_cfg(name="openai/gpt-4o", weight=1.0),
    ]
    ensemble = LLMEnsemble(cfgs)

    # _select_model with an override should return the matching model
    selected = ensemble._select_model(model="openai/gpt-4o")
    assert selected.model == "openai/gpt-4o"

    selected = ensemble._select_model(model="openai/gpt-4o-mini")
    assert selected.model == "openai/gpt-4o-mini"

    # A non-existent override should fall back to sampling (still returns a model)
    selected = ensemble._select_model(model="nonexistent/model")
    assert selected.model in ("openai/gpt-4o-mini", "openai/gpt-4o")


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------

def _make_anthropic_cfg(**overrides) -> LLMModelConfig:
    """Create an LLMModelConfig tailored for Anthropic testing.

    The ``provider`` attribute is set post-init because the installed
    openevolve dataclass doesn't declare it as a field yet.
    """
    provider = overrides.pop("provider", "anthropic")
    defaults = dict(
        name="anthropic/claude-sonnet-4-20250514",
        api_base="https://api.anthropic.com",
        api_key="test-anthropic-key",
        system_message="You are a helpful assistant.",
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        timeout=30,
        retries=2,
        retry_delay=1,
        weight=1.0,
    )
    defaults.update(overrides)
    cfg = LLMModelConfig(**defaults)
    cfg.provider = provider
    return cfg


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_provider_instantiation(mock_anthropic_cls):
    """AnthropicLLM can be created with config."""
    cfg = _make_anthropic_cfg()
    provider = AnthropicLLM(cfg)

    # Model name should have the prefix stripped
    assert provider.model == "claude-sonnet-4-20250514"
    assert provider.temperature == 0.7
    assert provider.retries == 2
    assert provider.api_key == "test-anthropic-key"
    assert provider.last_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # Verify the Anthropic client was constructed
    mock_anthropic_cls.assert_called_once()


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_implements_interface(mock_anthropic_cls):
    """AnthropicLLM is a subclass of LLMInterface."""
    cfg = _make_anthropic_cfg()
    provider = AnthropicLLM(cfg)

    assert isinstance(provider, LLMInterface)
    assert hasattr(provider, "generate")
    assert hasattr(provider, "generate_with_context")
    assert callable(provider.generate)
    assert callable(provider.generate_with_context)


def _make_anthropic_stream_mock(mock_client, text_answer="Hello from Claude!",
                                 thinking=None, input_tokens=100, output_tokens=50):
    """Build a mock for client.messages.stream() that emits synthesized
    `text` / `thinking` delta events followed by content_block_stop markers
    (matches the Anthropic SDK's real event stream)."""
    events = []
    if thinking is not None:
        think_delta = MagicMock()
        think_delta.type = "thinking"
        think_delta.thinking = thinking
        events.append(think_delta)
        think_stop = MagicMock()
        think_stop.type = "content_block_stop"
        think_stop.content_block = MagicMock(type="thinking", thinking=thinking)
        events.append(think_stop)

    text_delta = MagicMock()
    text_delta.type = "text"
    text_delta.text = text_answer
    events.append(text_delta)
    text_stop = MagicMock()
    text_stop.type = "content_block_stop"
    text_stop.content_block = MagicMock(type="text", text=text_answer)
    events.append(text_stop)

    final_usage = MagicMock()
    final_usage.input_tokens = input_tokens
    final_usage.output_tokens = output_tokens
    final_usage.cache_creation_input_tokens = 0
    final_usage.cache_read_input_tokens = 0
    final_msg = MagicMock()
    final_msg.usage = final_usage

    stream_obj = MagicMock()
    stream_obj.__iter__ = lambda self: iter(events)
    stream_obj.get_final_message.return_value = final_msg

    cm = MagicMock()
    cm.__enter__ = lambda self: stream_obj
    cm.__exit__ = lambda self, *a: None
    mock_client.messages.stream.return_value = cm


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_generate_with_context(mock_anthropic_cls):
    """Anthropic makes correct streaming API call (system separate, messages list)."""
    mock_client = MagicMock()
    _make_anthropic_stream_mock(mock_client, text_answer="Hello from Claude!")
    mock_anthropic_cls.return_value = mock_client

    cfg = _make_anthropic_cfg()
    provider = AnthropicLLM(cfg)

    result = asyncio.run(
        provider.generate_with_context(
            system_message="Be helpful.",
            messages=[{"role": "user", "content": "Say hello"}],
        )
    )

    assert result == "Hello from Claude!"
    call_kwargs = mock_client.messages.stream.call_args[1]
    assert call_kwargs["system"] == [
        {
            "type": "text",
            "text": "Be helpful.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_model_name_stripping(mock_anthropic_cls):
    """'anthropic/claude-sonnet-4.6' becomes 'claude-sonnet-4.6' for native API."""
    cfg = _make_anthropic_cfg(name="anthropic/claude-sonnet-4.6")
    provider = AnthropicLLM(cfg)
    assert provider.model == "claude-sonnet-4.6"

    # A name without a prefix should be left as-is
    cfg2 = _make_anthropic_cfg(name="claude-haiku-3")
    provider2 = AnthropicLLM(cfg2)
    assert provider2.model == "claude-haiku-3"


@patch("reps.llm.anthropic.anthropic.Anthropic")
def test_anthropic_token_usage_normalization(mock_anthropic_cls):
    """input_tokens/output_tokens normalized to prompt_tokens/completion_tokens."""
    mock_client = MagicMock()
    _make_anthropic_stream_mock(mock_client, text_answer="response",
                                 input_tokens=200, output_tokens=75)
    mock_anthropic_cls.return_value = mock_client

    cfg = _make_anthropic_cfg()
    provider = AnthropicLLM(cfg)

    asyncio.run(
        provider.generate_with_context(
            system_message="System",
            messages=[{"role": "user", "content": "Hi"}],
        )
    )

    assert provider.last_usage == {
        "prompt_tokens": 200,
        "completion_tokens": 75,
        "total_tokens": 275,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
