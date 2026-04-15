"""
Tests for the REPS LLM layer: base interface, OpenRouter provider, and ensemble.
"""

from unittest.mock import patch, MagicMock

import pytest
from openevolve.config import LLMModelConfig

from reps.llm.base import LLMInterface
from reps.llm.openrouter import OpenRouterLLM
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
