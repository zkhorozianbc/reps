"""Tests for reps.config module."""

import pytest


def test_config_from_dict_basic():
    from reps.config import Config
    config = Config.from_dict({"max_iterations": 50, "llm": {"primary_model": "test-model", "api_key": "test"}})
    assert config.max_iterations == 50


def test_config_reps_section():
    from reps.config import Config
    config = Config.from_dict({"reps": {"enabled": True, "batch_size": 5, "reflection": {"enabled": True, "top_k": 3}}})
    assert config.reps.enabled is True
    assert config.reps.batch_size == 5


def test_config_provider_field():
    from reps.config import Config
    config = Config.from_dict({"provider": "anthropic", "llm": {"primary_model": "test", "api_key": "test"}})
    assert config.provider == "anthropic"


def test_config_provider_defaults_to_openrouter():
    from reps.config import Config
    config = Config.from_dict({})
    assert config.provider == "openrouter"
    assert config.llm.api_base == "https://openrouter.ai/api/v1"


def test_config_harness_field():
    from reps.config import Config
    config = Config.from_dict({"harness": "openevolve"})
    assert config.harness == "openevolve"


def test_config_from_yaml(tmp_path):
    from reps.config import Config
    yaml_content = "max_iterations: 100\nharness: reps\nprovider: anthropic\nllm:\n  primary_model: test\n  api_key: test\nreps:\n  enabled: true\n"
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    config = Config.from_yaml(yaml_file)
    assert config.max_iterations == 100
    assert config.harness == "reps"
    assert config.provider == "anthropic"


def test_config_openrouter_finalizes_base_and_model_provider():
    from reps.config import Config

    config = Config.from_dict({
        "provider": "openrouter",
        "llm": {"primary_model": "anthropic/claude-sonnet-4.6", "api_key": "test"},
    })

    assert config.llm.api_base == "https://openrouter.ai/api/v1"
    assert {model.provider for model in config.llm.models} == {"openrouter"}
    assert {model.provider for model in config.llm.evaluator_models} == {"openrouter"}


def test_config_openrouter_uses_provider_env_key_when_omitted(monkeypatch):
    from reps.config import Config

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    config = Config.from_dict({
        "provider": "openrouter",
        "llm": {"primary_model": "anthropic/claude-sonnet-4.6"},
    })

    assert config.llm.api_key == "test-openrouter-key"
    assert {model.api_key for model in config.llm.models} == {"test-openrouter-key"}


def test_load_config_default_does_not_use_openai_key_for_openrouter(monkeypatch):
    from reps.config import load_config

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")

    config = load_config()

    assert config.provider == "openrouter"
    assert config.llm.api_key is None


def test_config_rejects_inconsistent_provider_base():
    from reps.config import Config

    with pytest.raises(ValueError, match="api_base.*openrouter"):
        Config.from_dict({
            "provider": "openrouter",
            "llm": {
                "primary_model": "anthropic/claude-sonnet-4.6",
                "api_key": "test",
                "api_base": "https://api.anthropic.com/v1",
            },
        })


def test_config_rejects_inconsistent_provider_env_key(monkeypatch):
    from reps.config import Config

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Config.from_dict({
            "provider": "openrouter",
            "llm": {
                "primary_model": "anthropic/claude-sonnet-4.6",
                "api_key": "${OPENAI_API_KEY}",
            },
        })


def test_config_rejects_unknown_yaml_fields():
    from reps.config import Config

    with pytest.raises(ValueError, match="unknown"):
        Config.from_dict({"llm": {"primary_model": "test", "api_key": "test", "temperatuer": 0.8}})


def test_config_rejects_invalid_enum_like_fields():
    from reps.config import Config

    with pytest.raises(ValueError, match="harness"):
        Config.from_dict({"harness": "repz"})

    with pytest.raises(ValueError, match="selection_strategy"):
        Config.from_dict({"database": {"selection_strategy": "roulette"}})

    with pytest.raises(ValueError, match="generation_mode"):
        Config.from_dict({
            "reps": {
                "workers": {
                    "types": [
                        {
                            "name": "bad-worker",
                            "impl": "single_call",
                            "generation_mode": "rewrite",
                        }
                    ]
                }
            }
        })


def test_config_yaml_paths_resolve_relative_to_config_dir(tmp_path):
    from reps.config import Config

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    yaml_file = config_dir / "test.yaml"
    yaml_file.write_text(
        "initial_program: ../bench/initial_program.py\n"
        "evaluator_path: ../bench/evaluator.py\n"
        "output: ../results\n"
        "llm:\n"
        "  primary_model: test\n"
        "  api_key: test\n"
    )

    config = Config.from_yaml(yaml_file)

    assert config.initial_program == str((tmp_path / "bench" / "initial_program.py").resolve())
    assert config.evaluator_path == str((tmp_path / "bench" / "evaluator.py").resolve())
    assert config.output == str((tmp_path / "results").resolve())


def test_openrouter_smoke_config_exercises_learning_path(monkeypatch):
    from pathlib import Path

    from reps.config import Config

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    repo_root = Path(__file__).resolve().parents[1]
    config = Config.from_yaml(repo_root / "experiment/configs/smoke_openrouter.yaml")

    assert config.provider == "openrouter"
    assert config.max_iterations >= 6
    assert config.llm.primary_model == "anthropic/claude-sonnet-4.6"
    assert config.llm.max_tokens >= 8000
    assert config.reps.batch_size >= 3
    assert config.reps.reflection.enabled is True
    assert config.reps.revisitation.enabled is True
    assert config.reps.convergence.enabled is True
    assert config.reps.sota.enabled is True
    assert config.reps.annotations.enabled is True
    assert config.reps.annotations.dead_end_awareness is True

    workers = config.reps.workers.types
    assert {worker.role for worker in workers} >= {"exploiter", "explorer", "crossover"}
    assert {worker.model_id for worker in workers} == {"anthropic/claude-sonnet-4.6"}
