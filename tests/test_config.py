"""Tests for reps.config module."""


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
