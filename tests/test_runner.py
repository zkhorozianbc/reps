"""Tests for the REPS experiment runner."""
import pytest
from pathlib import Path


def test_runner_parses_harness_reps(tmp_path):
    from reps.runner import load_experiment_config
    yaml_content = "harness: reps\nprovider: openrouter\nmax_iterations: 10\nllm:\n  primary_model: test\n  api_key: test\nreps:\n  enabled: true\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = load_experiment_config(str(config_file))
    assert config.harness == "reps"
    assert config.provider == "openrouter"


def test_runner_parses_harness_openevolve(tmp_path):
    from reps.runner import load_experiment_config
    yaml_content = "harness: openevolve\nmax_iterations: 10\nllm:\n  primary_model: test\n  api_key: test\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = load_experiment_config(str(config_file))
    assert config.harness == "openevolve"


def test_runner_main_function_exists():
    from reps.runner import main
    assert callable(main)
