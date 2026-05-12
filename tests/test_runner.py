"""Tests for the REPS experiment runner."""
import argparse
import logging
import json
import os

import pytest
from pathlib import Path


class _DeterministicLLM:
    def __init__(self, cfg):
        self.model = cfg.name
        self.last_usage = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context("", [{"role": "user", "content": prompt}])

    async def generate_with_context(self, system_message, messages, **kwargs):
        return "```python\nVALUE = 1\n```"


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


def test_apply_overrides_updates_expanded_model_config():
    from reps.config import Config
    from reps.runner import _apply_overrides

    config = Config.from_dict({"llm": {"primary_model": "test", "api_key": "test"}})

    _apply_overrides(config, ["llm.temperature=0.91"])

    assert config.llm.temperature == 0.91
    assert config.llm.models[0].temperature == 0.91
    assert config.llm.evaluator_models[0].temperature == 0.91


def test_apply_overrides_rejects_unknown_paths():
    from reps.config import Config
    from reps.runner import _apply_overrides

    config = Config.from_dict({"llm": {"primary_model": "test", "api_key": "test"}})

    with pytest.raises(ValueError, match="Unknown override path"):
        _apply_overrides(config, ["llm.temperatuer=0.91"])


def test_apply_overrides_finalizes_after_batch_for_provider_switch():
    from reps.config import Config
    from reps.runner import _apply_overrides

    config = Config.from_dict({
        "provider": "openrouter",
        "llm": {
            "primary_model": "anthropic/claude-sonnet-4.6",
            "api_key": "openrouter-key",
        },
    })

    _apply_overrides(
        config,
        [
            "provider=anthropic",
            "llm.api_base=null",
            "llm.api_key=anthropic-key",
        ],
    )

    assert config.provider == "anthropic"
    assert config.llm.api_base is None
    assert config.llm.api_key == "anthropic-key"
    assert {model.provider for model in config.llm.models} == {"anthropic"}
    assert {model.api_key for model in config.llm.models} == {"anthropic-key"}


def test_positive_int_rejects_zero_and_negative():
    from reps.runner import _positive_int

    assert _positive_int("3") == 3
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        _positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        _positive_int("-1")


def test_load_dotenv_loads_config_and_cwd_ancestors_without_overwriting(tmp_path, monkeypatch):
    from reps.runner import _load_dotenv

    project_dir = tmp_path / "project"
    config_dir = project_dir / "configs"
    config_dir.mkdir(parents=True)
    (project_dir / ".env").write_text("ROOT_ONLY=root\nSHARED=root\n")
    (config_dir / ".env").write_text("CONFIG_ONLY=config\nSHARED=config\n")
    monkeypatch.delenv("ROOT_ONLY", raising=False)
    monkeypatch.delenv("CONFIG_ONLY", raising=False)
    monkeypatch.delenv("SHARED", raising=False)

    loaded = _load_dotenv(config_dir, project_dir)

    assert loaded == config_dir / ".env"
    assert os.environ["CONFIG_ONLY"] == "config"
    assert os.environ["ROOT_ONLY"] == "root"
    assert os.environ["SHARED"] == "config"


def test_next_run_dir_skips_non_numeric_entries(tmp_path):
    from reps.runner import _next_run_dir

    (tmp_path / "run_latest").mkdir()
    (tmp_path / "run_002").mkdir()

    run_dir = Path(_next_run_dir(str(tmp_path)))

    assert run_dir == tmp_path / "run_003"
    assert run_dir.is_dir()


def test_next_run_dir_handles_only_non_numeric_entries(tmp_path):
    from reps.runner import _next_run_dir

    (tmp_path / "run_latest").mkdir()

    run_dir = Path(_next_run_dir(str(tmp_path)))

    assert run_dir == tmp_path / "run_001"
    assert run_dir.is_dir()


def test_resolve_run_inputs_uses_config_output_and_relative_paths(tmp_path):
    from reps.config import Config
    from reps.runner import _resolve_run_inputs

    config = Config.from_dict({
        "initial_program": "initial_program.py",
        "evaluator_path": "evaluator.py",
        "output": "configured-results",
        "llm": {"primary_model": "test", "api_key": "test"},
    })
    config_file = tmp_path / "configs" / "config.yaml"
    config_file.parent.mkdir()
    config_file.write_text("llm:\n  primary_model: test\n  api_key: test\n")

    initial, evaluator, output = _resolve_run_inputs(
        config,
        config_path=str(config_file),
        initial_program_arg=None,
        evaluator_arg=None,
        output_arg=None,
    )

    assert initial == str((config_file.parent / "initial_program.py").resolve())
    assert evaluator == str((config_file.parent / "evaluator.py").resolve())
    assert output == str((config_file.parent / "configured-results").resolve())


@pytest.mark.asyncio
async def test_run_reps_file_logging_is_isolated_between_runs(tmp_path):
    from reps.config import Config, LLMModelConfig
    from reps.runner import run_reps

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    initial = task_dir / "initial_program.py"
    initial.write_text("VALUE = 0\n")
    evaluator = task_dir / "evaluator.py"
    evaluator.write_text(
        "def evaluate(program_path):\n"
        "    return {'combined_score': 1.0}\n"
    )

    def make_config():
        cfg = Config()
        cfg.max_iterations = 1
        cfg.log_level = "INFO"
        cfg.diff_based_evolution = False
        cfg.database.num_islands = 1
        cfg.evaluator.parallel_evaluations = 1
        cfg.evaluator.max_concurrent_iterations = 1
        cfg.evaluator.timeout = 10
        cfg.evaluator.max_retries = 0
        cfg.evaluator.cascade_evaluation = False
        cfg.llm.models = [
            LLMModelConfig(
                name="deterministic-local",
                provider="local",
                init_client=_DeterministicLLM,
            )
        ]
        cfg.llm.evaluator_models = list(cfg.llm.models)
        cfg.reps.enabled = False
        return cfg

    run_one = tmp_path / "run_one"
    run_two = tmp_path / "run_two"
    run_one.mkdir()
    run_two.mkdir()

    await run_reps(make_config(), str(initial), str(evaluator), str(run_one))
    await run_reps(make_config(), str(initial), str(evaluator), str(run_two))
    logging.getLogger("leak_probe").warning("second-run-only-marker")

    first_log = (run_one / "run.log").read_text()
    assert f"Log file: {run_two / 'run.log'}" not in first_log
    assert "second-run-only-marker" not in first_log

    child_programs = [
        json.loads(path.read_text())
        for path in (run_one / "programs").glob("*.json")
        if path.name != "initial.json" and not path.name.endswith(".trace.json")
    ]
    metadata = json.loads((run_one / "metadata.json").read_text())
    assert metadata["metric_call_count"] == 2
    assert child_programs
    assert child_programs[0]["metadata"]["reps_meta"]["tokens_in"] == 1
    assert child_programs[0]["metadata"]["reps_meta"]["tokens_out"] == 1
