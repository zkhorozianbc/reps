"""Tests for the REPS experiment runner."""
import logging
import json

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
    assert child_programs
    assert child_programs[0]["metadata"]["reps_meta"]["tokens_in"] == 1
    assert child_programs[0]["metadata"]["reps_meta"]["tokens_out"] == 1
