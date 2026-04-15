# REPS Refactor: Self-Contained Harness + Anthropic Provider

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make REPS a self-contained harness with no top-level `openevolve/` directory, able to run experiments against both vanilla OpenEvolve (pip-installed) and REPS, with native Anthropic API support alongside OpenRouter.

**Architecture:** REPS owns its full evolution loop, LLM client, config loading, and prompt building — all extracted from the vendored `openevolve/` code. The `openevolve` pip package is used only for running vanilla baseline experiments via its CLI. A lightweight provider abstraction lets configs select `provider: openrouter` (default, OpenAI-compatible) or `provider: anthropic` (native SDK).

**Tech Stack:** Python 3.10+, uv, openai SDK, anthropic SDK, dacite, pyyaml, numpy, pytest

---

## File Structure

### New/Modified files in `reps/`:

| File | Responsibility |
|------|---------------|
| `reps/config.py` | All config dataclasses (REPS + evolution). Extracted from `openevolve/openevolve/config.py`. Adds `provider` field. |
| `reps/controller.py` | Full evolution loop with REPS orchestration. Extracted from `openevolve/openevolve/process_parallel.py`. |
| `reps/llm/base.py` | `LLMInterface` ABC — `generate()` and `generate_with_context()` |
| `reps/llm/openrouter.py` | OpenAI-compatible LLM client. Extracted from `openevolve/openevolve/llm/openai.py` (without manual mode). |
| `reps/llm/anthropic.py` | Native Anthropic API client implementing `LLMInterface`. |
| `reps/llm/ensemble.py` | Model ensemble. Extracted from `openevolve/openevolve/llm/ensemble.py`. |
| `reps/llm/__init__.py` | Exports. |
| `reps/prompt_sampler.py` | Prompt template building. Extracted from `openevolve/openevolve/prompt/sampler.py`. |
| `reps/database.py` | Program dataclass + ProgramDatabase. Extracted from `openevolve/openevolve/database.py`. |
| `reps/evaluator.py` | Evaluator. Extracted from `openevolve/openevolve/evaluator.py`. |
| `reps/utils.py` | Utility functions: `safe_numeric_average`, `apply_diff`, `extract_diffs`, `format_diff_summary`, `parse_full_rewrite`, `extract_code_language`. Extracted from `openevolve/openevolve/utils/`. |
| `reps/runner.py` | CLI entry point. `harness=reps` runs REPS controller; `harness=openevolve` delegates to `openevolve-run`. |
| `reps/prompt_templates/` | Default prompt templates. Copied from `openevolve/openevolve/prompts/defaults/`. |

### Existing files in `reps/` (unchanged or minor fixes):

| File | Change |
|------|--------|
| `reps/__init__.py` | Update exports |
| `reps/iteration_config.py` | No change |
| `reps/reflection_engine.py` | Replace `from openevolve.utils.metrics_utils import safe_numeric_average` → `from reps.utils import safe_numeric_average` |
| `reps/metrics_logger.py` | Same import fix |
| `reps/worker_pool.py` | No change |
| `reps/convergence_monitor.py` | No change |
| `reps/contract_selector.py` | No change |
| `reps/sota_controller.py` | No change |
| `reps/compute_signature.py` | No change |

### Top-level:

| File | Change |
|------|--------|
| `pyproject.toml` | Add dependencies: `openai`, `anthropic`, `pyyaml`, `dacite`, `tqdm`, `flask`. Add `[project.scripts] reps-run = "reps.runner:main"`. |
| `experiment/run_experiment.sh` | Rewrite: use `reps-run` for REPS, `openevolve-run` for baseline. |
| `experiment/configs/*.yaml` | Add `harness:` and `provider:` fields. |

### Deleted:

| Path | Reason |
|------|--------|
| `openevolve/` (entire directory) | Replaced by pip-installed `openevolve` package |

### Tests:

| File | Source |
|------|--------|
| `tests/test_config.py` | New: test REPS config loading, provider field |
| `tests/test_controller.py` | Ported from `openevolve/tests/test_process_parallel.py`, adapted to `reps.controller` |
| `tests/test_island_isolation.py` | Ported from `openevolve/tests/test_island_isolation.py`, adapted to `reps.controller` + `reps.database` |
| `tests/test_reps_features.py` | New: test REPS modules (reflection, convergence, contracts, SOTA, worker pool) |
| `tests/test_llm_providers.py` | New: test OpenRouter + Anthropic provider instantiation and interface |

---

## Task 1: Extract utility functions into `reps/utils.py`

**Files:**
- Create: `reps/utils.py`
- Test: `tests/test_utils.py`

These utilities are imported by `reflection_engine.py` and `metrics_logger.py` from `openevolve.utils.metrics_utils`. Extract them first to unblock the import fix.

- [ ] **Step 1: Write failing test for `safe_numeric_average`**

```python
# tests/test_utils.py
"""Tests for reps.utils extracted utility functions."""
import pytest
from reps.utils import safe_numeric_average


class TestSafeNumericAverage:
    def test_basic_average(self):
        assert safe_numeric_average({"a": 1.0, "b": 3.0}) == 2.0

    def test_single_value(self):
        assert safe_numeric_average({"score": 5.0}) == 5.0

    def test_empty_dict(self):
        assert safe_numeric_average({}) == 0.0

    def test_ignores_non_numeric(self):
        assert safe_numeric_average({"a": 1.0, "b": "text", "c": 3.0}) == 2.0

    def test_combined_score_key(self):
        """If combined_score is present, it should be used directly when called with get()."""
        metrics = {"combined_score": 0.9, "sub_a": 0.5, "sub_b": 0.3}
        # safe_numeric_average computes the average of ALL numeric values
        result = safe_numeric_average(metrics)
        expected = (0.9 + 0.5 + 0.3) / 3
        assert abs(result - expected) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reps.utils'`

- [ ] **Step 3: Read the source and extract utilities**

Read `openevolve/openevolve/utils/metrics_utils.py` to get the exact implementation of `safe_numeric_average`. Also read `openevolve/openevolve/utils/code_utils.py` to get `apply_diff`, `apply_diff_blocks`, `extract_diffs`, `format_diff_summary`, `split_diffs_by_target`, `parse_full_rewrite`. Also read `openevolve/openevolve/utils/format_utils.py` for `format_metrics_safe`, `format_improvement_safe`.

Create `reps/utils.py` containing all these functions. Keep the exact same signatures and logic — this is a copy, not a rewrite.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_utils.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reps/utils.py tests/test_utils.py
git commit -m "feat: extract utility functions from openevolve into reps/utils.py"
```

---

## Task 2: Extract LLM base interface and OpenRouter provider

**Files:**
- Create: `reps/llm/__init__.py`
- Create: `reps/llm/base.py`
- Create: `reps/llm/openrouter.py`
- Create: `reps/llm/ensemble.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write failing tests for LLM interface and OpenRouter provider**

```python
# tests/test_llm_providers.py
"""Tests for LLM provider abstraction."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


def test_openrouter_provider_instantiation():
    """Test that OpenRouterLLM can be instantiated with config."""
    from reps.llm.openrouter import OpenRouterLLM
    from reps.config import LLMModelConfig

    cfg = LLMModelConfig(
        name="anthropic/claude-sonnet-4.6",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        temperature=0.7,
        max_tokens=4096,
        timeout=60,
        retries=3,
        retry_delay=5,
    )
    llm = OpenRouterLLM(model_cfg=cfg)
    assert llm.model == "anthropic/claude-sonnet-4.6"
    assert llm.api_base == "https://openrouter.ai/api/v1"


def test_openrouter_implements_interface():
    """Test that OpenRouterLLM implements LLMInterface."""
    from reps.llm.base import LLMInterface
    from reps.llm.openrouter import OpenRouterLLM

    assert issubclass(OpenRouterLLM, LLMInterface)


def test_ensemble_instantiation():
    """Test that LLMEnsemble can be created with model configs."""
    from reps.llm.ensemble import LLMEnsemble
    from reps.config import LLMModelConfig

    cfg = LLMModelConfig(
        name="test-model",
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        temperature=0.7,
        max_tokens=4096,
        timeout=60,
        retries=3,
        retry_delay=5,
    )
    ensemble = LLMEnsemble([cfg])
    assert len(ensemble.models) == 1


def test_ensemble_model_override_selection():
    """Test that ensemble selects the right model on override."""
    from reps.llm.ensemble import LLMEnsemble
    from reps.config import LLMModelConfig

    cfgs = [
        LLMModelConfig(name="model-a", api_base="https://test.ai/v1", api_key="k",
                        temperature=0.5, max_tokens=100, timeout=10, retries=0, retry_delay=1),
        LLMModelConfig(name="model-b", api_base="https://test.ai/v1", api_key="k",
                        temperature=0.7, max_tokens=100, timeout=10, retries=0, retry_delay=1, weight=0.5),
    ]
    ensemble = LLMEnsemble(cfgs)
    selected = ensemble._select_model(model="model-b")
    assert selected.model == "model-b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_llm_providers.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `reps/llm/base.py`**

```python
# reps/llm/base.py
"""Abstract base class for LLM providers."""
from abc import ABC, abstractmethod
from typing import Dict, List


class LLMInterface(ABC):
    """Interface that all LLM providers must implement."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context."""
        ...
```

- [ ] **Step 4: Create `reps/llm/openrouter.py`**

Copy the `OpenAILLM` class from `openevolve/openevolve/llm/openai.py` into `reps/llm/openrouter.py`, renaming it to `OpenRouterLLM`. Drop the manual mode code (it's not used by REPS). Change `from openevolve.llm.base import LLMInterface` to `from reps.llm.base import LLMInterface`. Keep all the reasoning model detection, retry logic, and token tracking.

- [ ] **Step 5: Create `reps/llm/ensemble.py`**

Copy `openevolve/openevolve/llm/ensemble.py` into `reps/llm/ensemble.py`. Change imports to:
- `from reps.llm.base import LLMInterface`
- `from reps.llm.openrouter import OpenRouterLLM`
- `from reps.config import LLMModelConfig`

- [ ] **Step 6: Create `reps/llm/__init__.py`**

```python
# reps/llm/__init__.py
"""LLM providers for REPS."""
from reps.llm.base import LLMInterface
from reps.llm.openrouter import OpenRouterLLM
from reps.llm.ensemble import LLMEnsemble

__all__ = ["LLMInterface", "OpenRouterLLM", "LLMEnsemble"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_llm_providers.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add reps/llm/ tests/test_llm_providers.py
git commit -m "feat: extract LLM base interface, OpenRouter provider, and ensemble into reps/llm/"
```

---

## Task 3: Add native Anthropic provider

**Files:**
- Create: `reps/llm/anthropic.py`
- Modify: `reps/llm/__init__.py`
- Modify: `reps/llm/ensemble.py`
- Test: `tests/test_llm_providers.py` (add tests)

- [ ] **Step 1: Write failing tests for Anthropic provider**

Append to `tests/test_llm_providers.py`:

```python
def test_anthropic_provider_instantiation():
    """Test that AnthropicLLM can be instantiated with config."""
    from reps.llm.anthropic import AnthropicLLM
    from reps.config import LLMModelConfig

    cfg = LLMModelConfig(
        name="claude-sonnet-4-6",
        api_key="test-key",
        temperature=0.7,
        max_tokens=4096,
        timeout=60,
        retries=3,
        retry_delay=5,
    )
    llm = AnthropicLLM(model_cfg=cfg)
    assert llm.model == "claude-sonnet-4-6"


def test_anthropic_implements_interface():
    """Test that AnthropicLLM implements LLMInterface."""
    from reps.llm.base import LLMInterface
    from reps.llm.anthropic import AnthropicLLM

    assert issubclass(AnthropicLLM, LLMInterface)


def test_anthropic_generate_with_context():
    """Test Anthropic generate_with_context makes correct API call."""
    from reps.llm.anthropic import AnthropicLLM
    from reps.config import LLMModelConfig

    cfg = LLMModelConfig(
        name="claude-sonnet-4-6",
        api_key="test-key",
        temperature=0.7,
        max_tokens=4096,
        timeout=60,
        retries=3,
        retry_delay=5,
    )
    llm = AnthropicLLM(model_cfg=cfg)

    # Mock the Anthropic client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Generated response")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=20)
    llm.client.messages.create = MagicMock(return_value=mock_response)

    result = asyncio.run(llm.generate_with_context(
        system_message="You are helpful",
        messages=[{"role": "user", "content": "Hello"}],
    ))

    assert result == "Generated response"
    assert llm.last_usage["prompt_tokens"] == 10
    assert llm.last_usage["completion_tokens"] == 20

    # Verify the API was called with correct structure
    call_kwargs = llm.client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["system"] == "You are helpful"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


def test_anthropic_model_name_stripping():
    """Test that Anthropic provider strips provider prefix from model names."""
    from reps.llm.anthropic import AnthropicLLM
    from reps.config import LLMModelConfig

    cfg = LLMModelConfig(
        name="anthropic/claude-sonnet-4.6",
        api_key="test-key",
        temperature=0.7,
        max_tokens=4096,
        timeout=60,
        retries=3,
        retry_delay=5,
    )
    llm = AnthropicLLM(model_cfg=cfg)
    # Should strip "anthropic/" prefix for native API
    assert llm.model == "claude-sonnet-4.6"
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_llm_providers.py::test_anthropic_provider_instantiation -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reps.llm.anthropic'`

- [ ] **Step 3: Create `reps/llm/anthropic.py`**

```python
# reps/llm/anthropic.py
"""Native Anthropic API provider for REPS."""
import asyncio
import logging
from typing import Any, Dict, List, Optional

import anthropic

from reps.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMInterface):
    """LLM interface using the native Anthropic SDK."""

    def __init__(self, model_cfg=None):
        raw_model = model_cfg.name
        # Strip provider prefix (e.g. "anthropic/claude-sonnet-4.6" -> "claude-sonnet-4.6")
        self.model = raw_model.split("/", 1)[-1] if "/" in raw_model else raw_model
        self.temperature = model_cfg.temperature
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay

        self.client = anthropic.Anthropic(
            api_key=model_cfg.api_key,
            timeout=self.timeout,
            max_retries=self.retries if self.retries is not None else 0,
        )

        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.info(f"Initialized Anthropic LLM with model: {self.model}")

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message="",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        params: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_message:
            params["system"] = system_message
        if temperature is not None:
            params["temperature"] = temperature

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._call_api(params), timeout=timeout
                )
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.messages.create(**params)
        )

        if hasattr(response, "usage") and response.usage is not None:
            self.last_usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                "total_tokens": (
                    (getattr(response.usage, "input_tokens", 0) or 0)
                    + (getattr(response.usage, "output_tokens", 0) or 0)
                ),
            }
        else:
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return response.content[0].text
```

- [ ] **Step 4: Update `reps/llm/ensemble.py` to support provider dispatch**

The ensemble needs to create the right provider based on config. Modify the model initialization in `LLMEnsemble.__init__`:

```python
def _create_model(self, model_cfg):
    """Create the appropriate LLM based on model config."""
    if model_cfg.init_client:
        return model_cfg.init_client(model_cfg)

    # Check if this should use native Anthropic
    provider = getattr(model_cfg, "provider", None)
    if provider == "anthropic":
        from reps.llm.anthropic import AnthropicLLM
        return AnthropicLLM(model_cfg)

    # Default: OpenAI-compatible (OpenRouter, local vLLM, etc.)
    from reps.llm.openrouter import OpenRouterLLM
    return OpenRouterLLM(model_cfg)
```

- [ ] **Step 5: Update `reps/llm/__init__.py`**

```python
from reps.llm.base import LLMInterface
from reps.llm.openrouter import OpenRouterLLM
from reps.llm.anthropic import AnthropicLLM
from reps.llm.ensemble import LLMEnsemble

__all__ = ["LLMInterface", "OpenRouterLLM", "AnthropicLLM", "LLMEnsemble"]
```

- [ ] **Step 6: Run all LLM tests**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_llm_providers.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add reps/llm/anthropic.py reps/llm/__init__.py reps/llm/ensemble.py tests/test_llm_providers.py
git commit -m "feat: add native Anthropic API provider with model name prefix stripping"
```

---

## Task 4: Extract config, database, evaluator, and prompt sampler

**Files:**
- Create: `reps/config.py`
- Create: `reps/database.py`
- Create: `reps/evaluator.py`
- Create: `reps/prompt_sampler.py`
- Create: `reps/prompt_templates/` (copy from `openevolve/openevolve/prompts/defaults/`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config loading**

```python
# tests/test_config.py
"""Tests for REPS config loading."""
import pytest
import tempfile
from pathlib import Path


def test_config_from_dict_basic():
    """Test creating config from a basic dict."""
    from reps.config import Config
    config = Config.from_dict({
        "max_iterations": 50,
        "llm": {"primary_model": "test-model", "api_key": "test"},
    })
    assert config.max_iterations == 50


def test_config_reps_section():
    """Test that REPS config section loads correctly."""
    from reps.config import Config
    config = Config.from_dict({
        "reps": {
            "enabled": True,
            "batch_size": 5,
            "reflection": {"enabled": True, "top_k": 3},
            "workers": {"types": ["exploiter", "explorer"]},
        }
    })
    assert config.reps.enabled is True
    assert config.reps.batch_size == 5
    assert config.reps.reflection.top_k == 3


def test_config_provider_field():
    """Test that provider field is parsed from config."""
    from reps.config import Config
    config = Config.from_dict({
        "provider": "anthropic",
        "llm": {"primary_model": "claude-sonnet-4-6", "api_key": "test"},
    })
    assert config.provider == "anthropic"


def test_config_provider_defaults_to_openrouter():
    """Test that provider defaults to openrouter."""
    from reps.config import Config
    config = Config.from_dict({})
    assert config.provider == "openrouter"


def test_config_harness_field():
    """Test that harness field is parsed from config."""
    from reps.config import Config
    config = Config.from_dict({"harness": "openevolve"})
    assert config.harness == "openevolve"


def test_config_from_yaml(tmp_path):
    """Test loading config from YAML file."""
    from reps.config import Config
    yaml_content = """
max_iterations: 100
harness: reps
provider: anthropic
llm:
  primary_model: "claude-sonnet-4-6"
  api_key: "test"
reps:
  enabled: true
  batch_size: 10
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)
    config = Config.from_yaml(yaml_file)
    assert config.max_iterations == 100
    assert config.harness == "reps"
    assert config.provider == "anthropic"
    assert config.reps.enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_config.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Create `reps/config.py`**

Copy the full contents of `openevolve/openevolve/config.py` into `reps/config.py`. Make the following modifications:

1. Remove the `TYPE_CHECKING` import of `openevolve.llm.base.LLMInterface` — replace with `from reps.llm.base import LLMInterface`.
2. Add `provider: str = "openrouter"` field to `Config` class (valid values: `"openrouter"`, `"anthropic"`).
3. Add `harness: str = "reps"` field to `Config` class (valid values: `"reps"`, `"openevolve"`).
4. Add `provider` field to `LLMModelConfig`: `provider: Optional[str] = None` — when set, overrides the top-level provider for this specific model.
5. In `load_config()`, handle `ANTHROPIC_API_KEY` environment variable alongside `OPENAI_API_KEY`.

- [ ] **Step 4: Create `reps/database.py`**

Read `openevolve/openevolve/database.py` and copy its full contents into `reps/database.py`. Change all internal imports:
- `from openevolve.utils.metrics_utils import ...` → `from reps.utils import ...`
- `from openevolve.config import DatabaseConfig` → `from reps.config import DatabaseConfig`

- [ ] **Step 5: Create `reps/evaluator.py`**

Read `openevolve/openevolve/evaluator.py` and copy into `reps/evaluator.py`. Change imports:
- `from openevolve.config import EvaluatorConfig` → `from reps.config import EvaluatorConfig`
- `from openevolve.llm.ensemble import LLMEnsemble` → `from reps.llm.ensemble import LLMEnsemble`
- `from openevolve.prompt.sampler import PromptSampler` → `from reps.prompt_sampler import PromptSampler`
- Any `openevolve.utils.*` → `reps.utils`

- [ ] **Step 6: Create `reps/prompt_sampler.py`**

Read `openevolve/openevolve/prompt/sampler.py` and copy into `reps/prompt_sampler.py`. Change imports:
- `from openevolve.config import PromptConfig` → `from reps.config import PromptConfig`
- `from openevolve.prompt.templates import TemplateManager` → inline or extract template manager
- `from openevolve.utils.*` → `from reps.utils import ...`

Also read `openevolve/openevolve/prompt/templates.py` and either inline it or create `reps/prompt_templates.py`.

Copy the template files from `openevolve/openevolve/prompts/defaults/` into `reps/prompt_templates/`.

- [ ] **Step 7: Run config tests**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add reps/config.py reps/database.py reps/evaluator.py reps/prompt_sampler.py reps/prompt_templates/ tests/test_config.py
git commit -m "feat: extract config, database, evaluator, and prompt sampler into reps/"
```

---

## Task 5: Extract the REPS controller (evolution loop)

**Files:**
- Create: `reps/controller.py`
- Test: `tests/test_controller.py`

This is the core task — extracting the full evolution loop from `openevolve/openevolve/process_parallel.py` into `reps/controller.py`.

- [ ] **Step 1: Write failing tests for the controller**

Port `openevolve/tests/test_process_parallel.py` to `tests/test_controller.py`. Change all imports:
- `from openevolve.config import Config, ...` → `from reps.config import Config, ...`
- `from openevolve.database import Program, ProgramDatabase` → `from reps.database import Program, ProgramDatabase`
- `from openevolve.process_parallel import ProcessParallelController, SerializableResult` → `from reps.controller import ProcessParallelController, SerializableResult`

```python
# tests/test_controller.py
"""Tests for REPS process-based parallel controller."""
import asyncio
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

os.environ["OPENAI_API_KEY"] = "test"

from reps.config import Config, DatabaseConfig, EvaluatorConfig, LLMConfig, PromptConfig
from reps.database import Program, ProgramDatabase
from reps.controller import ProcessParallelController, SerializableResult


class TestProcessParallel(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.max_iterations = 10
        self.config.evaluator.parallel_evaluations = 2
        self.config.evaluator.timeout = 10
        self.config.database.num_islands = 2
        self.config.database.in_memory = True
        self.config.checkpoint_interval = 5

        self.eval_content = '''
def evaluate(program_path):
    return {"score": 0.5, "performance": 0.6}
'''
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(self.eval_content)

        self.database = ProgramDatabase(self.config.database)
        for i in range(3):
            program = Program(
                id=f"test_{i}",
                code=f"def func_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5 + i * 0.1, "performance": 0.4 + i * 0.1},
                iteration_found=0,
            )
            self.database.add(program)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_controller_initialization(self):
        controller = ProcessParallelController(self.config, self.eval_file, self.database)
        self.assertEqual(controller.num_workers, 2)
        self.assertIsNone(controller.executor)

    def test_controller_start_stop(self):
        controller = ProcessParallelController(self.config, self.eval_file, self.database)
        controller.start()
        self.assertIsNotNone(controller.executor)
        controller.stop()
        self.assertIsNone(controller.executor)

    def test_database_snapshot_creation(self):
        controller = ProcessParallelController(self.config, self.eval_file, self.database)
        snapshot = controller._create_database_snapshot()
        self.assertIn("programs", snapshot)
        self.assertIn("islands", snapshot)
        self.assertEqual(len(snapshot["programs"]), 3)

    def test_serializable_result(self):
        result = SerializableResult(
            child_program_dict={"id": "test", "code": "pass"},
            parent_id="parent",
            iteration_time=1.5,
            iteration=10,
        )
        self.assertEqual(result.child_program_dict["id"], "test")
        self.assertIsNone(result.error)

    def test_run_evolution_basic(self):
        async def run_test():
            controller = ProcessParallelController(self.config, self.eval_file, self.database)
            with patch.object(controller, "_submit_iteration") as mock_submit:
                mock_future = MagicMock()
                mock_result = SerializableResult(
                    child_program_dict={
                        "id": "child_1", "code": "def evolved(): return 1",
                        "language": "python", "parent_id": "test_0",
                        "generation": 1,
                        "metrics": {"score": 0.7, "performance": 0.8},
                        "iteration_found": 1,
                        "metadata": {"changes": "test", "island": 0},
                    },
                    parent_id="test_0", iteration_time=0.1, iteration=1,
                )
                mock_future.done.return_value = True
                mock_future.result.return_value = mock_result
                mock_future.cancel.return_value = True
                mock_submit.return_value = mock_future
                controller.start()
                result = await controller.run_evolution(start_iteration=1, max_iterations=1)
                mock_submit.assert_called_once_with(1, 0, None)
                self.assertIn("child_1", self.database.programs)
        asyncio.run(run_test())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_controller.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reps.controller'`

- [ ] **Step 3: Create `reps/controller.py`**

Copy the full contents of `openevolve/openevolve/process_parallel.py` into `reps/controller.py`. Change ALL imports:

- `from openevolve.config import Config, ...` → `from reps.config import Config, ...` (include all the REPS config classes)
- `from openevolve.database import Program, ProgramDatabase` → `from reps.database import Program, ProgramDatabase`
- `from openevolve.utils.metrics_utils import safe_numeric_average` → `from reps.utils import safe_numeric_average`
- `from openevolve.llm.ensemble import LLMEnsemble` → `from reps.llm.ensemble import LLMEnsemble`
- `from openevolve.prompt.sampler import PromptSampler` → `from reps.prompt_sampler import PromptSampler`
- `from openevolve.evaluator import Evaluator` → `from reps.evaluator import Evaluator`
- `from openevolve.utils.code_utils import ...` → `from reps.utils import ...`
- All `from reps.xxx import ...` for REPS modules stay the same

In `_worker_init()`, change all the config reconstruction imports to use `reps.config` instead of `openevolve.config`.

In `_lazy_init_worker_components()`, change:
- `from openevolve.llm.ensemble import LLMEnsemble` → `from reps.llm.ensemble import LLMEnsemble`
- `from openevolve.prompt.sampler import PromptSampler` → `from reps.prompt_sampler import PromptSampler`
- `from openevolve.evaluator import Evaluator` → `from reps.evaluator import Evaluator`

- [ ] **Step 4: Run tests**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_controller.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reps/controller.py tests/test_controller.py
git commit -m "feat: extract full REPS evolution controller from openevolve/process_parallel.py"
```

---

## Task 6: Port island isolation tests

**Files:**
- Create: `tests/test_island_isolation.py`

- [ ] **Step 1: Port the test file**

Copy `openevolve/tests/test_island_isolation.py` to `tests/test_island_isolation.py`. Change imports:
- `from openevolve.config import Config, DatabaseConfig, EvaluatorConfig` → `from reps.config import Config, DatabaseConfig, EvaluatorConfig`
- `from openevolve.database import ProgramDatabase, Program` → `from reps.database import ProgramDatabase, Program`
- `from openevolve.process_parallel import ProcessParallelController` → `from reps.controller import ProcessParallelController`

- [ ] **Step 2: Run tests**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_island_isolation.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_island_isolation.py
git commit -m "test: port island isolation tests to reps.controller + reps.database"
```

---

## Task 7: Write REPS feature module tests

**Files:**
- Create: `tests/test_reps_features.py`

- [ ] **Step 1: Write tests for all REPS feature modules**

```python
# tests/test_reps_features.py
"""Tests for REPS feature modules (F1-F8)."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from reps.iteration_config import IterationConfig, IterationResult
from reps.worker_pool import WorkerPool
from reps.convergence_monitor import ConvergenceMonitor, ConvergenceAction, classify_edit
from reps.contract_selector import ContractSelector, Contract
from reps.sota_controller import SOTAController, SearchRegime


class TestWorkerPool:
    def test_initialization(self):
        config = {
            "types": ["exploiter", "explorer", "crossover"],
            "initial_allocation": {"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15},
            "exploiter_temperature": 0.3,
            "explorer_temperature": 1.0,
            "random_seed": 42,
        }
        pool = WorkerPool(config)
        assert abs(sum(pool.allocation.values()) - 1.0) < 1e-6

    def test_build_iteration_config(self):
        config = {
            "types": ["exploiter", "explorer"],
            "initial_allocation": {"exploiter": 0.7, "explorer": 0.3},
            "random_seed": 42,
        }
        pool = WorkerPool(config)
        ic = pool.build_iteration_config(database=None, prompt_extras={"reflection": "test"})
        assert isinstance(ic, IterationConfig)
        assert ic.worker_type in ("exploiter", "explorer")
        assert ic.prompt_extras["reflection"] == "test"

    def test_record_result_and_yield(self):
        config = {"types": ["exploiter"], "initial_allocation": {"exploiter": 1.0}, "random_seed": 42}
        pool = WorkerPool(config)
        pool.record_result("exploiter", True)
        pool.record_result("exploiter", False)
        pool.record_result("exploiter", True)
        assert abs(pool.get_yield_rate("exploiter") - 2.0 / 3) < 1e-6


class TestConvergenceMonitor:
    def test_no_action_when_insufficient_data(self):
        monitor = ConvergenceMonitor({"window_size": 20, "enabled": True})
        results = [IterationResult(diff="def foo(): pass", worker_type="exploiter")]
        assert monitor.update(results) == ConvergenceAction.NONE

    def test_classify_edit(self):
        assert classify_edit("") == "empty"
        assert "function" in classify_edit("def my_function(): pass")
        assert "loop" in classify_edit("for i in range(10): x += i")


class TestContractSelector:
    def test_initialization_with_arms(self):
        config = {
            "enabled": True,
            "models": ["model-a", "model-b"],
            "temperatures": [0.3, 0.7],
            "random_seed": 42,
        }
        selector = ContractSelector(config)
        assert len(selector.arms) == 4  # 2 models x 2 temps

    def test_select_returns_contract(self):
        config = {
            "enabled": True,
            "models": ["model-a"],
            "temperatures": [0.7],
            "random_seed": 42,
        }
        selector = ContractSelector(config)
        contract = selector.select()
        assert isinstance(contract, Contract)
        assert contract.model_id == "model-a"
        assert contract.temperature == 0.7

    def test_update_posterior(self):
        config = {"enabled": True, "models": ["m"], "temperatures": [0.5], "random_seed": 42}
        selector = ContractSelector(config)
        selector.update("m", 0.5, success=True)
        assert selector.posteriors[("m", 0.5)]["alpha"] == 2.0

    def test_disabled_returns_none(self):
        selector = ContractSelector({"enabled": False, "models": [], "temperatures": []})
        assert selector.select() is None


class TestSOTAController:
    def test_regime_far(self):
        ctrl = SOTAController({"enabled": True, "target_score": 10.0})
        assert ctrl.get_regime(5.0) == SearchRegime.FAR

    def test_regime_near(self):
        ctrl = SOTAController({"enabled": True, "target_score": 10.0})
        assert ctrl.get_regime(9.5) == SearchRegime.NEAR

    def test_regime_polishing(self):
        ctrl = SOTAController({"enabled": True, "target_score": 10.0})
        assert ctrl.get_regime(9.9) == SearchRegime.POLISHING

    def test_format_for_prompt(self):
        ctrl = SOTAController({"enabled": True, "target_score": 10.0})
        ctrl.get_regime(8.0)
        text = ctrl.format_for_prompt()
        assert "SOTA" in text
        assert "10.0" in text

    def test_disabled_returns_mid(self):
        ctrl = SOTAController({"enabled": False})
        assert ctrl.get_regime(5.0) == SearchRegime.MID


class TestReflectionEngine:
    def test_format_for_prompt_empty(self):
        from reps.reflection_engine import ReflectionEngine
        engine = ReflectionEngine(llm_ensemble=MagicMock(), config={"enabled": True, "top_k": 3, "bottom_k": 2})
        assert engine.format_for_prompt() == ""

    def test_format_for_prompt_with_data(self):
        from reps.reflection_engine import ReflectionEngine
        engine = ReflectionEngine(llm_ensemble=MagicMock(), config={"enabled": True, "top_k": 3, "bottom_k": 2})
        engine._current_reflection = {
            "working_patterns": ["pattern A works"],
            "failing_patterns": ["pattern B fails"],
            "hypotheses": ["hypothesis 1"],
            "suggested_directions": ["try X"],
        }
        text = engine.format_for_prompt()
        assert "pattern A works" in text
        assert "pattern B fails" in text

    def test_parse_reflection_json(self):
        from reps.reflection_engine import ReflectionEngine
        engine = ReflectionEngine(llm_ensemble=MagicMock(), config={"enabled": True})
        result = engine._parse_reflection('{"working_patterns": ["a"], "failing_patterns": [], "hypotheses": [], "suggested_directions": []}')
        assert result["working_patterns"] == ["a"]
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_reps_features.py -v`
Expected: PASS (these test existing modules — should pass once import fix from Task 8 is done)

- [ ] **Step 3: Commit**

```bash
git add tests/test_reps_features.py
git commit -m "test: add comprehensive tests for REPS feature modules (F1-F8)"
```

---

## Task 8: Fix REPS module imports and update `__init__.py`

**Files:**
- Modify: `reps/reflection_engine.py`
- Modify: `reps/metrics_logger.py`
- Modify: `reps/__init__.py`

- [ ] **Step 1: Fix `reflection_engine.py` import**

In `reps/reflection_engine.py`, line 134, change:
```python
from openevolve.utils.metrics_utils import safe_numeric_average
```
to:
```python
from reps.utils import safe_numeric_average
```

- [ ] **Step 2: Fix `metrics_logger.py` import**

In `reps/metrics_logger.py`, line 104, change:
```python
from openevolve.utils.metrics_utils import safe_numeric_average
```
to:
```python
from reps.utils import safe_numeric_average
```

- [ ] **Step 3: Update `reps/__init__.py`**

```python
"""
REPS: Recursive Evolutionary Program Search

A self-contained harness for evolutionary code search with adaptive meta-cognition.
"""

from reps.iteration_config import IterationConfig, IterationResult
from reps.reflection_engine import ReflectionEngine
from reps.worker_pool import WorkerPool
from reps.convergence_monitor import ConvergenceMonitor, ConvergenceAction
from reps.contract_selector import ContractSelector, Contract
from reps.sota_controller import SOTAController, SearchRegime
from reps.metrics_logger import MetricsLogger

__all__ = [
    "IterationConfig",
    "IterationResult",
    "ReflectionEngine",
    "WorkerPool",
    "ConvergenceMonitor",
    "ConvergenceAction",
    "ContractSelector",
    "Contract",
    "SOTAController",
    "SearchRegime",
    "MetricsLogger",
]
```

- [ ] **Step 4: Run all tests so far**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add reps/reflection_engine.py reps/metrics_logger.py reps/__init__.py
git commit -m "fix: update REPS module imports from openevolve.utils to reps.utils"
```

---

## Task 9: Create the runner and update pyproject.toml

**Files:**
- Create: `reps/runner.py`
- Modify: `pyproject.toml`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing test for the runner**

```python
# tests/test_runner.py
"""Tests for the REPS experiment runner."""
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path


def test_runner_parses_harness_reps(tmp_path):
    """Test that runner dispatches to REPS controller for harness=reps."""
    from reps.runner import load_experiment_config
    yaml_content = """
harness: reps
provider: openrouter
max_iterations: 10
llm:
  primary_model: "anthropic/claude-sonnet-4.6"
  api_key: "test"
reps:
  enabled: true
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = load_experiment_config(str(config_file))
    assert config.harness == "reps"
    assert config.provider == "openrouter"


def test_runner_parses_harness_openevolve(tmp_path):
    """Test that runner recognizes openevolve harness."""
    from reps.runner import load_experiment_config
    yaml_content = """
harness: openevolve
max_iterations: 10
llm:
  primary_model: "test-model"
  api_key: "test"
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = load_experiment_config(str(config_file))
    assert config.harness == "openevolve"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Create `reps/runner.py`**

```python
# reps/runner.py
"""REPS Experiment Runner.

Entry point for running experiments with either the REPS harness or vanilla OpenEvolve.

Usage:
    reps-run <initial_program> <evaluator> --config <config.yaml> [--output <dir>] [--iterations N]
"""
import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from reps.config import Config, load_config


def load_experiment_config(config_path: str) -> Config:
    """Load experiment config from YAML."""
    return Config.from_yaml(config_path)


async def run_reps(config: Config, initial_program: str, evaluator: str, output_dir: str):
    """Run experiment with the REPS harness."""
    from reps.controller import ProcessParallelController
    from reps.database import ProgramDatabase
    from reps.llm.ensemble import LLMEnsemble

    # Set up logging
    logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO))
    logger = logging.getLogger(__name__)

    # Configure provider on model configs
    if config.provider == "anthropic":
        for model in config.llm.models:
            if model.provider is None:
                model.provider = "anthropic"
        for model in config.llm.evaluator_models:
            if model.provider is None:
                model.provider = "anthropic"

    # Initialize database
    db = ProgramDatabase(config.database)

    # Load and add initial program
    initial_code = Path(initial_program).read_text()
    from reps.database import Program
    initial_prog = Program(
        id="initial",
        code=initial_code,
        language=config.language or "python",
        metrics={},
        iteration_found=0,
    )
    db.add(initial_prog)

    # Initialize controller
    controller = ProcessParallelController(
        config=config,
        evaluation_file=evaluator,
        database=db,
        output_dir=output_dir,
    )

    # Initialize reflection engine with LLM ensemble
    if config.reps.enabled:
        llm_ensemble = LLMEnsemble(config.llm.models)
        controller._reps_init_reflection_engine(llm_ensemble)

    # Run evolution
    controller.start()
    try:
        best = await controller.run_evolution(
            start_iteration=1,
            max_iterations=config.max_iterations,
        )
        if best:
            logger.info(f"Best program: {best.id}, metrics: {best.metrics}")
    finally:
        controller.stop()


def run_openevolve(config_path: str, initial_program: str, evaluator: str, output_dir: str, iterations: int):
    """Run experiment with vanilla OpenEvolve (pip-installed package)."""
    cmd = [
        sys.executable, "-m", "openevolve.cli",
        initial_program, evaluator,
        "--config", config_path,
        "--output", output_dir,
        "--iterations", str(iterations),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="REPS Experiment Runner")
    parser.add_argument("initial_program", help="Path to initial program")
    parser.add_argument("evaluator", help="Path to evaluator script")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default="reps_output", help="Output directory")
    parser.add_argument("--iterations", type=int, default=None, help="Override max_iterations")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.iterations:
        config.max_iterations = args.iterations

    if config.harness == "openevolve":
        run_openevolve(args.config, args.initial_program, args.evaluator, args.output, config.max_iterations)
    else:
        asyncio.run(run_reps(config, args.initial_program, args.evaluator, args.output))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "reps"
version = "0.1.0"
description = "Recursive Evolutionary Program Search"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.22.0",
    "openai>=1.0.0",
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "dacite>=1.9.2",
    "tqdm>=4.64.0",
    "flask",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=22.0.0",
    "isort>=5.10.0",
]

[project.scripts]
reps-run = "reps.runner:main"

[tool.setuptools.packages.find]
include = ["reps*"]

[tool.setuptools.package-data]
reps = ["prompt_templates/*.txt", "prompt_templates/*.json"]

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests requiring external services",
]
addopts = "--strict-markers"
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add reps/runner.py pyproject.toml tests/test_runner.py
git commit -m "feat: add REPS experiment runner with harness/provider dispatch"
```

---

## Task 10: Update experiment configs

**Files:**
- Modify: `experiment/configs/circle_sonnet_reps.yaml`
- Modify: `experiment/configs/circle_sonnet_base.yaml`
- Modify: all other `experiment/configs/*.yaml`
- Modify: `experiment/run_experiment.sh`

- [ ] **Step 1: Add `harness` and `provider` fields to all experiment configs**

For each REPS config (e.g., `circle_sonnet_reps.yaml`), add at the top:
```yaml
harness: reps
provider: openrouter
```

For each baseline config (e.g., `circle_sonnet_base.yaml`), add at the top:
```yaml
harness: openevolve
provider: openrouter
```

- [ ] **Step 2: Update `experiment/run_experiment.sh`**

Replace the runner to use `reps-run` for REPS experiments and `openevolve-run` for baselines:

```bash
#!/bin/bash
# REPS Experiment Runner
# Runs baseline (vanilla OpenEvolve) vs REPS harness
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIGS="$ROOT/experiment/configs"
RESULTS="$ROOT/experiment/results"

# ... (keep the seed/iteration setup)

# --- Run baseline (vanilla OpenEvolve via pip package) ---
for seed in "${SEEDS[@]}"; do
    # ... same setup ...
    reps-run "$INITIAL" "$EVALUATOR" \
        --config "$SEED_CONFIG" \
        --iterations "$ITERATIONS" \
        --output "$OUTDIR"
done

# --- Run REPS ---
for seed in "${SEEDS[@]}"; do
    # ... same setup ...
    reps-run "$INITIAL" "$EVALUATOR" \
        --config "$SEED_CONFIG" \
        --iterations "$ITERATIONS" \
        --output "$OUTDIR"
done
```

- [ ] **Step 3: Commit**

```bash
git add experiment/configs/ experiment/run_experiment.sh
git commit -m "feat: update experiment configs with harness/provider fields"
```

---

## Task 11: Delete the vendored `openevolve/` directory

**Files:**
- Delete: `openevolve/` (entire directory)

- [ ] **Step 1: Verify all tests pass without touching openevolve/**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify imports are clean**

Run: `cd /Users/zkhorozian/code/reps && grep -r "from openevolve" reps/ tests/`
Expected: NO matches (all imports should use `reps.*`)

- [ ] **Step 3: Delete the vendored directory**

```bash
rm -rf openevolve/
```

- [ ] **Step 4: Run all tests again to confirm nothing depended on vendored code**

Run: `cd /Users/zkhorozian/code/reps && uv run python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove vendored openevolve/ directory — REPS is now self-contained"
```

---

## Task 12: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the quickstart section**

Update the README to reflect the new setup:
- Installation: `uv pip install -e .` (for REPS), `uv pip install openevolve` (for baseline comparison)
- Running: `reps-run <initial_program> <evaluator> --config <config.yaml>`
- Config: mention `harness:` and `provider:` fields
- Remove references to `openevolve/` directory and `cd openevolve`

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for self-contained REPS + provider options"
```

---

## Task 13: Final integration test — full test suite

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/zkhorozian/code/reps
uv run python -m pytest tests/ -v --tb=short
```
Expected: ALL PASS

- [ ] **Step 2: Verify the REPS package can be imported cleanly**

```bash
cd /Users/zkhorozian/code/reps
uv run python -c "
from reps.config import Config, load_config
from reps.controller import ProcessParallelController, SerializableResult
from reps.database import Program, ProgramDatabase
from reps.llm import LLMInterface, OpenRouterLLM, AnthropicLLM, LLMEnsemble
from reps.runner import load_experiment_config
from reps import ReflectionEngine, WorkerPool, ConvergenceMonitor, ContractSelector, SOTAController
print('All imports successful')
"
```
Expected: `All imports successful`

- [ ] **Step 3: Verify no references to old openevolve path remain**

```bash
grep -r "from openevolve" reps/ tests/ experiment/ --include="*.py" --include="*.sh"
grep -r "openevolve/" reps/ tests/ experiment/ --include="*.py" --include="*.yaml" --include="*.sh" | grep -v "openevolve-run" | grep -v "openevolve_output" | grep -v "pip install openevolve"
```
Expected: No matches (only references should be to the pip package name or CLI command)

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "chore: final cleanup — all tests pass, no vendored openevolve references"
```
