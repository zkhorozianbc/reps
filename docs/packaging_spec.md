# REPS Packaging Spec

## Goals

REPS is locally installable today (`uv pip install -e .`) and produces a clean
wheel (`uv build` → `dist/reps-0.1.0-py3-none-any.whl`), but is not yet
PyPI-publishable. This spec covers four hygiene fixes needed before a real
release: (1) clarifying what "gitignore experiment/" means and acting on it,
(2) trimming the hard dependency list down to what `reps/` actually imports
and moving the rest into optional extras, (3) filling in the PyPI-required
metadata block in `pyproject.toml`, and (4) shipping a PEP 561 `py.typed`
marker so downstream type-checkers honour our annotations. The implementation
subagent will execute against the concrete diffs proposed below; this doc is
spec-only.

## Sub-spec 1: Gitignore `experiment/`

### Current state

`/home/user/reps/.gitignore` only ignores **results**, not the whole
`experiment/` tree:

```
# /home/user/reps/.gitignore lines 13-17
# Experiment results — keep configs and best programs, skip logs/checkpoints
experiment/results/*/seed_*/logs/
experiment/results/*/seed_*/checkpoints/
experiment/results/*/*.log
experiment/results/*/seed_*/best/
```

`git ls-files experiment/` reports **871 tracked files**, including:

- `experiment/benchmarks/circle_packing/{evaluator.py, initial_program.py, initial_program_sota.py, packing_circle_max_sum_of_radii.py, system_prompt.md, verify_strict.py, visualize.py}` — the canonical benchmark referenced by the README result table and `experiment/results/circle_sonnet_reps/packing.png`.
- `experiment/benchmarks/circle_packing_n32/{evaluator.py, initial_program.py, system_prompt.md, visualize.py}`.
- All 23 yamls in `experiment/configs/` (the configs the README points users to with `reps-run`).
- A large set of `experiment/results/*/best_program.py` artifacts cited by the README's SOTA table.

The wheel **already** excludes `experiment/` entirely — `[tool.setuptools.packages.find] include = ["reps*"]` at `pyproject.toml:36-37` plus the absence of `experiment` in any `package-data` entry means `unzip -l dist/reps-0.1.0-py3-none-any.whl` shows only `reps/...` paths. This was verified by inspecting the wheel: only files under `reps/` are present.

### Decision tree

The user's instruction "experiment should be gitignored as well" is ambiguous. Two interpretations:

| Interpretation | Action | Trade-off |
|---|---|---|
| **A — destructive**: stop tracking *all* of `experiment/` | `git rm -r --cached experiment/`, then add `experiment/` to `.gitignore`. | Loses the benchmark code and result history from the public repo. README links break (the SOTA table references `experiment/results/circle_sonnet_reps/packing.png`). |
| **B — non-destructive**: ensure `experiment/` is excluded from *the wheel* and treat the request as a packaging concern | No change to tracked files. Confirm wheel exclusion (already true) and document. | Repo retains benchmarks for reproduction; wheel users get a clean install. |
| **C — hybrid**: move benchmarks to a tracked top-level `examples/` directory (still excluded from the wheel via `include = ["reps*"]`) and gitignore everything else under `experiment/` (results, logs, checkpoints) | Bigger refactor: `git mv experiment/benchmarks examples/` and update README install instructions and any internal paths. | Best long-term shape, but out of scope for a packaging-hygiene pass. |

### Recommendation

**Adopt B — non-destructive.** Concrete actions:

1. Edit `/home/user/reps/.gitignore` to broaden the `experiment/results/` rules so any new run output is ignored regardless of seed/path layout. Replace lines 13-17 with:

   ```
   # Experiment results — keep configs and best programs, skip ephemeral run output
   experiment/results/**/logs/
   experiment/results/**/checkpoints/
   experiment/results/**/*.log
   experiment/results/**/best/
   ```

   Rationale: the existing globs assume a `seed_*` subdir, but several tracked
   results live under `experiment/results/<run_name>/run_*/...` instead. The
   `**` form catches both layouts.

2. Add a single comment line in `pyproject.toml` near `[tool.setuptools.packages.find]` (around line 36) noting that `experiment/` is intentionally excluded from the wheel:

   ```toml
   # Wheel ships only the reps/ package; experiment/, tests/, docs/ are dev-only.
   [tool.setuptools.packages.find]
   include = ["reps*"]
   ```

3. Verify after-the-fact via the verification plan (§6).

4. Open question to escalate to the user: "Did you mean A (drop benchmarks from the repo) or B (just keep the wheel clean)? Defaulting to B; flip if A was intended." File this under §7 Open Decisions.

**Why default to B:** the README's headline result links to
`experiment/results/circle_sonnet_reps/packing.png` and `git mv` of benchmarks
would break the user-facing claim. The wheel is already clean. Until the user
confirms otherwise, the lower-blast-radius interpretation is right.

If the user picks A or C later, that's a follow-up spec — the pyproject and
`.gitignore` changes here don't preclude it.

## Sub-spec 2: Optional dependency extras

### Current state

`/home/user/reps/pyproject.toml:10-23`:

```toml
dependencies = [
    "numpy>=1.22.0",
    "openai>=1.0.0",
    "anthropic>=0.60.0",
    "pyyaml>=6.0",
    "dacite>=1.9.2",
    "tqdm>=4.64.0",
    "flask",
    "scipy>=1.10.0",
    "matplotlib>=3.5.0",
    "dspy>=3.1.3",
    "python-dotenv>=1.0.0",
    "bs4>=0.0.2",
]
```

### Audit: where each dep is actually imported

Determined by `grep -rn "^import X\|^from X" reps/ --include="*.py"` plus a wider scan across the repo. Results:

| Dep | Used in `reps/`? | Used in `experiment/`? | Used in `tests/`? | Used elsewhere? |
|---|---|---|---|---|
| `numpy` | yes (`reps/database.py:18`, `reps/embedding.py`, etc.) | yes (benchmarks) | yes | — |
| `openai` | yes (`reps/llm/openai_compatible.py:9`, `reps/embedding.py:9`, `reps/api/model.py:171`, `reps/program_summarizer.py:166`, `reps/workers/openai_tool_runner.py:26`) | — | yes | — |
| `anthropic` | yes (`reps/llm/anthropic.py:9`, `reps/api/model.py:160`, `reps/workers/anthropic_tool_runner.py:16-18`) | — | yes | — |
| `pyyaml` | yes (`reps/config.py:12`, `reps/runner.py:30`) | (configs are .yaml files but not imported) | yes | — |
| `dacite` | yes (`reps/config.py:11`) | — | yes | — |
| `tqdm` | **NO** — `grep` finds zero matches in `reps/` or `tests/` | — | — | — |
| `flask` | **NO** — `grep` finds zero matches anywhere in repo | — | — | — |
| `scipy` | **NO** in `reps/` | yes (`experiment/benchmarks/circle_packing/{initial_program_sota.py:4, packing_circle_max_sum_of_radii.py:213-220}`, plus `experiment/results/*/best_program.py` evolved artifacts) | — | — |
| `matplotlib` | **NO** in `reps/` | yes (`experiment/visualize.py:5-8`, `experiment/benchmarks/circle_packing*/visualize.py`, `experiment/benchmarks/circle_packing/packing_circle_max_sum_of_radii.py:83-84`) | — | — |
| `dspy` | yes — but **only** `reps/workers/dspy_react.py:16` | — | only via `inspect.getsource` (`tests/test_trace_reflection_integration.py:142`, doesn't actually import dspy) | — |
| `python-dotenv` | **NO** — `reps/runner.py:206` defines a hand-rolled `_load_dotenv()` that reads `.env` directly without using the `dotenv` package | — | — | docstring at `reps/runner.py:14` mentions "via python-dotenv" — this comment is stale |
| `bs4` | **NO** — `grep` finds zero matches anywhere in repo | — | — | — |

### Critical finding: `dspy` is eagerly imported

`reps/workers/__init__.py:17` has:

```python
from reps.workers import dspy_react  # noqa: F401
```

This is the registration mechanism — the `@register("dspy_react")` decorator at `reps/workers/dspy_react.py:84` only fires if the module is imported, and the module imports `dspy` at line 16. This means `dspy` is a *load-bearing* dependency unless we either (a) keep it in core deps, or (b) wrap the registration in a try/except so missing dspy degrades gracefully.

**Implementation note for the executor:** if we move `dspy` to an extra, we *must* also patch `reps/workers/__init__.py` to make the `dspy_react` import lazy/optional:

```python
# proposed shape (executor's job, not this spec's)
try:
    from reps.workers import dspy_react  # noqa: F401
except ImportError:
    pass  # dspy extra not installed; dspy_react worker unavailable
```

…and the controller path at `reps/controller.py:319` already lazy-imports `make_dspy_lm`, so it'll error at use-time with a clear `ImportError` if a YAML asks for `impl: dspy_react` without the extra. That's the right shape.

### Proposed grouping

Trim core to what `reps/` actually imports at runtime. Stage everything else
behind optional extras. New `[project]` block:

```toml
dependencies = [
    "numpy>=1.22.0",
    "openai>=1.0.0",
    "anthropic>=0.60.0",
    "pyyaml>=6.0",
    "dacite>=1.9.2",
    "httpx>=0.27",          # used by reps/workers/{anthropic,openai}_tool_runner.py
]

[project.optional-dependencies]
dspy = ["dspy>=3.1.3"]
benchmarks = [
    "scipy>=1.10.0",
    "matplotlib>=3.5.0",
]
all = [
    "dspy>=3.1.3",
    "scipy>=1.10.0",
    "matplotlib>=3.5.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=22.0.0",
    "isort>=5.10.0",
]
```

### Per-dep rationale

- **`flask` — drop entirely.** Zero imports anywhere in the repo; safe to remove without code changes.
- **`bs4` — drop entirely.** Same: zero imports.
- **`tqdm` — drop entirely.** Same: zero imports. Progress reporting in REPS goes through the metrics logger and stream-print modules, neither of which uses `tqdm`.
- **`python-dotenv` — drop entirely.** `_load_dotenv()` at `reps/runner.py:206-229` is a hand-rolled walker that reads `.env` directly via `Path.read_text()`. Stale comment at `reps/runner.py:14` should be updated to remove the "via python-dotenv" mention as part of this change. (Minor follow-up: the executor should fix the docstring.)
- **`scipy`, `matplotlib` → `benchmarks` extra.** Used only by `experiment/benchmarks/*` and `experiment/visualize.py`. Users who `pip install reps` and bring their own evaluator don't need them; users running the bundled benchmarks do.
- **`dspy` → `dspy` extra.** Only the optional `dspy_react` worker uses it. Most users running `single_call`, `anthropic_tool_runner`, or `openai_tool_runner` impls don't need it. Requires the lazy-import patch in `reps/workers/__init__.py:17` (see "Critical finding" above).
- **`httpx` → promote to core.** Currently it's a transitive of `openai`/`anthropic` so installs work today, but `reps/workers/anthropic_tool_runner.py:17` and `reps/workers/openai_tool_runner.py:24` import it directly. Best practice is to declare what you import; do so explicitly.
- **`numpy`, `openai`, `anthropic`, `pyyaml`, `dacite` — keep in core.** All directly imported by `reps/` and required for the v1 public API to function.

### Resulting install size

Removing 6 transitive trees (`flask` pulls Werkzeug/Jinja2/MarkupSafe/itsdangerous/click/blinker; `scipy` pulls a 50MB BLAS-linked binary; `matplotlib` pulls Pillow/kiwisolver/contourpy/cycler; `bs4` pulls soupsieve; `tqdm` is small but unused; `python-dotenv` is small but unused) from a default install is a meaningful win — likely cuts the cold install from ~300MB to ~80MB, mostly numpy.

## Sub-spec 3: PyPI metadata

### Current state

`pyproject.toml` (lines 5-9) only has the four absolute-minimum fields:

```toml
[project]
name = "reps"
version = "0.1.0"
description = "Recursive Evolutionary Program Search"
requires-python = ">=3.12"
```

Missing for a real PyPI publish:
- `authors` — required for the package author credit on PyPI.
- `license` / `license-files` — required for SPDX licence display.
- `readme` — without this, PyPI shows the bare `description` only.
- `urls` — Homepage / Repository / Issues / Documentation.
- `classifiers` — Python version, license, dev status, audience, topic. PyPI uses these for filtering.
- `keywords` — search hints.

### Proposed additions

Insert the following inside the existing `[project]` block, immediately after
the `description` line and before `requires-python`. Every value is concrete;
`{TBD}` markers flag fields that need a human decision (covered in §7).

```toml
[project]
name = "reps"                       # see §7 Open Decisions: distribution name conflict on PyPI
version = "0.1.0"
description = "Recursive Evolutionary Program Search — LLM-driven evolutionary code search with reflection, worker diversity, convergence detection, and SOTA steering."
readme = "README.md"
requires-python = ">=3.12"
license = { file = "LICENSE" }      # see §7 Open Decisions: license choice
authors = [
    {name = "{TBD}", email = "{TBD}"},      # see §7 Open Decisions
]
keywords = [
    "evolutionary-search",
    "llm",
    "code-generation",
    "program-synthesis",
    "alphaevolve",
    "openevolve",
    "reflection",
    "claude",
    "openai",
    "anthropic",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",      # update if licence choice changes
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Code Generators",
    "Typing :: Typed",
]

[project.urls]
Homepage = "https://github.com/zkhorozianbc/reps"
Repository = "https://github.com/zkhorozianbc/reps"
Issues = "https://github.com/zkhorozianbc/reps/issues"
Documentation = "https://github.com/zkhorozianbc/reps#readme"
```

### Notes for the executor

- **`license = { file = "LICENSE" }`** is the modern form that points
  setuptools at the actual LICENSE file we'll write to repo root.
  Avoids hardcoding the SPDX id in two places and matches the
  release spec's recommendation. The file-form has been supported
  since setuptools 42 (matching `requires = ["setuptools>=42"]` at
  `pyproject.toml:2`), so no setuptools bump needed.
- **`readme = "README.md"`** assumes the executor adds a `[tool.setuptools]` `include-package-data = true` or similar so the README ships in the sdist. The wheel typically embeds README into `PKG-INFO` automatically; verify in §6.
- **No `LICENSE` file exists** at `/home/user/reps/LICENSE` today — `ls /home/user/reps/LICENSE*` returned nothing. The executor must create one before publishing. See §7.
- **Distribution name** — `name = "reps"` is almost certainly taken on PyPI (`reps` is a common short word). The Python module name (`import reps`) is independent of the PyPI distribution name, so we can publish under e.g. `reps-search` while users still write `import reps`. See §7 for candidates.

### Cross-spec interaction

- The `License :: OSI Approved :: MIT License` classifier and the
  shipped LICENSE file must agree with whichever license is chosen.
  If the user picks Apache-2.0 instead, both the classifier line and
  the LICENSE file content change.
- `Typing :: Typed` classifier becomes valid only after Sub-spec 4 (`py.typed`) lands. Adding it now is fine — it's an aspirational marker — but the two should ship in the same release.

## Sub-spec 4: `py.typed` marker (PEP 561)

### Current state

REPS uses type hints throughout (e.g. `reps/api/model.py:75-202`,
`reps/api/optimizer.py:42-413`, the entire `reps/api/*` surface). Without
a PEP 561 marker, downstream type-checkers (mypy, pyright, pyre) treat
the installed package as untyped and fall back to `Any` for all REPS
symbols, defeating the type hints that already exist.

### Proposed change

Two edits.

**Edit 1.** Create an empty file `/home/user/reps/reps/py.typed` (zero bytes).

**Edit 2.** Update `pyproject.toml:39-40` from:

```toml
[tool.setuptools.package-data]
reps = ["prompt_templates/*.txt", "prompt_templates/*.json"]
```

to:

```toml
[tool.setuptools.package-data]
reps = ["prompt_templates/*.txt", "prompt_templates/*.json", "py.typed"]
```

That's it. Two-line change.

### Why

Per PEP 561, packages signal "I ship inline type hints — please use them"
by including a `py.typed` marker file in the package root. Without it,
mypy `--strict` users in downstream projects will see e.g.
`reps.Optimizer` as `Any` and lose all the type safety we baked in.

### Cross-spec interaction

- The `Typing :: Typed` PyPI classifier added in Sub-spec 3 is the public-facing twin of this marker. Ship both together.
- `[tool.setuptools.package-data]` already includes the prompt-template assets — adding `py.typed` to the same list is a single-token change.

## Verification plan

After the implementation subagent applies the changes, run these commands in order to confirm each sub-spec is correct.

### Sub-spec 1 (gitignore)

```bash
# 1. New result paths under any layout are ignored.
mkdir -p /tmp/reps-gi-check && cd /tmp/reps-gi-check
git -C /home/user/reps check-ignore -v experiment/results/foo/run_001/logs/x.log experiment/results/foo/run_001/checkpoints/y.json
# Expect: both paths reported as matched by the new globs.

# 2. Tracked benchmarks remain.
git -C /home/user/reps ls-files experiment/benchmarks/circle_packing/ | wc -l
# Expect: 7 (or whatever the current count is) — must NOT be 0.

# 3. Wheel still excludes experiment/.
cd /home/user/reps && uv build && unzip -l dist/reps-0.1.0-py3-none-any.whl | grep -c "experiment/"
# Expect: 0.
```

### Sub-spec 2 (extras)

```bash
# 1. Core install: no scipy/matplotlib/dspy/flask/bs4/tqdm/dotenv pulled in.
cd /tmp && uv venv .venv-core --python 3.12 && source .venv-core/bin/activate
uv pip install /home/user/reps
uv pip list | grep -E "scipy|matplotlib|dspy|flask|beautifulsoup4|tqdm|dotenv"
# Expect: empty output. (httpx may show up as transitive of openai; that's fine.)

# 2. Public API still imports cleanly with bare core deps.
python -c "import reps; print(reps.Optimizer, reps.Model, reps.OptimizationResult)"
# Expect: three class reprs printed. No ImportError.

# 3. dspy_react worker degrades gracefully.
python -c "from reps.workers import registry; print('dspy_react' in registry._WORKERS)"
# Expect: False (dspy not installed → registration skipped).
# Then invoking it via YAML should ImportError at use-time with a clear message.

# 4. Extras install correctly.
uv pip install '/home/user/reps[dspy]'
python -c "from reps.workers import registry; print('dspy_react' in registry._WORKERS)"
# Expect: True.

uv pip install '/home/user/reps[benchmarks]'
python -c "import scipy.optimize, matplotlib.pyplot; print('ok')"
# Expect: 'ok'.

uv pip install '/home/user/reps[all]'
# Expect: both above succeed.
```

### Sub-spec 3 (metadata)

```bash
# 1. Wheel metadata reflects new fields.
cd /home/user/reps && uv build
unzip -p dist/reps-0.1.0-py3-none-any.whl reps-0.1.0.dist-info/METADATA | head -40
# Expect: Author, License, Project-URL, Classifier, Keywords, Description-Content-Type
# all present and non-empty (modulo {TBD} placeholders, which the user fills in).

# 2. twine check passes.
uv pip install twine
twine check dist/reps-0.1.0*
# Expect: "PASSED" for both the wheel and the sdist.

# 3. README renders cleanly.
unzip -p dist/reps-0.1.0-py3-none-any.whl reps-0.1.0.dist-info/METADATA | grep -A2 "Description-Content-Type"
# Expect: "text/markdown" content-type and the README body following.
```

### Sub-spec 4 (`py.typed`)

```bash
# 1. Marker file ships in the wheel.
unzip -l /home/user/reps/dist/reps-0.1.0-py3-none-any.whl | grep py.typed
# Expect: one line, "reps/py.typed".

# 2. Downstream type-checker sees the types.
mkdir /tmp/reps-typed-check && cd /tmp/reps-typed-check
python -m venv .venv && source .venv/bin/activate
pip install /home/user/reps mypy
cat > probe.py <<'EOF'
import reps
opt: reps.Optimizer = reps.Optimizer(model="anthropic/claude-sonnet-4.6")
reveal_type(opt.optimize)  # mypy should report a Callable signature, not Any.
EOF
mypy probe.py
# Expect: "Revealed type is 'def (initial: builtins.str, evaluate: ...) -> reps.api.result.OptimizationResult'"
# (or similar concrete signature — the key is NOT 'Any').
```

### Suite-wide regression

```bash
cd /home/user/reps && uv run python -m pytest tests/
# Expect: the full test suite passes (no regressions from the dependency trim or import-pattern changes).
```

If any test that asserts on `dspy_react` registration fails, the executor needs to either (a) install the `dspy` extra in the test environment, or (b) mark the test as requiring the extra (`@pytest.mark.skipif(...)`).

## Open decisions

These cannot be resolved by the design subagent alone — they need a human (or the user). Surface them in the implementation PR description.

### 7.1 Gitignore intent (Sub-spec 1)

> Did "experiment should be gitignored as well" mean **A** (drop benchmarks from the repo entirely) or **B** (just keep the wheel clean — already done)?

Default chosen: **B**. If the answer is A, follow up with a destructive `git rm -r --cached experiment/` plus README link surgery. If the answer is C (move benchmarks to `examples/`), that's a separate refactor spec.

### 7.2 PyPI distribution name (Sub-spec 3)

The bare name `reps` is almost certainly taken on PyPI (it's an English word and a common acronym). The Python module name (`import reps`) is **independent** of the PyPI distribution name, so we can publish under any of these and users still write `import reps`:

| Candidate | Pros | Cons |
|---|---|---|
| `reps-search` | Descriptive (evolutionary **search**); hyphenated names are common on PyPI | Slightly long |
| `reps-ai` | Trendy; signals LLM-driven | Generic; risks AI-buzzword fatigue |
| `repsearch` | Memorable, single token | Less searchable |
| `reps-evo` | Short, signals evolutionary | Cryptic |
| `recursive-evolutionary-program-search` | Spells out the acronym | Mouthful |

Recommendation: **`reps-search`** — descriptive, available (per a quick PyPI namesearch), and clearly signals what the package does.

Action: bump `name = "reps"` → `name = "reps-search"` (or chosen alternative) in `pyproject.toml:6`. Module imports stay `import reps`.

### 7.3 License choice (Sub-spec 3)

No `LICENSE` file exists at `/home/user/reps/LICENSE` today. Need:

1. A licence decision (MIT, Apache-2.0, BSD-3-Clause are the typical OSS choices for libraries; Apache-2.0 has explicit patent grants which matter for ML projects).
2. A `LICENSE` text file at the repo root (committed to git, included in the sdist).
3. Matching `License :: OSI Approved :: ... License` classifier in `pyproject.toml` and `LICENSE` file at repo root.

**Note:** REPS is forked from OpenEvolve (per the README). Check OpenEvolve's licence first — if it's MIT or Apache, REPS can match; if it's GPL, REPS must comply with copyleft requirements. The licence choice is **load-bearing** for both legal compliance and PyPI metadata.

Recommendation: **Apache-2.0** (default for ML libraries, has patent-grant language that matters in this space), pending confirmation of OpenEvolve's licence.

### 7.4 Authors / email (Sub-spec 3)

Need real values for the `authors = [{name = "{TBD}", email = "{TBD}"}]` block. The git remote is `github.com/zkhorozianbc/reps`, so the author is presumably the GitHub user `zkhorozianbc`. Get their preferred display name and a contact email (a `noreply` GitHub email like `<id>+zkhorozianbc@users.noreply.github.com` is acceptable if they don't want to publish a personal email).

### 7.5 Stale docstring follow-up (Sub-spec 2)

`reps/runner.py:14` says "Automatically loads a sibling `.env` file (via python-dotenv) so env-var…" — but the implementation at line 206 doesn't use `python-dotenv`. When the executor drops `python-dotenv` from deps, they should also fix this docstring to "Automatically loads a sibling `.env` file (via a small built-in walker)…" or similar.

### 7.6 Cross-spec ordering

The four sub-specs have light coupling:

- Sub-spec 3 (`Typing :: Typed` classifier) and Sub-spec 4 (`py.typed` marker) should ship in the **same release** so the classifier doesn't lie.
- Sub-spec 2's `dspy` → extra move requires the lazy-import patch in `reps/workers/__init__.py:17`. Without that patch, removing `dspy` from core deps will break `import reps`. The executor must apply both changes in a single commit (or land the lazy-import patch first and the deps trim second).
- Sub-spec 1 is independent of the other three.
- Sub-spec 3's licence choice (open decision 7.3) blocks publishing but not the rest of the work — the executor can land Sub-specs 1, 2, and 4 plus the non-licence parts of Sub-spec 3 immediately, leaving licence + authors as a follow-up commit gated on user input.

Suggested commit order:

1. (Sub-spec 4) Add `reps/py.typed` + the one-line `package-data` update. Smallest possible change.
2. (Sub-spec 1) Broaden `.gitignore` globs + add explanatory comment in `pyproject.toml`.
3. (Sub-spec 2) Lazy-import `dspy_react` in `reps/workers/__init__.py`; trim core deps; add extras; fix stale docstring at `reps/runner.py:14`.
4. (Sub-spec 3) Add `readme`, `keywords`, `classifiers`, `urls` to `pyproject.toml`; add `LICENSE` file; fill in `authors` once user provides values; flip `name` to chosen distribution name once user picks.

Each commit is independently verifiable against §6.
