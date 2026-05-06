# REPS Versioning + Release Workflow Spec

## 1. Goals

REPS just shipped its v1 Python API on `main` with `version = "0.1.0"` in
[`pyproject.toml`](../pyproject.toml). The repo has no `CHANGELOG`, no
`.github/workflows/`, no version-bump policy, and no PyPI publish path.
This spec defines four tightly-scoped artifacts: a version-bump policy
keyed to REPS's actual API surface and shipped commits, a CHANGELOG
format with a drafted initial entry, two GitHub Actions workflows (CI
+ PyPI release) sketched as concrete YAML, and the pre-1.0 disclaimers
that need to land in `README.md` and `pyproject.toml`. Goal is a system
where a maintainer cuts a release by editing `pyproject.toml` +
`CHANGELOG.md`, tagging `vX.Y.Z`, and pushing — and the workflow
publishes to PyPI without human intervention.

Out of scope (cross-referenced, not decided here):

- The PyPI distribution name (currently `reps` in `pyproject.toml` —
  may not be available on PyPI; rename decision belongs to a packaging
  spec).
- The `LICENSE` file (MIT vs Apache-2.0 — required by PyPI but a
  separate decision; belongs to the packaging spec).

## 2. Sub-spec 1: Version bump policy

### 2.1 Pre-1.0 era (current)

REPS is `0.1.0`. The Python API just shipped; users haven't yet built
non-trivial things on top of `reps.Optimizer`. We need a policy that's
strict enough to be predictable but loose enough to let us iterate on
the API shape without a forced 1.0 commitment.

**Rule (pre-1.0, while `version = "0.x.y"`):**

| Bump            | Triggers (pre-1.0)                                                       |
|-----------------|---------------------------------------------------------------------------|
| **Patch (0.1.X)** | Bug fixes; docs-only; internal refactors with no public API change; test additions; CI changes; commit-history housekeeping. |
| **Minor (0.X.0)** | ANY public API change — additive (new symbol, new optional kwarg, new dataclass field) OR breaking (rename, removal, default-value flip, signature change). |
| **Major (1.0.0)** | The one-time graduation to "stable API" (see §5.2 for the gating criteria). After 1.0, switch to strict semver. |

The pre-1.0 collapse of "additive" and "breaking" into one bucket
(minor) is deliberate. It lets us rename / refactor without burning a
fake major (every refactor would otherwise be a 1.0, then 2.0, then
3.0). The cost is that pre-1.0 minor bumps are not safe to consume
blindly — see §5.

**Rule (post-1.0, target state, while `version = "X.Y.Z"` with X ≥ 1):**

Strict [semver](https://semver.org):

| Bump        | Triggers (post-1.0)                                                                                              |
|-------------|-------------------------------------------------------------------------------------------------------------------|
| **Patch (X.Y.Z)** | Bug fixes; docs; internal refactors; test additions.                                                            |
| **Minor (X.Y.0)** | Additive only: new public symbol, new optional kwarg with a backward-compatible default, new dataclass field, new method on an existing public class. |
| **Major (X.0.0)** | Anything that breaks consumer code: rename, removal, signature change, default-value flip, raised exception type changing, `EvaluationResult` field removal or rename, etc. |

### 2.2 Reference table — actual REPS changes so far

Worked examples grounded in commits already on `main`. These show how
the policy would have been applied if we'd been at 1.0 already, so
it's clear what counts as what.

| Change                                                                | Commit       | Pre-1.0 bump | Post-1.0 bump | Notes |
|------------------------------------------------------------------------|--------------|--------------|---------------|-------|
| `reps.LM` → `reps.Model` rename                                        | `ab2517f`    | minor        | **major**     | Public class rename. Pre-1.0 we collapse to minor; post-1.0 this is a hard break — a deprecation alias would have been required. |
| `reps.REPS` → `reps.Optimizer` rename                                  | `642ab15`    | minor        | **major**     | Same shape: public class rename. |
| `Optimizer(model=...)` accepts `str` in addition to `reps.Model`       | `ab2517f`    | minor        | minor         | Additive: widens the accepted type union. Existing `model=Model(...)` calls keep working. |
| Add `reps.ModelKwargs` TypedDict + `**Unpack[ModelKwargs]` on Optimizer | `ab2517f`   | minor        | minor         | Additive new public symbol + new accepted kwargs. |
| Add `per_instance_scores`, `feedback` fields on `EvaluationResult`     | `102952e`    | minor        | minor         | Additive (defaulted to `None`). Existing constructors unaffected. |
| GEPA Phase 6 minibatch shipped, then reverted                          | `31e1482`…`d0ad5c0` | patch (net) | **major** (net)| If we'd shipped the `evaluate(code, instances=...)` contract change at 1.x, the revert itself would be a major (we removed a public capability). In practice both shipped in 0.x and netted to a no-op on `main`. |
| Provider kwargs forwarding fix                                         | `df77b02`    | patch        | patch         | Bug fix in existing public path (kwargs were declared, not actually forwarded). |
| Adversarial test coverage                                              | `9121c7b`    | patch        | patch         | Tests only; no public surface change. |
| Docs (api spec annotations, GEPA plan)                                 | `9eb3c6c`, `277fc5b` | patch | patch | Docs-only. |

### 2.3 Edge-case rules

These are the questions that always come up at release time. Pin the
answers now.

1. **Adding a constructor kwarg with a default** — minor. The kwarg
   must default to the current behavior (omitting it is a no-op).
   Example: adding `Optimizer(..., enable_xyz=False)` is minor; adding
   `Optimizer(..., enable_xyz=True)` flips behavior for existing users
   and is **major** (post-1.0). If you really want enable_xyz on by
   default at 1.x, ship two minors: first as `enable_xyz=False`, then a
   later **major** flips the default.

2. **Changing the default value of an existing kwarg** — **major**
   (post-1.0). Default values are part of the contract. Existing users
   who relied on the default get different behavior on `pip upgrade`,
   which is a silent break.

   Pre-1.0: minor. Document the change loudly in `CHANGELOG.md` under
   `Changed`.

3. **Adding a field to a public dataclass (`OptimizationResult`,
   `EvaluationResult`)** — minor. Existing callers that read the old
   fields are unaffected. Existing callers that *construct* the
   dataclass with positional args are at risk, so prefer adding fields
   at the end with a default.

4. **Removing a field from a public dataclass** — **major**. Even
   pre-1.0 we should call this out as `Removed` in the CHANGELOG
   because it silently breaks readers.

5. **Renaming a public symbol** — **major** (post-1.0). The migration
   path is "deprecate first, remove next major":

   - Minor X.Y.0: ship the new name; keep the old name as a thin alias
     that imports from the new name. Add a `DeprecationWarning` on the
     old import path.
   - Major (X+1).0.0: remove the old name.

   Pre-1.0 we skip the alias and just rename (as we did with
   `LM`→`Model` and `REPS`→`Optimizer`). This is the main thing
   consumers need to know about the pre-1.0 era.

6. **Raising a different exception type from a public method** —
   **major**. `try/except ValueError:` callers will silently miss the
   new exception.

7. **Adding a new accepted return shape from `evaluate(code)`** — minor.
   The current contract accepts `float | dict | EvaluationResult`
   ([python_api_spec.md §`optimize()`](python_api_spec.md)). Adding a
   fourth accepted shape is additive.

8. **Tightening accepted types** (e.g. evaluator return shape no longer
   accepts `float`) — **major**.

9. **Internal refactors that touch `reps.internal.*` re-exports** —
   `reps.internal` is a documented escape hatch in
   `python_api_spec.md`. Treat its exposed names (`ReflectionEngine`,
   `WorkerPool`, `ConvergenceMonitor`, etc.) as **public for
   versioning purposes**. Renames or signature changes inside
   `reps.internal` are major (post-1.0).

10. **Anything in `reps/` not re-exported via `reps.__init__` or
    `reps.api.__init__` or `reps.internal.__init__`** — internal.
    Renames and signature changes are not versioned. The CLI
    (`reps-run`) and YAML config schema (`reps.config.Config`) are
    public.

11. **YAML config schema changes** — `reps.config.Config` is consumed
    by both the CLI and `Optimizer.from_config()`. Adding a field with
    a default = minor; removing a field or changing a default =
    **major**.

### 2.4 Release mechanics

Maintainer steps for any release:

1. Edit `pyproject.toml` `version` to the new value.
2. Move `[Unreleased]` section in `CHANGELOG.md` to a new dated
   `[X.Y.Z] - YYYY-MM-DD` heading; add a fresh empty `[Unreleased]`.
3. Commit: `chore(release): vX.Y.Z`.
4. Tag: `git tag vX.Y.Z && git push origin main vX.Y.Z`.
5. The `release.yml` workflow (§4) takes over from there.

No need to bump version on every PR — only at release time. That keeps
the version in `pyproject.toml` aligned with the latest published
release rather than HEAD.

## 3. Sub-spec 2: CHANGELOG.md format

### 3.1 Format choice — Keep a Changelog

Use [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Sections
per version, in this fixed order, omit empty ones:

- **Added** — new public symbols, new kwargs, new fields.
- **Changed** — behavior changes that are NOT breaking; default
  changes that the team has decided are minor under §2.3 rule 2
  (pre-1.0 only).
- **Deprecated** — soon-to-be-removed symbols still present.
- **Removed** — public symbols no longer present.
- **Fixed** — bug fixes.
- **Security** — security-relevant fixes.

### 3.2 Mapping from REPS commit conventions to CHANGELOG sections

Recent commit log shows REPS uses Conventional Commits-ish prefixes.
Map them as follows:

| Commit prefix             | CHANGELOG section | Notes |
|---------------------------|-------------------|-------|
| `feat(...)`               | Added             | If the feat is a *change* to existing behavior rather than new surface, file under Changed. |
| `fix(...)`                | Fixed             |  |
| `refactor(...)`           | Changed           | Only if user-visible. Internal refactors aren't logged. |
| `docs(...)`               | (omit)            | Docs aren't release-noteworthy unless they document a user-visible contract change — in which case the contract change itself is the entry. |
| `revert: ...`             | Removed           | Or Fixed if reverting a bug introduction. |
| `test(...)` / `chore(...)`| (omit)            |  |
| Renames / removals        | Changed + Removed | Both — the new name under Changed, the old under Removed. |

### 3.3 Drafted initial CHANGELOG.md

Write to `/home/user/reps/CHANGELOG.md`:

```markdown
# Changelog

All notable changes to REPS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0.0. Until then, the project follows the pre-1.0 policy
documented in [`docs/release_spec.md`](docs/release_spec.md): minor bumps
may include breaking changes; only patch bumps are safe to consume blindly.

## [Unreleased]

## [0.1.0] - 2026-05-05

First public release. Ships the v1 Python API
([`docs/python_api_spec.md`](docs/python_api_spec.md)) and GEPA Phases 1-5
([`docs/gepa_implementation_plan.md`](docs/gepa_implementation_plan.md)).

### Added

- `reps.Optimizer` — single-class entry point for evolutionary code
  search; constructed with `model=...` plus optional knobs, runs via
  `optimizer.optimize(initial, evaluate)` (`aaae09a`, renamed from
  `reps.REPS` in `642ab15`).
- `reps.Model` — sync LLM facade over `reps.llm.*` providers
  (Anthropic, OpenAI, OpenRouter); model strings parsed as
  `"<provider>/<id>"` with env-var fallback for `api_key` (`3ef3bb4`,
  renamed from `reps.LM` in `ab2517f`).
- `reps.ModelKwargs` — `TypedDict(total=False)` of optional `Model`
  construction kwargs; spread as `**Unpack[ModelKwargs]` on
  `Optimizer` so common-case users skip the `Model` constructor
  entirely (`ab2517f`).
- `reps.OptimizationResult` — return type from `optimize()`; carries
  `best_code`, `best_score`, `best_metrics`, `best_per_instance_scores`,
  `best_feedback`, `iterations_run`, `total_metric_calls`,
  `total_tokens`, `output_dir` (`aaae09a`).
- `reps.EvaluationResult` — re-exported at top level; evaluators may
  return `float`, `dict`, or `EvaluationResult` and the harness coerces
  consistently (`102952e`).
- `EvaluationResult.per_instance_scores` and `EvaluationResult.feedback`
  fields — power Pareto selection, trace reflection, and merge (GEPA
  Phase 1.1, `102952e`).
- `Optimizer.from_config(cfg: reps.config.Config)` — escape hatch for
  power users who need knobs the simple constructor doesn't expose
  (`aaae09a`).
- `reps.internal.*` — documented re-export surface for advanced
  internals (`ReflectionEngine`, `WorkerPool`, `ConvergenceMonitor`,
  etc.) so existing direct importers keep working (`aaae09a`).
- Pareto-frontier selection — `selection_strategy="pareto"` or
  `"mixed"` with `pareto_fraction=...` on `Optimizer`; chooses parents
  by per-instance domination instead of MAP-Elites bins (GEPA Phase 2,
  `5dea23b`, `7e73e61`).
- Trace-grounded reflection — `trace_reflection=True` on `Optimizer`
  emits a per-candidate LLM-generated mutation directive from the
  parent's specific failures (GEPA Phase 3, `c9bda73`, `1fdacfe`).
- System-aware merge — `merge=True` on `Optimizer` selects crossover
  partners whose strengths complement the primary's weaknesses on
  disjoint instance dimensions (GEPA Phase 4, `f89fb8f`, `b1db148`).
- Ancestry-aware reflection — `lineage_depth=N` on `Optimizer` extends
  trace reflection with N generations of parent context (GEPA Phase 5,
  `2420e69`).
- `circle_packing` benchmark emits four sub-scores (`validity`,
  `boundary`, `overlap`, `sum_radii_progress`) as
  `per_instance_scores` plus `feedback` (`eac9f83`).

### Fixed

- `provider_kwargs` are now actually forwarded to the underlying SDK
  client when constructing `reps.Model` (previously declared but
  swallowed) (`df77b02`).
- `EvaluationResult.from_dict` peels top-level `per_instance_scores`
  and `feedback` keys into the dedicated dataclass fields rather than
  leaving them buried in `metrics` (`df77b02`).

### Removed

- GEPA Phase 6 (minibatch evaluation with promotion) — shipped in
  `31e1482`, `877d57d`, `8441b72`, `2bed1a1`, `13f6bf5`, then reverted
  in `d0ad5c0` because the `evaluate(code, instances=...)` contract
  coupled the harness to a benchmark-side instance registry that most
  REPS benchmarks don't have. Cascade evaluation
  (`evaluate_stage1` → `evaluate`) covers the same fast-fail use case
  without polluting the public contract. See
  [`docs/gepa_implementation_plan.md` "Phase 6 — reverted"](docs/gepa_implementation_plan.md).

[Unreleased]: https://github.com/zkhorozianbc/reps/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zkhorozianbc/reps/releases/tag/v0.1.0
```

(Replace the GitHub URL with whatever ends up in the packaging spec.)

### 3.4 Maintenance discipline

- The `[Unreleased]` block is updated **in the same PR** that introduces
  a user-visible change. Don't defer it to release time — that's how
  you forget what shipped.
- A PR that doesn't touch the public surface needn't update CHANGELOG
  (consistent with §3.2: docs/test/chore commits are omitted).
- At release time, the only mechanical step is renaming `[Unreleased]`
  → `[X.Y.Z] - YYYY-MM-DD`, plus updating the bottom `[X.Y.Z]:` link.
- A pre-commit or CI lint check that the version in `pyproject.toml`
  matches the topmost dated CHANGELOG heading is nice-to-have but not
  required for v1 of this workflow.

## 4. Sub-spec 3: GitHub Actions workflows

Two workflows in `.github/workflows/`:

- `test.yml` — runs on every PR and on pushes to `main`.
- `release.yml` — runs only on tag push matching `v*.*.*`.

### 4.1 `.github/workflows/test.yml`

```yaml
name: tests

on:
  pull_request:
  push:
    branches: [main]

jobs:
  pytest:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.12"]
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Install REPS in editable mode with dev extras
        run: uv pip install --system -e ".[dev]"

      - name: Run pytest
        run: uv run python -m pytest tests/ -v
        env:
          # The previous 2 env-key-requiring failures are eliminated
          # by the test_skip spec's lazy summarizer construction in
          # production code (reps/controller.py) — no skipif markers
          # needed. CI passes without ANTHROPIC_API_KEY because the
          # controller no longer eagerly constructs the summarizer
          # LLM at __init__ time. See docs/test_skip_spec.md.
          PYTHONUNBUFFERED: "1"
```

Notes on matrix and Python versions:

- The project pins `requires-python = ">=3.12"` in `pyproject.toml`.
  Don't matrix over 3.10/3.11 — those aren't supported. If the policy
  changes, expand the matrix here.
- A future addition: `runs-on: [ubuntu-latest, macos-latest]` once
  someone confirms macOS is a target.

Cross-reference: a sibling `tests/test_skip` spec is designing the
`skipif` markers for the two pre-existing tests that hit
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` as more than just monkeypatched
env vars. Until that lands, this workflow may flake on those tests in
a fresh CI environment. That's a hard prerequisite for `test.yml`
landing on `main`.

### 4.2 `.github/workflows/release.yml`

```yaml
name: release

on:
  push:
    tags:
      - "v*.*.*"

jobs:
  build:
    name: Build sdist + wheel
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Verify tag matches pyproject.toml version
        run: |
          TAG_VERSION="${GITHUB_REF_NAME#v}"
          PYPROJECT_VERSION=$(uv run python -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
          if [ "$TAG_VERSION" != "$PYPROJECT_VERSION" ]; then
            echo "::error::Tag version ($TAG_VERSION) does not match pyproject.toml version ($PYPROJECT_VERSION)"
            exit 1
          fi

      - name: Build sdist + wheel
        run: uv build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    name: Publish to PyPI
    needs: build
    runs-on: ubuntu-latest
    # Trusted Publishing requires the workflow to declare an environment
    # AND id-token: write. Both are set up via PyPI project settings.
    environment:
      name: pypi
      # NOTE: the URL hardcodes the PyPI distribution name. If the
      # packaging spec resolves to a name other than `reps` (e.g.
      # `reps-search` per its recommendation), the executor MUST
      # update this URL to match. The `[project].name` field in
      # pyproject.toml is the source of truth.
      url: https://pypi.org/p/reps
    permissions:
      id-token: write
    steps:
      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI via Trusted Publishing
        uses: pypa/gh-action-pypi-publish@release/v1
        # No `password:` line — Trusted Publishing uses OIDC.
```

Key design points:

- **One trigger only**: tag push matching `v*.*.*`. Any other path
  (PR, manual dispatch, push to main) does NOT publish. This is
  paranoia about accidental releases — the only way to publish is to
  cut a tag, which is an explicit human action.
- **Tag-vs-pyproject.toml validation**: the build job fails fast if
  someone tags `v0.2.0` but forgot to bump `pyproject.toml`. This
  catches the most common pre-release human error.
- **Two-job split (build → publish)**: lets a future maintainer add
  signing, attestation, or sdist verification between the two without
  rewiring the publish step. Also means a build failure doesn't waste
  a PyPI environment trip.
- **No API token stored in GitHub Secrets**: this workflow uses
  [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
  which is OIDC-based. GitHub mints a short-lived token, PyPI
  verifies the workflow identity, no secret rotation. This is the
  current PyPI best practice.
- **Environment `pypi`**: a GitHub environment named `pypi` must be
  created in repo Settings → Environments. The environment can have
  required reviewers (e.g. require maintainer approval before
  publish), which is a cheap second human-in-the-loop guard.

### 4.3 PyPI Trusted Publishing — prerequisite setup

This workflow can't run until PyPI is configured. The one-time setup
is documented at
[docs.pypi.org/trusted-publishers](https://docs.pypi.org/trusted-publishers/),
but the steps for REPS:

1. **Manual first upload** — Trusted Publishing requires the project
   to already exist on PyPI. The first release is uploaded manually
   from a maintainer's laptop, using a one-time PyPI API token:

   ```bash
   uv build
   uv publish --token pypi-...
   ```

   (Or `python -m twine upload dist/*`.) This creates the project
   entry on PyPI with version `0.1.0` and uploads the wheel + sdist.

2. **Configure trusted publisher on PyPI** — go to
   `https://pypi.org/manage/project/<project-name>/settings/publishing/`,
   add a "pending GitHub publisher" or (if the repo is already known)
   an "active" publisher with these values:

   | Field           | Value                                                |
   |-----------------|------------------------------------------------------|
   | Owner           | `zkhorozianbc` (or whatever the repo owner is)       |
   | Repository name | `reps`                                               |
   | Workflow name   | `release.yml`                                        |
   | Environment     | `pypi`                                               |

3. **Create the GitHub environment** — repo Settings → Environments →
   New → name `pypi`. Optional: add a maintainer as required reviewer.

4. **First automated release** — bump to `0.1.1` (or whatever the next
   version is), tag `v0.1.1`, push tag. The workflow runs end-to-end.

After step 4 succeeds once, all subsequent releases are tag-and-go.

### 4.4 What's deliberately NOT in these workflows

- **Linting / formatting / type-checking** — REPS already declares
  `black` and `isort` in `[project.optional-dependencies].dev`. These
  could be added to `test.yml`, but adding them to the release path
  blocks release on cosmetic issues. Keep them out for now; add a
  separate `lint.yml` if/when desired.
- **TestPyPI publish** — see verification plan in §6 for how to
  exercise the workflow against TestPyPI before pointing it at PyPI.
- **Auto-bump on merge** — explicitly avoided. Releases are a
  deliberate human action, not a side effect of merging to main.
- **GitHub release creation** — could add
  `softprops/action-gh-release@v2` after the publish step to create a
  GitHub Release with the CHANGELOG section as body. Nice-to-have;
  not load-bearing. Defer until first release ships.

## 5. Sub-spec 4: Pre-1.0 disclaimers

### 5.1 README.md

Add a short subsection under the existing `## Install` heading (or just
above it), per the pattern already endorsed by
[`docs/readme_spec.md` §198](readme_spec.md):

```markdown
## Status: pre-1.0

REPS is pre-1.0. The Python API
([`docs/python_api_spec.md`](docs/python_api_spec.md)) shipped recently
and may still evolve. Per [`docs/release_spec.md`](docs/release_spec.md),
minor version bumps (0.1 → 0.2) may include breaking changes during
the pre-1.0 era. Pin to a specific minor version (e.g.
`reps==0.1.*`) if you need stability across upgrades. Strict semver
applies once REPS reaches 1.0.0.
```

### 5.2 1.0 graduation criteria

Pin them now so we don't argue at the time:

- At least 5 external users (i.e. non-maintainers) have shipped a
  non-trivial program search using `reps.Optimizer.optimize` — "shipped"
  meaning the optimized artifact made it into a real workflow, not
  just a notebook spike.
- No outstanding renames or signature changes the maintainers know they
  want to make. (If the team currently believes `Optimizer.optimize`
  should be `Optimizer.search`, do that rename first, then 1.0.)
- The `reps.internal.*` re-exports list has been audited: anything
  there that we don't actually want to commit to long-term gets moved
  out before 1.0.
- The deferred-to-v1.5 list at the bottom of `python_api_spec.md` has
  been triaged — items that turned out to be load-bearing for real
  users are either shipped or explicitly cut.
- A `LICENSE` file is in place (PyPI requires this regardless, but
  enshrine it as a 1.0 gate too).

When all five hold, cut `1.0.0`.

### 5.3 pyproject.toml additions

Add these fields to the `[project]` table to communicate maturity to
PyPI consumers:

```toml
[project]
name = "reps"
version = "0.1.0"
description = "Recursive Evolutionary Program Search"
readme = "README.md"
requires-python = ">=3.12"
license = { file = "LICENSE" }                      # blocked on packaging spec
authors = [
    { name = "Zach Khorozian", email = "..." },     # email TBD (open decision §7)
]
classifiers = [
    "Development Status :: 3 - Alpha",              # the pre-1.0 signal
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",       # or Apache; depends on packaging spec
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/zkhorozianbc/reps"
Changelog = "https://github.com/zkhorozianbc/reps/blob/main/CHANGELOG.md"
Issues = "https://github.com/zkhorozianbc/reps/issues"
```

The `Development Status :: 3 - Alpha` classifier is the
machine-readable version of the README disclaimer. When 1.0 ships,
flip it to `Development Status :: 5 - Production/Stable`.

The `license`, `authors[].email`, and the precise repo URL are all
gated on decisions outside this spec — the packaging spec owns the
license file, §7 lists the open author / URL questions.

## 6. Verification plan

Before pointing `release.yml` at PyPI, verify each piece:

### 6.1 Local build sanity

```bash
cd /home/user/reps
uv build
ls dist/        # expect: reps-0.1.0.tar.gz, reps-0.1.0-py3-none-any.whl
uv run python -m twine check dist/*    # validates README rendering for PyPI
```

`dist/` already contains `reps-0.1.0.tar.gz` and
`reps-0.1.0-py3-none-any.whl` from a prior local build, so the basic
build path is known to work. Re-running on a clean checkout confirms
reproducibility.

### 6.2 TestPyPI dry run

TestPyPI is a separate index that mirrors PyPI's behavior. Use it to
exercise the full workflow end-to-end without touching the real index.

1. Manually upload `0.1.0` to TestPyPI:

   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/ --token <test-pypi-token>
   ```

2. Configure TestPyPI Trusted Publishing for the GitHub repo (same
   procedure as PyPI, just `test.pypi.org` instead of `pypi.org`).

3. Add a `.github/workflows/release-testpypi.yml` that's identical to
   `release.yml` but with:

   - Trigger: tag matching `v*.*.*-rc.*` (release candidates only).
   - `environment.name: testpypi` and
     `environment.url: https://test.pypi.org/p/reps`.
   - Action input: `repository-url: https://test.pypi.org/legacy/`.

4. Cut a `v0.1.1-rc.1` tag. Confirm the workflow publishes to TestPyPI.

5. Install from TestPyPI to confirm the wheel is sane:

   ```bash
   uv pip install --index-url https://test.pypi.org/simple/ \
                  --extra-index-url https://pypi.org/simple/ \
                  reps==0.1.1rc1
   uv run python -c "import reps; print(reps.Optimizer)"
   ```

   The `--extra-index-url` is needed because TestPyPI doesn't host
   transitive dependencies (`numpy`, `anthropic`, etc.).

If steps 1-5 all succeed, the production `release.yml` will work the
same way against PyPI.

### 6.3 Test workflow validation

Before merging `test.yml`, run it against an open PR to confirm:

- pytest collects all 111 tests.
- The 2 env-key-requiring tests are skipped (cross-reference the
  test_skip spec). If they're not skipped, this PR blocks on that spec
  shipping first.
- Total CI time is reasonable (target: under 3 minutes for the full
  suite).

### 6.4 Tag-vs-pyproject.toml validation

The `release.yml` build job has an explicit version-match check
(§4.2). To verify it works:

- Cut a deliberately-mismatched tag against a sandbox repo (e.g. tag
  `v9.9.9` while `pyproject.toml` says `0.1.0`). Confirm the workflow
  fails at the verification step, not at PyPI upload.

## 7. Open decisions

These are not blockers for this spec but must be resolved before the
first real release:

1. **Author email** — `[project].authors[].email` is required by some
   PyPI consumers and shows up on the project page. Confirm with
   maintainer.
2. **GitHub repository URL** — likely `github.com/zkhorozianbc/reps`
   (per the existing README). Confirm — if the repo moves to an org
   before publish, all references in this spec, the CHANGELOG link
   footer, and `pyproject.toml` `[project.urls]` need updating.
3. **PyPI distribution name** — `reps` is the current `[project].name`
   in `pyproject.toml`. The name may already be taken on PyPI; the
   packaging spec owns the rename decision (`reps-search`,
   `reps-llm`, etc. all candidates). The release workflow refers to
   whatever name lands in `pyproject.toml`, so this is a downstream
   decision.
4. **License file** — required by PyPI; MIT vs Apache-2.0 owned by
   the packaging spec. Update `[project].license` and
   `[project].classifiers` to match once decided.
5. **PyPI Trusted Publishing setup** — see §4.3. One-time manual
   first upload + configure publisher on `pypi.org`. Owned by whoever
   has the PyPI account that will own the project namespace.
6. **TestPyPI account** — recommended for §6.2 dry run; same person
   as #5 typically owns this.
7. **Required reviewers on the `pypi` GitHub environment** — defaults
   to no reviewers (any tag push publishes). Decide whether to require
   maintainer approval as an extra guard.
8. **GitHub Release auto-creation** — see §4.4. Defer until first
   release succeeds.
9. **CHANGELOG enforcement** — optional CI check that PRs touching
   `reps/api/`, `reps/__init__.py`, `reps/config.py`, `reps/internal/`,
   or `reps/evaluation_result.py` also touch `CHANGELOG.md`. Useful
   discipline; defer the actual lint until the workflow is in regular
   use.

## 8. Summary checklist

Files this spec leads to (created in implementation, not by this
spec):

- [ ] `/home/user/reps/CHANGELOG.md` — initial content from §3.3.
- [ ] `/home/user/reps/.github/workflows/test.yml` — content from §4.1.
- [ ] `/home/user/reps/.github/workflows/release.yml` — content from §4.2.
- [ ] `/home/user/reps/README.md` — append "Status: pre-1.0" subsection
      from §5.1.
- [ ] `/home/user/reps/pyproject.toml` — add `[project]` metadata
      fields (`readme`, `license`, `authors`, `classifiers`,
      `[project.urls]`) per §5.3.

Out-of-band setup steps (not files):

- [ ] Manually upload `0.1.0` (or a future first version) to PyPI to
      claim the namespace.
- [ ] Configure PyPI Trusted Publishing for the GitHub repo +
      `release.yml` + `pypi` environment.
- [ ] Create the `pypi` GitHub Environment in repo settings.
- [ ] (Optional) Repeat the above for TestPyPI for dry-run rehearsals.

Once all files land and out-of-band setup is done, releasing is a
three-step ritual: bump `pyproject.toml` + `CHANGELOG.md`, commit,
tag `vX.Y.Z`, push.
